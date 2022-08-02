import torch
from torch.utils.data import Dataset
from datasets_turntaking.utils import (
    load_waveform,
    get_audio_info,
    time_to_frames,
    find_island_idx_len,
)

from vap_turn_taking.utils import vad_list_to_onehot, get_activity_history


def get_ipu_ends(
    vad,
    ipu_pause_frames,
    ipu_min_frames,
    audio_duration_frames,
    audio_context_frames=-1,
):
    def get_channel_ipu_ends(vad_channel):
        """get ipus only based on a single channel"""
        # ipu = deepcopy(vad_channel)
        ipu = vad_channel
        n_frames = vad_channel.shape[0]
        starts, dur, v = find_island_idx_len(ipu)

        # Pause silences below threshold (ipu_pause_frames)
        # are filled to join vad-segments to IPU
        pause_starts = starts[v == 0]
        pause_dur = dur[v == 0]
        fill_starts = pause_starts[pause_dur < ipu_pause_frames]
        fill_dur = pause_dur[pause_dur < ipu_pause_frames]

        # Fill silences below `ipu_pause_frames`
        for s, d in zip(fill_starts, fill_dur):
            ipu[s : s + d] = 1

        # get new values for the filled "ipus"
        starts, dur, v = find_island_idx_len(ipu)
        # focus on the active segments (vadvalue => 1)
        starts = starts[v == 1]
        dur = dur[v == 1]

        # check which IPU segments are above the threshold
        keep = dur >= ipu_min_frames
        starts = starts[keep]
        dur = dur[keep]
        ends = starts + dur

        # check that the end is not before the required audio context
        if audio_context_frames > 0:
            keep = ends >= audio_context_frames
            ends = ends[keep]

        # remove ipus to close to end of dialog (no future information)

        max_frame = n_frames - (audio_duration_frames - audio_context_frames)
        keep = ends < max_frame
        ends = ends[keep]
        return ends

    ends0 = get_channel_ipu_ends(vad[:, 0])
    ends1 = get_channel_ipu_ends(vad[:, 1])

    # ipu = torch.stack((ipu0, ipu1), dim=-1) # _, ipu0 = get_channel_ipus...
    v = torch.cat((ends0, ends1))
    s = torch.cat((torch.zeros_like(ends0), torch.ones_like(ends1)))
    ipu_ends, perm = v.sort()
    speakers = s[perm]
    return ipu_ends, speakers


def get_ipu_indices(
    dataset,
    clip_duration,
    vad_hop_time,
    ipu_pause_time,
    ipu_min_time,
    audio_context_time,
):
    ipu_pause_frames = int(ipu_pause_time / vad_hop_time)
    ipu_min_frames = int(ipu_min_time / vad_hop_time)
    audio_context_frames = int(audio_context_time / vad_hop_time)
    audio_duration_frames = int(clip_duration / vad_hop_time)
    # print("ipu_pause_frames: ", ipu_pause_frames)
    # print("ipu_min_frames: ", ipu_min_frames)
    # print("audio_context_frames: ", audio_context_frames)

    map_to_dset_idx = []
    map_to_start = []
    for i, (audio_path, vad) in enumerate(zip(dataset["audio_path"], dataset["vad"])):
        duration = get_audio_info(audio_path)["duration"]
        vad_frames = vad_list_to_onehot(
            vad,
            hop_time=vad_hop_time,
            duration=duration,
            channel_last=True,
        )
        ipu_ends, _ = get_ipu_ends(
            vad=vad_frames,
            ipu_pause_frames=ipu_pause_frames,
            ipu_min_frames=ipu_min_frames,
            audio_duration_frames=audio_duration_frames,
            audio_context_frames=audio_context_frames,
        )

        ipu_end_time = ipu_ends * vad_hop_time
        start_time = ipu_end_time - audio_context_time

        # end_time = start_time + clip_duration
        map_to_dset_idx += [i] * start_time.shape[0]
        map_to_start += start_time.tolist()
    return map_to_dset_idx, map_to_start


def get_sliding_window_indices(dataset, clip_duration, audio_step_time):
    def get_n_segments(duration):
        """Number of segments present in a dialog of `duration` seconds."""
        return int((duration - clip_duration) / audio_step_time + 1)

    start = 0
    map_to_dset_idx = []
    # map_to_step = []
    map_to_start = []
    for i, path in enumerate(dataset["audio_path"]):
        duration = get_audio_info(path)["duration"]
        n_clips = get_n_segments(duration)
        end = start + n_clips
        map_to_dset_idx += [i] * (end - start)
        # map_to_step += list(range(0, n_clips))
        map_to_start += torch.arange(0, duration, audio_step_time)[:n_clips].tolist()
        start += end
    return map_to_dset_idx, map_to_start


class DialogAudioDataset(Dataset):
    def __init__(
        self,
        dataset,
        feature_extractor=None,
        type="sliding",
        # AUDIO #################################
        sample_rate=16000,
        audio_mono=True,
        audio_duration=10,
        audio_normalize=True,
        # VAD #################################
        vad=True,
        vad_hz=100,
        vad_horizon=2,
        vad_history=False,
        vad_history_times=[60, 30, 10, 5],
        # Sliding #################################
        audio_overlap=2,  # Sliding Window
        # IPU #################################
        ipu_pause_time=0.1,
        ipu_min_time=0.4,
        audio_context_time=5,
        # DSET #################################
        flip_channels=True,
        flip_probability=0.5,
        transforms=None,
    ):
        super().__init__()
        self.dataset = dataset  # Hugginface datasets
        self.feature_extractor = feature_extractor
        self.transforms = transforms

        # Audio (waveforms)
        self.sample_rate = sample_rate
        self.audio_mono = audio_mono
        self.audio_duration = audio_duration
        self.audio_overlap = audio_overlap  # TODO: SPECIAL
        self.audio_step_time = audio_duration - audio_overlap
        self.audio_normalize = audio_normalize
        self.audio_normalize_threshold = 0.05

        # VAD parameters
        self.vad = vad  # use vad or not
        self.vad_hz = vad_hz
        self.vad_hop_time = 1.0 / vad_hz

        # Vad prediction labels
        self.horizon_time = vad_horizon
        self.vad_horizon = time_to_frames(vad_horizon, hop_time=self.vad_hop_time)

        # Vad history
        self.vad_history = vad_history
        self.vad_history_times = vad_history_times
        self.vad_history_frames = (
            (torch.tensor(vad_history_times) / self.vad_hop_time).long().tolist()
        )

        # IPU
        self.ipu_pause_time = ipu_pause_time
        self.ipu_min_time = ipu_min_time
        self.audio_context_time = audio_context_time

        # Dset
        self.flip_channels = flip_channels
        self.flip_probability = flip_probability

        self.map_to_dset_idx, self.map_to_start_time = self.get_sample_maps(type)

    def get_sample_maps(self, type="sliding"):
        if type == "ipu":
            map_to_dset_idx, map_to_start_time = get_ipu_indices(
                self.dataset,
                clip_duration=self.audio_duration,
                vad_hop_time=self.vad_hop_time,
                ipu_pause_time=self.ipu_pause_time,
                ipu_min_time=self.ipu_min_time,
                audio_context_time=self.audio_context_time,
            )
        else:
            map_to_dset_idx, map_to_start_time = get_sliding_window_indices(
                self.dataset, self.audio_duration, self.audio_step_time
            )
        return map_to_dset_idx, map_to_start_time

    def __repr__(self):
        s = "DialogSlidingWindow"
        s += f"\n\tsample_rate: {self.sample_rate}"
        s += f"\n\taudio_mono: {self.audio_mono}"
        s += f"\n\taudio_duration: {self.audio_duration}"
        s += f"\n\taudio_overlap: {self.audio_overlap}"
        s += f"\n\taudio_step_time: {self.audio_step_time}"
        s += f"\n\taudio_normalize: {self.audio_normalize}"
        s += f"\n\taudio_normalize_threshold: {self.audio_normalize_threshold}"

        # VAD parameters
        s += f"\n\tvad_hz: {self.vad_hz}"
        s += f"\n\tvad_hop_time: {self.vad_hop_time}"

        # Vad prediction labels
        s += f"\n\tvad_horizon: {self.vad_horizon}"

        # Vad history
        s += f"\n\tvad_history: {self.vad_history}"
        s += f"\n\tvad_history_times: {self.vad_history_times}"
        s += f"\n\tvad_history_frames: {self.vad_history_frames}"

        # Dset
        s += f"\n\tflip_channels: {self.flip_channels}"
        s += f"\n\tflip_probability: {self.flip_probability}"
        s += "\n" + "-" * 40
        return s

    def __len__(self):
        return len(self.map_to_dset_idx)

    def get_dialog_sample(self, idx, flip=False):
        d = self.dataset[idx]
        return self.get_full_sample(d, flip=flip)

    def get_full_sample(self, b, flip=False):
        """Get the sample from the dialog"""
        # Loads the dialog waveform (stereo) and normalize/to-mono for each
        # smaller segment in loop below
        waveform, _ = load_waveform(
            b["audio_path"],
            sample_rate=self.sample_rate,
            normalize=self.audio_normalize,
            mono=self.audio_mono,
        )

        if self.vad:
            # TODO: extract relevant vad directly
            # Extract Vad-frames based on a list of speaker activity
            # channel_vad: [(start, end), (start,end), ...]
            # [ch0_vad, ch1_vad]
            # for both speakers
            # duration of entire dialog
            duration = get_audio_info(b["audio_path"])["duration"]
            all_vad_frames = vad_list_to_onehot(
                b["vad"],
                hop_time=self.vad_hop_time,
                duration=duration,
                channel_last=True,
            )
            if flip:
                all_vad_frames = torch.stack(
                    (all_vad_frames[:, 1], all_vad_frames[:, 0]), dim=-1
                )

        if flip:
            if not self.audio_mono:
                waveform = torch.stack((waveform[1], waveform[0]))

        # dict to return
        ret = {
            "waveform": waveform,
            "dataset": [b["dataset"]],
            "session": [b["session"]],
        }

        if self.feature_extractor is not None:
            ret["features"] = self.feature_extractor(waveform)

        ##############################################
        # History
        ##############################################
        if self.vad and self.vad_history:
            # history up until the current features arrive
            vh, _ = get_activity_history(
                all_vad_frames,
                bin_end_frames=self.vad_history_frames,
                channel_last=True,
            )
            # We define the history as the ratio of speaker A (=0)
            ret["vad_history"] = vh[..., 0].unsqueeze(0)

        ##############################################
        # VAD
        ##############################################
        # add "silent" lookahead
        # add horizon after end (silence)
        if self.vad:
            lookahead = torch.zeros((self.vad_horizon, 2))
            ret["vad"] = torch.cat((all_vad_frames, lookahead)).unsqueeze(0)
        return ret

    def dialog_to_batch(self, d, audio_overlap=5, audio_duration=10, batch_size=16):
        step = audio_duration - audio_overlap

        # waveform sizes
        sample_size = int(self.sample_rate * audio_duration)
        sample_step = int(self.sample_rate * step)

        # VA sizes
        vh_size = int(self.vad_hz * audio_duration)
        vf_size = int(self.vad_hz * (audio_duration + self.horizon_time))
        v_step = int(self.vad_hz * step)

        # we assume that all data contains a batch dimension
        w = d["waveform"][0].unfold(dimension=0, size=sample_size, step=sample_step)
        vf = d["vad"][0].unfold(dimension=0, size=vf_size, step=v_step).permute(0, 2, 1)
        vh = (
            d["vad_history"][0]
            .unfold(dimension=0, size=vh_size, step=v_step)
            .permute(0, 2, 1)
        )
        # print("w: ", tuple(w.shape))
        # print("vf: ", tuple(vf.shape))  # may be one extra step
        # print("vh: ", tuple(vh.shape))

        assert (
            w.shape[0] == vf.shape[0]
        ), f"w != vf | {w.shape} != {vf.shape} -> Not the same number of samples"
        assert (
            vf.shape[0] == vh.shape[0]
        ), f"vf != vh | {vf.shape} != {vh.shape} -> Not the same number of samples"

        n_samples = w.shape[0]
        if n_samples <= batch_size:
            batch = {"waveform": w, "vad": vf, "vad_history": vh}
            batches = [batch]
            return batches

        # split into batches of appropriate sizes

        n_batch_samples = 0
        batches = []
        for s in range(0, n_samples, batch_size):
            e = s + batch_size
            vf_tmp = vf[s:e]
            n_batch_samples += vf_tmp.shape[0]
            batches.append(
                {
                    "waveform": w[s:e],
                    "vad": vf_tmp,
                    "vad_history": vh[s:e],
                }
            )
        assert n_batch_samples == n_samples, "Did not get  all samples"

        return batches

    def get_sample(self, b, start_time, end_time):
        """Get the sample from the dialog"""
        # Loads the dialog waveform (stereo) and normalize/to-mono for each
        # smaller segment in loop below
        waveform, _ = load_waveform(
            b["audio_path"],
            sample_rate=self.sample_rate,
            start_time=start_time,
            end_time=end_time,
            normalize=self.audio_normalize,
            mono=self.audio_mono,
        )

        # VAD-frame of relevant part
        if self.vad:
            start_frame = time_to_frames(start_time, self.vad_hop_time)
            end_frame = time_to_frames(end_time, self.vad_hop_time)
            duration = get_audio_info(b["audio_path"])["duration"]

            # TODO: extract relevant vad directly
            # Extract Vad-frames based on a list of speaker activity
            # channel_vad: [(start, end), (start,end), ...]
            # [ch0_vad, ch1_vad]
            # for both speakers
            # duration of entire dialog
            all_vad_frames = vad_list_to_onehot(
                b["vad"],
                hop_time=self.vad_hop_time,
                duration=duration,
                channel_last=True,
            )

        if self.flip_channels and torch.rand(1) > self.flip_probability:
            if self.vad:
                all_vad_frames = torch.stack(
                    (all_vad_frames[:, 1], all_vad_frames[:, 0]), dim=-1
                )
            if not self.audio_mono:
                waveform = torch.stack((waveform[1], waveform[0]))

        if not self.audio_mono:
            waveform = waveform.unsqueeze(
                0
            )  # add batch dim (2, n_samples) -> (1, 2, n_samples)

        # dict to return
        ret = {
            "waveform": waveform,
            "dataset": b["dataset"],
            "session": b["session"],
        }

        if self.feature_extractor is not None:
            ret["features"] = self.feature_extractor(waveform)

        ##############################################
        # History
        ##############################################
        if self.vad and self.vad_history:
            # history up until the current features arrive
            vad_history, _ = get_activity_history(
                all_vad_frames,
                bin_end_frames=self.vad_history_frames,
                channel_last=True,
            )
            # ret["vad_history"] = vad_history[start_frame:end_frame].unsqueeze(0)
            # vad history is always defined as speaker 0 activity
            ret["vad_history"] = vad_history[start_frame:end_frame][..., 0].unsqueeze(0)

        ##############################################
        # VAD label
        ##############################################
        # time with vadlabel:   32 batch 5.027
        # time without vadlabel: 32 batch 2.666

        ##############################################
        # VAD
        ##############################################
        if self.vad:
            if end_frame + self.vad_horizon > all_vad_frames.shape[0]:
                lookahead = torch.zeros(
                    (self.vad_horizon + 1, 2)
                )  # add horizon after end (silence)
                all_vad_frames = torch.cat((all_vad_frames, lookahead))
            ret["vad"] = all_vad_frames[
                start_frame : end_frame + self.vad_horizon
            ].unsqueeze(0)
        return ret

    def __getitem__(self, idx):
        dset_idx = self.map_to_dset_idx[idx]
        start_time = self.map_to_start_time[idx]
        end_time = start_time + self.audio_duration
        b = self.dataset[dset_idx]
        d = self.get_sample(b, start_time, end_time)

        if self.transforms is not None:
            n_frames = d["vad_history"].shape[1]
            vad = d["vad"][:, :n_frames]
            d["waveform"] = self.transforms(d["waveform"], vad=vad)
        return d


if __name__ == "__main__":
    from datasets_turntaking.dialog_audio_dm import get_dialog_audio_datasets

    # from datasets_turntaking.features.plot_utils import plot_vad_sample
    # import sounddevice as sd
    # from tqdm import tqdm

    # dset_hf = get_dialog_audio_datasets(datasets=["switchboard"], split="val")
    dset_hf = get_dialog_audio_datasets(
        datasets=["fisher", "switchboard"], split="train"
    )

    dset = DialogAudioDataset(
        dataset=dset_hf, type="sliding", vad_history=True, vad_hz=50
    )
    # dset = DialogAudioDataset(dataset=dset_hf, type='ipu', vad_history=True, vad_hz=50)
    print(dset)
    print("N: ", len(dset))

    batch = dset[100]
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")

    d = dset_hf[0]
    end_time = 180
    n = torch.tensor(d["dialog"]["end"])
    n = n[n <= end_time]
    n = len(n)
    speaker = d["dialog"]["speaker"][:n]
    text = d["dialog"]["text"][:n]
    start = d["dialog"]["start"][:n]
    end = d["dialog"]["end"][:n]

    idx = 299
    d = dset[idx]
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")

    # fig, ax = plot_vad_sample(
    #     waveform=d["waveform"][0],
    #     vad=d["vad"][0].t(),
    #     vad_labels=d["vad_label"][0],
    #     vad_current_frame=None,
    #     vad_bins=256,
    #     sample_rate=dset.sample_rate,
    #     ax=None,
    #     figsize=(16, 5),
    #     plot=True,
    # )
    # sd.play(d["waveform"][0], samplerate=dset.sample_rate)
