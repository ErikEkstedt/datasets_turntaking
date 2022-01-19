import torch
from torch.utils.data import Dataset

from datasets_turntaking.features.vad import VadProjection, VAD
from datasets_turntaking.utils import load_waveform, get_audio_info, time_to_frames


def vad_list_to_onehot(vad_list, hop_time, duration, channel_last=False):
    n_frames = time_to_frames(duration, hop_time) + 1
    if isinstance(vad_list[0][0], list):
        n_channels = len(vad_list)
        vad_tensor = torch.zeros((n_channels, n_frames))
        for ch, ch_vad in enumerate(vad_list):
            for v in ch_vad:
                s = time_to_frames(v[0], hop_time)
                e = time_to_frames(v[1], hop_time)
                vad_tensor[ch, s:e] = 1.0
    else:
        vad_tensor = torch.zeros((1, n_frames))
        for v in vad_list:
            s = time_to_frames(v[0], hop_time)
            e = time_to_frames(v[1], hop_time)
            vad_tensor[:, s:e] = 1.0

    if channel_last:
        vad_tensor = vad_tensor.permute(1, 0)

    return vad_tensor


class DialogSlidingWindow(Dataset):
    def __init__(
        self,
        dataset,
        feature_extractor=None,
        # AUDIO #################################
        sample_rate=16000,
        audio_mono=True,
        audio_duration=10,
        audio_normalize=True,
        audio_overlap=2,  # Special
        # VAD #################################
        vad_hz=100,
        vad_bin_times=[0.2, 0.4, 0.6, 0.8],
        vad_threshold_ratio=0.5,
        vad_history=False,
        vad_history_times=[60, 30, 10, 5],
        # DSET #################################
        flip_channels=True,
        flip_probability=0.5,
    ):
        super().__init__()
        self.dataset = dataset  # Hugginface datasets
        self.feature_extractor = feature_extractor

        # Audio (waveforms)
        self.sample_rate = sample_rate
        self.audio_mono = audio_mono
        self.audio_duration = audio_duration
        self.audio_overlap = audio_overlap  # TODO: SPECIAL
        self.audio_step_time = audio_duration - audio_overlap
        self.audio_normalize = audio_normalize
        self.audio_normalize_threshold = 0.05

        # VAD parameters
        self.vad_hz = vad_hz
        self.vad_hop_time = 1.0 / vad_hz
        self.vad_bin_times = vad_bin_times
        self.vad_bin_sizes = [
            time_to_frames(vt, hop_time=self.vad_hop_time) for vt in vad_bin_times
        ]
        self.vad_threshold_ratio = vad_threshold_ratio

        # Vad prediction labels
        self.vad_frame_pred = sum(self.vad_bin_sizes)
        self.vad_codebook = VadProjection(
            n_bins=2 * len(self.vad_bin_sizes),
            bin_sizes=self.vad_bin_sizes,
            threshold_ratio=self.vad_threshold_ratio,
        )

        # Vad history
        self.vad_history = vad_history
        self.vad_history_times = vad_history_times
        self.vad_history_frames = (
            (torch.tensor(vad_history_times) / self.vad_hop_time).long().tolist()
        )

        # Dset
        self.flip_channels = flip_channels
        self.flip_probability = flip_probability

        self.map_to_dset_idx, self.map_to_step = self.get_all_indices()

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
        s += f"\n\tvad_bin_times: {self.vad_bin_times}"
        s += f"\n\tvad_bin_sizes: {self.vad_bin_sizes}"
        s += f"\n\tvad_threshold_ratio: {self.vad_threshold_ratio}"

        # Vad prediction labels
        s += f"\n\tvad_frame_pred: {self.vad_frame_pred}"

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

    def get_n_segments(self, duration):
        """
        Number of segments present in a dialog of `duration` seconds.

        We must subtract 1 to get the correct number of segments.
            n = duration / self.audio_step_time
        finds how many START position exists inside the given duration.
        However, we must ensure that the end is present as well.
        """
        return int(duration / self.audio_step_time) - 1

    def get_all_indices(self):
        start = 0
        map_to_dset_idx = []
        map_to_step = []
        for i, path in enumerate(self.dataset["audio_path"]):
            duration = get_audio_info(path)["duration"]
            n_clips = self.get_n_segments(duration)
            end = start + n_clips
            map_to_dset_idx += [i] * (end - start)
            map_to_step += list(range(0, n_clips))
            start += end
        return map_to_dset_idx, map_to_step

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
        start_frame = time_to_frames(start_time, self.vad_hop_time)
        end_frame = time_to_frames(end_time, self.vad_hop_time)

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

        if self.flip_channels and torch.rand(1) > self.flip_probability:
            all_vad_frames = torch.stack(
                (all_vad_frames[:, 1], all_vad_frames[:, 0]), dim=-1
            )
            if not self.audio_mono:
                waveform = torch.stack((waveform[1], waveform[0]))

        # dict to return
        ret = {
            "waveform": waveform,
            "dataset_name": b["dataset_name"],
            "session": b["session"],
        }

        if self.feature_extractor is not None:
            ret["features"] = self.feature_extractor(waveform)

        ##############################################
        # History
        ##############################################
        if self.vad_history:
            # history up until the current features arrive
            vad_history, _ = VAD.get_activity_history(
                all_vad_frames,
                bin_end_frames=self.vad_history_frames,
                channel_last=True,
            )
            ret["vad_history"] = vad_history[start_frame:end_frame].unsqueeze(0)

        ##############################################
        # VAD label
        ##############################################
        ret["vad_label"] = self.vad_codebook.vad_to_idx(
            all_vad_frames[start_frame + 1 : end_frame + self.vad_frame_pred]
        ).unsqueeze(0)

        # only care about relevant part
        ret["vad"] = all_vad_frames[start_frame:end_frame].unsqueeze(0)
        return ret

    def get_start_end_time(self, n_segment):
        start = int(n_segment * self.audio_step_time)
        end = start + self.audio_duration
        return start, end

    def __getitem__(self, idx):
        dset_idx = self.map_to_dset_idx[idx]
        n_segment = self.map_to_step[idx]
        start_time, end_time = self.get_start_end_time(n_segment)
        b = self.dataset[dset_idx]
        d = self.get_sample(b, start_time, end_time)

        if d["waveform"].shape[1] != int(self.audio_duration * self.sample_rate):
            print(b["session"])
            print(d["waveform"].shape)
            print("dset_idx: ", dset_idx)
            print("n_segment: ", n_segment)
        return d


if __name__ == "__main__":

    from datasets_turntaking.dialog_audio.dm_dialog_audio import (
        get_dialog_audio_datasets,
    )
    from datasets_turntaking.features.plot_utils import plot_vad_sample
    import sounddevice as sd
    from tqdm import tqdm

    dset_hf = get_dialog_audio_datasets(datasets=["switchboard"], split="val")
    dset = DialogSlidingWindow(dataset=dset_hf, vad_history=True, vad_hz=50)
    print(dset)
    dset_idx = 19
    n_segment = 36
    idx = 720

    b = dset.dataset[dset_idx]

    idx = 737
    print(dset.map_to_dset_idx[idx])
    print(dset.map_to_step[idx])
    d = dset[idx]

    d = dset

    duration = get_audio_info(b["audio_path"])["duration"]
    print("duration: ", duration)

    n_clips = dset.get_n_segments(duration)

    x, _ = load_waveform(b["audio_path"], sample_rate=16000)

    start_time, end_time = dset.get_start_end_time(n_segment)

    for d in tqdm(dset):
        pass

    # idx = 299
    # d = dset[idx]
    # for k, v in d.items():
    #     if isinstance(v, torch.Tensor):
    #         print(f"{k}: {tuple(v.shape)}")
    #     else:
    #         print(f"{k}: {v}")
    #
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
