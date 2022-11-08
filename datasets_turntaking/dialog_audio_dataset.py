from os import environ, makedirs
from os.path import join, exists
from torch.utils.data import Dataset
from typing import Any, Callable, Dict, Optional, List, Tuple, Union
from tqdm import tqdm
import torch

import datasets_turntaking.features.functional as DF
import datasets_turntaking.features.transforms as DT
from datasets_turntaking.utils import read_json, write_json

# omit verbose `datasets` info
# WARNING: Setting verbosity level by hand...
environ["DATASETS_VERBOSITY"] = "error"
from datasets import concatenate_datasets


from datasets_turntaking.dataset.spoken_dialog import (
    load_fisher,
    load_switchboard,
    load_vacation_interview,
)
from datasets_turntaking.utils import (
    load_waveform,
    get_audio_info,
    time_to_frames,
    find_island_idx_len,
)
from vap_turn_taking.utils import vad_list_to_onehot, get_activity_history


"""
Dataset speed/memory/distributed

Source: https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662

* [x] Use tensors/arrays instead of lists:
    - https://github.com/pytorch/pytorch/issues/13246#issuecomment-893198671

"""


def load_spoken_dialog_audio_dataset(datasets: List[str], split: str, **kwargs):
    dset = []
    for dataset in datasets:
        if dataset == "fisher":
            dset.append(load_fisher(split=split, format_turns=False, **kwargs))
        elif dataset == "switchboard":
            dset.append(load_switchboard(split=split, format_turns=False))
        elif dataset == "vacation_interview":
            dset.append(load_vacation_interview(split=split))
    assert (
        len(dset) > 0
    ), f"Must load at least one dataset ['fisher', 'switchboard']. Got {datasets}"
    dset = concatenate_datasets(dset)
    return dset


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


def get_sliding_window_indices(dataset, clip_duration: float, audio_step_time: float):
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


def get_full_indices(dataset):
    map_to_dset_idx = []
    map_to_start = []
    for i in range(len(dataset)):
        map_to_dset_idx.append(i)
        map_to_start.append(0)
    return map_to_dset_idx, map_to_start


def get_events_windows(
    dataset,
    clip_duration: float,
    vad_hop_time: float,
    min_context_time: float,
    include_bc: bool = False,
    savepath: str = "data",
):
    """
    https://github.com/pytorch/pytorch/issues/13246#issuecomment-893198671
    """
    from vap_turn_taking.events import TurnTakingEventsNew

    # Can't tell which split we are (we are simply a dataset)
    # so we approximate the "type" of data by num_rows
    dsets = dataset.unique("dataset")
    dsets.sort()
    dsets = "_".join(dsets)  # e.g. fisher_switchboard
    name = dsets + f"_{dataset.num_rows}"
    if include_bc:
        name += f"_bc"
    name += f"_ad{clip_duration}_mc{min_context_time}"

    makedirs(savepath, exist_ok=True)
    filename = join(savepath, name + ".json")
    if exists(filename):
        clips = read_json(filename)
        print("Loaded events datasamples -> ", filename)
        return clips["map_to_dset_idx"], clips["map_to_start"]

    event_conf = {
        "sh_pre_cond_time": 1.0,
        "sh_post_cond_time": 1.0,
        "sh_prediction_region_on_active": True,
        "bc_pre_cond_time": 1.0,
        "bc_post_cond_time": 1.0,
        "bc_max_duration": 1.0,
        "bc_negative_pad_left_time": 1.0,
        "bc_negative_pad_right_time": 2.0,
        "prediction_region_time": 0.5,
        "long_onset_region_time": 0.2,
        "long_onset_condition_time": 1.0,
        "min_context_time": min_context_time,
        "metric_time": 0.05,
        "metric_pad_time": 0.0,
        "max_time": 9999,
        "frame_hz": 50,
        "equal_hold_shift": True,
    }
    eventer = TurnTakingEventsNew(**event_conf)

    map_to_dset_idx = []
    map_to_start = []
    for dataset_idx, d in tqdm(
        enumerate(dataset), total=len(dataset), desc="find events"
    ):

        # d = dataset[1]
        vad_list = d["vad_list"]
        duration = get_audio_info(d["audio_path"])["duration"]
        va, _ = get_vad(
            vad_list=vad_list,
            duration=duration,
            start_time=0,
            end_time=duration,
            hop_time=vad_hop_time,
            horizon_time=0,
            history_include=False,
        )
        events = eventer(va, max_time=duration)

        interesting = events["shift"][0]
        if include_bc:
            interesting += events["short"][0]

        if len(interesting) == 0:
            continue

        interesting.sort()
        interesting = torch.tensor(interesting)[:, :-1]  # 'remove' speaker column
        # convert to time from frames
        interesting = (interesting * vad_hop_time).round()
        # subtract the context
        inter = interesting - min_context_time
        # TODO: Something something add as many events in single clip as possible
        # maybe using: diff = inter[1:] - inter[:-1]
        for start in inter[:, 0]:
            start = start.item()
            if start + clip_duration > duration:
                start = duration - clip_duration
            map_to_dset_idx.append(dataset_idx)
            map_to_start.append(start)
    write_json(
        {"map_to_dset_idx": map_to_dset_idx, "map_to_start": map_to_start}, filename
    )
    print("Saved events datasamples -> ", filename)
    return map_to_dset_idx, map_to_start


# This was not faster than original way...
def vad_list_to_oh(
    vad_list: List[List[Tuple[float, float]]],
    hop_time: float,
    duration: float,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    channel_first: bool = False,
):
    start_time = start_time if start_time is not None else 0.0
    end_time = end_time if end_time is not None else duration
    duration = end_time - start_time
    n_frames = time_to_frames(duration, hop_time)
    vad_tensor = torch.zeros((n_frames, 2))
    if start_time == 0 and end_time == duration:
        # Entire vad requested -> simply iterate over each entry
        for ch, ch_vad in enumerate(vad_list):
            for start, end in ch_vad:
                s = time_to_frames(start, hop_time)
                e = time_to_frames(end, hop_time)
                vad_tensor[s:e, ch] = 1.0
    else:
        for ch in range(2):
            vad_ch_tensor = torch.tensor(vad_list[ch])

            ##########################################
            # Ends prior to start_time are not valid
            ##########################################
            valid_ends = torch.where(vad_ch_tensor[:, 1] > start_time)[0]
            # print("Valid ends: ", valid_ends)
            if len(valid_ends) > 0:
                first_valid_index = valid_ends[0].item()
            else:
                # No entries valid for this channel (default is zero)
                continue

            ##########################################
            # Starts after 'end_time' are not valid
            ##########################################
            valid_starts = torch.where(vad_ch_tensor[:, 0] < end_time)[0]
            if len(valid_starts) > 0:
                last_valid_index = valid_starts[-1].item()
                for start, end in vad_list[ch][
                    first_valid_index : last_valid_index + 1
                ]:
                    # offset with start_time
                    start -= start_time
                    end -= start_time
                    if start < 0:
                        start = 0
                    s = time_to_frames(start, hop_time)
                    e = time_to_frames(end, hop_time)
                    vad_tensor[s:e, ch] = 1.0

    if channel_first:
        vad_tensor = vad_tensor.permute(1, 0)

    return vad_tensor


def get_vad(
    vad_list: List[Any],
    duration: float,
    hop_time: float,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    horizon_time: float = 2,
    history_include: bool = False,
    history_times: List[int] = [60, 30, 10, 5],
) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
    start_frame = time_to_frames(start_time, hop_time)
    end_frame = time_to_frames(end_time, hop_time)
    horizon_frames = time_to_frames(horizon_time, hop_time)

    all_vad_frames = vad_list_to_onehot(
        vad_list,
        hop_time=hop_time,
        duration=duration,
        channel_last=True,
    )

    ##############################################
    # History
    ##############################################
    vah = None
    if history_include:
        history_frames = (torch.tensor(history_times) / hop_time).long().tolist()
        # history up until the current features arrive
        vah, _ = get_activity_history(
            all_vad_frames,
            bin_end_frames=history_frames,
            channel_last=True,
        )
        # vad history is always defined as speaker 0 activity
        vah = vah[start_frame:end_frame][..., 0].unsqueeze(0)

    ##############################################
    # VAD
    ##############################################
    # end_frame + horizon spans after dialog end -> pad with zeros
    if all_vad_frames.shape[0] < end_frame + horizon_frames:
        lookahead = torch.zeros((horizon_frames + 1, 2))
        all_vad_frames = torch.cat((all_vad_frames, lookahead))

    va = all_vad_frames[start_frame : end_frame + horizon_frames]

    # Add batch dimension
    va = va.unsqueeze(0)
    return va, vah


class DialogAudioDataset(Dataset):
    TYPES = ["sliding", "events", "full"]

    def __init__(
        self,
        dataset,
        feature_extractor: Optional[Callable] = None,
        type: str = "sliding",
        # AUDIO #################################
        sample_rate: int = 16000,
        audio_mono: bool = True,
        audio_duration: int = 10,
        audio_normalize: bool = True,  # VAD ################################# vad: bool = True,
        vad_hz: int = 50,
        vad_horizon_time: float = 2,
        vad_history: bool = False,
        vad_history_times: List[int] = [60, 30, 10, 5],
        # Sliding #################################
        audio_overlap: int = 2,  # Sliding Window
        # IPU #################################
        ipu_pause_time: float = 0.1,
        ipu_min_time: float = 0.4,
        audio_context_time: int = 5,
        # DSET #################################
        flip_channels: bool = False,
        flip_probability: float = 0.5,
        mask_vad: bool = False,
        mask_vad_probability: float = 0.5,
        mask_vad_scale: float = 0.1,
        transforms: Optional[Callable] = None,
    ):
        super().__init__()
        self.dataset = dataset  # Hugginface datasets
        self.feature_extractor = feature_extractor
        self.transforms = transforms
        self.dset_type = type

        # Audio (waveforms)
        self.sample_rate = sample_rate
        self.audio_mono = audio_mono
        self.audio_duration = audio_duration
        self.audio_overlap = audio_overlap  # TODO: SPECIAL
        self.audio_step_time = audio_duration - audio_overlap
        self.audio_normalize = audio_normalize
        self.audio_normalize_threshold = 0.05
        self.n_samples = sample_rate * audio_duration

        # VAD parameters
        self.vad = True  # use vad or not
        self.vad_hz = vad_hz
        self.vad_hop_time = 1.0 / vad_hz

        # Vad prediction labels
        self.horizon_time = vad_horizon_time
        self.vad_horizon = time_to_frames(vad_horizon_time, hop_time=self.vad_hop_time)
        # Vad history
        self.vad_history = vad_history
        self.vad_history_times = vad_history_times

        # IPU
        self.ipu_pause_time = ipu_pause_time
        self.ipu_min_time = ipu_min_time
        self.audio_context_time = audio_context_time

        # Dset
        self.flip_channels = flip_channels
        self.flip_probability = flip_probability
        if flip_channels:
            self.batch_flipper = DT.FlipBatch()

        self.mask_vad = mask_vad
        self.mask_vad_probability = mask_vad_probability
        self.mask_vad_scale = mask_vad_scale

        self.map_to_dset_idx, self.map_to_start_time = self.get_sample_maps(type)

    def get_sample_maps(
        self, type: str = "sliding"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if type == "ipu":
            map_to_dset_idx, map_to_start_time = get_ipu_indices(
                self.dataset,
                clip_duration=self.audio_duration,
                vad_hop_time=self.vad_hop_time,
                ipu_pause_time=self.ipu_pause_time,
                ipu_min_time=self.ipu_min_time,
                audio_context_time=self.audio_context_time,
            )
        elif type == "events":
            map_to_dset_idx, map_to_start_time = get_events_windows(
                self.dataset,
                clip_duration=self.audio_duration,
                vad_hop_time=self.vad_hop_time,
                min_context_time=self.audio_duration // 2,
            )
        elif type == "full":
            map_to_dset_idx, map_to_start_time = get_full_indices(self.dataset)
        else:
            map_to_dset_idx, map_to_start_time = get_sliding_window_indices(
                self.dataset, self.audio_duration, self.audio_step_time
            )
        return torch.tensor(map_to_dset_idx), torch.tensor(map_to_start_time)

    def __repr__(self) -> str:
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
        s += f"\n\tvad_horizon: {self.vad_horizon}"
        s += f"\n\tvad_history: {self.vad_history}"
        s += f"\n\tvad_history_times: {self.vad_history_times}"

        # Dset
        s += f"\n\tflip_channels: {self.flip_channels}"
        s += f"\n\tflip_probability: {self.flip_probability}"
        s += f"\n\tmask_vad: {self.mask_vad}"
        s += f"\n\tmask_vad_probability: {self.mask_vad_probability}"
        s += "\n" + "-" * 40
        return s

    def __len__(self) -> int:
        return len(self.map_to_dset_idx)

    def get_dialog_sample(self, idx) -> Dict[str, Any]:
        d = self.dataset[idx]
        return self.get_sample(d)

    def get_sample(
        self, b, start_time: Optional[float] = None, end_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """Get the sample from the dialog"""
        # dict to return
        ret = {
            "dataset": b["dataset"],
            "session": b["session"],
        }
        duration = get_audio_info(b["audio_path"])["duration"]

        if start_time is None:
            start_time = 0

        if end_time is None:
            end_time = duration

        # Loads the dialog waveform (stereo) and normalize/to-mono for each
        # smaller segment in loop below
        ret["waveform"], _ = load_waveform(
            b["audio_path"],
            sample_rate=self.sample_rate,
            start_time=start_time,
            end_time=end_time,
            mono=self.audio_mono,
        )
        # TODO: why did an entry yield 32_000_2 instead of 32_000_0?

        if not self.dset_type == "full":
            ret["waveform"] = ret["waveform"][..., : self.n_samples]

        # VAD-frame of relevant part
        if self.vad:
            va, vah = get_vad(
                vad_list=b["vad_list"],
                duration=duration,
                start_time=start_time,
                end_time=end_time,
                hop_time=self.vad_hop_time,
                horizon_time=self.horizon_time,
                history_include=self.vad_history,
                history_times=self.vad_history_times,
            )

            ret["vad"] = va
            if vah is not None:
                ret["vad_history"] = vah

        # add batch dim (2, n_samples) -> (1, 2, n_samples) or (1, n_samples) -> (1, 1, n_samples)
        ret["waveform"] = ret["waveform"].unsqueeze(0)
        if self.feature_extractor is not None:
            ret["features"] = self.feature_extractor(ret["waveform"])

        return ret

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a dict with the following keys:
            waveform:       torch.Tensor, (1, n_channels, n_samples)
            vad:            torch.Tensor, (1, n_frames+vad_horizon_frames, n_channels)
            vad_history:    Optional, torch.Tensor, (1, n_frames, len(vad_history_times))
            dataset:        str, name of dataset
            session:        str, name/id of session
        """
        dset_idx = self.map_to_dset_idx[idx].item()
        start_time = self.map_to_start_time[idx].item()
        end_time = (
            None if self.dset_type == "full" else start_time + self.audio_duration
        )
        b = self.dataset[dset_idx]
        d = self.get_sample(b, start_time, end_time)

        if self.mask_vad and torch.rand(1) <= self.mask_vad_probability:
            d["waveform"] = DF.mask_around_vad(
                d["waveform"],
                d["vad"],
                vad_hz=self.vad_hz,
                sample_rate=self.sample_rate,
                scale=self.mask_vad_scale,
            )

        if self.transforms is not None:
            # n_frames = d["vad"].shape[1] - self.vad_horizon
            # d["waveform"] = self.transforms(d["waveform"], vad=d["vad"][:, :n_frames])
            d["waveform"] = self.transforms(d["waveform"])

        if self.flip_channels and torch.rand(1) <= self.flip_probability:
            d = self.batch_flipper(d)

        return d


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from datasets_turntaking.features.plot_utils import plot_batch_sample

    dataset = load_spoken_dialog_audio_dataset(
        # ["switchboard", "fisher"], split="train", min_word_vad_diff=0.1
        ["vacation_interview"],
        split="train",
        min_word_vad_diff=0.1,
    )
    dset = DialogAudioDataset(
        dataset=dataset,
        # type="sliding",
        # type="events",
        type="full",
        vad_history=False,
        vad_hz=50,
        audio_mono=False,
        mask_vad=False,
        mask_vad_probability=0.5,
    )
    d = dset[0]
    print("waveform: ", tuple(d["waveform"].shape))

    print("dset: ", dset)
    print("Length: ", len(dset))
    for idx in range(10):
        # idx = int(torch.randint(0, len(dset), (1,)).item())
        batch = dset[idx]
        print(idx)
        # w = DF.mask_around_vad(
        #     batch["waveform"], batch["vad"][:, :-100], vad_hz=50, sample_rate=16000
        # )
        # w = masker(batch["waveform"], batch["vad"])
        # batch['waveform'] = waveform_mask_with_vad(batch['waveform'], batch['vad'][:, :-100])
        # fig, ax = plot_batch_sample(
        #     waveform=batch["waveform"][0],
        #     # waveform=w[0],
        #     vad=batch["vad"][0, :-100],
        #     sample_rate=dset.sample_rate,
        #     plot=False,
        # )
        # # sd.play(batch["waveform"][0].t().numpy(), samplerate=16000)
        # plt.show()
