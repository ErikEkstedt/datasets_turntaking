from argparse import ArgumentParser
from os.path import join, expanduser
from os import cpu_count, environ
from typing import Optional, Dict
import math
import random
from copy import deepcopy

# omit verbose `datasets` info
# WARNING: Setting verbosity level by hand...
environ["DATASETS_VERBOSITY"] = "error"

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from omegaconf import OmegaConf
from datasets import concatenate_datasets

from datasets_turntaking.switchboard import load_switchboard
from datasets_turntaking.utils import (
    load_waveform,
    find_island_idx_len,
    repo_root,
    OmegaConfArgs,
    load_config,
)
from datasets_turntaking.features.vad import VadProjection, VAD

# TODO: transforms -> noise
# TODO: Config
# TODO: VAD-history


# Hardcoded for now
# Change these to your swithcboard/CALLHOME audio roots
# Not publicly available to download :(
DATASETS = ["switchboard"]
swb_audio_root = join(expanduser("~"), "projects/data/switchboard/audio")
swb_ext = ".wav"


def get_dialog_audio_datasets(datasets, split):
    """
    Load multiple dataset (of Huggingface `datasets` type) and concatenate to
    a single dataset.
    """
    dsets = []
    for d in datasets:
        if d == "switchboard":
            dsets.append(
                load_switchboard(split, audio_root=swb_audio_root, ext=swb_ext)
            )
        elif d == "callhome":
            raise NotImplementedError("Callhome is not yet implemented")
        else:
            raise NotImplementedError(f"{d} is not yet implemented")
    dsets = concatenate_datasets(dsets)
    return dsets


# For VAD frames
def samples_to_frames(s, hop_len):
    return int(s / hop_len)


def frames_to_time(f, hop_time):
    return f * hop_time


def time_to_frames(t, hop_time):
    return int(t / hop_time)


def time_to_samples(t, sample_rate):
    return int(t * sample_rate)


def get_channel_ipus(
    vad_channel, ipu_pause_frames, ipu_min_frames, audio_context_frames=-1
):
    ipu = deepcopy(vad_channel)
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
    return ends, ipu


def get_ipu_ends(vad, ipu_pause_frames, ipu_min_frames, audio_context_frames=-1):
    ends0, _ = get_channel_ipus(
        vad[:, 0], ipu_pause_frames, ipu_min_frames, audio_context_frames
    )
    ends1, _ = get_channel_ipus(
        vad[:, 1], ipu_pause_frames, ipu_min_frames, audio_context_frames
    )

    # ipu = torch.stack((ipu0, ipu1), dim=-1) # _, ipu0 = get_channel_ipus...
    v = torch.cat((ends0, ends1))
    s = torch.cat((torch.zeros_like(ends0), torch.ones_like(ends1)))
    ipu_ends, perm = v.sort()
    speakers = s[perm]
    return ipu_ends, speakers


def print_dm(data_conf, args):
    print("-" * 60)
    print("Dataloader")
    for k, v in data_conf["dataset"].items():
        print(f"  {k}: {v}")
    print("  batch_size: ", args.batch_size)
    print("  num_workers: ", args.num_workers)
    print()


class DialogIterable(IterableDataset):
    def __init__(
        self,
        dataset,
        audio_mono=True,
        audio_duration=10,
        audio_normalize=True,
        sample_rate=16000,
        vad_hop_time=0.01,
        vad_bin_sizes=[20, 40, 60, 80],
        vad_threshold_ratio=0.5,
        vad_history=False,
        vad_history_times=[60, 30, 10, 5],
        flip_channels=True,
        flip_probability=0.5,
        shuffle=False,
    ):
        super().__init__()
        self.dataset = dataset  # included datasets

        # Audio (waveforms)
        self.audio_mono = audio_mono
        self.audio_duration = audio_duration
        # normalize waveform chunks w/w.abs().max()
        self.audio_normalize = audio_normalize
        # should not normalize audio with less intensity
        self.audio_normalize_threshold = 0.05

        self.flip_channels = flip_channels
        self.flip_probability = flip_probability

        self.sample_rate = sample_rate
        self.shuffle = shuffle

        # Defines the audio chunk size
        self.samples_per_chunk = int(self.audio_duration * self.sample_rate)

        # VAD parameters
        self.vad_hop_time = vad_hop_time  # 0.01s=100Hz
        self.vad_bin_sizes = vad_bin_sizes
        self.vad_threshold_ratio = vad_threshold_ratio
        # samples per frame in the vad-frame representation
        self.vad_hop_samples = int(self.vad_hop_time * self.sample_rate)
        self.vad_frames_per_chunk = samples_to_frames(
            self.samples_per_chunk, hop_len=self.vad_hop_samples
        )

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
            (torch.tensor(vad_history_times) / vad_hop_time).long().tolist()
        )

    def get_indices(self):
        """
        SOURCE: https://pytorch.org/docs/stable/data.html#iterable-style-datasets
        """
        indices = list(range(len(self.dataset)))
        worker_info = get_worker_info()
        if worker_info is not None:
            end = len(indices)
            per_worker = math.ceil((end / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, end)
            indices = indices[iter_start:iter_end]
        return indices

    def _normalize_audio(self, wav):
        if wav.shape[0] > 1:
            if wav[0].abs().max() > self.audio_normalize_threshold:
                wav[0] /= wav[0].abs().max()
            if wav[1].abs().max() > self.audio_normalize_threshold:
                wav[1] /= wav[1].abs().max()
        else:
            if wav.abs().max() > self.audio_normalize_threshold:
                wav /= wav.abs().max()
        return wav

    def load_defaults(self, b):
        # Loads the dialog waveform (stereo) and normalize/to-mono for each
        # smaller segment in loop below
        x, _ = load_waveform(b["audio_path"], sample_rate=self.sample_rate)

        # Extract Vad-frames based on a list of speaker activity
        # channel_vad: [(start, end), (start,end), ...]
        # [ch0_vad, ch1_vad]
        # for both speakers
        # duration of entire dialog
        duration = x.shape[-1] / self.sample_rate
        vad_frames = VAD.vad_list_to_onehot(
            b["vad"],
            sample_rate=self.sample_rate,
            hop_length=self.vad_hop_samples,
            duration=duration,
            channel_last=True,
        )

        vad_history = None
        if self.vad_history:
            # history up until the current features arrive
            vad_history, _ = VAD.get_activity_history(
                vad_frames,
                bin_end_frames=self.vad_history_frames,
                channel_last=True,
            )
        return x, vad_frames, vad_history, duration

    def finalize_sample(
        self, wav, vad_frames, vad_history, focus_speaker, b, f_start, f_end
    ):
        # Normalize over the particular segment
        if self.audio_normalize:
            wav = self._normalize_audio(wav)

        if self.audio_mono:
            wav = wav.mean(dim=0).unsqueeze(0)

        if self.vad_history:
            tmp_vad_history = vad_history[f_start:f_end]

        ###############################################################
        # Include the prediction horizon and pad with zeros
        # if there is no horizon (end of dialog)
        # (+1) includes the last frame
        tmp_vad = vad_frames[f_start : f_end + self.vad_frame_pred + 1, :]
        if tmp_vad.shape[0] < self.vad_frames_per_chunk + self.vad_frame_pred:
            diff = (
                self.vad_frames_per_chunk + 1 + self.vad_frame_pred - tmp_vad.shape[0]
            )
            z = torch.zeros((diff, 2))
            tmp_vad = torch.cat((tmp_vad, z), dim=-2)

        if tmp_vad.shape[0] != self.vad_frames_per_chunk + self.vad_frame_pred + 1:
            print(
                "VAD:",
                tmp_vad.shape[0],
                self.vad_frames_per_chunk,
                self.vad_frame_pred,
            )

        # flip vad and wav prior to label extraction
        if self.flip_channels and torch.rand(1).item() > self.flip_probability:
            if not self.audio_mono:
                wav = torch.stack((wav[1], wav[0]))
            tmp_vad = torch.stack((tmp_vad[:, 1], tmp_vad[:, 0]), dim=-1)
            # Vad history is the residual if flipped
            # i.e.
            # speaker A spoke 1 (100%) -> speaker B spoke 0 (0%)
            # speaker A spoke 0.3 (30%) -> speaker B spoke 0.7 (70%)
            if self.vad_history:
                tmp_vad_history = 1 - tmp_vad_history

        # Extract vad labels using the extra horizon
        # and make sure it is the same size as the "input" vad
        tmp_vad_labels = self.vad_codebook.vad_to_idx(tmp_vad[..., 1:, :])

        # Force size to be of maximum size
        tmp_vad = tmp_vad[..., : self.vad_frames_per_chunk, :]
        tmp_vad_labels = tmp_vad_labels[: self.vad_frames_per_chunk]

        ret = {
            "waveform": wav,
            "vad": tmp_vad,  # add batch dimension
            "vad_label": tmp_vad_labels,
            "speaker": focus_speaker,
            "dataset_name": b["dataset_name"],
            "session": b["session"],
        }

        if self.vad_history:
            ret["vad_history"] = tmp_vad_history

        return ret


class DialogSlidingWindow(DialogIterable):
    def __init__(
        self,
        audio_overlap=5,
        audio_include_ratio=0.4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Time params used for sliding window
        self.audio_overlap = audio_overlap
        self.audio_include_ratio = audio_include_ratio

        # Sample params used for sliding window
        self.samples_overlap = int(self.audio_overlap * self.sample_rate)
        self.samples_step = self.samples_per_chunk - self.samples_overlap

    def _include_end(self, n_samples, samples_per_chunk, step, ratio=0.4):
        """
        Calculate whether to include the last chunk of audio.

        If unfold omits too much samples at the end we will take the last `samples_per_chunk`
        and append to the segments.
        """
        r = (n_samples - samples_per_chunk) / step + 1
        r -= int(r)
        if r >= ratio:
            return True
        return False

    def get_chunks(self, x):
        # Extract chunks of predetermined size
        chunks = x.unfold(
            dimension=-1, size=self.samples_per_chunk, step=self.samples_step
        )

        # VAD
        vs = torch.arange(x.shape[-1])
        vs_chunks = vs.unfold(
            dimension=0, size=self.samples_per_chunk, step=self.samples_step
        )

        # add the end of the audio if 'sufficient' information was omitted
        if self._include_end(
            x.shape[-1],
            self.samples_per_chunk,
            self.samples_step,
            ratio=self.audio_include_ratio,
        ):
            chunks = torch.cat(
                (chunks, x[:, -self.samples_per_chunk :].unsqueeze(1)), dim=1
            )
            vs_chunks = torch.cat(
                (vs_chunks, vs[-self.samples_per_chunk :].unsqueeze(0))
            )

        # only need first sample
        vs_chunks = [v[0] for v in vs_chunks]
        return chunks, vs_chunks

    def get_tmp_wav_and_frames(self, j, chunks, vs_chunks):
        wav = chunks[..., j, :]

        # Get the start/end sample from vs_chunks
        # and transform into frames
        s_start = vs_chunks[j]
        f_start = samples_to_frames(s_start, self.vad_hop_samples)
        f_end = f_start + self.vad_frames_per_chunk
        return wav, f_start, f_end

    def __iter__(self):
        """
        Iterate over overlapping segments in a dialog, over all dialogs in the dataset.
        """

        # Choose the dialog order
        # This will be different across the workers
        # see self.get_indices and the link.
        indices = self.get_indices()
        if self.shuffle:
            random.shuffle(indices)

        # Iterate over all dialogs in the dataset
        # Extract the audio and chunk into segments (and all other processing)
        # Simply yield a single datapoint to the `DataLoder` (recognizes automatically
        # the dataset as an `IterableDataset, no extra effort recquired).
        for i in indices:
            b = self.dataset[i]
            x, vad_frames, vad_history, duration = self.load_defaults(b)
            chunks, vs_chunks = self.get_chunks(x)

            # Iterate over each audio chunk in the dialog
            # and yield to a default DataLoader
            for j in range(chunks.shape[1]):
                wav, f_start, f_end = self.get_tmp_wav_and_frames(j, chunks, vs_chunks)

                # omit if invalid
                if wav is None:
                    continue

                ret = self.finalize_sample(
                    wav,
                    vad_frames,
                    vad_history,
                    focus_speaker=None,
                    b=b,
                    f_start=f_start,
                    f_end=f_end,
                )
                yield ret


class DialogIPU(DialogIterable):
    def __init__(
        self,
        *args,
        audio_context_duration=8,
        ipu_min_time=1,
        ipu_pause_time=0.2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # the datawindow is of size `self.audio_duration`
        # We "center" the end of the IPUs at `self.audio_context_duration`
        self.audio_context_duration = audio_context_duration
        self.audio_context_frames = time_to_frames(
            audio_context_duration, self.vad_hop_time
        )
        self.audio_duration_frames = time_to_frames(
            self.audio_duration, self.vad_hop_time
        )

        # IPU params
        self.ipu_min_time = ipu_min_time
        self.ipu_pause_time = ipu_pause_time
        self.ipu_min_frames = time_to_frames(ipu_min_time, self.vad_hop_time)
        self.ipu_pause_frames = time_to_frames(ipu_pause_time, self.vad_hop_time)

    def get_window_start_end(self, ipu_end):
        start = ipu_end - self.audio_context_duration
        end = start + self.audio_duration
        if start < 0:
            return None, None
        return start, end

    def get_tmp_wav_and_frames(self, focus_frame, x):
        f_start = focus_frame - self.audio_context_frames
        f_end = f_start + self.vad_frames_per_chunk

        # Get times (seconds)
        ipu_end_time = frames_to_time(focus_frame, self.vad_hop_time)
        tmp_start_time, tmp_end_time = self.get_window_start_end(ipu_end_time)

        if tmp_start_time is None:
            return None, None, None
            # continue

        ###############################################################
        # get waveform segment (samples)
        tmp_start_sample = time_to_samples(tmp_start_time, self.sample_rate)
        tmp_end_sample = time_to_samples(tmp_end_time, self.sample_rate)
        wav = x[:, tmp_start_sample:tmp_end_sample]

        # Add silence at end if not entire chunk exists
        if wav.shape[-1] != self.samples_per_chunk:
            diff = wav.shape[-1] - self.samples_per_chunk
            if diff < 0:
                wav = torch.cat((wav, torch.zeros(2, -diff)), dim=-1)
            else:
                wav = wav[:, : self.samples_per_chunk]
            # continue
        return wav, f_start, f_end

    def __iter__(self):
        """
        Iterate over IPU "centered" (not necessary in the middle) sequences, in a dialog, over all dialogs in the dataset.
        """

        # Choose the dialog order
        # This will be different across the workers
        # see self.get_indices and the link.
        indices = self.get_indices()
        if self.shuffle:
            random.shuffle(indices)

        # Iterate over all dialogs in the dataset
        # Extract the audio and chunk into segments (and all other processing)
        # Simply yield a single datapoint to the `DataLoder` (recognizes automatically
        # the dataset as an `IterableDataset, no extra effort recquired).
        for i in indices:
            b = self.dataset[i]
            x, vad_frames, vad_history, duration = self.load_defaults(b)
            # Find IPU
            ipu_ends, speakers = get_ipu_ends(
                vad=vad_frames,
                ipu_pause_frames=self.ipu_pause_frames,
                ipu_min_frames=self.ipu_min_frames,
                audio_context_frames=self.audio_context_frames,
            )

            # IPUs later than 'max_valid_frame' contains too little future information
            max_valid_frame = vad_frames.shape[0] - (
                self.vad_frames_per_chunk - self.audio_context_frames
            )

            # Iterate over each IPU end in the dialog
            # and yield to a default DataLoader
            # omit last IPU
            for focus_frame, focus_speaker in zip(ipu_ends, speakers):
                if focus_frame >= max_valid_frame:
                    # print("Max focus frame reached!")
                    # print("focus_frame: ", focus_frame)
                    # print("max_valid_frame: ", max_valid_frame)
                    continue

                wav, f_start, f_end = self.get_tmp_wav_and_frames(focus_frame, x)

                # omit invalid IPUs
                if wav is None:
                    continue

                ret = self.finalize_sample(
                    wav,
                    vad_frames,
                    vad_history,
                    focus_speaker=focus_speaker,
                    b=b,
                    f_start=f_start,
                    f_end=f_end,
                )
                yield ret


class DialogAudioDM(pl.LightningDataModule):
    def __init__(
        self,
        datasets,
        type="sliding",
        audio_mono=True,
        audio_duration=20,
        audio_normalize=True,
        audio_overlap=5,
        audio_include_ratio=0.4,
        audio_context_duration=8,
        ipu_min_time=1,
        ipu_pause_time=0.2,
        sample_rate=16000,
        vad_hop_time=0.01,
        vad_bin_sizes=[20, 40, 60, 80],
        vad_threshold_ratio=0.5,
        vad_history=False,
        vad_history_times=[60, 30, 10, 5],
        shuffle_training_data=True,
        batch_size=4,
        num_workers=0,
        pin_memory=True,
    ):
        super().__init__()
        self.datasets = datasets  # names of datasets
        self.type = type

        # IterableDataset
        # Audio (waveforms)
        self.audio_mono = audio_mono
        self.audio_duration = audio_duration
        self.sample_rate = sample_rate

        # Sliding Window Dataset
        self.audio_overlap = audio_overlap
        self.audio_normalize = audio_normalize
        self.audio_include_ratio = audio_include_ratio

        # IPU Dataset
        self.audio_context_duration = audio_context_duration
        self.ipu_min_time = ipu_min_time
        self.ipu_pause_time = ipu_pause_time

        # VAD
        self.vad_hop_time = vad_hop_time
        self.vad_bin_sizes = vad_bin_sizes
        self.vad_threshold_ratio = vad_threshold_ratio
        self.vad_history = vad_history
        self.vad_history_times = vad_history_times

        # DataLoder
        self.shuffle_training_data = shuffle_training_data
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers

    @property
    def config(self):
        return OmegaConf.create(
            {
                "datasets": self.datasets,
                "audio_mono": self.audio_mono,
                "audio_duration": self.audio_duration,
                "audio_overlap": self.audio_overlap,
                "audio_normalize": self.audio_normalize,
                "audio_include_ratio": self.audio_include_ratio,
                "audio_context_duration": self.audio_context_duration,
                "ipu_min_time": self.ipu_min_time,
                "ipu_pause_time": self.ipu_pause_time,
                "sample_rate": self.sample_rate,
                "vad_hop_time": self.vad_hop_time,
                "vad_bin_sizes": self.vad_bin_sizes,
                "vad_threshold_ratio": self.vad_threshold_ratio,
                "vad_history": self.vad_history,
                "vad_history_times": self.vad_history_times,
                "shuffle_training_data": self.shuffle_training_data,
            }
        )

    def write_config(self, path="DialogAudioDM_config.yaml"):
        conf = self.config
        OmegaConf.save(config=conf, f=path)

    def prepare_data(self):
        """
        loads the data over all splits.
        Using huggingface datasets which may process/download and cache the data or used cache versions.

        Doing this here to make sure that all the data is accessable before any training, evaluation etc.
        However, uses the same call as is done in `self.setup()`

        So this might be unnecessary given we use Huggingface `datasets` ...

        To avoid the `datasets` logging warnings set `DATASETS_VERBOSITY=error` in your terminal ENV.
        """
        for split in ["train", "validation", "test"]:
            _ = get_dialog_audio_datasets(
                datasets=self.datasets,
                split=split,
            )

    def _dataset(self, dset, shuffle=False):
        if self.type == "ipu":
            dset = DialogIPU(
                dataset=dset,
                audio_mono=self.audio_mono,
                audio_duration=self.audio_duration,
                audio_normalize=self.audio_normalize,
                audio_context_duration=self.audio_context_duration,
                ipu_min_time=self.ipu_min_time,
                ipu_pause_time=self.ipu_pause_time,
                sample_rate=self.sample_rate,
                vad_hop_time=self.vad_hop_time,
                vad_bin_sizes=self.vad_bin_sizes,
                vad_threshold_ratio=self.vad_threshold_ratio,
                vad_history=self.vad_history,
                vad_history_times=self.vad_history_times,
                shuffle=shuffle,
            )
        else:
            dset = DialogSlidingWindow(
                dataset=dset,
                audio_mono=self.audio_mono,
                audio_duration=self.audio_duration,
                audio_overlap=self.audio_overlap,
                audio_normalize=self.audio_normalize,
                audio_include_ratio=self.audio_include_ratio,
                sample_rate=self.sample_rate,
                vad_hop_time=self.vad_hop_time,
                vad_bin_sizes=self.vad_bin_sizes,
                vad_threshold_ratio=self.vad_threshold_ratio,
                vad_history=self.vad_history,
                vad_history_times=self.vad_history_times,
                shuffle=shuffle,
            )
        return dset

    def setup(self, stage: Optional[str] = "fit"):
        """Loads the datasets"""
        if stage == "test":
            test_hf_dataset = get_dialog_audio_datasets(
                datasets=self.datasets, split="test"
            )
            self.test_dset = self._dataset(test_hf_dataset)
        else:  # if stage == "fit" or stage is None:
            train_hf_dataset = get_dialog_audio_datasets(
                datasets=self.datasets, split="train"
            )
            self.train_dset = self._dataset(
                train_hf_dataset, shuffle=self.shuffle_training_data
            )
            val_hf_dataset = get_dialog_audio_datasets(
                datasets=self.datasets, split="val"
            )
            self.val_dset = self._dataset(val_hf_dataset)

    def collate_fn(self, batch):
        waveforms = []
        vad = []
        vad_history = []
        vad_label = []
        dset_names = []
        sessions = []
        for b in batch:
            waveforms.append(b["waveform"])
            dset_names.append(b["dataset_name"])
            sessions.append(b["session"])

            if "vad" in b:
                vad.append(b["vad"])

            if "vad_history" in b:
                vad_history.append(b["vad_history"])

            if "vad_label" in b:
                vad_label.append(b["vad_label"])

        ret = {
            "waveform": torch.cat(waveforms),
            "dset_name": dset_names,
            "session": sessions,
        }
        if len(vad) > 0:
            ret["vad"] = torch.stack(vad)

        if len(vad_history) > 0:
            ret["vad_history"] = torch.stack(vad_history)

        if len(vad_label) > 0:
            ret["vad_label"] = torch.stack(vad_label)

        return ret

    def train_dataloader(self):
        return DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=False,
        )

    @staticmethod
    def default_config_path():
        return join(repo_root(), "config/dset_dialog_audio.yaml")

    @staticmethod
    def load_config(path=None, args=None, format="dict") -> Dict:
        if path is None:
            path = DialogAudioDM.default_config_path()
        return load_config(path, args=args, format=format)

    @staticmethod
    def add_data_specific_args(parent_parser):
        """argparse arguments for SoSIModel (based on yaml-config)"""
        parser = parent_parser.add_argument_group("ULMProjection")
        parser.add_argument("--data_conf", default=None, type=str)
        parser.add_argument("--batch_size", default=4, type=int)
        parser.add_argument("--num_workers", default=cpu_count(), type=int)

        # A workaround for OmegaConf + WandB-Sweeps
        conf = DialogAudioDM.load_config()
        parser = OmegaConfArgs.add_argparse_args(parser, conf)
        return parent_parser


def quick_load_dm(batch_size=1, num_workers=0):
    parser = ArgumentParser()
    parser = DialogAudioDM.add_data_specific_args(parser)
    args = parser.parse_args()
    args.batch_size = batch_size
    args.num_workers = num_workers
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("-" * 50)
    argdict = vars(args)
    dm = DialogAudioDM(
        datasets=argdict["dataset.datasets"],
        audio_duration=argdict["dataset.audio_duration"],
        audio_overlap=argdict["dataset.audio_overlap"],
        audio_normalize=argdict["dataset.audio_normalize"],
        audio_include_ratio=argdict["dataset.audio_include_ratio"],
        sample_rate=argdict["dataset.sample_rate"],
        vad_hop_time=argdict["dataset.vad_hop_time"],
        vad_bin_sizes=argdict["dataset.vad_bin_sizes"],
        batch_size=batch_size,
        num_workers=num_workers,
    )
    dm.prepare_data()
    return dm


def quick_load_dataloader(split="val", batch_size=1, num_workers=0, vad_history=False):
    parser = ArgumentParser()
    parser = DialogAudioDM.add_data_specific_args(parser)
    args = parser.parse_args()
    args.batch_size = batch_size
    args.num_workers = num_workers
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("-" * 50)
    argdict = vars(args)
    dm = DialogAudioDM(
        datasets=argdict["dataset.datasets"],
        audio_duration=argdict["dataset.audio_duration"],
        audio_overlap=argdict["dataset.audio_overlap"],
        audio_normalize=argdict["dataset.audio_normalize"],
        audio_include_ratio=argdict["dataset.audio_include_ratio"],
        sample_rate=argdict["dataset.sample_rate"],
        vad_hop_time=argdict["dataset.vad_hop_time"],
        vad_bin_sizes=argdict["dataset.vad_bin_sizes"],
        vad_history=vad_history,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    dm.prepare_data()
    dloader = None
    if split in ["train", "val", "validation"]:
        dm.setup()
        if split in ["val", "validation"]:
            print("Loaded Validation")
            dloader = dm.val_dataloader()
        else:
            print("Loaded Train")
            dloader = dm.train_dataloader()
    elif split == "test":
        print("Loaded Test")
        dm.setup("test")
        dloader = dm.test_dataloader()
    return dloader


class DEBUG:
    @staticmethod
    def debug_dset_sliding():
        dset_hf = get_dialog_audio_datasets(datasets=["switchboard"], split="val")
        dset = DialogSlidingWindow(
            dataset=dset_hf,
            audio_duration=10,
            audio_overlap=2,
            sample_rate=16000,
            audio_include_ratio=0.4,
            vad_history=True,
        )

        batch = next(iter(dset))
        print("SLIDING")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {tuple(v.shape)}")
            else:
                print(f"{k}: {v}")

    @staticmethod
    def debug_dset_ipu():
        dset_hf = get_dialog_audio_datasets(datasets=["switchboard"], split="val")
        dset = DialogIPU(
            dset_hf,
            audio_duration=10,
            audio_context_duration=8,
            ipu_min_time=0.8,
            ipu_pause_time=0.2,
            vad_history=True,
        )
        print("IPU")
        b = next(iter(dset))
        for k, v in b.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {tuple(v.shape)}")
            else:
                print(f"{k}: {v}")

    @staticmethod
    def debug_dm():
        """
        Sliding:
            train: 94922
            val: 10965

        IPU:
            train: 209531 (~2.2 times overlap 2 seconds)
            val: 23558
        """
        import matplotlib.pyplot as plt
        from tqdm import tqdm
        from datasets_turntaking.features.plot_utils import plot_vad_sample

        parser = ArgumentParser()
        parser = DialogAudioDM.add_data_specific_args(parser)
        args = parser.parse_args()
        data_conf = DialogAudioDM.load_config(path=args.data_conf, args=args)
        data_conf["dataset"]["vad_history"] = True
        data_conf["dataset"]["type"] = "sliding"
        print_dm(data_conf, args)
        dm = DialogAudioDM(
            datasets=data_conf["dataset"]["datasets"],
            type=data_conf["dataset"]["type"],
            audio_duration=data_conf["dataset"]["audio_duration"],
            audio_normalize=data_conf["dataset"]["audio_normalize"],
            audio_overlap=data_conf["dataset"]["audio_overlap"],
            audio_include_ratio=data_conf["dataset"]["audio_include_ratio"],
            audio_context_duration=data_conf["dataset"]["audio_context_duration"],
            ipu_min_time=data_conf["dataset"]["ipu_min_time"],
            ipu_pause_time=data_conf["dataset"]["ipu_pause_time"],
            sample_rate=data_conf["dataset"]["sample_rate"],
            vad_hop_time=data_conf["dataset"]["vad_hop_time"],
            vad_bin_sizes=data_conf["dataset"]["vad_bin_sizes"],
            vad_history=data_conf["dataset"]["vad_history"],
            vad_history_times=data_conf["dataset"]["vad_history_times"],
            batch_size=100,
            num_workers=4,
        )
        dm.prepare_data()
        dm.setup()
        n = 0
        pbar = tqdm(dm.train_dataloader(), desc="N: 0")
        for d in pbar:
            n += d["vad"].shape[0]
            pbar.desc = f"N: {n}"
        print("train: ", n)
        n = 0
        pbar = tqdm(dm.val_dataloader(), desc="N: 0")
        for d in pbar:
            n += d["vad"].shape[0]
            pbar.desc = f"N: {n}"
        print("val: ", n)

        # Try out a dataloder with the awesome iterable dataset
        dloader = dm.val_dataloader()
        diter = iter(dloader)
        batch = next(diter)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {tuple(v.shape)}")
            else:
                print(f"{k}: {v})")

        b = 1
        _ = plot_vad_sample(
            batch["waveform"][b],
            vad=batch["vad"][b].permute(1, 0),
            vad_labels=batch["vad_label"][b],
            sample_rate=16000,
            plot=False,
        )
        # sd.play(batch["waveform"][b], samplerate=16000)
        plt.show()


if __name__ == "__main__":
    DEBUG.debug_dset_sliding()
    # DEBUG.debug_dset_ipu()
