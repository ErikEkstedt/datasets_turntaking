from argparse import ArgumentParser
from typing import Optional
from os.path import join, expanduser, exists
from os import listdir, cpu_count
import shutil

import pytorch_lightning as pl
from datasets import concatenate_datasets, load_from_disk
from torch.nn.utils.rnn import pad_sequence

import torch
from torch.utils.data import DataLoader

from datasets_turntaking.utils import load_waveform, time_to_frames_samples
from datasets_turntaking.dataset.speech import load_multiple_datasets

CACHE_PATH = join(expanduser("~"), ".cache/datasets_turntaking/speech")


def get_dataset(
    split="train",
    savepath=None,
    datasets=["lj_speech", "librispeech", "vctk"],
    load_from_cache_file=True,
    overwrite=False,
    debug=False,
):
    if savepath is None:
        savepath = CACHE_PATH

    def create_split(split):
        split_path = join(savepath, "_".join(datasets), split)
        if (
            overwrite
            or not load_from_cache_file
            or not exists(split_path)
            or len(listdir(split_path)) == 0
        ):
            dsets = load_multiple_datasets(datasets, split)
            if debug:
                for dset in dsets:
                    dset = dset.select(range(50))

            # Concatenate
            dataset = concatenate_datasets(dsets)
            dataset.set_format(type="torch")

            # Remove if it exists in order to overwrite
            if overwrite and exists(split_path):
                shutil.rmtree(split_path)
            dataset.save_to_disk(split_path)
        else:
            dataset = load_from_disk(split_path)
        return dataset

    if split == "all":
        dsets = []
        for split in ["train", "val", "test"]:
            dsets.append(create_split(split))
    else:
        dsets = create_split(split)
    return dsets


def get_collate(features=["waveform", "text", "f0", "vad"], sample_rate=8000):
    def collate_fn(batch):
        ret = {
            "id": [],
            "file": [],
            "dataset": [],
        }
        if "waveform" in features:
            ret["waveform"] = []
            ret["n_samples"] = []

        if "f0" in features:
            ret["f0"] = []
            ret["f0_frames"] = []

        if "vad" in features:
            ret["vad"] = []
            ret["vad_frames"] = []

        for b in batch:
            ret["id"].append(b["id"])
            ret["file"].append(b["file"])
            ret["dataset"].append(b["dataset"])

            if "waveform" in features:
                x, _ = load_waveform(b["file"], sample_rate=sample_rate)
                ret["waveform"].append(x.squeeze(0))
                ret["n_samples"].append(ret["waveform"][-1].shape[-1])

            if "f0" in features:
                f0 = torch.load(
                    b["file"].replace(".flac", "_f0.pt").replace(".wav", "_f0.pt")
                )
                ret["f0"].append(f0)
                ret["f0_frames"].append(ret["f0"][-1].shape[-1])

            if "vad" in features:
                vad = torch.load(
                    b["file"].replace(".flac", "_vad.pt").replace(".wav", "_vad.pt")
                )
                ret["vad"].append(vad)
                ret["vad_frames"].append(ret["vad"][-1].shape[-1])

        if "waveform" in features:
            ret["waveform"] = pad_sequence(ret["waveform"], batch_first=True)
            ret["n_samples"] = torch.tensor(ret["n_samples"])

        if "vad" in features:
            ret["vad"] = pad_sequence(ret["vad"], batch_first=True)
            ret["vad_frames"] = torch.tensor(ret["vad_frames"])

        if "f0" in features:
            ret["f0"] = pad_sequence(ret["f0"], batch_first=True)
            ret["f0_frames"] = torch.tensor(ret["f0_frames"])

        return ret

    return collate_fn


class SpeechAudioModule(pl.LightningDataModule):
    DATASETS = ["lj_speech", "librispeech", "vctk"]

    def __init__(
        self,
        sample_rate=8000,
        duration=-1,
        pred_duration=0,
        features=["f0", "vad"],
        savepath=None,
        batch_size=2,
        num_workers=1,
        pin_memory=True,
        overwrite=False,
        datasets=DATASETS,
        load_from_cache_file=True,
        num_proc=1,
        debug=False,
    ):
        super().__init__()
        self.debug = debug
        self.duration = duration
        self.pred_duration = pred_duration
        self.sample_rate = sample_rate

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers

        # `datasets` parameters
        self.load_from_cache_file = load_from_cache_file
        if num_proc > 1:
            print("Not implemented for multiple processes. num_proc -> 1")
            num_proc = 1
        self.num_proc = num_proc

        # Datasets
        if datasets is None:
            datasets = self.DATASETS
        else:
            for dset in datasets:
                assert (
                    dset in self.DATASETS
                ), f"Must prepare dataset to be of correct format. Use {self.DATASETS}"
        self.datasets = datasets
        self.datasets.sort()  # sort for consistency

        if savepath is None:
            savepath = CACHE_PATH
        self.savepath = savepath
        self.overwrite = overwrite

        # F0 Features
        self.features = features
        frame_length = int(self.sample_rate * 0.05)
        hop_length = int(self.sample_rate * 0.02)
        self.f0_params = {
            "fmin": 60,
            "fmax": 500,
            "frame_length": frame_length,
            "hop_length": hop_length,
        }

    def get_split_path(self, split="train"):
        name = f"_vad_f0"
        return join(self.savepath + name, split)

    def prepare_data(self):
        """Concatenates multiple datasets"""
        _ = get_dataset(
            split="all",
            datasets=self.datasets,
            load_from_cache_file=self.load_from_cache_file,
            overwrite=self.overwrite,
            debug=self.debug,
        )

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "test":
            self.test_dset = get_dataset("test", datasets=self.datasets)
        elif stage == "all":
            self.train_dset = get_dataset("train", datasets=self.datasets)
            self.val_dset = get_dataset("val", datasets=self.datasets)
            self.test_dset = get_dataset("test", datasets=self.datasets)
        else:  # if stage == "fit" or stage is None:
            self.train_dset = get_dataset("train", datasets=self.datasets)
            self.val_dset = get_dataset("val", datasets=self.datasets)

    def get_random_segment(self, clip_duration):
        max_end = clip_duration - self.pred_duration
        max_start = max_end - self.duration

        s = torch.rand(1) * max_start
        e = s + self.duration

        pred_end = e + self.pred_duration
        return s, e, pred_end

    def collate_fn(self, batch):
        ret = {
            "id": [],
            "file": [],
            "dataset": [],
            "waveform": [],
            "n_samples": [],
        }

        if "f0" in self.features:
            ret["f0"] = []
            ret["f0_frames"] = []

        if "vad" in self.features:
            ret["vad"] = []
            ret["vad_frames"] = []

        for b in batch:
            ret["id"].append(b["id"])
            ret["file"].append(b["file"])
            ret["dataset"].append(b["dataset"])

            x, _ = load_waveform(b["file"], sample_rate=self.sample_rate)
            clip_duration = x.shape[-1] / self.sample_rate

            start_frame = 0
            end_frame = time_to_frames_samples(
                clip_duration, self.sample_rate, self.f0_params["hop_length"]
            )

            if self.duration > 0:
                # Choose sub clip in segment
                start_time, end_time, pred_end_time = self.get_random_segment(
                    clip_duration
                )
                if end_time < self.duration:
                    continue

                # samples
                start_sample = int(start_time * self.sample_rate)
                end_sample = int(end_time * self.sample_rate)
                ret["waveform"].append(
                    x[0, start_sample:end_sample]
                )  # 0 => only single channel
            else:
                ret["waveform"].append(x[0])  # 0 => only single channel

            ret["n_samples"].append(ret["waveform"][-1].shape[-1])

            # frames
            if self.duration > 0:
                start_frame = time_to_frames_samples(
                    start_time, self.sample_rate, self.f0_params["hop_length"]
                )
                end_frame = time_to_frames_samples(
                    end_time, self.sample_rate, self.f0_params["hop_length"]
                )
                # pred_end_frame = time_to_frames_samples(
                #     pred_end_time, self.sample_rate, self.f0_params["hop_length"]
                # )

            if "f0" in self.features:
                f0 = torch.load(
                    b["file"].replace(".flac", "_f0.pt").replace(".wav", "_f0.pt")
                )
                ret["f0"].append(f0[start_frame:end_frame])  # 0 => only single channel
                ret["f0_frames"].append(ret["f0"][-1].shape[-1])

            if "vad" in self.features:
                vad = torch.load(
                    b["file"].replace(".flac", "_vad.pt").replace(".wav", "_vad.pt")
                )
                ret["vad"].append(
                    vad[start_frame:end_frame]
                )  # 0 => only single channel
                ret["vad_frames"].append(ret["vad"][-1].shape[-1])

        ret["waveform"] = pad_sequence(ret["waveform"], batch_first=True)
        ret["n_samples"] = torch.tensor(ret["n_samples"])

        if "vad" in self.features:
            ret["vad"] = pad_sequence(ret["vad"], batch_first=True)
            ret["vad_frames"] = torch.tensor(ret["vad_frames"])

        if "f0" in self.features:
            ret["f0"] = pad_sequence(ret["f0"], batch_first=True)
            ret["f0_frames"] = torch.tensor(ret["f0_frames"])

        return ret

    def train_dataloader(self):
        return DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        n_cpus = cpu_count()
        parser.add_argument(
            "--datasets", type=str, nargs="+", default=SpeechAudioModule.DATASETS
        )
        parser.add_argument("--duration", default=-1, type=float)
        parser.add_argument(
            "--features", action="store", type=str, nargs="*", default=[]
        )
        parser.add_argument("--batch_size", default=4, type=int)
        parser.add_argument("--num_workers", default=n_cpus, type=int)
        parser.add_argument("--pin_memory", default=True, type=bool)

        # arguments for `datasets` library
        parser.add_argument("--savepath", default=None, type=str)
        parser.add_argument("--overwrite", default=False, type=bool)
        parser.add_argument("--load_from_cache_file", default=True, type=bool)
        parser.add_argument("--num_proc", default=n_cpus, type=int)
        return parser


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from datasets_turntaking.features.plot_utils import plot_waveform

    parser = ArgumentParser()
    parser = SpeechAudioModule.add_data_specific_args(parser)
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    args.batch_size = 16
    dm = SpeechAudioModule(
        features=["f0"],
        datasets=["lj_speech", "librispeech"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        overwrite=args.overwrite,
        num_proc=args.num_proc,
        debug=False,
    )
    dm.prepare_data()
    dm.setup()
    dloader = dm.val_dataloader()
    print("val: ", len(dloader))
    batch = next(iter(dloader))

    for k, v in batch.items():
        if isinstance(v, list):
            print(k, len(v))
        else:
            print(k, v.shape)
            if v.ndim == 1:
                print(v)
