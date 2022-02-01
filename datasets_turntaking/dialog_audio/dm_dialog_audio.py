from os import cpu_count, environ
from os.path import join
from typing import Optional, Dict

# omit verbose `datasets` info
# WARNING: Setting verbosity level by hand...
environ["DATASETS_VERBOSITY"] = "error"

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from datasets import concatenate_datasets

from datasets_turntaking.callhome import load_callhome
from datasets_turntaking.switchboard import load_switchboard
from datasets_turntaking.dialog_audio.dataset import DialogAudioDataset
from datasets_turntaking.utils import repo_root, OmegaConfArgs, load_config


DEFAULT_CONFIG = join(repo_root(), "config/dset_dialog_audio.yaml")


def get_dialog_audio_datasets(datasets, split):
    """
    Load multiple dataset (of Huggingface `datasets` type) and concatenate to
    a single dataset.
    """
    dsets = []
    for d in datasets:
        if d == "switchboard":
            dsets.append(load_switchboard(split))
        elif d == "callhome":
            dsets.append(load_callhome(split))
        else:
            raise NotImplementedError(f"{d} is not yet implemented")
    dsets = concatenate_datasets(dsets)
    return dsets


class DialogAudioDM(pl.LightningDataModule):
    def __init__(
        self,
        datasets,
        type="sliding",  # ipu
        audio_mono=True,
        audio_duration=10,
        audio_normalize=True,
        audio_overlap=2,
        audio_include_ratio=0.4,
        audio_context_duration=8,
        ipu_min_time=1,
        ipu_pause_time=0.2,
        sample_rate=16000,
        vad_hz=100,
        vad_bin_times=[0.2, 0.4, 0.6, 0.8],
        vad_threshold_ratio=0.5,
        vad_history=False,
        vad_history_times=[60, 30, 10, 5],
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
        self.vad_hz = vad_hz
        self.vad_bin_times = vad_bin_times
        self.vad_threshold_ratio = vad_threshold_ratio
        self.vad_history = vad_history
        self.vad_history_times = vad_history_times

        # DataLoder
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers

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

    def _dataset(self, dset):
        return DialogAudioDataset(
            dataset=dset,
            feature_extractor=None,
            type=self.type,
            audio_mono=self.audio_mono,
            audio_duration=self.audio_duration,
            audio_overlap=self.audio_overlap,
            audio_normalize=self.audio_normalize,
            sample_rate=self.sample_rate,
            vad_hz=self.vad_hz,
            vad_bin_times=self.vad_bin_times,
            vad_threshold_ratio=self.vad_threshold_ratio,
            vad_history=self.vad_history,
            vad_history_times=self.vad_history_times,
            flip_channels=True,
            flip_probability=0.5,
        )

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
            self.train_dset = self._dataset(train_hf_dataset)
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
            ret["vad"] = torch.cat(vad)

        if len(vad_history) > 0:
            ret["vad_history"] = torch.cat(vad_history)

        if len(vad_label) > 0:
            ret["vad_label"] = torch.cat(vad_label)

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

    def __repr__(self):
        s = "DialogAudioDM"
        s += f"\n\tbatch_size: {self.batch_size}"
        s += f"\n\tpin_memory: {self.pin_memory}"
        s += f"\n\tnum_workers: {self.num_workers}"

        if hasattr(self, "train_dset"):
            s += "\n\t" + ("-" * 10) + "\n"
            s += str(self.train_dset)
        elif hasattr(self, "test_dset"):
            s += "\n\t" + ("-" * 10) + "\n"
            s += str(self.train_dset)
        return s

    @staticmethod
    def print_dm(data_conf, args=None):
        print("-" * 60)
        print("Dataloader")
        for k, v in data_conf["dataset"].items():
            print(f"  {k}: {v}")
        if args is not None:
            print("  batch_size: ", args.batch_size)
            print("  num_workers: ", args.num_workers)
        print()

    @staticmethod
    def default_config_path():
        return DEFAULT_CONFIG

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


if __name__ == "__main__":

    data_conf = DialogAudioDM.load_config()
    DialogAudioDM.print_dm(data_conf)

    data_conf["dataset"]["vad_hz"] = 100
    dm = DialogAudioDM(
        datasets=data_conf["dataset"]["datasets"],
        type=data_conf["dataset"]["type"],
        sample_rate=data_conf["dataset"]["sample_rate"],
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        audio_overlap=data_conf["dataset"]["audio_overlap"],
        # audio_context_duration=data_conf["dataset"]["audio_context_duration"],
        # ipu_min_time=data_conf["dataset"]["ipu_min_time"],
        # ipu_pause_time=data_conf["dataset"]["ipu_pause_time"],
        vad_hz=data_conf["dataset"]["vad_hz"],
        vad_bin_times=data_conf["dataset"]["vad_bin_times"],
        vad_history=data_conf["dataset"]["vad_history"],
        vad_history_times=data_conf["dataset"]["vad_history_times"],
        batch_size=16,
        num_workers=0,
    )
    dm.prepare_data()
    dm.setup()
    print(dm)

    print("\nBATCH DATASET")
    d = dm.val_dset[0]
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")

    print("\nBATCH DATALOADER")
    batch = next(iter(dm.train_dataloader()))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")

    #############################
    # from tqdm import tqdm
    #
    # for batch in tqdm(dm.train_dataloader()):
    #     pass
