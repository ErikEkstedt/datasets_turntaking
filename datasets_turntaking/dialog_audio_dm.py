from os import cpu_count, environ
from os.path import join
from torch.utils.data import DataLoader
from typing import Optional, Dict
import pytorch_lightning as pl
import torch

# omit verbose `datasets` info
# WARNING: Setting verbosity level by hand...
environ["DATASETS_VERBOSITY"] = "error"

from datasets_turntaking.dialog_audio_dataset import DialogAudioDataset
from datasets_turntaking.dataset.spoken_dialog import load_spoken_dataset
from datasets_turntaking.utils import repo_root, OmegaConfArgs, load_config


DEFAULT_CONFIG = join(repo_root(), "config/dset_dialog_audio.yaml")


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
        vad=True,
        vad_hz=100,
        vad_horizon=2,
        vad_history=False,
        vad_history_times=[60, 30, 10, 5],
        flip_channels=True,
        train_files=None,
        val_files=None,
        test_files=None,
        batch_size=4,
        num_workers=0,
        pin_memory=True,
        transforms=None,
    ):
        super().__init__()
        self.datasets = datasets  # names of datasets
        self.type = type
        self.transforms = transforms

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
        self.vad = vad
        self.vad_hz = vad_hz
        self.vad_horizon = vad_horizon
        self.vad_history = vad_history
        self.vad_history_times = vad_history_times
        self.flip_channels = flip_channels

        # Dataset Files
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files

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
            _ = load_spoken_dataset(
                datasets=self.datasets,
                split=split,
            )

    def _dataset(self, dset, split="train"):
        # Only flip during training...
        if split == "train":
            flip = self.flip_channels
        else:
            flip = False

        return DialogAudioDataset(
            dataset=dset,
            transforms=self.transforms,
            feature_extractor=None,
            type=self.type,
            audio_mono=self.audio_mono,
            audio_duration=self.audio_duration,
            audio_overlap=self.audio_overlap,
            audio_normalize=self.audio_normalize,
            sample_rate=self.sample_rate,
            vad=self.vad,
            vad_hz=self.vad_hz,
            vad_horizon=self.vad_horizon,
            vad_history=self.vad_history,
            vad_history_times=self.vad_history_times,
            flip_channels=flip,
            flip_probability=0.5,
        )

    def setup(self, stage: Optional[str] = "fit"):
        """Loads the datasets"""

        if stage in (None, "fit"):
            train_hf_dataset = load_spoken_dataset(
                datasets=self.datasets,
                split="train",
                train_files=self.train_files,
                val_files=self.val_files,
                test_files=self.test_files,
            )
            self.train_dset = self._dataset(train_hf_dataset, split="train")
            val_hf_dataset = load_spoken_dataset(
                datasets=self.datasets,
                split="val",
                train_files=self.train_files,
                val_files=self.val_files,
                test_files=self.test_files,
            )
            self.val_dset = self._dataset(val_hf_dataset, split="val")

        if stage in (None, "test"):
            test_hf_dataset = load_spoken_dataset(
                datasets=self.datasets,
                split="test",
                train_files=self.train_files,
                val_files=self.val_files,
                test_files=self.test_files,
            )
            self.test_dset = self._dataset(test_hf_dataset, split="test")

    def collate_fn(self, batch):
        waveforms = []
        vad = []
        vad_history = []
        vad_label = []
        dset_names = []
        sessions = []
        for b in batch:
            waveforms.append(b["waveform"])
            dset_names.append(b["dataset"])
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
        s += f"\n\tAudio"
        s += f"\n\tmono: {self.audio_mono}"
        s += f"\n\tduration: {self.audio_duration}"
        s += f"\n\toverlap: {self.audio_overlap}"
        s += f"\n\tVA"
        s += f"\n\tvad_hz: {self.vad_hz}"
        s += f"\n\tvad_history: {self.vad_history}"
        s += f"\n\tDataset"
        s += f"\n\tdatasets: {self.datasets}"
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
        parser.add_argument("--train_files", default=None, type=str)
        parser.add_argument("--val_files", default=None, type=str)
        parser.add_argument("--test_files", default=None, type=str)

        # A workaround for OmegaConf + WandB-Sweeps
        conf = DialogAudioDM.load_config()
        parser = OmegaConfArgs.add_argparse_args(parser, conf)
        return parent_parser


if __name__ == "__main__":

    from os.path import join

    data_conf = DialogAudioDM.load_config()

    train_files = None
    val_files = None
    data_conf["dataset"]["vad_hz"] = 50
    dm = DialogAudioDM(
        datasets=["switchboard", "fisher"],
        type=data_conf["dataset"]["type"],
        sample_rate=data_conf["dataset"]["sample_rate"],
        audio_mono=data_conf["dataset"]["audio_mono"],
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        audio_overlap=data_conf["dataset"]["audio_overlap"],
        vad_hz=data_conf["dataset"]["vad_hz"],
        vad_horizon=data_conf["dataset"]["vad_horizon"],
        vad_history=data_conf["dataset"]["vad_history"],
        vad_history_times=data_conf["dataset"]["vad_history_times"],
        train_files=train_files,
        val_files=val_files,
        batch_size=4,
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
