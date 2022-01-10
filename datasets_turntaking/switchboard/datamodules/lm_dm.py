from argparse import ArgumentParser
from os.path import join, expanduser, exists
from os import cpu_count
from typing import Optional

from datasets import load_dataset, load_from_disk
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader
from datasets_turntaking.utils import repo_root

DATASET_SCRIPT = join(repo_root(), "datasets_turntaking/switchboard/switchboard.py")
F0_MEAN_PATH = join(repo_root(), "datasets_turntaking/switchboard/f0_means.json")
AUDIO_ROOT = join(expanduser("~"), "projects/data/switchboard/audio")
CACHE_PATH = join(expanduser("~"), ".cache/datasets_turntaking/switchboard/lm")


class LMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer,
        dataset_path=None,
        max_length=1024,
        batch_size=8,
        num_workers=4,
        pin_memory=False,
        num_proc=4,
        load_from_cache_file=True,
        batched=True,
        audio_root=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        if dataset_path is not None:
            self.dataset_path = dataset_path
        else:
            self.dataset_path = CACHE_PATH

        if tokenizer is not None:
            self.dataset_path = join(self.dataset_path, tokenizer.name_or_path)

        # datasets: dataset.map(encode, ...)
        self.num_proc = num_proc
        self.load_from_cache_file = load_from_cache_file
        self.batched = batched

        # Collate fn
        self.audio_root = audio_root
        self.max_length = max_length

        # DataLoaders
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    # TODO: Handle backchannels and within
    def _encode(self, examples):
        ret = self.tokenizer(examples["dialog.text"])
        _ = ret.pop("attention_mask")
        return ret

    def _split_path(self, split):
        return join(self.dataset_path, split)

    def prepare_data(self):
        for split in ["train", "validation", "test"]:
            split_path = self._split_path(split)
            if not exists(split_path) or not self.load_from_cache_file:
                assert (
                    self.tokenizer is not None
                ), "Dataset requires processing with tokenizer!"
                dataset = load_dataset(
                    DATASET_SCRIPT,
                    split=split,
                    name="default",
                )
                dataset = dataset.flatten()
                dataset = dataset.map(
                    self._encode,
                    batched=self.batched,
                    load_from_cache_file=self.load_from_cache_file,
                    num_proc=self.num_proc,
                )
                dataset = dataset.remove_columns(
                    [
                        "session",
                        "audio_path",
                        "dialog.id",
                        "dialog.text",
                        "dialog.speaker",
                        "dialog.start",
                        "dialog.end",
                        "dialog.words",
                        "dialog.within",
                        "dialog.backchannel",
                    ]
                )
                dataset.set_format(type="torch")
                dataset.save_to_disk(split_path)

    def setup(self, stage: Optional[str] = None):
        if stage == "test":
            self.test_dset = load_from_disk(self._split_path("test"))
        else:
            self.train_dset = load_from_disk(self._split_path("train"))
            self.val_dset = load_from_disk(self._split_path("validation"))

    # TODO:
    # Train on entire dialog albeit over max input size?
    # handle  in trainer or here?
    def collate_fn(self, batch):
        ret = self.tokenizer.pad(
            {"input_ids": [b["input_ids"][: self.max_length] for b in batch]}
        )
        ret["speaker_ids"] = self.tokenizer.pad(
            {"input_ids": [b["speaker_ids"][: self.max_length] for b in batch]}
        )["input_ids"]
        for k, v in ret.items():
            ret[k] = torch.tensor(v)
        return ret

    def _dataloader(self, dset, shuffle=True):
        return DataLoader(
            dset,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self._dataloader(self.train_dset, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dset, shuffle=False)

    def test_dataloader(self):
        return self._dataloader(self.test_dset, shuffle=False)

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        n_cpus = cpu_count()
        parser.add_argument("--batch_size", default=4, type=int)
        parser.add_argument("--num_workers", default=n_cpus, type=int)
        parser.add_argument("--pin_memory", default=True, type=bool)
        parser.add_argument("--savepath", default=None, type=str)
        parser.add_argument("--overwrite", default=False, type=bool)
        parser.add_argument("--dataset_path", default=None, type=str)
        parser.add_argument(
            "--max_length",
            default=500,
            type=int,
            help="maximum length of sequences (applied in `collate_fn`)",
        )
        # arguments for `datasets` library
        parser.add_argument("--load_from_cache_file", default=True, type=bool)
        parser.add_argument("--num_proc", default=n_cpus, type=int)
        return parser


if __name__ == "__main__":
    from convlm.turngpt.tokenizer import SpokenDialogTokenizer

    parser = ArgumentParser()
    parser = LMDataModule.add_data_specific_args(parser)
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    print("Loading tokenizer...")
    tokenizer = SpokenDialogTokenizer("gpt2")
    print("Done")

    args.audio_root = "/home/erik/projects/data/switchboard/audio"
    dm = LMDataModule(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        audio_root=args.audio_root,
        max_length=args.max_length,
        num_proc=args.num_proc,
        dataset_path=args.dataset_path,
        load_from_cache_file=args.load_from_cache_file,
    )
    dm.prepare_data()
    dm.setup("fit")
    print("train: ", len(dm.train_dset))
    print("val: ", len(dm.val_dset))
    dloader = dm.train_dataloader()
    batch = next(iter(dloader))
    for k, v in batch.items():
        if k == "session":
            print(v)
        else:
            print(k, type(v), v.shape)
