from argparse import ArgumentParser
from os.path import expanduser, join, exists
from os import listdir, cpu_count
import re
import shutil
from typing import Optional

import torch
from torch.utils.data import DataLoader

from datasets import concatenate_datasets, load_from_disk
import pytorch_lightning as pl

from datasets_turntaking.dataset import load_multiple_datasets

CACHE_PATH = join(expanduser("~"), ".cache/datasets_turntaking/conversational")


class ConversationalDM(pl.LightningDataModule):
    DATASETS = [
        "curiosity_dialogs",
        "daily_dialog",
        "multi_woz_v22",
        "meta_woz",
        "taskmaster1",
        "taskmaster2",
        "taskmaster3",
    ]

    def __init__(
        self,
        tokenizer,
        datasets=None,
        savepath=CACHE_PATH,
        batch_size=2,
        max_length=256,
        num_workers=1,
        pin_memory=True,
        overwrite=False,
        include_dialog=False,
        load_from_cache_file=True,
        num_proc=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.pin_memory = pin_memory
        self.num_workers = num_workers

        # `datasets` parameters
        self.load_from_cache_file = load_from_cache_file
        self.num_proc = num_proc
        self.include_dialog = include_dialog

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
        self.savepath = join(savepath, self.tokenizer.name_or_path)
        self.overwrite = overwrite

    def get_split_path(self, split):
        return join(self.savepath, split)

    def filter_empty_turns(self, examples):
        """
        return only dialogs with no empty turns
        """
        for utterance in examples["dialog"]:
            if utterance == "" or not re.search(r"\w", utterance):  # utt is empty
                return False
        return True

    def encode(self, examples):
        """omit `attention_mask`"""
        t = self.tokenizer(examples["dialog"])
        return {"input_ids": t["input_ids"], "speaker_ids": t["speaker_ids"]}

    def prepare_data(self):
        """Concatenates multiple datasets"""

        for split in ["train", "validation", "test"]:
            split_path = self.get_split_path(split)

            if (
                self.overwrite
                or not self.load_from_cache_file
                or not exists(split_path)
                or len(listdir(split_path)) == 0
            ):

                # Remove if it exists in order to overwrite
                if self.overwrite and exists(split_path):
                    shutil.rmtree(split_path)

                dsets = load_multiple_datasets(self.datasets, split)
                dataset = concatenate_datasets(dsets)
                print("filter empty turns")
                dataset = dataset.filter(self.filter_empty_turns)
                dataset = dataset.map(
                    self.encode,
                    batched=True,
                    load_from_cache_file=self.load_from_cache_file,
                    num_proc=self.num_proc,
                )
                dataset.set_format(type="torch")
                dataset.save_to_disk(split_path)

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dset = load_from_disk(self.get_split_path("train"))
            self.val_dset = load_from_disk(self.get_split_path("validation"))

        if stage == "test":
            self.test_dset = load_from_disk(self.get_split_path("test"))

    def collate_fn(self, batch):
        ret = self.tokenizer.pad(
            {"input_ids": [b["input_ids"][: self.max_length] for b in batch]}
        )
        ret["speaker_ids"] = self.tokenizer.pad(
            {"input_ids": [b["speaker_ids"][: self.max_length] for b in batch]}
        )["input_ids"]
        for k, v in ret.items():
            ret[k] = torch.tensor(v)

        if self.include_dialog:
            ret["dialog"] = [b["dialog"] for b in batch]
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
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        n_cpus = cpu_count()
        parser.add_argument(
            "--datasets", type=str, nargs="+", default=ConversationalDM.DATASETS
        )
        parser.add_argument("--savepath", default=None, type=str)
        parser.add_argument("--overwrite", default=False, type=bool)
        parser.add_argument(
            "--max_length",
            default=500,
            type=int,
            help="maximum length of sequences (applied in `collate_fn`)",
        )
        # arguments for `datasets` library
        parser.add_argument("--batch_size", default=4, type=int)
        parser.add_argument("--num_workers", default=n_cpus, type=int)
        parser.add_argument("--pin_memory", default=True, type=bool)
        parser.add_argument("--load_from_cache_file", default=True, type=bool)
        parser.add_argument("--num_proc", default=n_cpus, type=int)
        return parser


def main():
    # https://github.com/ErikEkstedt/TurnGPT
    from turngpt.tokenizer import SpokenDialogTokenizer

    parser = ArgumentParser()
    parser = ConversationalDM.add_data_specific_args(parser)
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    tokenizer = SpokenDialogTokenizer()

    dm = ConversationalDM(
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        savepath=args.savepath,
        overwrite=args.overwrite,
        datasets=["curiosity_dialogs"],
        load_from_cache_file=args.load_from_cache_file,
        num_proc=args.num_proc,
        include_dialog=True,
    )
    dm.prepare_data()
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(k, type(v), tuple(v.shape))
        else:
            print(k, v)


if __name__ == "__main__":

    # Debugging

    from datasets_turntaking.dataset.switchboard import load_switchboard
    from datasets_turntaking.dataset.fisher import load_fisher

    split = "val"
    dsets = [load_switchboard(split)]
    dsets.append(load_fisher(split))

    d = dset[264]
    print("d: ", list(d.keys()))
    a = d["dialog"][0]
    b = d["dialog"][1]
    print(a["text"])

    for tt in a["text"]:
        print(tt)

    a["text"]

    from datasets_turntaking.utils import load_waveform

    x, sr = load_waveform(d["audio_path"])

    for i, utt in enumerate(b["text"]):
        if "[mn]" in utt:
            print(i, utt)

    import sounddevice as sd

    i = 82
    t = b["text"][i]
    start = b["start"][i]
    s = int(sr * (start))
    d = int(sr * 5)
    print(t)
    sd.play(x[1, s : s + d], samplerate=sr)
