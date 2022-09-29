from argparse import ArgumentParser
from os.path import expanduser, join, exists
from os import listdir, cpu_count
import re
import shutil
from typing import Optional

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from datasets import load_from_disk
from datasets_turntaking.dataset import load_multiple_datasets


CACHE_PATH = join(expanduser("~"), ".cache/datasets_turntaking/conversational")
SPOKEN_DATASETS = ["fisher", "switchboard"]
POSSIBLE_DATASETS = [
    "curiosity_dialogs",
    "daily_dialog",
    "multi_woz_v22",
    "meta_woz",
    "taskmaster1",
    "taskmaster2",
    "taskmaster3",
    "switchboard",
    "fisher",
]


def sort_and_combine_utterances(dialog):
    """
    Used in spoken datasets [switchboard, fisher]


    1. Collect all utterances from both speakers in a single list.
    2. Sort list by starting time.
    3. Combine adjacent utterances from the same speaker.
    4. Remove time information
    """

    # 1 and 2
    all_utterances_sorted = []
    for speaker, cc in enumerate(dialog):
        for t, s, e in zip(cc["text"], cc["start"], cc["end"]):
            all_utterances_sorted.append(
                {"start": s, "end": e, "text": t, "speaker": speaker}
            )
    all_utterances_sorted.sort(key=lambda x: x["start"])

    # 3
    combined_utterances = [all_utterances_sorted[0]]
    for utt in all_utterances_sorted[1:]:
        if utt["speaker"] == combined_utterances[-1]["speaker"]:
            combined_utterances[-1]["text"] += " " + utt["text"]
            combined_utterances[-1]["end"] = utt["end"]
        else:
            combined_utterances.append(utt)

    # 4
    utterances = [t["text"] for t in combined_utterances]

    return utterances


class ConversationalDM(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer,
        datasets=["daily_dialog"],
        columns=["dataset", "dialog"],  # features prior to tokenization
        savepath=CACHE_PATH,
        batch_size=10,
        max_length=256,  # the maximum size of the batch
        keep_length=64,  # keep end if over this length
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
        self.keep_length = keep_length
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        assert keep_length < max_length, "Error: `keep_length` < `max_length`"

        # `datasets` parameters
        self.load_from_cache_file = load_from_cache_file
        self.num_proc = num_proc if num_proc is not None else cpu_count()
        self.include_dialog = include_dialog

        # Datasets
        for dset in datasets:
            assert (
                dset in POSSIBLE_DATASETS
            ), f"Must prepare dataset to be of correct format. Use {POSSIBLE_DATASETS}"
        self.datasets = datasets
        self.columns = columns
        # removes restarts from spoken dialog data ("h- hello" -> "hello")
        self.remove_restarts = True
        self.datasets.sort()  # sort for consistency
        self.savepath = savepath
        self.overwrite = overwrite

    def get_dset_path(self, split):
        name = "_".join(self.datasets)
        name += f"_{self.tokenizer.name_or_path.replace('/', '-')}"
        name += f"_L{self.max_length}"
        return join(self.savepath, name, split)

    def filter_empty_turns(self, examples):
        """
        return only dialogs with no empty turns
        """
        for utterance in examples["dialog"]:
            if utterance == "" or not re.search(r"\w", utterance):  # utt is empty
                return False
        return True

    def encode_dataset(self, examples):
        """omit `attention_mask`"""
        if examples["dataset"] in SPOKEN_DATASETS:
            examples["dialog"] = sort_and_combine_utterances(examples["dialog"])

        t = self.tokenizer(examples["dialog"])
        examples["input_ids"] = t["input_ids"]
        examples["speaker_ids"] = t["speaker_ids"]
        return examples

    def encode_dataset_fixed_size(self, examples):
        """
        Tokenizes the dataset and organize to appropriate lengths (`self.max_length`)
        """
        if examples["dataset"] in SPOKEN_DATASETS:
            examples["dialog"] = sort_and_combine_utterances(examples["dialog"])
        t = self.tokenizer(examples["dialog"])
        inp_ids = t["input_ids"]
        sp_ids = t["speaker_ids"]

        # Split to appropriate lengths
        input_ids = []
        speaker_ids = []
        dataset = []

        for batch in range(len(inp_ids)):
            tmp_inps = inp_ids[batch]
            tmp_sp = sp_ids[batch]
            for i in range(0, len(tmp_inps), self.max_length):
                size = min(len(tmp_inps), i + self.max_length) - i
                if size >= self.keep_length:
                    input_ids.append(tmp_inps[i : i + self.max_length])
                    speaker_ids.append(tmp_sp[i : i + self.max_length])
                    if "dataset" in examples:
                        dataset.append(str(examples["dataset"][batch]))
                else:
                    break

        ret = {"input_ids": input_ids, "speaker_ids": speaker_ids}
        if len(dataset) > 0:
            ret["dataset"] = dataset
        return ret

    def prepare_data(self):
        """Concatenates multiple datasets"""

        for split in ["train", "validation", "test"]:
            dset_path = self.get_dset_path(split)

            if (
                self.overwrite
                or not self.load_from_cache_file
                or not exists(dset_path)
                or len(listdir(dset_path)) == 0
            ):

                # Remove if it exists in order to overwrite
                if self.overwrite and exists(dset_path):
                    shutil.rmtree(dset_path)

                dataset = load_multiple_datasets(
                    self.datasets,
                    split=split,
                    columns=self.columns,
                    remove_restarts=self.remove_restarts,
                )

                print("#" * 40)
                print("Filter empty turns")
                print("#" * 40)
                dataset = dataset.filter(self.filter_empty_turns)
                print("#" * 40)
                print(f"TOKENIZE {split} DATASET: ", self.tokenizer.name_or_path)
                print("#" * 40)
                dataset = dataset.map(
                    self.encode_dataset_fixed_size,
                    batched=True,
                    remove_columns=dataset.column_names,
                    load_from_cache_file=self.load_from_cache_file,
                    num_proc=self.num_proc,
                )
                dataset.set_format("torch")
                dataset.save_to_disk(dset_path)

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dset = load_from_disk(self.get_dset_path("train"))
            self.val_dset = load_from_disk(self.get_dset_path("validation"))
        elif stage == "all":
            self.train_dset = load_from_disk(self.get_dset_path("train"))
            self.val_dset = load_from_disk(self.get_dset_path("validation"))
            self.test_dset = load_from_disk(self.get_dset_path("test"))
        elif stage == "test":
            self.test_dset = load_from_disk(self.get_dset_path("test"))

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
        parser.add_argument("--datasets", type=str, nargs="+", default=["daily_dialog"])
        parser.add_argument("--savepath", default=CACHE_PATH, type=str)
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


def basic_use():
    """Basic use for ConversationalDM"""
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
    from turngpt.tokenizer import SpokenDialogTokenizer

    tokenizer = SpokenDialogTokenizer()

    dm = ConversationalDM(
        tokenizer,
        datasets=["switchboard", "fisher"],
        max_length=256,
        # overwrite=True,
        batch_size=20,
    )
    dm.prepare_data()
    dm.setup()

    d = dm.train_dset[1]

    t = tokenizer.decode(d["input_ids"])

    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {len(v)}")

    batch = next(iter(dm.train_dataloader()))
    print("batch: ", batch.keys())
    print(batch["input_ids"].shape)
