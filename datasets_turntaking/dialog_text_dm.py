from argparse import ArgumentParser
from os.path import expanduser, join, exists
from os import cpu_count
import re
from typing import Optional, List

import torch
from torch.utils.data import DataLoader

from datasets import logging, concatenate_datasets, load_from_disk
import pytorch_lightning as pl

from datasets_turntaking.dataset.spoken_dialog import load_fisher, load_switchboard
from datasets_turntaking.dataset.written_dialog import (
    load_taskmaster1,
    load_taskmaster2,
    load_taskmaster3,
    load_curiosity_dialogs,
    load_daily_dialog,
    load_metawoz,
    load_multiwoz_v22,
)

logger = logging.get_logger(__name__)

CACHE_PATH = join(expanduser("~"), ".cache/datasets_turntaking/dialog_text")
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


def dataset_name(datasets, tokenizer, split, max_length, keep_length, overlap_length):
    datasets.sort()  # for name consistency

    # Datasets
    name = "_".join([d[:3] for d in datasets])
    name += "_text"
    name += f"_{tokenizer.name_or_path}"
    if max_length > 0:
        name += f"_max{max_length}_keep{keep_length}_ovrl{overlap_length}"
    name += f"_{split}"
    savepath = join(CACHE_PATH, name)
    return savepath


def tokenize_dataset(sample, tokenizer):
    t = tokenizer(sample["dialog"])
    sample["input_ids"] = t["input_ids"]
    sample["speaker_ids"] = t["speaker_ids"]
    return sample


def force_max_len_dataset(sample, max_length, overlap_length, keep_length):
    batch_inp = sample["input_ids"]
    batch_sp = sample["speaker_ids"]
    batch_dataset = sample.get("dataset", None)
    batch_session = sample.get("session", None)
    batch_audio_path = sample.get("audio_path", None)

    step_len = max_length - overlap_length

    # Split to appropriate lengths
    input_ids, speaker_ids = [], []
    dataset, session, audio_path = [], [], []
    for batch in range(len(batch_inp)):
        tmp_inps, tmp_sp = batch_inp[batch], batch_sp[batch]
        for i in range(0, len(tmp_inps), step_len):
            size = min(len(tmp_inps), i + max_length) - i
            if size >= keep_length:
                input_ids.append(tmp_inps[i : i + max_length])
                speaker_ids.append(tmp_sp[i : i + max_length])
                if batch_dataset is not None:
                    dataset.append(batch_dataset[batch])
                if batch_session is not None:
                    session.append(batch_session[batch])
                if batch_audio_path is not None:
                    audio_path.append(batch_audio_path[batch])
            else:
                break

    sample["input_ids"] = input_ids
    sample["speaker_ids"] = speaker_ids
    if len(dataset) > 0:
        sample["dataset"] = dataset
    if len(session) > 0:
        sample["session"] = session
    if len(audio_path) > 0:
        sample["audio_path"] = audio_path
    return sample


def filter_empty_turns(d):
    """
    return only dialogs with no empty turns
    """
    for utterance in d["dialog"]:
        if utterance == "" or not re.search(r"\w", utterance):  # utt is empty
            return False
    return True


def load_text_dataset(
    datasets,
    tokenizer,
    split: str = "train",
    max_length: int = 256,
    keep_length: int = 20,
    overlap_length: int = 10,
    omit_overlap_within: bool = True,
    omit_backchannels: bool = False,
    overwrite: bool = False,
    load_from_cache_file: bool = True,
    num_proc: Optional[int] = None,
    savepath: Optional[str] = None,
):
    if savepath is None:
        savepath = dataset_name(
            datasets, tokenizer, split, max_length, keep_length, overlap_length
        )

    if exists(savepath) and not overwrite:
        logger.info(f"LOAD PREPROCESSED DATASET: {savepath}")
        return load_from_disk(savepath)

    dsets = []
    for d in datasets:
        if d == "fisher":
            dsets.append(
                load_fisher(
                    split, omit_overlap_within, omit_backchannels, format_turns=True
                )
            )
        elif d == "switchboard":
            dsets.append(
                load_switchboard(
                    split, omit_overlap_within, omit_backchannels, format_turns=True
                )
            )
        elif d == "curiosity_dialogs":
            dsets.append(load_curiosity_dialogs(split))
        elif d == "daily_dialog":
            dsets.append(load_daily_dialog(split))
        elif d == "multi_woz_v22":
            dsets.append(load_multiwoz_v22(split))
        elif d == "meta_woz":
            dsets.append(load_metawoz(split))
        elif d == "taskmaster1":
            dsets.append(load_taskmaster1(split))
        elif d == "taskmaster2":
            dsets.append(load_taskmaster2(split))
        elif d == "taskmaster3":
            dsets.append(load_taskmaster3(split))
        else:
            raise NotImplementedError(f"Not installed: {d}")

    dset = concatenate_datasets(dsets)
    dset = dset.filter(filter_empty_turns, desc="Filter Empty Turns")
    num_proc = num_proc if num_proc is not None else cpu_count()
    dset = dset.map(
        tokenize_dataset,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        load_from_cache_file=load_from_cache_file,
        num_proc=num_proc,
        desc="Tokenize",
    )
    dset = dset.map(
        force_max_len_dataset,
        batched=True,
        fn_kwargs={
            "max_length": max_length,
            "overlap_length": overlap_length,
            "keep_length": keep_length,
        },
        load_from_cache_file=load_from_cache_file,
        remove_columns=["dialog", "vad"],
        num_proc=num_proc,
        desc="Force Max Length",
    )
    dset.set_format(
        "torch", columns=["input_ids", "speaker_ids"], output_all_columns=True
    )

    print(f"SAVE TO DISK: {savepath}")
    dset.save_to_disk(savepath)
    return dset


# TODO: Better formatting see dataset/spoken_dialog/utils.py
# Instead of format_sort_and_combine_utterances below.
def concatenate_dsets(dsets, columns=["dialog", "dataset"]):
    """
    Concatenate and simplify
    """
    dset = concatenate_datasets(dsets)
    remove = []
    for c in dset.column_names:
        if c not in columns:
            remove.append(c)
    if len(remove) > 0:
        dset = dset.remove_columns(remove)
    return dset


def format_sort_and_combine_utterances(d):
    """
    Used in spoken datasets [switchboard, fisher]
    1. Collect all utterances from both speakers in a single list.
    2. Sort list by starting time.
    3. Combine adjacent utterances from the same speaker.
    4. Remove time information
    5. Update dialogs
    """

    dialog = d["dialog"]

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

    d["dialog"] = utterances

    return d


def load_text_dataset_old(
    datasets,
    tokenizer,
    split="train",
    columns=["dialog", "dataset"],
    overwrite=False,
    load_from_cache_file=True,
    num_proc=cpu_count(),
    max_length=-1,
    keep_length=64,
    savepath=None,
    **kwargs,
):
    if savepath is None:
        savepath = dataset_name(datasets, tokenizer, split, max_length, keep_length)

    if exists(savepath) and not overwrite:
        logger.info(f"LOAD PREPROCESSED DATASET: {savepath}")
        return load_from_disk(savepath)

    def encode_dataset_fixed_size(d):
        """
        Tokenizes the dataset and organize to appropriate lengths (`self.max_length`)
        """
        t = tokenizer(d["dialog"])
        inp_ids = t["input_ids"]
        sp_ids = t["speaker_ids"]

        # Split to appropriate lengths
        input_ids = []
        speaker_ids = []
        dataset = []
        if max_length > 0:
            for batch in range(len(inp_ids)):
                tmp_inps = inp_ids[batch]
                tmp_sp = sp_ids[batch]
                for i in range(0, len(tmp_inps), max_length):
                    size = min(len(tmp_inps), i + max_length) - i
                    if size >= keep_length:
                        input_ids.append(tmp_inps[i : i + max_length])
                        speaker_ids.append(tmp_sp[i : i + max_length])
                        if "dataset" in d:
                            dataset.append(str(d["dataset"][batch]))
                    else:
                        break
        else:
            input_ids = inp_ids
            speaker_ids = sp_ids
            if "dataset" in d:
                dataset = d["dataset"]

        ret = {"input_ids": input_ids, "speaker_ids": speaker_ids}
        if len(dataset) > 0:
            ret["dataset"] = dataset
        return ret

    dsets = []
    for d in datasets:
        if d in ["fisher", "switchboard"]:
            if d == "fisher":
                dset = load_fisher(split=split)
            else:
                dset = load_switchboard(split=split)
            dset = dset.map(format_sort_and_combine_utterances)
            dsets.append(dset)
        elif d == "curiosity_dialogs":
            dsets.append(load_curiosity_dialogs(split))
        elif d == "daily_dialog":
            dsets.append(load_daily_dialog(split))
        elif d == "multi_woz_v22":
            dsets.append(load_multiwoz_v22(split))
        elif d == "meta_woz":
            dsets.append(load_metawoz(split))
        elif d == "taskmaster1":
            dsets.append(load_taskmaster1(split))
        elif d == "taskmaster2":
            dsets.append(load_taskmaster2(split))
        elif d == "taskmaster3":
            dsets.append(load_taskmaster3(split))
        else:
            raise NotImplementedError(f"Not installed: {d}")

    dataset = concatenate_dsets(dsets, columns=columns)

    print("#" * 40)
    print("Filter empty turns")
    print("#" * 40)
    dataset = dataset.filter(filter_empty_turns)

    print("#" * 40)
    print(f"TOKENIZE {split} DATASET: ", tokenizer.name_or_path)
    print("#" * 40)
    dataset = dataset.map(
        encode_dataset_fixed_size,
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=load_from_cache_file,
        num_proc=num_proc,
    )
    print("DATASET: ", dataset)
    dataset.set_format(
        "torch", columns=["input_ids", "speaker_ids"], output_all_columns=True
    )
    print(f"SAVE TO DISK: {savepath}")
    dataset.save_to_disk(savepath)
    return dataset


class DialogTextDM(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer,
        datasets: List[str] = ["daily_dialog"],
        columns: List[str] = ["dataset", "dialog"],  # features prior to tokenization
        savepath: Optional[str] = None,
        batch_size: int = 10,
        max_length: int = 256,  # the maximum size of the batch
        keep_length: int = 64,  # keep end if over this length
        num_workers: int = -1,
        pin_memory: bool = True,
        overwrite: bool = False,
        load_from_cache_file: bool = True,
        num_proc: Optional[int] = None,
    ):
        super().__init__()
        assert keep_length < max_length, "Error: `keep_length` < `max_length`"
        for dset in datasets:
            assert (
                dset in POSSIBLE_DATASETS
            ), f"Must prepare dataset to be of correct format. Use {POSSIBLE_DATASETS}"

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = cpu_count() if num_workers < 0 else max_length
        self.keep_length = keep_length
        self.max_length = max_length
        self.pin_memory = pin_memory

        # `datasets` parameters
        self.load_from_cache_file = load_from_cache_file
        self.num_proc = num_proc if num_proc is not None else cpu_count()
        self.overwrite = overwrite

        self.datasets = datasets
        self.datasets.sort()  # sort for consistency
        self.columns = columns
        self.savepath = savepath

    def prepare_data(self):
        """
        Call dataset in prepare data too make sure data is downloaded, formatted, tokenized, etc...
        """
        for split in ["train", "validation", "test"]:
            _ = load_text_dataset(
                datasets=self.datasets,
                tokenizer=self.tokenizer,
                split=split,
                columns=["dialog", "dataset"],
                overwrite=self.overwrite,
                load_from_cache_file=self.load_from_cache_file,
                num_proc=self.num_proc,
                max_length=self.max_length,
                keep_length=self.keep_length,
            )

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None or stage == "all":
            self.train_dset = load_text_dataset(
                datasets=self.datasets,
                tokenizer=self.tokenizer,
                split="train",
                columns=["dialog", "dataset"],
                overwrite=self.overwrite,
                load_from_cache_file=self.load_from_cache_file,
                num_proc=self.num_proc,
                max_length=self.max_length,
                keep_length=self.keep_length,
            )
            self.val_dset = load_text_dataset(
                datasets=self.datasets,
                tokenizer=self.tokenizer,
                split="val",
                columns=["dialog", "dataset"],
                overwrite=self.overwrite,
                load_from_cache_file=self.load_from_cache_file,
                num_proc=self.num_proc,
                max_length=self.max_length,
                keep_length=self.keep_length,
            )

        if stage == "all" or stage == "test":
            self.test_dset = load_text_dataset(
                datasets=self.datasets,
                tokenizer=self.tokenizer,
                split="test",
                columns=["dialog", "dataset"],
                overwrite=self.overwrite,
                load_from_cache_file=self.load_from_cache_file,
                num_proc=self.num_proc,
                max_length=self.max_length,
                keep_length=self.keep_length,
            )

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

    print("Load tokenizer...")
    tokenizer = SpokenDialogTokenizer()

    dset = load_text_dataset(
        datasets=["fisher", "switchboard", "daily_dialog", "curiosity_dialogs"],
        tokenizer=tokenizer,
    )
    d = dset[0]
    t = tokenizer.decode(d["input_ids"])
    print(t)

    # print("Load DM...")
    # dm = DialogTextDM(
    #     tokenizer,
    #     datasets=["switchboard", "fisher", "daily_dialog", "curiosity_dialogs"],
    #     max_length=256,
    #     batch_size=20,
    #     # overwrite=True,
    # )
    # dm.prepare_data()
    #
    # dm.setup()
    #
    # d = dm.train_dset[0]
    # t = tokenizer.decode(d["input_ids"])
    #
    # print(t)
    #
    # for k, v in d.items():
    #     if isinstance(v, torch.Tensor):
    #         print(f"{k}: {tuple(v.shape)}")
    #     else:
    #         print(f"{k}: {len(v)}")
    #
    # batch = next(iter(dm.train_dataloader()))
    # print("batch: ", batch.keys())
    # print(batch["input_ids"].shape)
