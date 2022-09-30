from datasets import load_dataset, concatenate_datasets
from os import cpu_count
from os.path import join
from typing import List

from datasets_turntaking.utils import repo_root
from datasets_turntaking.dataset.spoken_dialog.utils import format_spoken_dialogs

ROOT = join(repo_root(), "datasets_turntaking/dataset/spoken_dialog")
DSET_PATHS = {
    "switchboard": join(ROOT, "switchboard/switchboard.py"),
    "fisher": join(ROOT, "fisher/fisher.py"),
    "callhome": join(ROOT, "callhome/callhome.py"),
    "vacation_interview": join(ROOT, "vacation_interview/vacation_interview.py"),
}


def load_spoken_dataset(
    datasets,
    split,
    num_proc=cpu_count(),
    load_from_cache_file=True,
    format=False,
    **custom_kwargs
) -> List:

    if split == "val":
        split = "validation"
    dsets = []
    for d in datasets:
        if d in list(DSET_PATHS.keys()):
            dsets.append(
                load_dataset(
                    DSET_PATHS[d], name="default", split=split, **custom_kwargs
                )
            )

    assert len(dsets) > 0, "Must load at least one dataset"
    dataset = concatenate_datasets(dsets)
    if format:
        dataset = dataset.map(
            format_spoken_dialogs,
            batched=False,
            load_from_cache_file=load_from_cache_file,
            num_proc=num_proc,
        )
    return dataset


def load_switchboard(split="train", **kwargs):
    if split == "val":
        split = "validation"
    return load_dataset(DSET_PATHS["switchboard"], split=split, **kwargs)


def load_fisher(split="train", **kwargs):
    if split == "val":
        split = "validation"
    return load_dataset(DSET_PATHS["fisher"], split=split, **kwargs)


def load_callhome(split="train"):
    if split == "val":
        split = "validation"
    return load_dataset(DSET_PATHS["fisher"], split=split)


def load_vacation_interview(split="train"):
    return load_dataset(DSET_PATHS["fisher"], split=split)
