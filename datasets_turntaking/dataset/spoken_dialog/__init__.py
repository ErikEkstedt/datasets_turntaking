from datasets import load_dataset
from os import cpu_count
from os.path import join

from datasets_turntaking.utils import repo_root
from datasets_turntaking.dataset.spoken_dialog.utils import format_dialog_turns

# Backchannel/filler/acknowledgement
CAND_WORDS = [
    "oh",
    "yeah",
    "right",
    "really",
    "sure",
    "mhm",
    "mm",
    "um-hum",
    "uh-huh",
]
BC_CANDS = CAND_WORDS + [a + " " + b for a in CAND_WORDS for b in CAND_WORDS]
ROOT = join(repo_root(), "datasets_turntaking/dataset/spoken_dialog")
DSET_PATHS = {
    "switchboard": join(ROOT, "switchboard/switchboard.py"),
    "fisher": join(ROOT, "fisher/fisher.py"),
    "callhome": join(ROOT, "callhome/callhome.py"),
    "vacation_interview": join(ROOT, "vacation_interview/vacation_interview.py"),
}

# Loads the default datasets (configured as huggingfacee datasets)
def load_switchboard(
    split="train",
    omit_overlap_within=True,
    omit_backchannels=False,
    num_proc=None,
    load_from_cache_file=True,
    format_turns=False,
    **kwargs
):
    if split == "val":
        split = "validation"

    dset = load_dataset(DSET_PATHS["switchboard"], split=split, **kwargs)

    if format_turns:
        num_proc = num_proc if num_proc is not None else cpu_count()
        dset = dset.map(
            format_dialog_turns,
            batched=False,
            fn_kwargs={
                "omit_overlap_within": omit_overlap_within,
                "omit_backchannels": omit_backchannels,
            },
            load_from_cache_file=load_from_cache_file,
            num_proc=num_proc,
            desc="SWB: Format Turns",
        )
    return dset


def load_fisher(
    split="train",
    omit_overlap_within=True,
    omit_backchannels=False,
    num_proc=None,
    load_from_cache_file=True,
    format_turns=False,
    **kwargs
):
    if split == "val":
        split = "validation"

    dset = load_dataset(DSET_PATHS["fisher"], split=split, **kwargs)
    if format_turns:
        num_proc = num_proc if num_proc is not None else cpu_count()
        dset = dset.map(
            format_dialog_turns,
            batched=False,
            fn_kwargs={
                "omit_overlap_within": omit_overlap_within,
                "omit_backchannels": omit_backchannels,
            },
            load_from_cache_file=load_from_cache_file,
            num_proc=num_proc,
            desc="Fisher Format Turns",
        )
    return dset


def load_callhome(split="train"):
    if split == "val":
        split = "validation"
    raise NotImplementedError("Callhome is not tested")


def load_vacation_interview(split="train"):
    if split == "val":
        split = "validation"
    raise NotImplementedError("Callhome is not tested")
