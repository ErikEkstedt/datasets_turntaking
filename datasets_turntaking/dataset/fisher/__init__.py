from os.path import join
from datasets import load_dataset
from datasets_turntaking.utils import repo_root

DATASET_SCRIPT = join(repo_root(), "datasets_turntaking/dataset/fisher/fisher.py")


def load_fisher(split="train", **kwargs):
    if split == "val":
        split = "validation"

    dset = load_dataset(DATASET_SCRIPT, name="default", split=split, **kwargs)
    return dset
