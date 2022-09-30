from os.path import join
from datasets import load_dataset
from datasets_turntaking.utils import repo_root

DATASET_SCRIPT = join(
    repo_root(), "datasets_turntaking/dataset/spoken_dialog/fisher/fisher.py"
)


def load_fisher(split="train", **kwargs):
    if split == "val":
        split = "validation"

    return load_dataset(DATASET_SCRIPT, name="default", split=split, **kwargs)
