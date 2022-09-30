from os.path import join
from datasets import load_dataset
from datasets_turntaking.utils import repo_root


DATASET_SCRIPT = join(
    repo_root(), "datasets_turntaking/dataset/spoken_dialog/switchboard/switchboard.py"
)


def load_switchboard(split="train", **kwargs):
    if split == "val":
        split = "validation"
    return load_dataset(DATASET_SCRIPT, name="default", split=split, **kwargs)
