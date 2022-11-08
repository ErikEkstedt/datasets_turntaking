from os.path import join
from datasets import load_dataset
from datasets_turntaking.utils import repo_root

DATASET_SCRIPT = join(
    repo_root(),
    "datasets_turntaking/dataset/spoken_dialog/vacation_interview/vacation_interview.py",
)


def load_vacation_interview(split="train"):
    return load_dataset(DATASET_SCRIPT, name="default", split="train")
