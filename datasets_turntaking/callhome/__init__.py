from os.path import join, expanduser
from datasets import load_dataset
from datasets_turntaking.utils import repo_root

DATASET_SCRIPT = join(repo_root(), "datasets_turntaking/callhome/callhome.py")
DATA_DIR = join(expanduser("~"), "projects/data/callhome")


def load_callhome(split="train"):
    if split == "val":
        split = "validation"

    def process_and_add_name(examples):
        examples["dataset_name"] = "callhome"
        return examples

    dset = load_dataset(
        DATASET_SCRIPT,
        data_dir=DATA_DIR,
        split=split,
        # download_mode="force_redownload",
    )
    dset = dset.map(process_and_add_name)
    return dset
