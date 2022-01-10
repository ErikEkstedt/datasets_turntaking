from datasets import load_dataset

from os.path import join, expanduser
from datasets_turntaking.utils import repo_root

DATASET_SCRIPT = join(repo_root(), "datasets_turntaking/callhome/callhome.py")
DATA_DIR = join(expanduser("~"), "projects/data/callhome")


if __name__ == "__main__":
    dset = load_dataset(
        DATASET_SCRIPT,
        data_dir=DATA_DIR,
        split="test",
        download_mode="force_redownload",
    )

    d = dset[0]
    print(d)
