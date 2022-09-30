from os.path import join
from datasets import load_dataset
from datasets_turntaking.utils import repo_root

DATASET_SCRIPT = join(
    repo_root(), "datasets_turntaking/dataset/spoken_dialog/fisher/fisher.py"
)


def load_fisher(
    split="train",
    train_files=None,
    val_files=None,
    test_files=None,
):
    if split == "val":
        split = "validation"

    def process_and_add_name(examples):
        examples["dataset"] = "fisher"
        return examples

    dset = load_dataset(
        DATASET_SCRIPT,
        name="default",
        split=split,
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
    )
    dset = dset.map(process_and_add_name)
    return dset
