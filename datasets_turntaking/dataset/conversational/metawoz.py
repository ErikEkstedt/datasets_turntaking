from datasets import load_dataset
from os.path import join
from datasets_turntaking.utils import read_json, repo_root, write_json

SPLIT_PATH = join(
    repo_root(), "datasets_turntaking/conversational/splits/metawoz_splits.json"
)


def _random_splits():
    """randomly selects validation/train splits.
    This was used once to construct meta
    """
    import random

    dset = load_dataset("meta_woz", split="train")
    idx = list(range(len(dset)))
    random.shuffle(idx)
    train_split = int(len(idx) * 0.9)
    train_idx = idx[:train_split]
    val_idx = idx[train_split:]
    write_json({"train": train_idx, "validation": val_idx}, "metawoz_splits.json")


def load_metawoz(split="train"):
    """
    Only contain splits "train" and "test" so we split the train split to include "validation".

    we shuffle the idx of the datset and select 10% of training data to be
    validation. these were saved to disc "metawoz_splits.json" for consistency.

    The dataset does not include speaker-id information so we assume that
    consecutive utterances are from different speakers.
    """

    def add_dataset_name(examples):
        examples["dataset_name"] = "meta_woz"
        return examples

    remove_metawoz = ["id", "user_id", "bot_id", "domain", "task_id"]
    # No validation split by default.
    # Simply use the first 90% of train-split for training and the last 10% as validation
    if split == "test":
        dset = load_dataset("meta_woz", split=split)
    else:
        if split == "train":
            idx = read_json(SPLIT_PATH)["train"]
        elif split in ["validation", "val"]:
            idx = read_json(SPLIT_PATH)["validation"]
        split = "train"  # only contains train or test
        dset = load_dataset("meta_woz", split=split)
        dset = dset.select(idx)
    dset = dset.remove_columns(remove_metawoz)
    dset = dset.rename_column("turns", "dialog")
    dset = dset.map(add_dataset_name)
    return dset


if __name__ == "__main__":
    dset = load_metawoz("train")

    dset = load_dataset("meta_woz", split="train")

    d = dset[0]
