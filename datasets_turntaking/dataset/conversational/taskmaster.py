from os.path import join
from datasets import load_dataset
from datasets import concatenate_datasets
from datasets_turntaking.utils import repo_root, read_json, write_json

SPLIT_PATH2 = join(
    repo_root(),
    "datasets_turntaking/dataset/conversational/splits/taskmaster2_splits.json",
)
SPLIT_PATH3 = join(
    repo_root(),
    "datasets_turntaking/dataset/conversational/splits/taskmaster3_splits.json",
)


def get_every_other_speaker_dialog(dialog):
    """process to make sure we have every other speaker"""
    new_dialog = [dialog[0]["text"]]
    last_speaker = dialog[0]["speaker"]
    for turn in dialog[1:]:
        if turn["speaker"] == last_speaker:
            new_dialog[-1] += " " + turn["text"]
        else:
            new_dialog.append(turn["text"])
            last_speaker = turn["speaker"]
    return new_dialog


def _split_taskmaster():
    """
    Simple script to randomly choose splits.
    Writes to json and loads for consistency.

    Only used "once".
    """
    import random

    # used for taskmaster 2 as well
    dset = load_dataset("taskmaster3", split="train")

    idx = list(range(len(dset)))
    random.shuffle(idx)

    train_split = int(len(idx) * 0.9)
    valtest_split = len(idx) - train_split
    val_split = valtest_split // 2

    train_idx = idx[:train_split]
    val_idx = idx[train_split : train_split + val_split]
    test_idx = idx[train_split + val_split :]

    write_json({"train": train_idx, "val": val_idx, "test": test_idx}, SPLIT_PATH3)


# Splits done!
def load_taskmaster1(split="train"):
    """
    Includes 2 configs: ['one_person_dialogs', 'woz_dialogs']

    but only 'one_person_dialogs' words out-of-the-box.
    'woz_dialogs' contains duplicate keys!!

    """

    def process_and_add_name(examples):
        examples["dataset_name"] = "taskmaster1"
        examples["dialog"] = get_every_other_speaker_dialog(examples["dialog"])
        return examples

    if split == "val":
        split = "validation"
    # all_names = ['one_person_dialogs', 'woz_dialogs']
    # only 'one_person_dialogs' does not contain duplicate keys!!
    dset = load_dataset("taskmaster1", name="one_person_dialogs", split=split)
    remove_taskmaster1 = ["conversation_id", "instruction_id"]
    dset = dset.remove_columns(remove_taskmaster1)
    dset = dset.rename_column("utterances", "dialog")
    dset = dset.map(process_and_add_name)
    return dset


def load_taskmaster2(split="train"):
    def process_and_add_name(examples):
        examples["dataset_name"] = "taskmaster2"
        examples["dialog"] = get_every_other_speaker_dialog(examples["dialog"])
        return examples

    if split == "val":
        split = "validation"

    # all_configs = ["flights", "food-ordering", "hotels", "movies", "music", "restaurant-search", "sports"]
    # unvalid_names = ["hotels", "movies", "music", "sports"]
    valid_names = ["flights", "food-ordering", "restaurant-search"]

    dsets = []
    for name in valid_names:
        dset = load_dataset("taskmaster2", name=name, split="train")
        dset = dset.remove_columns(["conversation_id", "instruction_id"])
        dset = dset.rename_column("utterances", "dialog")
        dsets.append(dset)
    dset = concatenate_datasets(dsets)
    dset = dset.map(process_and_add_name)

    # choose correct splits
    splits = read_json(SPLIT_PATH2)
    if split == "test":
        dset = dset.select(splits["test"])
    elif split == "validation":
        dset = dset.select(splits["val"])
    else:
        dset = dset.select(splits["train"])
    return dset


def load_taskmaster3(split="train"):
    """
    Only contains "train" split so we manually split into "validation" and test.
    """

    def process_and_add_name(examples):
        examples["dataset_name"] = "taskmaster3"
        examples["dialog"] = get_every_other_speaker_dialog(examples["dialog"])
        return examples

    if split == "val":
        split = "validation"

    dset = load_dataset("taskmaster3", split="train")
    remove_taskmaster3 = [
        "conversation_id",
        "vertical",
        "instructions",
        "scenario",
    ]
    dset = dset.remove_columns(remove_taskmaster3)
    dset = dset.rename_column("utterances", "dialog")
    dset = dset.map(process_and_add_name)

    # choose correct splits
    splits = read_json(SPLIT_PATH3)
    if split == "test":
        dset = dset.select(splits["test"])
    elif split == "validation":
        dset = dset.select(splits["val"])
    else:
        dset = dset.select(splits["train"])
    return dset


if __name__ == "__main__":

    print("Training data")
    dset = load_taskmaster1()  # all splits a go!
    print("taskmaster 1: ", len(dset))
    dset = load_taskmaster2()  # all splits a go!
    print("taskmaster 2: ", len(dset))
    dset = load_taskmaster3()  # all splits a go!
    print("taskmaster 3: ", len(dset))
