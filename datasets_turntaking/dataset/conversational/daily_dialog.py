from datasets import load_dataset


def load_daily_dialog(split="train", add_dataset_name=True):
    if split == "val":
        split = "validation"

    dset = load_dataset("daily_dialog", split=split)

    # Add dataset name
    # remove_daily_dialog = ["act", "emotion"]
    def _add_dataset_name(examples):
        examples["dataset"] = "daily_dialog"
        return examples

    if add_dataset_name:
        dset = dset.map(_add_dataset_name)

    return dset


if __name__ == "__main__":

    dset = load_daily_dialog("val")
