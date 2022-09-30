from datasets import load_dataset


def load_daily_dialog(split="train"):
    """
    splits = ['train', 'validation', 'test']
    already contains correct `dialog` field

    and got no speaker-id information? Assume that utterances changes speaker every time.
    """

    def add_dataset(examples):
        examples["dataset"] = "daily_dialog"
        return examples

    if split == "val":
        split = "validation"

    remove_daily_dialog = ["act", "emotion"]

    dset = load_dataset("daily_dialog", split=split)
    dset = dset.remove_columns(remove_daily_dialog)
    dset = dset.map(add_dataset)
    return dset


if __name__ == "__main__":

    dset = load_daily_dialog("val")
