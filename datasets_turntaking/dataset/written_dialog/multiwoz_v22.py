from datasets import load_dataset


def load_multiwoz_v22(split="train"):
    """
    Splits: ['train', 'validation', 'test']

    WARNING
    -------
    NonMatchingChecksumError: Checksums didn't match for dataset source files:
        https://github.com/huggingface/datasets/issues/1876

    use `ignore_verifications=True` for now
    """

    def add_dataset(examples):
        examples["dataset"] = "multi_woz_v22"

        # ensure correct speaker shifts
        dialog = [examples["dialog"][0]]
        last_speaker = examples["turns.speaker"][0]
        for turn, speaker in zip(examples["dialog"][1:], examples["turns.speaker"][1:]):
            if speaker == last_speaker:
                dialog[-1] += " " + turn
            else:
                dialog.append(turn)
                last_speaker = speaker
        examples["dialog"] = dialog
        return examples

    if split == "val":
        split = "validation"

    remove_multi_woz = [
        "dialogue_id",
        "services",
        "turns.turn_id",
        "turns.speaker",
        # "turns.utterance",
        "turns.frames",
        "turns.dialogue_acts",
    ]

    dset = load_dataset("multi_woz_v22", ignore_verifications=True, split=split)
    dset = dset.flatten()
    dset = dset.rename_column("turns.utterance", "dialog")
    dset = dset.map(add_dataset)
    dset = dset.remove_columns(remove_multi_woz)
    return dset


if __name__ == "__main__":

    dset = load_multiwoz_v22("test")
