from datasets import load_dataset


def load_curiosity_dialogs(split="train"):
    """
    Splits: ['train', 'val', 'test']
    """

    def add_dataset_name(examples):
        examples["dataset_name"] = "curiosity_dialogs"

        # ensure consecutive speaker change
        dialog = [examples["dialog"][0]]
        last_speaker = examples["messages.sender"][0]
        for speaker, turn in zip(
            examples["messages.sender"][1:], examples["dialog"][1:]
        ):
            if speaker == last_speaker:
                dialog[-1] += " " + turn
            else:
                dialog.append(turn)
                last_speaker = speaker
        examples["dialog"] = dialog
        return examples

    remove_curiosity = [
        "messages.liked",
        "messages.sender",
        "messages.facts",
        "messages.message_id",
        "messages.dialog_acts",
        "known_entities",
        "focus_entity",
        "dialog_id",
        "inferred_steps",
        "created_time",
        "aspects",
        "first_aspect",
        "second_aspect",
        "shuffle_facts",
        "related_entities",
        "tag",
        "user_id",
        "assistant_id",
        "is_annotated",
        "user_dialog_rating",
        "user_other_agent_rating",
        "assistant_dialog_rating",
        "assistant_other_agent_rating",
        "reported",
        "annotated",
    ]

    if split == "validation":
        split = "val"

    dset = load_dataset("curiosity_dialogs", split=split)
    dset = dset.flatten()
    dset = dset.rename_column("messages.message", "dialog")
    dset = dset.map(add_dataset_name)
    dset = dset.remove_columns(remove_curiosity)
    return dset


if __name__ == "__main__":
    dset = load_curiosity_dialogs("test")
