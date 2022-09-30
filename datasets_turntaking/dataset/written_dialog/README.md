# Conversational Datasets

The goal is to combine all conversational dataset that we can find using
huggingface [datasets](https://github.com/huggingface/datasets). In order to
concatenate datasets they require the same `column_names` and `features` so the
first step to add a dataset is to "transform" it to a general structure. As a
starting point we only utilize the actual text, omitting much useful
information such as *entities*, *intents*, etc.

The simple combined dataset simply uses two `column_names`, the **dialog**
field which consists of a list of string representing the dialog, and
**dataset** which is simply a string with the corresponding dataset name.
All other fields are omitted `dataset.remove_columns([dataset specific list of
column names to remove])`.



One desiderata is that the speaker changes over consecutive turns so we process
the dataset using `dataset.map`, with a function that inserts the **dataset**
and handles consecutive turns:

```python
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

def process_and_add_name(examples):
    examples["dataset"] = "taskmaster2"
    examples["dialog"] = get_every_other_speaker_dialog(examples["dialog"])
    return examples
```

Where we iterate over the turns of the dialog and combine consecutive
utterances from the same speaker.


A function, `def load_<dataset-name>(split)`, which loads a dataset and formats it
correctly follows the steps below:

1. Load the original dataset
2. Remove unneccessary `column_names`
3. Change name of relevant field to `dialog`.
    - i.e. `dset = dset.rename_column("utterances", "dialog")`
4. Process the dataset as shown in code block above.
    - `dset = dset.map(process_and_add_name)`
5. Splits
    - if the dataset naturally contain splits for 'train' and 'validation' those are used
    - else we have to manually split

#### Example

```python
def load_curiosity_dialogs(split="train"):
    """
    Splits: ['train', 'val', 'test']
    """

    def add_dataset(examples):
        examples["dataset"] = "curiosity_dialogs"
        return examples

    remove_curiosity = [ "messages.liked", ...,  "annotated"]

    if split == "validation":
        split = "val"

    dset = load_dataset("curiosity_dialogs", split=split)
    dset = dset.flatten()
    dset = dset.remove_columns(remove_curiosity)
    dset = dset.rename_column("messages.message", "dialog")
    dset = dset.map(add_dataset)
    return dset
```
