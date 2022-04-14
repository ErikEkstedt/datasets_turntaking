from datasets import load_dataset

from os.path import join
from datasets_turntaking.utils import repo_root

DATASET_SCRIPT = join(repo_root(), "datasets_turntaking/speech/librispeech_asr.py")


def load_librispeech(split="train"):
    if split == "train":
        split = "train.360"

    if split == "val":
        split = "validation"

    def process_and_add_name(examples):
        examples["dataset_name"] = "librispeech_asr"
        return examples

    # dset = load_dataset("librispeech_asr", name="clean", split=split)
    dset = load_dataset(DATASET_SCRIPT, name="clean", split=split)
    dset = dset.remove_columns(["speaker_id", "chapter_id"])
    dset = dset.map(process_and_add_name)
    return dset


if __name__ == "__main__":
    dset = load_librispeech()
