from os.path import join
from datasets import load_dataset

from .librispeech import load_librispeech
from .lj_speech import load_lj_speech
from datasets_turntaking.utils import repo_root


def load_vctk(split):
    if split == "val":
        split = "validation"

    def process_and_add_name(examples):
        examples["dataset"] = "vctk"
        return examples

    DATASET_SCRIPT = join(repo_root(), "datasets_turntaking/speech/vctk.py")

    dset = load_dataset(DATASET_SCRIPT, name="default", split=split)
    dset = dset.remove_columns(["speaker_id", "utterance_id"])
    dset = dset.map(process_and_add_name)
    return dset


def load_multiple_datasets(datasets, split):
    dsets = []
    for d in datasets:
        if d == "lj_speech":
            dsets.append(load_lj_speech(split))
        elif d == "librispeech":
            dsets.append(load_librispeech(split))
        elif d == "vctk":
            dsets.append(load_vctk(split))
    return dsets
