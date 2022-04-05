from os.path import join
from datasets import load_dataset

from datasets_turntaking.fisher.utils import get_text_context
from datasets_turntaking.utils import load_waveform
from datasets_turntaking.utils import repo_root

DATASET_SCRIPT = join(repo_root(), "datasets_turntaking/fisher/fisher.py")


def load_fisher(
    split="train",
    train_files=None,
    val_files=None,
    test_files=None,
):
    if split == "val":
        split = "validation"

    def process_and_add_name(examples):
        examples["dataset_name"] = "fisher"
        return examples

    dset = load_dataset(
        DATASET_SCRIPT,
        name="default",
        split=split,
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
    )
    # dset = dset.remove_columns(["speaker_id", "chapter_id"])
    dset = dset.map(process_and_add_name)
    return dset


if __name__ == "__main__":

    # import sounddevice as sd

    dset = load_dataset(DATASET_SCRIPT, name="default", split="train")
    d = dset[1]

    c = get_text_context(d["dialog"], end=120, start=60)

    # x, sr = load_waveform(d["audio_path"])
    # s = min(d["dialog"]["A"]["start"][0], d["dialog"]["B"]["start"][0])
    # n = len(d["dialog"]["A"]["text"])
    # print(n)
    # n = len(d["dialog"]["B"]["start"])
    # print(n)

    dialog = d["dialog"]

    #
    # # start_sample = int(sr * s)
    # # x = x[:, start_sample:]
    # # sd.play(x[:, : int(120 * sr)].t(), samplerate=8000)
    #
    # sd.stop()
    #
    # print(d["dialog"]["A"]["text"][2])
    #
    # text = d["dialog"]["A"]["text"][2]
    #
    # s = int(sr * d["dialog"]["A"]["start"][2])
    # e = int(sr * d["dialog"]["A"]["end"][2])
    # sd.play(x[0, s:e].t(), samplerate=8000)
    #
    # print("-" * 50)
    # print(d["dialog"]["B"]["text"][:5])
    # d["dialog"]["A"]["text"][:5]
    # d["dialog"]["A"]["start"][:5]
    # d["dialog"]["B"]["text"][:5]
    # d["dialog"]["B"]["start"][:5]

    # Check alignment (waveform/spectrogram + start/end-times)
    # Check content .... listen...
