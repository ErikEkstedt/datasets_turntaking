from os.path import join
from copy import deepcopy
from datasets import Value, Sequence
from datasets import load_dataset

from datasets_turntaking.utils import repo_root
from datasets_turntaking.dataset.conversational import (
    load_daily_dialog,
    load_curiosity_dialogs,
    load_multiwoz_v22,
    load_metawoz,
    load_taskmaster1,
    load_taskmaster2,
    load_taskmaster3,
)

ROOT = repo_root()
SCRIPT_PATHS = {
    "switchboard": join(ROOT, "datasets_turntaking/dataset/switchboard/switchboard.py"),
    "fisher": join(ROOT, "datasets_turntaking/dataset/fisher/fisher.py"),
}

DIALOG_AUDIO_FEATURES = {
    "dataset": Value("string"),
    "session": Value("string"),
    "audio_path": Value("string"),
    "vad": [[Sequence(Value("float"))]],
    "dialog": [
        Sequence(
            {
                "start": Value("float"),
                "end": Value("float"),
                "text": Value("string"),
            }
        )
    ],
}

BACKCHANNEL_CANDIDATES = [
    "yeah",
    "um-hum",
    "uh-huh",
    "right",
    "oh",
    # two-word bcs
    "oh yeah",
    "yeah yeah",
    "right right",
    "oh really",
    "um-hum um-hum",
    "uh-huh uh-huh",
    "oh uh-huh",
]


def load_multiple_datasets(datasets, split, **kwargs):
    dsets = []
    for d in datasets:
        if d in ["fisher", "switchboard"]:
            dsets.append(load_dataset(SCRIPT_PATHS[d], split=split, **kwargs))

            # dset = load_dataset(DATASET_SCRIPT, name="default", split=split, **kwargs)
        # elif d == "switchboard":
        #     dsets.append(load_switchboard(split, **kwargs))
        elif d == "curiosity_dialogs":
            dsets.append(load_curiosity_dialogs(split, **kwargs))
        elif d == "daily_dialog":
            dsets.append(load_daily_dialog(split, **kwargs))
        elif d == "multi_woz_v22":
            dsets.append(load_multiwoz_v22(split, **kwargs))
        elif d == "meta_woz":
            dsets.append(load_metawoz(split, **kwargs))
        elif d == "taskmaster1":
            dsets.append(load_taskmaster1(split, **kwargs))
        elif d == "taskmaster2":
            dsets.append(load_taskmaster2(split, **kwargs))
        elif d == "taskmaster3":
            dsets.append(load_taskmaster3(split, **kwargs))
        else:
            raise NotImplementedError(f"Not installed: {d}")
    return dsets


def format_to_utterances(d):
    A = [
        {"text": t, "start": s, "end": e, "speaker": 0}
        for t, s, e in zip(
            d["dialog"][0]["text"],
            d["dialog"][0]["start"],
            d["dialog"][0]["end"],
        )
    ]
    B = [
        {"text": t, "start": s, "end": e, "speaker": 1}
        for t, s, e in zip(
            d["dialog"][1]["text"],
            d["dialog"][1]["start"],
            d["dialog"][1]["end"],
        )
    ]

    utterances = A + B
    utterances.sort(key=lambda x: x["start"])
    return utterances


def is_backchannel(utt):
    return utt["text"] in BACKCHANNEL_CANDIDATES


def join_utterances(utt1, utt2):
    utt = deepcopy(utt1)
    utt["text"] += " " + utt2["text"]
    if "words" in utt:
        utt["words"] += utt2["words"]
    utt["end"] = utt2["end"]
    return utt


def is_overlap_within(current, prev):
    start_within = prev["start"] <= current["start"] <= prev["end"]
    end_within = prev["start"] <= current["end"] <= prev["end"]
    return start_within and end_within


def refine_dialog(utterances):  # , vad=None):
    """
    Refine the dialog by omitting `overlap_within` and `backchannel`
    speech, both of which are added to the current/major utterance. Keeps
    the original fields for text, words, start, end, speaker.
    i.e:
        refined[i] = {
                'id',
                'speaker',
                'text',
                'words',
                'start',
                'end',
                'backchannel',
                'within'
                }
    """

    # First utterance
    first = utterances[0]
    first["backchannel"] = []
    first["backchannel_start"] = []
    first["within"] = []
    first["within_start"] = []
    refined = [first]
    last_speaker = first["speaker"]

    for current in utterances[1:]:
        if is_backchannel(current):
            refined[-1]["backchannel"].append(current["text"])
            refined[-1]["backchannel_start"].append(current["start"])
        elif is_overlap_within(current, refined[-1]):
            refined[-1]["within"].append(current["text"])
            refined[-1]["within_start"].append(current["start"])
        else:
            if current["speaker"] == last_speaker:
                refined[-1] = join_utterances(refined[-1], current)
            else:
                current["backchannel"] = []
                current["backchannel_start"] = []
                current["within"] = []
                current["within_start"] = []
                refined.append(current)
                last_speaker = current["speaker"]
    return refined


if __name__ == "__main__":

    from datasets import concatenate_datasets

    split = "validation"
    # dsets = load_multiple_datasets(["fisher", "switchboard"], split=split)
    dsets = load_multiple_datasets(["switchboard"], split=split)

    def encode(d):
        utterances = format_to_utterances(d)
        d["dialog"] = refine_dialog(utterances)
        return d

    dataset = concatenate_datasets(dsets)
    dataset = dataset.map(
        encode,
        batched=False,
        # load_from_cache_file=self.load_from_cache_file,
        num_proc=4,
    )

    d = dataset[1]
    for u in d["dialog"]:
        print(
            u["speaker"],
            "(",
            round(u["start"], 2),
            round(u["end"], 2),
            ") ->",
            u["text"],
        )
        if len(u["within"]) > 0:
            print("   WI ->", u["within"], u["within_start"])
        if len(u["backchannel"]) > 0:
            print("   BC ->", u["backchannel"], u["backchannel_start"])
        input()
