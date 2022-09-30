from copy import deepcopy


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

BACKCHANNEL_MAP = {
    "uh-huh": "uhuh",
    "huh-uh": "uhuh",
    "uh-hum": "mhm",
    "uh-hums": "mhm",
    "um-hum": "mhm",
    "hum-um": "mhm",
    "uh-oh": "uhoh",
}


def format_spoken_dialogs(d):
    """
    Processes spoken datasets (fisher, swithcboard) to a similar format as other
    conversational (daily_dialog, ...) written datasets.
    """
    utterances = format_to_utterances(d)
    d["utterances"] = refine_dialog(utterances)
    d["dialog"] = [u["text"] for u in d["utterances"]]
    return d


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
    utt["end"] = utt2["end"]

    if "words" in utt:
        utt["words"] += utt2["words"]

    if "backchannel" in utt2:
        utt["backchannel"] += utt2["backchannel"]

    if "backchannel_start" in utt2:
        utt["backchannel_start"] += utt2["backchannel_start"]

    if "within" in utt2:
        utt["within"] += utt2["within"]

    if "within_start" in utt2:
        utt["within_start"] += utt2["within_start"]

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
    first["backchannel_end"] = []
    first["within"] = []
    first["within_start"] = []
    first["within_end"] = []
    refined = [first]
    last_speaker = first["speaker"]

    for current in utterances[1:]:
        if is_backchannel(current):
            refined[-1]["backchannel"].append(current["text"])
            refined[-1]["backchannel_start"].append(current["start"])
            refined[-1]["backchannel_end"].append(current["end"])
        elif is_overlap_within(current, refined[-1]):
            refined[-1]["within"].append(current["text"])
            refined[-1]["within_start"].append(current["start"])
            refined[-1]["within_end"].append(current["end"])
        else:
            if current["speaker"] == last_speaker:
                refined[-1] = join_utterances(refined[-1], current)
            else:
                current["backchannel"] = []
                current["backchannel_start"] = []
                current["backchannel_end"] = []
                current["within"] = []
                current["within_start"] = []
                current["within_end"] = []
                refined.append(current)
                last_speaker = current["speaker"]
    return refined
