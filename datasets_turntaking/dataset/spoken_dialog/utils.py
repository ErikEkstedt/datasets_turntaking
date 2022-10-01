# Backchannel/filler/acknowledgement
CAND_WORDS = [
    "oh",
    "yeah",
    "right",
    "really",
    "sure",
    "mhm",
    "mm",
    "um-hum",
    "uh-huh",
]
BC_CANDS = CAND_WORDS + [a + " " + b for a in CAND_WORDS for b in CAND_WORDS]


def speaker_join_and_sort(dialog):
    all_utterances_sorted = []
    for speaker, cc in enumerate(dialog):
        for t, s, e in zip(cc["text"], cc["start"], cc["end"]):
            all_utterances_sorted.append(
                {"start": s, "end": e, "text": t, "speaker": speaker}
            )
    all_utterances_sorted.sort(key=lambda x: x["start"])
    return all_utterances_sorted


def is_overlap_within(current, prev):
    start_within = prev["start"] <= current["start"] <= prev["end"]
    end_within = prev["start"] <= current["end"] <= prev["end"]
    return start_within and end_within


def is_backchannel(current):
    return current["text"] in BC_CANDS


def collapse_dialog(
    utterances, omit_overlap_within=False, omit_backchannels=False, verbose=False
):
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

    n_bc = 0
    n_ov = 0
    refined = [utterances[0]]
    for current in utterances[1:]:
        if omit_overlap_within and is_overlap_within(current, refined[-1]):
            n_ov += 1
            continue

        if omit_backchannels and is_backchannel(current):
            n_bc += 1
            continue

        if current["speaker"] == refined[-1]["speaker"]:
            refined[-1]["text"] += " " + current["text"]
            refined[-1]["end"] = current["end"]
        else:
            refined.append(current)
    if verbose:
        if n_ov > 0:
            print("Omitted Overlaps: ", n_ov)
        if n_bc > 0:
            print("Omitted BCs: ", n_bc)
    return refined


def dialog_to_text_list(dialog):
    return [d["text"] for d in dialog]


def format_dialog_turns(sample, omit_overlap_within, omit_backchannels):
    dialog = speaker_join_and_sort(sample["dialog"])
    dialog = collapse_dialog(dialog, omit_overlap_within, omit_backchannels)
    utterances = dialog_to_text_list(dialog)
    sample["dialog"] = utterances
    return sample
