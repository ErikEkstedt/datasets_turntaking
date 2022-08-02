import torch
from os.path import join
import re
from datasets_turntaking.utils import read_txt


SPEAKER2CHANNEL = {"A": 0, "B": 1}


# TODO:Specific cleaning of text?
def fisher_regexp(s, remove_restarts=False):
    """

    See information about annotations at:
    * https://catalog.ldc.upenn.edu/docs/LDC2004T19/fe_03_readme.txt

    Regexp
    ------

    * Special annotations:  ["[laughter]", "[noise]", "[lipsmack]", "[sigh]"]
    * double paranthesis "((...))" was not heard by the annotator
      and can be empty but if not empty the annotator made their
      best attempt of transcribing what was said.
    * What's "[mn]" ? (can't find source?)
        * Inaudible
        * seems to be backchannel or laughter
        * oh/uh-huh/mhm/hehe
    * Names/accronyms (can't find source?)
        * t._v. = TV
        * m._t._v. = MTV
    """

    # Noise
    s = re.sub(r"\[noise\]", "", s)
    # laughter
    s = re.sub(r"\[laughter\]", "", s)
    # lipsmack
    s = re.sub(r"\[lipsmack\]", "", s)
    # sigh
    s = re.sub(r"\[sigh\]", "", s)
    # [mn] inaubible?
    s = re.sub(r"\[mn\]", "", s)

    # clean restarts
    # if remove_restarts=False "h-" -> "h"
    # if remove_restarts=True  "h-" -> ""
    if remove_restarts:
        s = re.sub(r"(\w+)-\s", " ", s)
        s = re.sub(r"(\w+)-$", r"", s)
    else:
        s = re.sub(r"(\w+)-\s", r"\1 ", s)
        s = re.sub(r"(\w+)-$", r"\1", s)

    # doubble paranthesis (DP) with included words
    # sometimes there is DP inside another DP
    s = re.sub(r"\(\(((.*?)+)\)\)", r"\1", s)
    s = re.sub(r"\(\(((.*?)+)\)\)", r"\1", s)

    # empty doubble paranthesis
    s = re.sub(r"\(\(\s*\)\)", "", s)

    # Names/accronyms
    s = re.sub(r"\.\_", "", s)

    # remove punctuation
    # (not included in annotations but artifacts from above)
    s = re.sub(r"\.", "", s)

    # remove double spacing on last
    s = re.sub(r"\s\s+", " ", s)
    return s.strip()  # remove whitespace start/end


def get_audio_path(nnn, root, ext=".sph"):
    dir = nnn[:3]
    n = int(nnn)
    if n < 800:
        d = 1
    else:
        a = n - 800
        d = a // 900 + 2
    return join(root, f"fisher_eng_tr_sp_d{d}/audio/{dir}/fe_03_{nnn}{ext}")


def get_transcript_path(nnn, root):
    dir = nnn[:3]
    return join(root, "fe_03_p1_tran/data/trans", f"{dir}/fe_03_{nnn}.txt")


def get_paths(nnn, root, ext=".sph"):
    audio_path = get_audio_path(nnn, root, ext=ext)
    transcript = get_transcript_path(nnn, root)
    return transcript, audio_path


def load_transcript(path, apply_regexp=True, remove_restarts=False):
    """
    Load the speakers as appropriate channels
    """
    # anno = {"A": [], "B": []}
    anno = [[], []]
    for row in read_txt(path):
        if row == "":
            continue

        split_row = row.split(" ")

        if split_row[0] == "#":
            continue

        s = float(split_row[0])
        e = float(split_row[1])
        speaker = split_row[2].replace(":", "")
        channel = SPEAKER2CHANNEL[speaker]
        text = " ".join(split_row[3:])

        if apply_regexp:
            text = fisher_regexp(text, remove_restarts=remove_restarts)

        # Omit empty
        if len(text) == 0:
            continue

        anno[channel].append({"start": s, "end": e, "text": text})
    return anno


def extract_vad_list(anno):
    vad = [[], []]
    for channel in [0, 1]:
        for utt in anno[channel]:
            s, e = utt["start"], utt["end"]
            vad[channel].append((s, e))
    return vad


def get_text_context(dialog, end, start=0):
    """
    Extract dialog context with utterances starting/ending after/before `start`/`end`

    This is on an utterance level and so will omit longer utterances that end after `end`, even though
    a lot of words may be included in [start, end].
    """
    context = [[], []]
    for channel in [0, 1]:
        ends = torch.tensor(dialog[channel]["end"])
        end_idx = torch.where(ends <= end)[0][-1].item()
        start_idx = 0
        if start > 0:
            starts = torch.tensor(dialog[channel]["start"])
            start_idx = torch.where(starts >= start)[0][0].item()
        context[channel].append(
            {
                "text": dialog[channel]["text"][start_idx : end_idx + 1],
                "start": dialog[channel]["start"][start_idx : end_idx + 1],
                "end": dialog[channel]["end"][start_idx : end_idx + 1],
            }
        )
    return context


if __name__ == "__main__":

    root = "/home/erik/projects/data/Fisher"
    nnn = "00001"
    trans_path, audio_path = get_paths(nnn, root, ext=".sph")

    anno = load_transcript(trans_path)
    # audio = load_waveform(audio_path) # can't load .sph with torchaudio
    vad = extract_vad_list(anno)

    p1 = get_audio_path("00001", root)
    p2 = get_audio_path("05100", root)
    p3 = get_audio_path("05300", root)

    s = "hello [noise] are [laughter] (( uh-huh )) you (( watching tv on )) m._t._v. (( oh that's not old ))"
    s = "h- he- hello uh-huh ver-"
    # s = "hello [noise] how [laughter] are (( ))"
    # s = re.sub(r"\(\(((\s\w*)+)\)\)", r"\1", s)
    # s = re.sub(r"\(\(((.*?)+)\)\)", r"\1", s)
    # print(s)
    s = fisher_regexp(s, True)
    print(s)

    # s = "cette plus facile (( au senegal aussi parce que je pouvais ))"
    s = "'(( demander beaucoup (( )) parce que ))'"
    # s = "'(( demander beaucoup (( bla )) parce que ))'"
    # s = 'demander beaucoup (( parce que ))'
    d = fisher_regexp(s)
    print(d)
