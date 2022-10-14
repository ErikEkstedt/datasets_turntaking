import torch
from os.path import join, exists
import re
from datasets_turntaking.utils import read_txt
from typing import Dict, List, Union


"""

1. Download audio data
2. Download transcript data
3. Extract into $FISHER_DATA_ROOT
4. Convert all audio '.sph' files to '.wav'


Fisher Audio files are downloaded in two files from LDC 
1. `fisher_eng_tr_sp_LDC2004S13.zip.001`,
2. `fisher_eng_tr_sp_LDC2004S13.zip.002`
and concatenated to a single file: `cat fisher_eng_tr_sp_LDC2004S13.zip.00* > fisher.zip`
which is then extracted into $FISHER_DATA_ROOT to produce the following structure:

$FISHER_DATA_ROOT/
└── fisher_eng_tr_sp_d1
    └── audio
        ├── ...
        └── 000
            ├── fe_03_00001.wav
            └── ...

The transcripts are downloaded as `fe_03_p1_tran_LDC2004T19.tgz` and extracted to $FISHER_DATA_ROOT
to produce the following structure:

$FISHER_TRANS_ROOT/
└── fe_03_p1_tran
    └── data
        ├── bbn_orig
        │   ├── ...
        │   └── 058
        └── trans
            └── 000
                ├── fe_03_00001.txt
                └── ...
"""

SPEAKER2CHANNEL = {"A": 0, "B": 1}

REL_TRANSCRIPT_ROOT = "fe_03_p1_tran/data/trans"
REL_WORD_LEVEL_ROOT = "fisher_transcripts_word_level"


def get_data_paths(nnn: str, root: str) -> Dict:
    sub_dir = nnn[:3]
    n = int(nnn)
    audio_dir_num = 1
    if n >= 800:
        audio_dir_num = (n - 800) // 900 + 2
    audio_path = join(
        root,
        f"fisher_eng_tr_sp_d{audio_dir_num}",
        "audio",
        sub_dir,
        f"fe_03_{nnn}.wav",
    )
    trans_path = join(root, REL_TRANSCRIPT_ROOT, sub_dir, f"fe_03_{nnn}.txt")
    word_path_a = join(root, REL_WORD_LEVEL_ROOT, sub_dir, f"fe_03_{nnn}_A_words.txt")
    word_path_b = join(root, REL_WORD_LEVEL_ROOT, sub_dir, f"fe_03_{nnn}_B_words.txt")
    return {
        "audio": audio_path,
        "utterance": trans_path,
        "word": {"A": word_path_a, "B": word_path_b},
    }


def extract_channel_dialog(
    path: str, ipu_thresh: float = 0.05
) -> List[Dict[str, Union[int, float, str]]]:
    transcript = read_txt(path)
    tmp_dialog = []
    # First entry
    start, end, word = transcript[0].split()
    start = float(start)
    end = float(end)
    last_end = end
    entry = {"start": start, "end": end, "text": word}
    for start_end_word in transcript[1:]:
        start, end, word = start_end_word.split()
        start = float(start)
        end = float(end)
        if start - last_end < ipu_thresh:
            # concatenate with previous
            entry["text"] += " " + word
            entry["end"] = end
        else:
            # New entry
            tmp_dialog.append(entry)
            entry = {"start": start, "end": end, "text": word}
        last_end = end
    # Add last entry
    tmp_dialog.append(entry)
    return tmp_dialog


def extract_vad_list_from_words(nnn, root, min_word_vad_diff=0.05):
    paths = get_data_paths(nnn, root)
    if not (exists(paths["word"]["A"]) and exists(paths["word"]["B"])):
        return None
    utt_a = read_txt(paths["word"]["A"])
    utt_b = read_txt(paths["word"]["B"])
    vad_list = [[], []]
    for channel, utterances in enumerate([utt_a, utt_b]):
        start, end, _ = utterances[0].split()
        start, end = float(start), float(end)
        for start_end_word in utterances[1:]:
            s, e, _ = start_end_word.split()
            s, e = float(s), float(e)
            if s - end < min_word_vad_diff:
                end = e
            else:
                vad_list[channel].append((round(start, 2), round(end, 2)))
                start = s
                end = e
        # add last entry
        vad_list[channel].append((start, end))
    return vad_list


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

    vad_list = extract_vad_list_from_words(nnn, root=root, min_word_vad_diff=0.05)

    # Session names in Fisher
    nnn = "00001"
    paths = get_data_paths(nnn, root)
    dialog = load_transcript(paths["utterance"])

    # dialog = [[A-dialog], [B-dialog]]
    a = dialog[0]

    a_utterances = extract_channel_dialog(paths["word"]["A"], ipu_thresh=0.05)
    b_utterances = extract_channel_dialog(paths["word"]["B"], ipu_thresh=0.05)

    trans_path, audio_path = get_paths(nnn, root, ext=".sph")

    anno = load_transcript(trans_path)

    # audio = load_waveform(audio_path) # can't load .sph with torchaudio
    vad_list = extract_vad_list(anno)

    # p1 = get_audio_path("00001", root)
    # p2 = get_audio_path("05100", root)
    # p3 = get_audio_path("05300", root)
    #
    # s = "hello [noise] are [laughter] (( uh-huh )) you (( watching tv on )) m._t._v. (( oh that's not old ))"
    # s = "h- he- hello uh-huh ver-"
    # # s = "hello [noise] how [laughter] are (( ))"
    # # s = re.sub(r"\(\(((\s\w*)+)\)\)", r"\1", s)
    # # s = re.sub(r"\(\(((.*?)+)\)\)", r"\1", s)
    # # print(s)
    # s = fisher_regexp(s, True)
    # print(s)
    #
    # # s = "cette plus facile (( au senegal aussi parce que je pouvais ))"
    # s = "'(( demander beaucoup (( )) parce que ))'"
    # # s = "'(( demander beaucoup (( bla )) parce que ))'"
    # # s = 'demander beaucoup (( parce que ))'
    # d = fisher_regexp(s)
    # print(d)
