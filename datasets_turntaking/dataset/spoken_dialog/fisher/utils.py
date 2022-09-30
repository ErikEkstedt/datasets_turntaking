import torch
from os.path import join
from datasets_turntaking.utils import read_txt


SPEAKER2CHANNEL = {"A": 0, "B": 1}


# TODO:Specific cleaning of text?


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


def load_transcript(path):
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
