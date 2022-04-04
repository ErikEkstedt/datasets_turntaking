from os.path import join
from datasets_turntaking.utils import read_txt


def get_audio_path(nnn, root):
    n = int(nnn)
    dir = nnn[:3]
    d = int(n / 900 + 1)
    return join(root, f"fisher_eng_tr_sp_d{d}/audio/{dir}/fe_03_{nnn}.sph")


def get_transcript_path(nnn, root):
    dir = nnn[:3]
    return join(root, "fe_03_p1_tran/data/trans", f"{dir}/fe_03_{nnn}.txt")


def get_paths(nnn, root):
    audio_path = get_audio_path(nnn, root)
    transcript = get_transcript_path(nnn, root)
    return transcript, audio_path


def load_transcript(path):
    anno = {"A": [], "B": []}
    for row in read_txt(path):
        if row == "":
            continue

        split_row = row.split(" ")

        if split_row[0] == "#":
            continue

        s = float(split_row[0])
        e = float(split_row[1])
        speaker = split_row[2].replace(":", "")
        text = " ".join(split_row[3:])
        anno[speaker].append({"start": s, "end": e, "text": text})
    return anno


def extract_vad_list(anno):
    vad = [[], []]
    for channel, speaker in zip([0, 1], ["A", "B"]):
        for utt in anno[speaker]:
            s, e = utt["start"], utt["end"]
            vad[channel].append((s, e))
    return vad


if __name__ == "__main__":

    root = "/home/erik/projects/data/Fisher"
    nnn = "00001"
    trans_path, audio_path = get_paths(nnn, root)

    anno = load_transcript(trans_path)
    # audio = load_waveform(audio_path) # can't load .sph with torchaudio
    vad = extract_vad_list(anno)
