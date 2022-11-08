from os.path import join, basename, exists
from glob import glob
from datasets_turntaking.utils import read_json


SPEAKER2CHANNEL = {"agent": 0, "user": 1}

AGENTS = ["baseline", "prediction"]


def get_sessions(root):
    sessions = []
    for session in [basename(d) for d in glob(join(root, "session*"))]:
        for agent in AGENTS:
            sessions.append(session + "/" + agent)
    sessions.sort()
    return sessions


def get_paths(root, session):
    audio_path = join(root, session, "dialog.wav")
    dialog_path = join(root, session, "dialog.json")
    return audio_path, dialog_path


def load_transcript(path):
    dialog = read_json(path)
    anno = [[], []]
    for turn in dialog["turns"]:
        channel = SPEAKER2CHANNEL[turn["name"]]
        anno[channel].append(
            {
                "start": float(turn["start_time"]),
                "end": float(turn["end_time"]),
                "text": turn["utterance"],
            }
        )
    return anno


def extract_vad_list(anno):
    vad = [[], []]
    for channel in [0, 1]:
        for utt in anno[channel]:
            s, e = round(utt["start"], 2), round(utt["end"], 2)
            vad[channel].append((s, e))
    return vad


if __name__ == "__main__":
    from datasets_turntaking.utils import load_waveform, read_json
    from os import listdir
    from os.path import expanduser

    root = join(expanduser("~"), "projects/data/vacation_interview")
    listdir(root)
    sessions = get_sessions(root)
    print(sessions)

    for session in sessions:
        audio_path, dialog_path = get_paths(root, session)
        waveform, sr = load_waveform(audio_path, sample_rate=16000)
        dialog = load_transcript(dialog_path)
        vad_list = extract_vad_list(dialog)
        break
