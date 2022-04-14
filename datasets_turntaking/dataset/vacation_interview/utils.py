from os.path import join, basename, exists
from glob import glob

from datasets_turntaking.utils import read_json


SPEAKER2CHANNEL = {"agent": 0, "user": 1}


def get_sessions(root):
    return [
        basename(d).replace(".json", "") for d in glob(join(root, "dialogs/*.json"))
    ]


def get_paths(session, root):
    audio_path = join(root, "audio", session + ".wav")
    transcript = join(root, "dialogs", session + ".json")
    vad_path = join(root, "vad", session + ".json")
    return transcript, audio_path, vad_path


def get_vad_path(session, root):
    vad_path = join(root, "vad", session + ".json")
    if exists(vad_path):
        return vad_path
    return False


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
            s, e = utt["start"], utt["end"]
            vad[channel].append((s, e))
    return vad


if __name__ == "__main__":

    from datasets_turntaking.utils import load_waveform, read_json
    import sounddevice as sd

    root = "/home/erik/projects/data/projection_dialogs"

    sessions = get_sessions(root)

    p = get_vad_path("session_0_baseline", root)
    read_json(p)

    sessions = get_sessions(root)
    session = sessions[0]
    path, audio_path, vad_path = get_paths(session, root)

    anno = load_transcript(path)
    vad = read_json(vad_path)

    x, sr = load_waveform(audio_path, normalize=True)
    sd.play(x[1], samplerate=sr)
