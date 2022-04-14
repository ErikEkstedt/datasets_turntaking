from os.path import basename, join
from os import makedirs
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

from datasets_turntaking.utils import load_waveform, write_json

try:
    from pyannote.audio import Pipeline
except:
    raise ModuleNotFoundError(
        "Requires `pyannote` please install: https://github.com/pyannote/pyannote-audio"
    )


def extract_vad(root, vadder):
    vad_root = join(root, "vad")
    makedirs(vad_root, exist_ok=True)
    audio_files = glob(join(args.root, "audio/*.wav"))
    audio_files.sort()
    for filepath in tqdm(audio_files, desc=vad_root):
        session = basename(filepath).replace(".wav", "")
        x, sr = load_waveform(filepath, sample_rate=16000)

        vad_list = [[], []]
        for channel in range(x.shape[0]):
            audio_in_memory = {"waveform": x[channel : channel + 1], "sample_rate": sr}
            out = vadder(audio_in_memory)
            for segment in out.get_timeline():
                vad_list[channel].append(
                    [round(segment.start, 2), round(segment.end, 2)]
                )

        vad_path = join(vad_root, session + ".json")
        write_json(vad_list, vad_path)
    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root", type=str)
    args = parser.parse_args()

    # args.root = "/home/erik/projects/data/projection_dialogs"
    # root = args.root

    vadder = Pipeline.from_pretrained("pyannote/voice-activity-detection")
    extract_vad(args.root, vadder)
