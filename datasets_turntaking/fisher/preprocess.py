from argparse import ArgumentParser
from os.path import exists
from tqdm import tqdm
from datasets_turntaking.preprocess_utils import sph_to_wav, delete_path
from datasets_turntaking.fisher.utils import get_audio_path
from datasets_turntaking.fisher.fisher import TOTAL_FILES


def to_wav_and_delete(nnn, root):
    filepath = get_audio_path(nnn, root, ext=".sph")

    if exists(filepath):
        _ = sph_to_wav(filepath)
        delete_path(filepath)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--root", type=str)
    args = parser.parse_args()

    for n in tqdm(range(1, TOTAL_FILES + 1), desc="sph to wav"):
        nnn = str(n).zfill(5)
        to_wav_and_delete(nnn, root=args.root)
