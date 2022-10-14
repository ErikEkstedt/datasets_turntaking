from os.path import join, expanduser
from datasets import load_dataset


if __name__ == "__main__":

    DSET_PATH = "/home/erik/projects/CCConv/datasets_turntaking/datasets_turntaking/dataset/spoken_dialog/fisher/fisher.py"
    word_paths = join(
        expanduser("~"), "projects/data/Fisher/fisher_transcripts_word_level"
    )
    root = join(expanduser("~"), "projects/data/Fisher")
    dset = load_dataset(
        DSET_PATH, name="default", split="validation", word_level_transcripts=word_paths
    )
