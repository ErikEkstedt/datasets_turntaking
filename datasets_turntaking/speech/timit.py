from os.path import join
from glob import glob
from torch.utils.data import Dataset
from datasets_turntaking.utils import load_waveform


def read_phn(path, encoding="utf-8"):
    starts, ends, phonemes = [], [], []
    with open(path, "r", encoding=encoding) as f:
        for line in f.readlines():
            s, e, p = line.strip().split(" ")
            starts.append(int(s))
            ends.append(int(e))
            phonemes.append(p)
    return {"start": starts, "end": ends, "phoneme": phonemes}


def read_text(path, encoding="utf-8"):
    starts, ends, text = [], [], []
    with open(path, "r", encoding=encoding) as f:
        for line in f.readlines():
            tmp = line.strip().split(" ")
            s, e = tmp[:2]
            t = " ".join(tmp[2:])
            starts.append(int(s))
            ends.append(int(e))
            text.append(t)
    return {"start": starts, "end": ends, "text": text}


class TimitDataset(Dataset):
    def __init__(self, files):
        super().__init__()
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wavpath = self.files[idx]
        x, sr = load_waveform(wavpath)
        phon = read_phn(wavpath.replace(".WAV", ".PHN").replace(".wav", ".phn"))
        text = read_text(wavpath.replace(".WAV", ".TXT").replace(".wav", ".txt"))[
            "text"
        ]
        return x, sr, phon, text


def get_timit_sample_dset(root, n=4):
    sample_files = glob(join(root, "TIMIT/TEST/DR1/FAKS0", "*[.wav .WAV]"))
    sample_files.sort()
    sample_files = sample_files[:n]
    return TimitDataset(sample_files)
