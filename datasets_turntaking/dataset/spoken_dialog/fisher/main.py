from datasets_turntaking.fisher import load_fisher

from datasets_turntaking.fisher.utils import get_text_context
from datasets_turntaking.utils import load_waveform
from datasets_turntaking.utils import repo_root


if __name__ == "__main__":

    # import sounddevice as sd

    dset = load_dataset(DATASET_SCRIPT, name="default", split="train")

    dset2 = load_fisher(split="train")
    d2 = dset2[1]

    c = get_text_context(d2["dialog"], end=120, start=60)

    # x, sr = load_waveform(d["audio_path"])
    # s = min(d["dialog"]["A"]["start"][0], d["dialog"]["B"]["start"][0])
    # n = len(d["dialog"]["A"]["text"])
    # print(n)
    # n = len(d["dialog"]["B"]["start"])
    # print(n)
    # # start_sample = int(sr * s)
    # # x = x[:, start_sample:]
    # # sd.play(x[:, : int(120 * sr)].t(), samplerate=8000)
    #
    # sd.stop()
    #
    # print(d["dialog"]["A"]["text"][2])
    #
    # text = d["dialog"]["A"]["text"][2]
    #
    # s = int(sr * d["dialog"]["A"]["start"][2])
    # e = int(sr * d["dialog"]["A"]["end"][2])
    # sd.play(x[0, s:e].t(), samplerate=8000)
    #
    # print("-" * 50)
    # print(d["dialog"]["B"]["text"][:5])
    # d["dialog"]["A"]["text"][:5]
    # d["dialog"]["A"]["start"][:5]
    # d["dialog"]["B"]["text"][:5]
    # d["dialog"]["B"]["start"][:5]

    # Check alignment (waveform/spectrogram + start/end-times)
    # Check content .... listen...
