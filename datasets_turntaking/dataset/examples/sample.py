from os.path import join
import sounddevice as sd
import matplotlib.pyplot as plt

from datasets_turntaking.utils import load_waveform, repo_root
from datasets_turntaking.features.plot_utils import plot_melspectrogram


PATH = join(repo_root(), "datasets_turntaking/examples/assets/her.wav")

if __name__ == "__main__":

    x, sr = load_waveform(PATH, sample_rate=16000, normalize=True, mono=True)
    vad = [[], []]

    fig, ax = plt.subplots(1, 1)
    m = plot_melspectrogram(x[0], ax=ax, frame_time=0.1, hop_time=0.05, sample_rate=sr)
    plt.tight_layout()
    plt.pause(0.1)

    sd.play(x[0], samplerate=sr)
