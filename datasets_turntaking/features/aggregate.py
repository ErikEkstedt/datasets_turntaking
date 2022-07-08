from os.path import join
import math
import torch
from torch.distributions import Categorical
from datasets_turntaking.dialog_audio_dm import DialogAudioDM
from vap_turn_taking import VAP
from tqdm import tqdm

import matplotlib.pyplot as plt


def bits_to_nats(bits):
    return bits / torch.tensor([math.e]).log2()


def nats_to_bits(nats):
    return nats / torch.tensor([2.0]).log()


# Get frequency stats over labels


if __name__ == "__main__":

    conf = DialogAudioDM.load_config()
    conf["dataset"]["waveform"] = False
    conf["dataset"]["vad"] = True
    conf["dataset"]["vad_history"] = False
    conf["dataset"]["datasets"] = ["fisher", "switchboard"]
    conf["dataset"]["audio_duration"] = 20
    conf["dataset"]["vad_hz"] = 50
    conf["dataset"]["vad_horizon"] = 2
    conf["dataset"]["flip_channels"] = False
    dm = DialogAudioDM(
        datasets=conf["dataset"]["datasets"],
        type=conf["dataset"]["type"],
        sample_rate=conf["dataset"]["sample_rate"],
        waveform=conf["dataset"]["waveform"],
        audio_duration=conf["dataset"]["audio_duration"],
        audio_normalize=conf["dataset"]["audio_normalize"],
        audio_overlap=conf["dataset"]["audio_overlap"],
        vad_hz=conf["dataset"]["vad_hz"],
        vad_horizon=conf["dataset"]["vad_horizon"],
        vad_history=conf["dataset"]["vad_history"],
        vad_history_times=conf["dataset"]["vad_history_times"],
        flip_channels=conf["dataset"]["flip_channels"],
        batch_size=32,
        num_workers=4,
        pin_memory=False,
    )
    dm.prepare_data()
    dm.setup(None)
    print(dm)
    print(f"Number of samples for sliding window {dm.train_dset.audio_step_time}s step")
    print("Train: ", len(dm.train_dset))
    print("Val: ", len(dm.val_dset))
    print("Test: ", len(dm.test_dset))
    # VAP objective
    vapper = VAP(type="discrete", frame_hz=conf["dataset"]["vad_hz"])
    print("Hz: ", vapper.frame_hz)
    print("n_classes: ", vapper.n_classes)

    # Each 'batch' in the dataset contains the keys ['waveform', 'vad', 'dset_name', 'session']
    # dloader = dm.train_dataloader()
    freq = torch.zeros(vapper.n_classes)
    for i, batch in enumerate(
        tqdm(dm.train_dataloader(), desc="Extract TRAIN label freq")
    ):
        y = vapper.extract_label(batch["vad"])
        freq += y.flatten(0).bincount(minlength=vapper.n_classes)
    for i, batch in enumerate(tqdm(dm.val_dataloader(), desc="Extract VAL label freq")):
        y = vapper.extract_label(batch["vad"])
        freq += y.flatten(0).bincount(minlength=vapper.n_classes)
    for i, batch in enumerate(
        tqdm(dm.val_dataloader(), desc="Extract TEST label freq")
    ):
        y = vapper.extract_label(batch["vad"])
        freq += y.flatten(0).bincount(minlength=vapper.n_classes)
    path = "assets/data"
    torch.save(freq, join(path, "freq.pt"))

    # Sort based on most common (the enumeration is completely arbitrary from the start)
    n, labels = freq.sort(descending=True)
    p = n / n.sum()

    # freq = freq/freq.sum()
    freq_dist = Categorical(p)
    ebits = nats_to_bits(freq_dist.entropy())[0]
    print(f"Entropy: {freq_dist.entropy()} nats")
    print(f"Entropy: {ebits} bits")
    print(
        f"PPL: {freq_dist.perplexity()}"
    )  # (== freq_dist.entropy().exp()) ppl from nats is the same as from bits

    # Plot distribution
    xx = torch.arange(vapper.n_classes)
    fig, ax = plt.subplots(1, 1)
    # ax.plot(xx, p.log())
    ax.semilogy(xx, p, label="Frequency")
    ax.hlines(
        y=1 / freq_dist.perplexity(),
        xmin=0,
        xmax=len(xx),
        color="red",
        linestyle="dashed",
        label=f"Entropy: {round(freq_dist.entropy().item(), 3)}",
    )
    ax.legend()
    # ax.bar(xx, n.log())
    # ax.semilogy(xx, n)
    ax.set_title(f"Label Distribution: {freq_dist.entropy()}")
    plt.show()
