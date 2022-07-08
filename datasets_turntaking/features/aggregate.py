from os.path import join
import math
import torch
from tqdm import tqdm
from torch.distributions import Categorical

from datasets_turntaking.dialog_audio_dm import DialogAudioDM
from datasets_turntaking.utils import write_json
from vap_turn_taking import VAP
from vap_turn_taking.config.example_data import event_conf
from vap_turn_taking import TurnTakingEvents

import matplotlib.pyplot as plt


def bits_to_nats(bits):
    return bits / torch.tensor([math.e]).log2()


def nats_to_bits(nats):
    return nats / torch.tensor([2.0]).log()


def get_bincount(y, events, event_name="shift", n=None, n_classes=256):
    ev = events[event_name][:, :n]
    w = torch.where(ev)[:2]
    return y[w].flatten(0).bincount(minlength=n_classes)


def get_statistics(dm, vapper, savepath="assets/data"):
    """Iterate over the entire dataset to extract label statistics"""

    def iterate_data(stats, dm, vapper, eventer, split="train"):
        if split == "val":
            dloader = dm.val_dataloader()
        elif split == "test":
            dloader = dm.test_dataloader()
        else:
            dloader = dm.train_dataloader()

        # Iterate over the entire dataset
        for batch in tqdm(dloader, desc=f"Extract {split.upper()} label freq"):
            y = vapper.extract_label(batch["vad"])
            n = y.shape[1]
            events = eventer(vad=batch["vad"])
            # Frequencies at label specific events
            stats["frequency"] += (
                y[:, :n].flatten(0).bincount(minlength=vapper.n_classes)
            )
            stats["shift"] += get_bincount(
                y, events, event_name="shift", n=n, n_classes=vapper.n_classes
            )
            stats["hold"] += get_bincount(
                y, events, event_name="hold", n=n, n_classes=vapper.n_classes
            )
            stats["predict_shift"] += get_bincount(
                y,
                events,
                event_name="predict_shift_pos",
                n=n,
                n_classes=vapper.n_classes,
            )
            stats["predict_bc"] += get_bincount(
                y, events, event_name="predict_bc_pos", n=n, n_classes=vapper.n_classes
            )
        return stats

    eventer = TurnTakingEvents(
        hs_kwargs=event_conf["hs"],
        bc_kwargs=event_conf["bc"],
        metric_kwargs=event_conf["metric"],
        frame_hz=dm.vad_hz,
    )

    stats = {
        "frequency": torch.zeros(vapper.n_classes),
        "shift": torch.zeros(vapper.n_classes),
        "hold": torch.zeros(vapper.n_classes),
        "predict_shift": torch.zeros(vapper.n_classes),
        "predict_bc": torch.zeros(vapper.n_classes),
    }

    stats = iterate_data(stats, dm, vapper, eventer, split="test")
    stats = iterate_data(stats, dm, vapper, eventer, split="val")
    stats = iterate_data(stats, dm, vapper, eventer, split="train")

    torch.save(stats, join(savepath, "frequency_stats.pt"))
    json_stats = {}
    for stat, freqs in stats.items():
        json_stats[stat] = freqs.tolist()
    write_json(stats, join(savepath, "frequency_stats.json"))
    return stats


def plot_event_label(stats, plot=True):
    """
    Plot event specific label distribution subsets
    """
    # Shift
    ev = "shift"
    sidx = stats[ev].nonzero()
    sn = stats[ev][sidx].squeeze()
    sn = sn / sn.sum()
    # Hold
    ev = "hold"
    hidx = stats[ev].nonzero()
    hn = stats[ev][hidx].squeeze()
    hn = hn / hn.sum()
    # predShift
    ev = "predict_shift"
    psidx = stats[ev].nonzero()
    psn = stats[ev][psidx].squeeze()
    psn = psn / psn.sum()
    # Hold
    ev = "predict_bc"
    pbcidx = stats[ev].nonzero()
    pbcn = stats[ev][pbcidx].squeeze()
    pbcn = pbcn / pbcn.sum()
    fig, ax = plt.subplots(2, 2, sharey=True)
    ax[0, 0].plot(sn.sort(descending=True)[0], linewidth=2, color="g", label="shift")
    ax[0, 1].plot(hn.sort(descending=True)[0], linewidth=2, color="b", label="hold")
    ax[1, 0].plot(
        psn.sort(descending=True)[0],
        linewidth=2,
        color="darkgreen",
        label="predict-shift",
    )
    ax[1, 1].plot(
        pbcn.sort(descending=True)[0], linewidth=2, color="r", label="predict-bc"
    )
    for row in ax:
        for a in row:
            a.legend()

    if plot:
        plt.pause(0.01)
    return fig, ax


def plot_label_frequencies(freq, plot=True):
    """Plot label frequency distribution over entire dataset"""
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
    xx = torch.arange(len(freq))
    fig, ax = plt.subplots(1, 1)
    # ax.plot(xx, p.log())
    ax.semilogy(xx, p, label="Frequency")
    ax.hlines(
        y=1 / freq_dist.perplexity(),
        xmin=0,
        xmax=len(xx),
        color="red",
        linestyle="dashed",
        label=f"Entropy: {round(freq_dist.entropy().item(), 3)} \nPPL: {round(freq_dist.perplexity().item(), 3)}",
    )
    ax.legend()
    # ax.bar(xx, n.log())
    # ax.semilogy(xx, n)
    ax.set_title("Label Distribution")

    if plot:
        plt.pause(0.01)

    return fig, ax


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
    # Get label statistics
    stats = get_statistics(dm, vapper)

    stats = torch.load("assets/data/frequency_stats.pt")
    fig, ax = plot_label_frequencies(stats["frequency"], plot=False)
    plt.show()
    _ = plot_event_label(stats, plot=False)
    plt.show()
