from librosa.display import waveshow
import matplotlib.pyplot as plt

import torch
from torch.nn import Sequential
import torchaudio.transforms as AT


def plot_waveform(waveform, ax, sample_rate=None):
    waveshow(
        waveform.numpy(),
        sr=sample_rate,
        color="b",
        alpha=0.4,
        ax=ax,
        label="waveform",
    )
    return ax


def plot_melspectrogram(
    waveform, ax, n_mels=80, frame_time=0.05, hop_time=0.01, sample_rate=16000
):
    waveform = waveform.detach().cpu()

    # Features
    frame_length = int(frame_time * sample_rate)
    hop_length = int(hop_time * sample_rate)
    melspec = Sequential(
        AT.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=frame_length,
            hop_length=hop_length,
            n_mels=n_mels,
        ),
        AT.AmplitudeToDB(),
    )(waveform)

    # im = ax.imshow(melspec, aspect="auto", interpolation="none", origin="lower")
    im = ax.imshow(
        melspec,
        aspect="auto",
        interpolation="none",
        origin="lower",
        extent=(0, melspec.shape[1], 0, melspec.shape[0]),
    )
    return melspec


def plot_vad_list(vad, end_time, target_time=None, ax=None, plot=True):
    def draw_vad_segment(
        ax, y, start, end, color, linewidth=1, linestyle="solid", boundaries=True
    ):
        ax.hlines(y, start, end, color, linewidth=linewidth, linestyle=linestyle)
        if boundaries:
            ax.vlines(start, y + 0.05, y - 0.05, color, linewidth=1, linestyle="solid")
            ax.vlines(end, y + 0.05, y - 0.05, color, linewidth=1, linestyle="solid")

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(16, 3))
    ax.set_xlim([0, end_time])
    color = ["b", "g"]
    ys = [0.1, -0.1]  # reverse order to have speaker 0 on top
    for ch, ch_vad in enumerate(vad):
        for s, e in ch_vad:
            draw_vad_segment(ax, y=ys[ch], start=s, end=e, color=color[ch], linewidth=2)
    if target_time:
        ax.vlines(
            x=target_time, ymin=ys[1] - 0.05, ymax=ys[0] + 0.05, color="r", linewidth=2
        )
    plt.tight_layout()
    if plot:
        plt.pause(0.1)
    return fig, ax


def plot_vad_oh(
    vad_oh,
    ax=None,
    colors=["b", "orange"],
    yticks=["B", "A"],
    ylabel=None,
    alpha=1,
    label=None,
    legend_loc="best",
    plot=False,
):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    x = torch.arange(vad_oh.shape[-1]) + 0.5  # fill_between step = 'mid'
    if label is not None:
        ax.fill_between(
            x, 0, vad_oh[0], step="mid", alpha=alpha, color=colors[0], label=label[1]
        )
        ax.fill_between(
            x, 0, -vad_oh[1], step="mid", alpha=alpha, label=label[0], color=colors[1]
        )
        ax.legend(loc=legend_loc)
    else:
        ax.fill_between(
            x=x, y1=0, y2=vad_oh[0], step="mid", alpha=alpha, color=colors[0]
        )
        ax.fill_between(
            x=x, y1=0, y2=-vad_oh[1], step="mid", alpha=alpha, color=colors[1]
        )
    ax.hlines(y=0, xmin=0, xmax=len(x), color="k", linestyle="dashed")
    ax.set_xlim([0, vad_oh.shape[-1]])
    ax.set_ylim([-1.05, 1.05])

    if yticks is None:
        ax.set_yticks([])
    else:
        ax.set_yticks([-0.5, 0.5])
        ax.set_yticklabels(yticks)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    plt.tight_layout()
    if plot:
        plt.pause(0.1)
    return fig, ax


def plot_vad_sample(
    waveform,
    vad,
    vad_labels,
    vad_current_frame=None,
    vad_bins=256,
    sample_rate=16000,
    ax=None,
    figsize=(16, 5),
    plot=False,
):
    if ax is None:
        fig, ax = plt.subplots(4, 1, figsize=figsize)

    assert len(ax) >= 4, "Must provide at least 4 ax"

    mel = plot_melspectrogram(
        waveform,
        ax=ax[0],
        n_mels=80,
        frame_time=0.05,
        hop_time=0.01,
        sample_rate=sample_rate,
    )
    ax[0].set_yticks([])
    ax[0].set_xticks([])
    ax[0].set_ylabel("mel")

    w = waveform[::10]
    ax[1].plot(w)
    ax[1].set_xlim([0, len(w)])
    ax[1].set_ylim([-1.0, 1.0])
    ax[1].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_ylabel("waveform")

    _ = plot_vad_oh(
        vad, ax=ax[2], label=["A", "B"], legend_loc="upper right", plot=False
    )
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_ylabel("vad")

    ax[3].step(torch.arange(len(vad_labels)), vad_labels)
    ax[3].set_ylim([0, vad_bins])
    ax[3].set_xlim([0, len(vad_labels)])
    ax[3].set_yticks([])
    ax[3].set_ylabel("idx")

    if vad_current_frame is not None:
        ax[-2].vlines(x=vad_current_frame, ymin=-1, ymax=1, color="k", linewidth=2)
        # ax[-1].vlines(x=vad_current_frame, ymin=-1, ymax=1, color="k", linewidth=2)

    plt.tight_layout()
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.02, hspace=0.02
    )
    if plot:
        plt.pause(0.1)

    return fig, ax
