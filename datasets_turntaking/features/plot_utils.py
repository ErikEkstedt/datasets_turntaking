import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Optional, Tuple, List, Union
import matplotlib.pyplot as plt
import torch


def plot_stereo_mel_spec(
    mel_spec: torch.Tensor,
    ax: List[mpl.axes.Axes],
    vad: Optional[Union[None, torch.Tensor]] = None,
    no_ticks: bool = False,
    cmap: str = "inferno",
    interpolation: bool = True,
    plot: bool = False,
) -> List[mpl.axes.Axes]:

    assert (
        mel_spec.ndim == 3
    ), f"Expects single channel waveform of shape (N_channels, N_MELS, N_Frames). Got: {mel_spec.shape}"

    colors = ["b", "orange"]
    n_channels, n_mels, n_frames = mel_spec.shape

    for ch in range(n_channels):
        tmp_vad = None
        if vad is not None:
            tmp_vad = vad[ch]

        plot_mel_spec(
            mel_spec[ch],
            vad=tmp_vad,
            ax=ax[ch],
            no_ticks=no_ticks,
            cmap=cmap,
            vad_color=colors[ch],
            interpolation=interpolation,
        )
    if plot:
        plt.pause(0.1)
    return ax


def plot_mel_spec(
    mel_spec: torch.Tensor,
    ax: mpl.axes.Axes,
    vad: Optional[torch.Tensor] = None,
    no_ticks: bool = False,
    cmap: str = "viridis",
    interpolation: bool = True,
    vad_color: str = "blue",
    plot: bool = False,
) -> List[mpl.axes.Axes]:
    assert (
        mel_spec.ndim == 2
    ), f"Expects single channel waveform of shape (N_MELS, N_Frames). Got: {mel_spec.shape}"

    n_mels, n_frames = mel_spec.shape

    interp = None
    if not interpolation:
        interp = "none"

    ax.imshow(
        mel_spec,
        aspect="auto",
        origin="lower",
        interpolation=interp,
        cmap=cmap,
        vmin=-1.5,
        vmax=1.5,
    )
    if vad is not None:
        ax.plot(vad[:n_frames] * (n_mels - 1), alpha=0.9, linewidth=5, color=vad_color)
    if no_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if plot:
        plt.pause(0.1)
    return ax


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
    assert (
        vad_oh.ndim == 2
    ), f"Expects vad_oh of shape (n_frames, 2) but got: {tuple(vad_oh.shape)}"
    assert (
        vad_oh.shape[-1] == 2
    ), f"Expects vad_oh of shape (n_frames, 2) but got: {tuple(vad_oh.shape)}"

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    x = torch.arange(vad_oh.shape[0]) + 0.5  # fill_between step = 'mid'
    if label is not None:
        ax.fill_between(
            x, 0, vad_oh[:, 0], step="mid", alpha=alpha, color=colors[0], label=label[1]
        )
        ax.fill_between(
            x,
            0,
            -vad_oh[:, 1],
            step="mid",
            alpha=alpha,
            label=label[0],
            color=colors[1],
        )
        ax.legend(loc=legend_loc)
    else:
        ax.fill_between(
            x=x, y1=0, y2=vad_oh[:, 0], step="mid", alpha=alpha, color=colors[0]
        )
        ax.fill_between(
            x=x, y1=0, y2=-vad_oh[:, 1], step="mid", alpha=alpha, color=colors[1]
        )
    ax.hlines(y=0, xmin=0, xmax=len(x), color="k", linestyle="dashed")
    ax.set_xlim([0, vad_oh.shape[0]])
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


def plot_batch_sample(
    waveform: torch.Tensor,
    vad: Optional[torch.Tensor] = None,
    vad_hz: Optional[int] = 50,
    sample_rate: int = 16000,
    plot: bool = False,
) -> Tuple[Figure, List[Axes]]:
    """
    dset = DialogAudioDataset(
        dataset=dset_hf, type="sliding", vad_history=True, vad_hz=50, audio_mono=False
    )
    batch = dset[102]

    fig, ax = plot_batch_sample(
        waveform=batch["waveform"][0],
        vad=batch["vad"][0, :-100],
        sample_rate=dset.sample_rate,
        plot=False,
    )
    plt.show()
    """
    assert (
        waveform.ndim == 2
    ), f"Waveform must be of shape (n_channels, n_samples) but got {tuple(waveform.shape)}"
    if vad is not None:
        assert (
            vad.ndim == 2
        ), f"VAD must be of shape (n_frames, 2) but got {tuple(vad.shape)}"
        assert (
            vad.shape[-1] == 2
        ), f"VAD must contain TWO speakers of shape (n_frames, 2) but got {tuple(vad.shape)}"
    n_channels = waveform.shape[0]

    ii = 0
    n_figs = 3
    if vad is None:
        n_figs -= 1

    if n_channels == 1:
        fig, ax = plt.subplots(n_figs, 1, figsize=(9, 6))
        _ = plot_melspectrogram(waveform[0], ax=ax[ii])
        ax[0].set_ylabel("Melspectrogram")
        ax[0].set_yticks([])
        ax[0].set_xticks([])
        ii += 1
    else:
        n_figs += 1  # extra fig for second channel spectrogram
        fig, ax = plt.subplots(n_figs, 1, figsize=(9, 6))

        v0 = None
        if vad is not None:
            v0 = vad[:, 0]
        _ = plot_melspectrogram(waveform[0], ax=ax[ii], vad=v0, vad_hz=vad_hz)
        ax[ii].set_ylabel("Melspectrogram")
        ax[ii].set_yticks([])
        ax[ii].set_xticks([])
        ii += 1

        v1 = None
        if vad is not None:
            v1 = vad[:, 1]
        _ = plot_melspectrogram(
            waveform[1], ax=ax[ii], vad=v1, vad_hz=vad_hz, vad_color="orange"
        )
        ax[ii].set_ylabel("Melspectrogram")
        ax[ii].set_yticks([])
        ax[ii].set_xticks([])
        ii += 1

    if vad is not None:
        _ = plot_vad_oh(vad, ax=ax[ii])
        ax[ii].set_xticks([])
        ii += 1
    _ = plot_waveform(waveform, ax=ax[ii], sample_rate=sample_rate)
    ax[ii].set_yticks([])
    ax[ii].set_ylabel("Waveform")

    if plot:
        plt.pause(0.1)
    return fig, ax
