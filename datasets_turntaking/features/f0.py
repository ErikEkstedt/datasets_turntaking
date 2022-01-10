import math
import torch
import torchaudio
import torchaudio.functional as AF

from librosa import yin, pyin
from pysptk.sptk import swipe
import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT_np

# TODO: global F0 envelope -> phrase. smoothing over larger frames?


def YIN(
    y,
    fmin,
    fmax,
    sr,
    frame_length=2048,
    win_length=None,
    hop_length=None,
    trough_threshold=0.1,
    center=True,
    pad_mode="reflect",
):
    def _yin(y):
        return torch.from_numpy(
            yin(
                y.numpy(),
                fmin=fmin,
                fmax=fmax,
                sr=sr,
                frame_length=frame_length,
                win_length=win_length,
                hop_length=hop_length,
                trough_threshold=trough_threshold,
                center=center,
                pad_mode=pad_mode,
            )
        )

    if y.ndim == 1:
        return _yin(y)
    elif y.ndim == 2:
        f0 = []
        for x in y:
            f0.append(_yin(x))
        return torch.stack(f0)


def PYIN(
    y,
    fmin,
    fmax,
    sr,
    frame_length=2048,
    win_length=None,
    hop_length=None,
    n_thresholds=100,
    beta_parameters=(2, 18),
    boltzmann_parameter=2,
    resolution=0.1,
    max_transition_rate=35.92,
    switch_prob=0.01,
    no_trough_prob=0.01,
    fill_na=0,
    center=True,
    pad_mode="reflect",
):
    def _pyin(y):
        f0, _, _ = pyin(
            y.numpy(),
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=frame_length,
            win_length=win_length,
            hop_length=hop_length,
            n_thresholds=n_thresholds,
            beta_parameters=beta_parameters,
            boltzmann_parameter=boltzmann_parameter,
            resolution=resolution,
            max_transition_rate=max_transition_rate,
            switch_prob=switch_prob,
            no_trough_prob=no_trough_prob,
            fill_na=fill_na,
            center=center,
            pad_mode=pad_mode,
        )
        return torch.from_numpy(f0)

    if y.ndim == 1:
        return _pyin(y)
    elif y.ndim == 2:
        f0 = []
        for x in y:
            f0.append(_pyin(x))
        return torch.stack(f0)


def SWIPE(
    y,
    fmin,  # default in swipe
    fmax,  # default in swipe
    sr,
    hop_length,
    threshold=0.3,  # custom defualt (0.3 in swipe)
):
    def _swipe(y):
        return torch.from_numpy(
            swipe(
                y.contiguous().double().numpy(),
                fs=sr,
                hopsize=hop_length,
                min=fmin,
                max=fmax,
                threshold=threshold,
                otype="f0",
            )
        ).float()

    if y.ndim == 1:
        return _swipe(y)
    elif y.ndim == 2:  # (B, N)
        f0 = []
        for x in y:
            f0.append(_swipe(x))
        return torch.stack(f0)


def pYAAPT(
    waveform,
    sample_rate,
    fmin=60,
    fmax=400,
    frame_length=400,
    hop_length=200,
    interpolated=False,
):
    """pyaapt uses time in ms to define frame_length, frame_space"""

    n_frames = math.ceil(waveform.shape[-1] / hop_length)

    flength = int(1000 * frame_length / sample_rate)
    hlength = int(1000 * hop_length / sample_rate)

    def _pyaapt(waveform):
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()

        signal = basic.SignalObj(
            data=waveform,
            fs=sample_rate,
        )
        f = pYAAPT_np.yaapt(
            signal,
            f0_min=fmin,
            f0_max=fmax,
            frame_length=flength,
            frame_space=hlength,
            fft_length=frame_length,
        )
        vuv = torch.from_numpy(f.vuv).float()
        if interpolated:
            f = f.samp_interp
        else:
            f = f.samp_values

        f = torch.from_numpy(f).float()
        if len(f) != n_frames:
            n_diff = n_frames - len(f)
            if interpolated:
                diff = torch.ones((n_diff,)) * f[..., -1]
            else:
                diff = torch.zeros((n_diff,))
            f = torch.cat((f, diff))
            vuv = torch.cat(
                (
                    vuv,
                    torch.zeros(
                        n_diff,
                    ).bool(),
                )
            )
        return f, vuv

    if waveform.ndim == 1:
        return _pyaapt(waveform)
    f0, vuv = [], []
    for x in waveform:
        f_, vuv_ = _pyaapt(x)
        f0.append(f_)
        vuv.append(vuv_)
    return torch.stack(f0), torch.stack(vuv)


def f0_kaldi_torch(
    y, sr, fmin=60, fmax=400, frame_length=400, hop_length=200, **kwargs
):
    frame_length_ms = 1000 * frame_length / sr
    hop_length_ms = 1000 * hop_length / sr

    f0 = AF.compute_kaldi_pitch(
        y,
        sample_rate=sr,
        frame_length=frame_length_ms,
        frame_shift=hop_length_ms,
        min_f0=fmin,
        max_f0=fmax,
        **kwargs,
    )
    return f0[..., 1], f0[..., 0]


def f0_pyaapt(x, sample_rate, frame_length, hop_length):
    flength = float(int(1000 * frame_length / sample_rate))
    hlength = float(int(1000 * hop_length / sample_rate))

    # pad = torch.zeros((x.shape[0], frame_length // 2))
    # x = torch.cat([pad, x, pad], dim=-1)
    kwargs = {
        "frame_length": flength,
        "nccf_thresh1": 0.25,
        "tda_frame_length": flength,
        "frame_space": hlength,
    }

    if x.ndim == 1:
        signal = basic.SignalObj(data=x.numpy(), fs=sample_rate)
        f0 = torch.from_numpy(pYAAPT_np.yaapt(signal, **kwargs).samp_values).float()
    else:
        f0 = []
        for x_ in x:
            signal = basic.SignalObj(data=x_.numpy(), fs=sample_rate)
            f0.append(
                torch.from_numpy(pYAAPT_np.yaapt(signal, **kwargs).samp_values).float()
            )
        f0 = torch.stack(f0)
    return f0


def f0_norm_min_max(f0):
    f0 = f0.clone()
    if f0.ndim == 2:
        bsx, nsx = torch.where(f0 != 0)
        for i in range(f0.shape[0]):
            n = nsx[bsx == i]
            f0[i, n] = f0[i, n] - f0[i, n].mean()
            f0[i, n] -= f0[i, n].min()
            f0[i, n] = f0[i, n] / f0[i, n].abs().max()
    else:
        n = torch.where(f0 != 0)[0]
        f0[n] = f0[n] - f0[n].mean()
        f0[n] -= f0[n].min()
        f0[n] = f0[n] / f0[n].abs().max()
    return f0


def mean_subtracted_f0(f0, f0_mean):
    normed = torch.zeros_like(f0)
    bsx, nsx = torch.where(f0 != 0)
    for i in range(f0.shape[0]):
        n = nsx[bsx == i]
        f0_ = f0[i, n] - f0_mean[i]
        normed[i, n] = f0_
    return normed


def mean_ratio_f0(f0, f0_mean, log=True):
    """Text-Free Prosody"""
    normed = torch.zeros_like(f0)
    bsx, nsx = torch.where(f0 != 0)
    for i in range(f0.shape[0]):
        n = nsx[bsx == i]
        if log:
            f0_ = f0[i, n].log() - f0_mean[i].log()
        else:
            f0_ = f0[i, n] / f0_mean[i]
        normed[i, n] = f0_
    return normed


def quantize_f0(x):
    """
    We set the maximum length to
    be 32 and the bin width to be 1, resulting in 32
    bins. We quantize speaker-mean normalized log
    F0 lf into K = 32 bins such that each bin with
    boundaries [bi−1, bi] contains the same probability
    mass: P(lf ∈ [bi−1, bi]) = 1/K.
    """

    pros = Prosody(sample_rate=8000)
    f0_idx = pros.lab2idx["f0"]
    vuv_idx = pros.lab2idx["f0_vuv"]

    p = pros(x)

    fig, ax = plt.subplots(1, 1)
    im = ax.matshow(p, aspect="auto")
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    fig.colorbar(im, ax=ax)
    plt.pause(0.1)

    from sklearn.cluster import KMeans
    import einops

    # Kmeans modeling
    kmeans_model = KMeans(
        n_clusters=32,
        max_iter=10,
    )

    p = pros(x)
    lf = norm_f0(p[..., f0_idx], p[..., vuv_idx])
    boundaries = torch.linspace(-1.5, 1.5, 32)
    lfd = torch.bucketize(lf, boundaries)

    X = einops.rearrange(lfd, "b t -> (b t) 1")
    kmeans_model.fit(X=X)
    lfd1 = einops.rearrange(kmeans_model.predict(X), "(b t) -> b t", b=lfd.shape[0])


if __name__ == "__main__":

    from convlm.features.plots import plot_f0
    from convlm.features.features import rms_torch

    waveform_path = (
        "/home/erik/projects/data/switchboard/audio/swb1_d1/data/sw02001.wav"
    )
    waveform_path = "/home/erik/.cache/huggingface/datasets/downloads/extracted/92df0cd0fea86e61e83873cf7a55e7f85eab245d3710f4388b66ae3899d6ee7f/LibriSpeech/test-clean/61/70968/61-70968-0000.flac"
    x, sr = torchaudio.load(waveform_path)
    # x = x[:, int(sr*2):int(sr*30)]
    # f0, nccf = f0_kaldi_torch(x, sr, hop_length=160, frame_length=400)
    # x = x[:, :int(30*sr)]

    frame_length = 400
    hop_length = 200
    rms = rms_torch(x, frame_length=frame_length, hop_length=hop_length)
    f0 = f0_pyaapt(x, sample_rate=sr, frame_length=frame_length, hop_length=hop_length)
    # vuv = (f0 != 0).float()
    print("rms: ", rms.shape)
    print("f0: ", f0.shape)

    fig, ax = plot_f0(f0, f0, channel=0)

    fig, ax = plot_f0(f0[:, : int(8000 * 30)], vuv[:, : int(8000 * 30)], channel=0)
    frame_length = 400
    hop_length = 200
    f0_extractor = F0(
        sample_rate=sr,
        frame_length=frame_length,
        hop_length=hop_length,
        interpolated=True,
    )

    u = get_sample(0, 19, audio_root, dataset)
    frame_length_ms = 1000 * frame_length / sr
    hop_length_ms = 1000 * hop_length / sr
    print("frame_time: ", frame_length_ms)
    print("hop_time: ", hop_length_ms)

    f0, df0, f0_vuv = f0_extractor(u["audio"])
    fig, ax = plot_f0(f0, df0, channel=0)
