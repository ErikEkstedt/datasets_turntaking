import math
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
import torchaudio.transforms as AT
from scipy.signal import firwin


def bandpass(x, sample_rate=16000, order=150, f_hp=50, f_lp=1500, dec_factor=1):
    f1 = f_hp / (sample_rate / 2)
    f2 = f_lp / (sample_rate / 2)
    b = torch.from_numpy(firwin(order + 1, [f1, f2], pass_zero=False)).float()
    b = b.to(x.device)
    a = 1
    a = torch.zeros_like(b)
    a[0] = 1.0

    filtered = AF.lfilter(x, a_coeffs=a, b_coeffs=b, clamp=True, batching=True)

    if dec_factor != 1:
        # Decimate the filtered output.
        filtered = filtered[0 : x.shape[-1] : dec_factor]
        sample_rate = sample_rate / dec_factor
    return filtered, sample_rate


def compute_nlfer(
    x, sample_rate, f0_min=60, f0_max=400, frame_size=700, hop_length=200, n_fft=8192
):
    N_f0_min = round((f0_min * 2 / float(sample_rate)) * n_fft)
    N_f0_max = round((f0_max / float(sample_rate)) * n_fft)
    # S = AT.Spectrogram(
    #     n_fft=n_fft, win_length=frame_size, hop_length=hop_length, power=1
    # )(x)
    Specter = AT.Spectrogram(
        n_fft=n_fft, win_length=frame_size, hop_length=hop_length, power=1
    ).to(x.device)
    S = Specter(x)
    num = S[:, N_f0_min:N_f0_max].abs().sum(dim=1)
    den = num.mean(dim=-1, keepdim=True)
    return num / den


def compute_shc(
    x,
    sample_rate=16000,
    f0_min=60,
    f0_max=400,
    NH=3,  # number of harmonics
    WL=40,  # Hz
    n_fft=8192,
    hop_length=200,
    frame_size=700,
):
    sample_rate = float(sample_rate)
    # half of spectral window length (always centered around the focus freq bin)
    WL2 = WL // 2
    # frequency -> bins
    N_wl_half = round(n_fft * WL2 / sample_rate)
    N_f0_min = round(n_fft * f0_min / sample_rate)
    N_f0_max = round(n_fft * f0_max / sample_rate)

    # Energy Spectrogram
    # S = AT.Spectrogram(
    #     n_fft=n_fft, win_length=frame_size, hop_length=hop_length, power=1
    # )(x)
    Specter = AT.Spectrogram(
        n_fft=n_fft, win_length=frame_size, hop_length=hop_length, power=1
    ).to(x.device)
    S = Specter(x)

    # Calculate max/min frequency bins used for SHM
    shc = []
    for f in range(N_f0_min, N_f0_max):
        pi = S[:, f - N_wl_half : f + N_wl_half]
        for r in range(2, NH + 1):
            rf = r * f  # harmonic
            pi = pi * S[:, rf - N_wl_half : rf + N_wl_half]
        shc.append(pi.sum(dim=1))
    shc = torch.stack(shc, dim=-1)

    # Normalize
    shc /= shc.max(dim=-1, keepdim=True).values
    return shc


def compute_nccf(
    waveform: torch.Tensor,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
    freq_low: int,
    flip: bool = False,
) -> torch.Tensor:
    r"""
    Compute Normalized Cross-Correlation Function (NCCF).
    .. math::
        \phi_i(m) = \frac{\sum_{n=b_i}^{b_i + N-1} w(n) w(m+n)}{\sqrt{E(b_i) E(m+b_i)}},
    where
    :math:`\phi_i(m)` is the NCCF at frame :math:`i` with lag :math:`m`,
    :math:`w` is the waveform,
    :math:`N` is the length of a frame,
    :math:`b_i` is the beginning of frame :math:`i`,
    :math:`E(j)` is the energy :math:`\sum_{n=j}^{j+N-1} w^2(n)`.
    """

    # EPSILON = 10 ** (-9)
    EPSILON = 1e-9

    waveform_length = waveform.size()[-1]
    num_of_frames = int(math.ceil(waveform_length / hop_length))

    # Number of lags to check
    lags = int(math.ceil(sample_rate / freq_low))

    p = lags + num_of_frames * frame_length - waveform_length
    waveform = F.pad(waveform, (0, p))

    # Compute lags
    output_lag = []
    for lag in range(1, lags + 1):
        s_n = waveform[..., :-lag].unfold(-1, frame_length, hop_length)[
            ..., :num_of_frames, :
        ]
        s_nk = waveform[..., lag:].unfold(-1, frame_length, hop_length)[
            ..., :num_of_frames, :
        ]

        e0 = s_n.pow(2).sum(dim=-1)
        ek = s_nk.pow(2).sum(dim=-1)
        scale = 1.0 / (EPSILON + torch.sqrt(e0 * ek))
        output_frames = scale * (s_n * s_nk).sum(-1)

        output_lag.append(output_frames)
    nccf = torch.stack(output_lag, dim=-1)

    if flip:
        print("nccf: ", tuple(nccf.shape))
        nccf = torch.flip(nccf, dims=(nccf.ndim - 1,))
        print("nccf: ", tuple(nccf.shape))
    return nccf


def pyaapt(
    waveform,
    sample_rate=20000,
    f0_min=60,
    f0_max=400,
    vuv_threshold=0.75,
    filter_kwargs={"order": 150, "min_hz": 50, "max_hz": 1500, "dec_factor": 1},
    spec_kwargs={"frame_length": 700, "hop_length": 200, "n_fft": 8192},
    shc_kwargs={"NH": 3, "WL": 40},
):
    """PyTorch implementation of `amfm_decompy.yaapt`."""
    # 1. Preprocess
    # Add nonlinearity -> signal squared
    signal = torch.stack((waveform, waveform.pow(2)), dim=1)
    signal, sample_rate = bandpass(
        signal,
        sample_rate,
        order=filter_kwargs["order"],
        f_hp=filter_kwargs["min_hz"],
        f_lp=filter_kwargs["max_hz"],
        dec_factor=filter_kwargs["dec_factor"],
    )
    x, x_nonlinear = signal[:, 0], signal[:, 1]

    # 2. Spectrally based F0 track
    nlfer = compute_nlfer(
        signal[:, 0],  # filtered waveforms
        sample_rate,
        f0_min=f0_min,
        f0_max=f0_max,
        frame_size=spec_kwargs["frame_length"],
        hop_length=spec_kwargs["hop_length"],
        n_fft=spec_kwargs["n_fft"],
    )
    # threshold for voiced
    vuv = (nlfer > vuv_threshold).float()
    shc = compute_shc(
        # x_nonlinear,
        signal[:, 1],  # nonlinear filtered
        sample_rate,
        f0_min=f0_min,
        f0_max=f0_max,
        NH=shc_kwargs["NH"],  # number of harmonics
        WL=shc_kwargs["WL"],  # Hz
        hop_length=spec_kwargs["hop_length"],
        frame_size=spec_kwargs["frame_length"],
        n_fft=spec_kwargs["n_fft"],
    )

    # 3. Candidate F0 based on NCCF
    nccf_signal = compute_nccf(
        signal,
        sample_rate,
        frame_length=spec_kwargs["frame_length"],
        hop_length=spec_kwargs["hop_length"],
        freq_low=f0_min,
        flip=True,
    )
    nccf, nccf_nonlinear = nccf_signal[:, 0], nccf_signal[:, 1]

    return {
        "nlfer": nlfer,
        "shc": shc,
        "nccf": nccf,
        "nccf_nonlinear": nccf_nonlinear,
        "vuv": vuv,
        "signal": x,
    }


def figure_3(
    spectrum,
    shc,
    shc_lines=None,
    shc_thresh=0.2,
    same_length=False,
    plot=False,
    figsize=(16, 9),
):
    fig, ax = plt.subplots(2, 1, figsize=figsize)

    # Spectrum
    ax[0].plot(spectrum)
    ax[0].set_title("Spectrum")
    ax[0].set_ylabel("Amplitude")
    ax[0].set_xlabel("Frequency(Hz)")

    # Spectral harmonics correlation
    ax[1].plot(shc)

    if shc_lines is not None:
        ax[1].vlines(x=shc_lines, ymin=0, ymax=1, color="k", linewidth=0.5)
    else:
        candidate_idx = torch.where(shc >= shc_thresh)[0]
        if len(candidate_idx) > 0:
            ax[1].vlines(x=candidate_idx, ymin=0, ymax=1, color="k", linewidth=0.5)
    ax[1].set_title("Spectral harmonics correlation function")
    ax[1].set_ylabel("Amplitude")
    ax[1].set_xlabel("Frequency(Hz)")
    if same_length:
        ax[1].set_xlim(ax[0].get_xlim())

    plt.tight_layout()

    if plot:
        plt.pause(0.1)

    return fig, ax


def plot_intermediate(f0, vuv_threshold, mask=False, plot=True):
    print("NLFER: ", tuple(f0["nlfer"].shape))
    print("SHC: ", tuple(f0["shc"].shape))
    print("NCCF: ", tuple(f0["nccf"].shape))
    print("vuv: ", tuple(f0["vuv"].shape))

    imshow_kwargs = {
        "aspect": "auto",
        "origin": "lower",
        "interpolation": "none",
    }

    fig, ax = plt.subplots(3, 1)
    ax[0].set_title("Normalized low frequency energy ratio")
    ax[0].plot(f0["nlfer"][0], label="NLFER")
    ax[0].hlines(
        y=vuv_threshold, xmin=0, xmax=f0["vuv"].shape[-1], linestyle="dashed", color="k"
    )
    ax[0].plot(f0["vuv"][0], label="vuv")
    ax[0].legend(loc="upper left")
    ax[0].set_xlim([0, f0["nlfer"].shape[-1]])
    ax[0].set_xticks([])

    shc = f0["shc"][0].t()
    nccf = f0["nccf"][0].t()
    if mask:
        shc[:, torch.logical_not(f0["vuv"][0])] = 0.0
        nccf[:, torch.logical_not(f0["vuv"][0])] = -1.0

    ax[1].set_title("Spectral harmonics correlation")
    ax[1].imshow(shc, extent=(0, shc.shape[1], 0, shc.shape[0]), **imshow_kwargs)
    ax[1].set_xticks([])

    ax[2].set_title("Normalized cross correlation function")
    ax[2].imshow(nccf, extent=(0, nccf.shape[1], 0, nccf.shape[0]), **imshow_kwargs)

    plt.tight_layout()
    if plot:
        plt.pause(0.1)
    return fig, ax


def plot_shc_selection(shc, vuv, shc_thresh=0.2, plot=True):
    # vuv: nlfier >= NLFER_thresh1
    n_frames, n_bins = shc.shape[0], shc.shape[1]

    fig, ax = plt.subplots(3, 1)
    imshow_kwargs = {
        "aspect": "auto",
        "origin": "lower",
        "interpolation": "none",
    }

    # mask unvoiced below NLFER_thresh1
    shc[torch.logical_not(vuv)] = 0.0
    ax[0].set_title("Spectral harmonics correlation")
    ax[0].imshow(shc.t(), extent=(0, n_frames, 0, n_bins), **imshow_kwargs)
    ax[0].set_xticks([])

    # mask everything below peak thresh SHC_thresh
    shc[shc < shc_thresh] = 0.0
    ax[1].set_title("Peak thresh")
    ax[1].imshow(shc.t(), extent=(0, n_frames, 0, n_bins), **imshow_kwargs)
    # ax[1].set_xticks([])

    if plot:
        plt.pause(0.1)
    return fig, ax


def find_peaks(data, threshold, box_size=3):
    def maximum_filter(x, kernel_size):
        pad = (kernel_size - 1) // 2
        x = F.max_pool2d(x, kernel_size, stride=1, padding=pad)
        return x

    data_max = maximum_filter(data, box_size)
    peak_goodmask = data == data_max
    peak_goodmask = torch.logical_and(peak_goodmask, (data > threshold))
    locations = peak_goodmask.nonzero(as_tuple=True)
    peak_values = data[locations]
    return locations, peak_values


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from datasets_turntaking.utils import load_waveform
    import librosa
    from librosa import display
    import time
    import sounddevice as sd

    sample_rate = 20000
    waveform, sr = load_waveform(
        "assets/hello.wav", sample_rate=sample_rate, normalize=True, mono=True
    )
    waveform = torch.stack((waveform[0, 10000:], waveform[0, :-10000]))
    # waveform = torch.cat([waveform] * 50).to('cuda')
    # waveform = torch.cat([waveform] * 50)
    print("waveform: ", tuple(waveform.shape))

    sd.play(waveform[0], samplerate=sample_rate)

    # waveform = waveform.to("cuda")

    # WIP
    f0_min = 60
    f0_max = 400
    f0_mid = 150  # Hz
    # frame_time = 0.035  # original
    # hop_time = 0.01
    frame_time = 0.035  # original
    hop_time = 0.01
    vuv_threshold = 0.5
    filter_kwargs = {"order": 150, "min_hz": 50, "max_hz": 1500, "dec_factor": 1}
    spec_kwargs = {"frame_length": 700, "hop_length": 200, "n_fft": 8192}
    shc_kwargs = {"NH": 3, "WL": 40}
    t = time.time()
    f0 = pyaapt(
        waveform,
        sample_rate,
        f0_min,
        f0_max,
        vuv_threshold=vuv_threshold,
        filter_kwargs=filter_kwargs,
        spec_kwargs=spec_kwargs,
        shc_kwargs=shc_kwargs,
    )
    print("time: ", round(time.time() - t, 2))
    print("NLFER: ", tuple(f0["nlfer"].shape))
    print("SHC: ", tuple(f0["shc"].shape))
    print("NCCF: ", tuple(f0["nccf"].shape))
    print("NCCF_nonlinear: ", tuple(f0["nccf_nonlinear"].shape))
    print("vuv: ", tuple(f0["vuv"].shape))
    fig, ax = plot_intermediate(f0, vuv_threshold=vuv_threshold, mask=True)

    # SHC selection
    N_min = round(spec_kwargs["n_fft"] * f0_min / sample_rate)
    N_mid = round(spec_kwargs["n_fft"] * f0_mid / sample_rate)
    N_max = round(spec_kwargs["n_fft"] * f0_max / sample_rate)
    f0_cand_n_bins = N_max - N_min
    N_mid_rel = N_mid - N_min
    print("N_min: ", N_min)
    print("N_mid: ", N_mid)
    print("N_max: ", N_max)
    print("N_mid_rel: ", N_mid_rel)
    print("f0_cand_n_bins: ", f0_cand_n_bins)

    # find peaks
    # add doubling/halfing f0 candidates
    # if all peaks > f0_mid then add a candidate of half the frequency of the highest peak
    # if all peaks < f0_mid then add a candidate of double the frequency of the highest peak
    # shc -> vuv-filter -> shc-filter (shc_thresh) -> peak finder -> add additional candidates
    shc = f0["shc"].clone()
    nccf = f0["nccf"].clone()
    vuv = f0["vuv"]
    shc_thresh = 0.2
    n_frames, shc_bins = shc.shape[1], shc.shape[2]
    nccf_bins = nccf.shape[2]
    print("shc: ", tuple(shc.shape))
    print("vuv: ", tuple(vuv.shape))
    # mask nlfer/vuv
    shc[torch.logical_not(vuv)] = 0.0
    nccf[torch.logical_not(vuv)] = -1
    imshow_kwargs = {
        "aspect": "auto",
        "origin": "lower",
        "interpolation": "none",
    }

    # TODO: the peak finder does not work well for NCCF. misses clear peaks when box-size=3 (=1 does not do anything)
    # extract only the peak values
    loc, vals = find_peaks(shc, threshold=0.2)
    shc_peaks = torch.zeros_like(shc)
    shc_peaks[loc] = vals
    # extract only the peak values
    loc, vals = find_peaks(nccf + 1, threshold=1.4, box_size=3)
    nccf_peaks = torch.zeros_like(nccf)
    nccf_peaks[loc] = vals
    fig, ax = plt.subplots(2, 1)
    # mask everything below peak thresh SHC_thresh
    ax[0].set_title("SHC peaks")
    ax[0].imshow(shc_peaks[0].t(), extent=(0, n_frames, 0, shc_bins), **imshow_kwargs)
    # ax[1].set_xticks([])
    ax[1].set_title("NCCF peaks")
    ax[1].imshow(nccf_peaks[0].t(), extent=(0, n_frames, 0, nccf_bins), **imshow_kwargs)
    plt.pause(0.1)

    # this creates variable length windows on batched sequences
    # -> pad
    # Concatenate all voiced frames (omit unvoiced)
    peaks_concat = []
    for b in loc[0].unique():
        bidx = b == loc[0]
        frames = loc[1][bidx]
        tmp_batch = []
        for frame in frames.unique():
            fidx = frame == frames
            vidx = loc[2][bidx][fidx]
            tmp_values = vals[vidx]

    cpeaks = torch.zeros()
    # insert half/double candidates
    # median candidate for 7 point smoothed of highest merit in each frame

    fig, ax = plt.subplots(2, 1)
    # mask everything below peak thresh SHC_thresh
    ax[0].set_title("Peak thresh")
    ax[0].imshow(shc[0].t(), extent=(0, n_frames, 0, n_bins), **imshow_kwargs)
    # ax[1].set_xticks([])
    ax[1].set_title("Peaks")
    ax[1].imshow(peaks[0].t(), extent=(0, n_frames, 0, n_bins), **imshow_kwargs)
    plt.pause(0.1)
