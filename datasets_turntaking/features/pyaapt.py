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
    shc /= shc.max(dim=-2, keepdim=True).values
    return shc


def compute_nccf(
    waveform: torch.Tensor,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
    freq_low: int,
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

    # Number of lags to check
    lags = int(math.ceil(sample_rate / freq_low))
    waveform_length = waveform.size()[-1]
    num_of_frames = int(math.ceil(waveform_length / hop_length))

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

        # # Done in torchaudio.functional.functional._compute_nccf
        # e0 = (EPSILON + torch.norm(s_n, p=2, dim=-1)).pow(2)
        # ek = (EPSILON + torch.norm(s_nk, p=2, dim=-1)).pow(2)
        # output_frames = (s_n * s_nk).sum(-1) / e0 / ek

        e0 = s_n.pow(2).sum(dim=-1)
        ek = s_nk.pow(2).sum(dim=-1)
        den = EPSILON + torch.sqrt(e0 * ek)
        output_frames = (s_n * s_nk).sum(-1) / den

        # den = e0*ek
        # output_frames = (s1*s2).sum(-1) / den
        # output_frames = (
        #     (s1 * s2).sum(-1)
        #     / (EPSILON + torch.norm(s1, p=2, dim=-1)).pow(2)
        #     / (EPSILON + torch.norm(s2, p=2, dim=-1)).pow(2)
        # )

        # output_lag.append(output_frames.unsqueeze(-1))
        output_lag.append(output_frames)
    nccf = torch.stack(output_lag, dim=-1)
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


def nlfer_orig(signal, frame_size, frame_jump, nfft, parameters):
    import numpy.lib.stride_tricks as stride_tricks
    from scipy.signal.windows import hann

    def stride_matrix(vector, n_lin, n_col, hop):
        data_matrix = stride_tricks.as_strided(
            vector,
            shape=(n_lin, n_col),
            strides=(vector.strides[0] * hop, vector.strides[0]),
        )
        return data_matrix

    # ---------------------------------------------------------------
    # Set parameters.
    # ---------------------------------------------------------------
    N_f0_min = np.around((parameters["f0_min"] * 2 / float(signal.new_fs)) * nfft)
    N_f0_max = np.around((parameters["f0_max"] / float(signal.new_fs)) * nfft)
    window = hann(frame_size + 2)[1:-1]
    data = np.zeros((signal.size))  # Needs other array, otherwise stride and
    data[:] = signal.filtered  # windowing will modify signal.filtered

    # ---------------------------------------------------------------
    # Main routine.
    # ---------------------------------------------------------------
    samples = np.arange(
        int(np.fix(float(frame_size) / 2)),
        signal.size - int(np.fix(float(frame_size) / 2)),
        frame_jump,
    )

    data_matrix = np.empty((len(samples), frame_size))
    data_matrix[:, :] = stride_matrix(data, len(samples), frame_size, frame_jump)
    data_matrix *= window

    specData = np.fft.rfft(data_matrix, nfft)
    frame_energy = np.abs(specData[:, int(N_f0_min - 1) : int(N_f0_max)]).sum(axis=1)
    # pitch.set_energy(frame_energy, parameters['nlfer_thresh1'])
    # pitch.set_frames_pos(samples)
    return frame_energy, samples


def compare():
    import numpy as np
    import matplotlib.pyplot as plt
    import amfm_decompy.basic_tools as basic
    from amfm_decompy.pYAAPT import BandpassFilter  # , yaapt
    from datasets_turntaking.utils import load_waveform
    import librosa
    from librosa import display

    sample_rate = 20000
    waveform, sr = load_waveform(
        "assets/hello.wav", sample_rate=sample_rate, normalize=True, mono=True
    )

    # WIP
    sample_rate = 20000
    f0_min = 60
    f0_max = 400
    filter_kwargs = {"order": 150, "min_hz": 50, "max_hz": 1500, "dec_factor": 1}
    spec_kwargs = {"frame_length": 700, "hop_length": 200, "n_fft": 8192}
    shc_kwargs = {"NH": 3, "WL": 40}
    f0 = pyaapt(
        waveform,
        sample_rate,
        f0_min,
        f0_max,
        filter_kwargs=filter_kwargs,
        spec_kwargs=spec_kwargs,
        shc_kwargs=shc_kwargs,
    )

    # ---------------------------------------------------------------
    # Create the signal objects and filter them.
    # ---------------------------------------------------------------

    # bandpass
    parameters = {"dec_factor": 1, "bp_forder": 150, "bp_low": 50, "bp_high": 1500}
    # nfler
    nlfer_params = {
        "frame_length": 35,
        "frame_space": 10,
        "fft_length": 8192,
        "f0_min": 60,
        "f0_max": 400,
        "NLFER_Thresh1": 0.75,
    }
    parameters.update(nlfer_params)
    nfft = parameters["fft_length"]
    frame_size = int(np.fix(parameters["frame_length"] * sample_rate / 1000))
    frame_jump = int(np.fix(parameters["frame_space"] * sample_rate / 1000))

    signal = basic.SignalObj(
        data=waveform[0].numpy(),
        fs=sample_rate,
    )
    nonlinear_sign = basic.SignalObj(signal.data ** 2, signal.fs)
    fir_filter = BandpassFilter(signal.fs, parameters)
    signal.filtered_version(fir_filter)
    nonlinear_sign.filtered_version(fir_filter)

    filtered, _ = bandpass(waveform, sample_rate)
    nonlinear_filtered, _ = bandpass(waveform.pow(2), sample_rate)

    # s = librosa.stft(signal[0, 0].numpy(), center=False)
    # S = librosa.stft(signal[0, 1].numpy(), center=False)
    # fig, ax = plt.subplots(2, 1, sharex=True)
    # img = display.specshow(
    #     librosa.amplitude_to_db(s, ref=np.max), y_axis="log", x_axis="time", ax=ax[0]
    # )
    # img = display.specshow(
    #     librosa.amplitude_to_db(S, ref=np.max), y_axis="log", x_axis="time", ax=ax[1]
    # )
    # ax[0].set_title("Power spectrogram")
    # fig.colorbar(img, ax=ax, format="%+2.0f dB")
    # plt.pause(0.1)

    ################################################################3
    non_equal = (
        (filtered.squeeze() - torch.from_numpy(signal.filtered).float()).abs() > 1e-5
    ).sum()
    print("bandpass error: ", non_equal)
    non_equal = (
        (
            nonlinear_filtered.squeeze()
            - torch.from_numpy(nonlinear_sign.filtered).float()
        ).abs()
        > 1e-5
    ).sum()
    print("bandpass nonlinear error: ", non_equal)
    ################################################################3

    threshold = 0.75
    pitch_frame_energy, samples = nlfer_orig(
        signal,
        frame_size=frame_size,
        frame_jump=frame_jump,
        nfft=nfft,
        parameters=parameters,
    )
    print("pitch_frame_energy: ", tuple(pitch_frame_energy.shape))
    print("samples: ", tuple(samples.shape))
    pitch_frame_energy = np.concatenate((np.zeros(2), pitch_frame_energy))
    pitch_mean = pitch_frame_energy.mean()
    pitch_frame_energy = pitch_frame_energy / pitch_mean
    pitch_vuv = pitch_frame_energy > threshold

    # NLFER
    x_nlfer = nlfer(filtered, sample_rate, frame_size, hop_length=frame_jump)
    vuv = x_nlfer > threshold
    # vuv = (x_nlfer > 0.5).float()

    fig, ax = plt.subplots(1, 1)
    ax.plot(x_nlfer[0], label="new", alpha=0.6)
    ax.plot(vuv[0], label="vuv", alpha=0.6)
    ax.plot(pitch_frame_energy, label="orig", alpha=0.6)
    ax.plot(pitch_vuv, label="orig vuv", linestyle="dashed", alpha=0.6)
    ax.legend()
    plt.pause(0.1)

    ################################################################3
    # SHC
    shc_ = shc(
        filtered,
        sample_rate,
        n_fft=8192,
        hop_length=frame_jump,
        frame_size=frame_size,
        f0_min=60,
        f0_max=400,
        NH=3,
        WL=40,
    )
    print("shc: ", tuple(shc_.shape))

    fig, ax = plt.subplots(2, 1)
    f0_max_bin = shc_.argmax(dim=1)[0]
    f0_max_bin[torch.logical_not(vuv[0])] = -15
    ax[0].plot(f0_max_bin, label="shc")
    ax[0].legend()
    ax[1].plot(vuv[0])
    plt.pause(0.1)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from datasets_turntaking.utils import load_waveform
    from torchaudio.functional.functional import _compute_nccf
    import time

    sample_rate = 16000
    waveform, sr = load_waveform(
        "assets/hello.wav", sample_rate=sample_rate, normalize=True, mono=True
    )
    # waveform = torch.cat([waveform] * 50).to('cuda')
    # waveform = torch.cat([waveform] * 50)
    print("waveform: ", tuple(waveform.shape))

    # waveform = waveform.to("cuda")

    # WIP
    f0_min = 60
    f0_max = 400
    # frame_time = 0.035  # original
    # hop_time = 0.01
    frame_time = 0.035  # original
    hop_time = 0.01
    vuv_threshold = 0.75
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

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(f0["nlfer"][0], label="nlfer", alpha=0.6)
    ax[0].plot(f0["vuv"][0], label="vuv", alpha=0.6)
    ax[0].legend()
    f0_max_bin = f0["shc"].argmax(dim=-1)[0]
    f0_max_bin[torch.logical_not(f0["vuv"][0])] = -15
    ax[1].plot(f0_max_bin, label="shc")
    ax[1].legend()
    plt.pause(0.1)
    input()
