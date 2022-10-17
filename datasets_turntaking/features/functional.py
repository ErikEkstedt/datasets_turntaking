import torch
import torch.nn.functional as F
import torchaudio.functional as AF

import einops


SAMPLE_RATE: int = 16_000
HOP_LENGTH: int = 320  # 0.02s @ 16khz
FRAME_LENGTH: int = 800  # 0.05s @ 16khz


def mask_around_vad(
    waveform: torch.Tensor,
    vad: torch.Tensor,
    vad_hz: int,
    sample_rate: int = SAMPLE_RATE,
    scale: float = 0.1,
) -> torch.Tensor:
    assert (
        vad.shape[-1] == 2
    ), f"Expects vad of shape (B, N_FRAMES, 2) but got {vad.shape}"

    v_mask = vad.permute(0, 2, 1)

    B, C, _ = waveform.shape
    if B > 1 and C > 1:
        w_tmp = einops.rearrange(waveform, "b c s -> (b c) s")
        v_mask = einops.rearrange(v_mask, "b c f -> (b c) f")
        if vad_hz != sample_rate:
            v_mask = AF.resample(v_mask, orig_freq=vad_hz, new_freq=sample_rate)
        w_tmp = w_tmp * v_mask[:, : w_tmp.shape[-1]] * scale
        waveform = einops.rearrange(w_tmp, "(b c) s -> b c s", b=B, c=C)
    else:
        if vad_hz != sample_rate:
            v_mask = AF.resample(v_mask, orig_freq=vad_hz, new_freq=sample_rate)
        waveform = waveform * v_mask[:, :, : waveform.shape[-1]] * scale
    return waveform


def zero_crossings(y: torch.Tensor) -> torch.Tensor:
    s = torch.signbit(y)
    s = s[..., :-1] != s[..., 1:]
    z = torch.zeros_like(s[..., :1])
    return torch.cat((z, s), dim=-1)


def zero_crossing_rate(
    y: torch.Tensor,
    frame_length: int = FRAME_LENGTH,
    hop_length: int = HOP_LENGTH,
    center: bool = True,
) -> torch.Tensor:
    if center:
        pad = int(frame_length // 2)
        y = F.pad(y, (pad, pad))
    s_frames = y.unfold(dimension=-1, size=frame_length, step=hop_length)
    cross = zero_crossings(s_frames)
    return cross.float().mean(dim=-1)


def rms_torch(
    y: torch.Tensor,
    frame_length: int = FRAME_LENGTH,
    hop_length: int = HOP_LENGTH,
    center: bool = True,
    mode: str = "reflect",
) -> torch.Tensor:
    if center:
        pad = int(frame_length // 2)
        if mode == "reflect":
            if y.ndim == 1:
                y = F.pad(y.view(1, 1, -1), [pad, pad], mode=mode)
            else:
                y = F.pad(y.view(y.shape[0], 1, -1), [pad, pad], mode=mode)
            y = y.squeeze()
        else:
            y = F.pad(y, (pad, pad))
    frames = y.unfold(dimension=-1, size=frame_length, step=hop_length)
    rms = frames.pow(2).mean(dim=-1).sqrt()
    if rms.ndim == 1:
        rms = rms.unsqueeze(0)
    return rms


def lpc_frames(
    waveform: torch.Tensor,
    frame_size: int,
    hop_size: int,
    padding: bool = True,
    window=torch.hann_window,
) -> torch.Tensor:
    if padding:
        # pad before to not 'overlook' into the future
        waveform = F.pad(waveform, (frame_size, 0), "constant", 0.0)
        # waveform = F.pad(waveform, (0, frame_size), "constant", 0.0)
    frames = waveform.unfold(dimension=-1, size=frame_size, step=hop_size)
    frames -= frames.mean(dim=-1, keepdim=True)
    frames *= window(frame_size)
    return frames


def __lpc(y: torch.Tensor, order) -> torch.Tensor:
    """

    PyTorch implementation of Librosa's LPC algorithm

        https://librosa.org/doc/main/generated/librosa.lpc.html#librosa-lpc

    based on the Paper:

        A New Autoregressive Spectrum Analysis Algorithm, LARRY MARPLE, 1980

        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1163429

    """
    # This implementation follows the description of Burg's algorithm given in
    # section III of Marple's paper referenced in the docstring.
    #
    # We use the Levinson-Durbin recursion to compute AR coefficients for each
    # increasing model order by using those from the last. We maintain two
    # arrays and then flip them each time we increase the model order so that
    # we may use all the coefficients from the previous order while we compute
    # those for the new one. These two arrays hold ar_coeffs for order M and
    # order M-1.  (Corresponding to a_{M,k} and a_{M-1,k} in eqn 5)

    # Add batch dimension
    if y.ndim < 2:
        y = y.unsqueeze(0)

    if y.ndim < 3:
        y = y.unsqueeze(0)

    # assert y.ndim == 3, "Made to process (Batch, Frames, Samples)"

    n_batch, n_frames, _ = y.size()
    ar_coeffs = torch.zeros(
        (n_batch, n_frames, order + 1),
        dtype=y.dtype,
        device=y.device,
        requires_grad=False,
    )
    ar_coeffs_prev = torch.zeros(
        (n_batch, n_frames, order + 1),
        dtype=y.dtype,
        device=y.device,
        requires_grad=False,
    )
    ar_coeffs[..., 0] = 1.0
    ar_coeffs_prev[..., 0] = 1.0

    # These two arrays hold the forward and backward prediction error. They
    # correspond to f_{M-1,k} and b_{M-1,k} in eqns 10, 11, 13 and 14 of
    # Marple. First they are used to compute the reflection coefficient at
    # order M from M-1 then are re-used as f_{M,k} and b_{M,k} for each
    # iteration of the below loop

    fwd_pred_error = y[..., 1:]
    bwd_pred_error = y[..., :-1]

    # DEN_{M} from eqn 16 of Marple.
    den = torch.einsum(
        "bfi,bfi -> bf", [fwd_pred_error, fwd_pred_error]
    ) + torch.einsum("bfi,bfi -> bf", [bwd_pred_error, bwd_pred_error])

    for i in range(order):
        if (den <= 0).sum() > 0:
            raise FloatingPointError("numerical error, input ill-conditioned?")
        # not_ill_cond = (den > 0).float()
        # den *= not_ill_cond

        # Eqn 15 of Marple, with fwd_pred_error and bwd_pred_error
        # corresponding to f_{M-1,k+1} and b{M-1,k} and the result as a_{M,M}
        # reflect_coeff = dtype(-2) * np.dot(bwd_pred_error, fwd_pred_error) / dtype(den)
        reflect_coeff = (
            -2.0
            * torch.einsum("bfi, bfi -> bf", [bwd_pred_error, fwd_pred_error])
            / den
        )

        # Now we use the reflection coefficient and the AR coefficients from
        # the last model order to compute all of the AR coefficients for the
        # current one.  This is the Levinson-Durbin recursion described in
        # eqn 5.
        # Note 1: We don't have to care about complex conjugates as our signals
        # are all real-valued
        # Note 2: j counts 1..order+1, i-j+1 counts order..0
        # Note 3: The first element of ar_coeffs* is always 1, which copies in the reflection coefficient at the end of the new AR coefficient array
        # after the preceding coefficients
        ar_coeffs_prev, ar_coeffs = ar_coeffs, ar_coeffs_prev
        for j in range(1, i + 2):
            ar_coeffs[..., j] = (
                ar_coeffs_prev[..., j]
                # + reflect_coeff[..., 0] * ar_coeffs_prev[..., i - j + 1]
                + reflect_coeff * ar_coeffs_prev[..., i - j + 1]
            )

        # Update the forward and backward prediction errors corresponding to
        # eqns 13 and 14.  We start with f_{M-1,k+1} and b_{M-1,k} and use them
        # to compute f_{M,k} and b_{M,k}
        fwd_pred_error_tmp = fwd_pred_error
        fwd_pred_error = fwd_pred_error + reflect_coeff.unsqueeze(-1) * bwd_pred_error
        bwd_pred_error = (
            bwd_pred_error + reflect_coeff.unsqueeze(-1) * fwd_pred_error_tmp
        )

        # SNIP - we are now done with order M and advance. M-1 <- M
        # Compute DEN_{M} using the recursion from eqn 17.
        # reflect_coeff = a_{M-1,M-1}      (we have advanced M)
        # den =  DEN_{M-1}                 (rhs)
        # bwd_pred_error = b_{M-1,N-M+1}   (we have advanced M)
        # fwd_pred_error = f_{M-1,k}       (we have advanced M)
        # den <- DEN_{M}                   (lhs)

        # librosa-numpy
        # q = dtype(1) - reflect_coeff ** 2
        # den = q * den - bwd_pred_error[-1] ** 2 - fwd_pred_error[0] ** 2

        q = 1.0 - reflect_coeff ** 2
        den = q * den - bwd_pred_error[..., -1] ** 2 - fwd_pred_error[..., 0] ** 2

        # Shift up forward error.
        #
        # fwd_pred_error <- f_{M-1,k+1}
        # bwd_pred_error <- b_{M-1,k}
        #
        # N.B. We do this after computing the denominator using eqn 17 but
        # before using it in the numerator in eqn 15.
        fwd_pred_error = fwd_pred_error[..., 1:]
        bwd_pred_error = bwd_pred_error[..., :-1]
    return ar_coeffs


def __window_frames(
    x: torch.Tensor,
    frame_length: int = FRAME_LENGTH,
    hop_length: int = HOP_LENGTH,
    padding: bool = True,
    window=torch.hann_window,
) -> torch.Tensor:
    if padding:
        # pad before to not 'overlook' into the future
        x = F.pad(x, (frame_length, 0), "constant", 0.0)

    frames = x.unfold(dimension=-1, size=frame_length, step=hop_length)
    frames -= frames.mean(dim=-1, keepdim=True)
    frames *= window(frame_length)
    return frames


def lpc(
    waveform: torch.Tensor,
    order: int = 2,
    frame_length: int = FRAME_LENGTH,
    hop_length: int = HOP_LENGTH,
    padding: bool = True,
    window=torch.hann_window,
) -> torch.Tensor:
    frames = __window_frames(waveform, frame_length, hop_length, padding, window)
    return __lpc(frames, order)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import scipy
    import librosa
    import numpy as np

    import torchaudio.transforms as AT

    y = torch.randn(32000).numpy()
    y, sr = librosa.load(librosa.ex("trumpet"), duration=0.020)
    y = torch.from_numpy(y)
    order = 2

    waveform, sr = librosa.load(librosa.ex("trumpet"), sr=16000, duration=0.1)
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    frame_length = int(0.04 * sr)
    hop_length = int(0.01 * sr)
    l = lpc(waveform, order=2, frame_length=frame_length, hop_length=hop_length)
    print(l)

    frames = __window_frames(waveform, frame_length, hop_length)
    frame_alphas = []
    for tmp_frame in frames[0]:
        frame_alphas.append(torch.from_numpy(librosa.lpc(tmp_frame.numpy(), order)))
    frame_alphas = torch.stack(frame_alphas)
    print(frame_alphas)

    alphas = librosa.lpc(y.numpy(), order)
    alphas2 = lpc(y, order)

    b = np.hstack([[0], -1 * alphas[1:]])
    y_hat = scipy.signal.lfilter(b, [1], y)

    fig, ax = plt.subplots()
    ax.plot(y)
    ax.plot(y_hat, linestyle="--")
    ax.legend(["y", "y_hat"])
    ax.set_title("LP Model Forward Prediction")
    plt.pause(0.1)
