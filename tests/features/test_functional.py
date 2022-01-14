import pytest
import torch
import librosa

from datasets_turntaking.features.functional import lpc, __lpc, __window_frames


@pytest.mark.features
@pytest.mark.functional
def test_lpc_single():
    y, _ = librosa.load(librosa.ex("trumpet"), duration=0.020)
    y = torch.from_numpy(y)
    order = 2

    alphas_librosa = torch.from_numpy(librosa.lpc(y.numpy(), order))
    alphas = __lpc(y, order).squeeze()  # remove batch/frame-dim -> numpy

    diff = (alphas_librosa - alphas).abs().mean()
    assert diff.sum() < 1e-5


@pytest.mark.features
@pytest.mark.functional
def test_lpc():

    order = 2
    waveform, sr = librosa.load(librosa.ex("trumpet"), sr=16000, duration=0.1)
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    frame_length = int(0.04 * sr)
    hop_length = int(0.01 * sr)
    l = lpc(waveform, order=2, frame_length=frame_length, hop_length=hop_length)[
        0
    ]  # first batch dim

    frames = __window_frames(waveform, frame_length, hop_length)
    frame_alphas = []
    for tmp_frame in frames[0]:
        frame_alphas.append(torch.from_numpy(librosa.lpc(tmp_frame.numpy(), order)))
    frame_alphas = torch.stack(frame_alphas)

    diff = (frame_alphas - l).abs()
    assert (diff > 1e-5).sum() == 0, "torch LPC != Librosa LPC with sensitivity 1e-5"
