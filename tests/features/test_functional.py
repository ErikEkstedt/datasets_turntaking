import pytest
import torch
import datasets_turntaking.features.functional as DF


VAD_HZ = 50
SAMPLE_RATE = 16_000


@pytest.fixture
def data():
    data = torch.load("assets/vap_data.pt")
    if torch.cuda.is_available():
        data["shift"]["vad"] = data["shift"]["vad"].to("cuda")
        data["bc"]["vad"] = data["bc"]["vad"].to("cuda")
        data["only_hold"]["vad"] = data["only_hold"]["vad"].to("cuda")
        data["shift"]["waveform"] = data["shift"]["waveform"].to("cuda")
        data["bc"]["waveform"] = data["bc"]["waveform"].to("cuda")
        data["only_hold"]["waveform"] = data["only_hold"]["waveform"].to("cuda")
    return data


@pytest.mark.functional
def test_mask_around_vad_single(data):
    vad = data["shift"]["vad"]
    waveform = data["shift"]["waveform"]
    masked_waveform = DF.mask_around_vad(
        waveform.clone(), vad, vad_hz=VAD_HZ, sample_rate=SAMPLE_RATE
    )
    ws = waveform.shape
    mws = masked_waveform.shape
    assert ws == mws, f"Expected masked waveform shape {ws} got {mws}"
    assert (
        waveform != masked_waveform
    ).sum() > 0, f"Expected different values but waveform==masked_waveform"


@pytest.mark.functional
def test_mask_around_vad(data):
    vad = torch.cat(
        (
            data["shift"]["vad"],
            data["only_hold"]["vad"],
            data["bc"]["vad"],
        )
    )
    waveform = torch.cat(
        (
            data["shift"]["waveform"],
            data["only_hold"]["waveform"],
            data["bc"]["waveform"],
        )
    )
    masked_waveform = DF.mask_around_vad(
        waveform.clone(), vad, vad_hz=VAD_HZ, sample_rate=SAMPLE_RATE
    )
    ws = waveform.shape
    mws = masked_waveform.shape
    assert ws == mws, f"Expected masked waveform shape {ws} got {mws}"
    assert (
        waveform != masked_waveform
    ).sum() > 0, f"Expected different values but waveform==masked_waveform"


@pytest.mark.functional
def test_mel_spec(data):
    waveform = data["shift"]["waveform"]
    mel_spec = DF.log_mel_spectrogram(waveform)


# @pytest.mark.features
# @pytest.mark.functional
# def test_lpc_single():
#     y, _ = librosa.load(librosa.ex("trumpet"), duration=0.020)
#     y = torch.from_numpy(y)
#     order = 2
#
#     alphas_librosa = torch.from_numpy(librosa.lpc(y.numpy(), order))
#     alphas = __lpc(y, order).squeeze()  # remove batch/frame-dim -> numpy
#
#     diff = (alphas_librosa - alphas).abs().mean()
#     assert diff.sum() < 1e-5
#
#
# @pytest.mark.features
# @pytest.mark.functional
# def test_lpc():
#     order = 2
#     waveform, sr = librosa.load(librosa.ex("trumpet"), sr=16000, duration=0.1)
#     waveform = torch.from_numpy(waveform).unsqueeze(0)
#     frame_length = int(0.04 * sr)
#     hop_length = int(0.01 * sr)
#     l = lpc(waveform, order=2, frame_length=frame_length, hop_length=hop_length)[
#         0
#     ]  # first batch dim
#
#     frames = __window_frames(waveform, frame_length, hop_length)
#     frame_alphas = []
#     for tmp_frame in frames[0]:
#         frame_alphas.append(torch.from_numpy(librosa.lpc(tmp_frame.numpy(), order)))
#     frame_alphas = torch.stack(frame_alphas)
#
#     diff = (frame_alphas - l).abs()
#     assert (diff > 1e-5).sum() == 0, "torch LPC != Librosa LPC with sensitivity 1e-5"
