import pytest
import torch
import datasets_turntaking.features.transforms as DT

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


@pytest.mark.transforms
def test_log_mel_spec(data):
    waveform = torch.cat(
        (
            data["shift"]["waveform"],
            data["only_hold"]["waveform"],
            data["bc"]["waveform"],
        )
    )
    t = DT.LogMelSpectrogram()
    mel_spec = t(waveform)


@pytest.mark.transforms
def test_mask_scale(data):
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
    t = DT.VadMaskScale(vad_hz=VAD_HZ, sample_rate=SAMPLE_RATE)

    masked_waveform = t(waveform.clone(), vad)

    ws = waveform.shape
    mws = masked_waveform.shape
    assert ws == mws, f"Expected masked waveform shape {ws} got {mws}"
    assert (
        waveform != masked_waveform
    ).sum() > 0, f"Expected different values but waveform==masked_waveform"


@pytest.mark.transforms
def test_flip_batch(data):
    batch = {}
    batch["vad"] = torch.cat(
        (
            data["shift"]["vad"],
            data["only_hold"]["vad"],
            data["bc"]["vad"],
        )
    )
    batch["waveform"] = torch.cat(
        (
            data["shift"]["waveform"],
            data["only_hold"]["waveform"],
            data["bc"]["waveform"],
        )
    )
    t = DT.FlipBatch()
    flipped = t(batch)

    assert list(flipped.keys()) == list(
        batch.keys()
    ), f"not same keys! Expected {list(flipped.keys())} got {list(batch.keys())}"

    for k, v in batch.items():
        if k in t.flippable:
            assert (flipped[k][:, 0] != v[:, 1]).sum() > 0, f"{k} same as non-flipped"
            if k == "waveform":
                assert (
                    flipped[k][:, 0] != v[:, 1]
                ).sum() > 0, f"{k} same as non-flipped"
                assert (
                    flipped[k][:, 1] != v[:, 0]
                ).sum() > 0, f"{k} same as non-flipped"
            if k == "vad":
                assert (
                    flipped[k][..., 0] != v[..., 1]
                ).sum() > 0, f"{k} same as non-flipped"
                assert (
                    flipped[k][..., 1] != v[..., 0]
                ).sum() > 0, f"{k} same as non-flipped"
            if k == "vad_history":
                assert (flipped[k] != 1 - v).sum() > 0, f"{k} same as non-flipped"
