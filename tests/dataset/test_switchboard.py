import pytest
from datasets_turntaking import DialogAudioDM


@pytest.mark.switchboard
@pytest.mark.dm
@pytest.mark.parametrize(
    ["type", "vad", "vad_history", "vad_hz", "sample_rate"],
    [
        ("sliding", True, True, 100, 16000),
        ("sliding", True, True, 50, 16000),
        ("sliding", False, False, 50, 16000),
        ("sliding", False, False, 50, 8000),
    ],
)
def test_dm(type, vad, vad_history, vad_hz, sample_rate):
    dm = DialogAudioDM(
        datasets=["switchboard"],
        type=type,
        sample_rate=sample_rate,
        vad_hz=vad_hz,
        vad_history=vad_history,
        batch_size=4,
        num_workers=0,
    )
    dm.prepare_data()
    dm.setup("fit")
    dm.setup("test")

    N = 5

    inputs = f"type: {type}, vad: {vad}, vad_history: {vad_history}, sample_rate: {sample_rate}, "

    try:
        for ii, batch in enumerate(dm.val_dataloader()):
            if ii == N:
                break
    except Exception as e:
        assert False, inputs + f"Validation dataloader broke {e}"

    try:
        for ii, batch in enumerate(dm.train_dataloader()):
            if ii == N:
                break
    except Exception as e:
        assert False, inputs + f"Train dataloader broke {e}"

    try:
        for ii, batch in enumerate(dm.test_dataloader()):
            if ii == N:
                break
    except Exception as e:
        assert False, inputs + f"Test dataloader broke {e}"
