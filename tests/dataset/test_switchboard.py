import pytest
from datasets_turntaking import DialogAudioDM


@pytest.mark.switchboard
@pytest.mark.dm
@pytest.mark.parametrize(
        ["type", "vad", "vad_history", "vad_hz", "sample_rate"], 
        [
            ("sliding", True, True, 100, 16000), 
            ("sliding", True, True, 10, 16000), 
            ("sliding", False, False, 100, 16000),
            ("sliding", False, False, 100, 8000)
            ])
def test_dm(type, vad, vad_history, vad_hz, sample_rate):
    dm = DialogAudioDM(
        datasets=["switchboard"],
        type=type,
        sample_rate=sample_rate,
        audio_duration=10,
        audio_normalize=True,
        audio_overlap=5,
        vad_hz=vad_hz,
        vad_horizon=2,
        vad_history=vad_history,
        vad_history_times=[60, 30, 15, 10, 5],
        batch_size=4,
        num_workers=0,
    )
    dm.prepare_data()
    dm.setup(None)

    N = 5

    inputs = f"type: {type}, vad: {vad}, vad_history: {vad_history}, sample_rate: {sample_rate}, "

    try:
        for ii, batch in enumerate(dm.val_dataloader()):
            if ii == N:
                break
    except Exception as e:
        assert False, inputs+f"Validation dataloader broke {e}"

    
    try:
        for ii, batch in enumerate(dm.train_dataloader()):
            if ii == N:
                break
    except Exception as e:
        assert False, inputs+f"Train dataloader broke {e}"

    try:
        for ii, batch in enumerate(dm.test_dataloader()):
            if ii == N:
                break
    except Exception as e:
        assert False, inputs+f"Test dataloader broke {e}"
