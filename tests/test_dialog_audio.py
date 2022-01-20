import pytest
from os import cpu_count

from datasets_turntaking import DialogAudioDM


@pytest.mark.slow
@pytest.mark.dialog_audio_dm
@pytest.mark.parametrize("type", ["sliding", "ipu"])
def test_dialog_audio_dm(type):

    data_conf = DialogAudioDM.load_config()
    DialogAudioDM.print_dm(data_conf)

    data_conf["dataset"]["vad_hz"] = 50
    dm = DialogAudioDM(
        datasets=data_conf["dataset"]["datasets"],
        type=type,
        sample_rate=data_conf["dataset"]["sample_rate"],
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        audio_overlap=data_conf["dataset"]["audio_overlap"],
        audio_context_duration=data_conf["dataset"]["audio_context_duration"],
        ipu_min_time=data_conf["dataset"]["ipu_min_time"],
        ipu_pause_time=data_conf["dataset"]["ipu_pause_time"],
        vad_hz=data_conf["dataset"]["vad_hz"],
        vad_bin_times=data_conf["dataset"]["vad_bin_times"],
        vad_history=data_conf["dataset"]["vad_history"],
        vad_history_times=data_conf["dataset"]["vad_history_times"],
        batch_size=16,
        num_workers=cpu_count(),
    )
    dm.prepare_data()
    dm.setup()

    for batch in dm.val_dataloader():
        pass

    for batch in dm.train_dataloader():
        pass
