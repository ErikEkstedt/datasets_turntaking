from datasets_turntaking.vacation_interview import load_vacation_interview
from datasets_turntaking.utils import load_waveform
from datasets_turntaking.dialog_audio.dm_dialog_audio import DialogAudioDM


def test_dset():
    dset = load_vacation_interview()
    print("dset: ", dset)
    d = dset[0]
    print("d: ", d.keys())
    x, sr = load_waveform(d["audio_path"])
    print("x: ", tuple(x.shape))


if __name__ == "__main__":

    dm = DialogAudioDM(
        datasets=["vacation_interview"],
        sample_rate=16000,
        vad_hz=100,
        vad_horizon=2,
        vad_history=True,
        vad_history_times=[60, 30, 10, 5],
        flip_channels=False,
        batch_size=4,
        num_workers=0,
    )
    dm.prepare_data()
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    batch.keys()
