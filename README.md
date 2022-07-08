# Datasets for Turn-Taking


## Installation

1. Create environment: `conda create -n datasets_turntaking python=3.9` and source environment
2. Install dependencies: `pip install -r requirements.txt`
    * Install [VAP](https://github.com/ErikEkstedt/vap_turn_taking)
      * Download: `git clone`
      * Install: `pip install -r requirements.txt` and `pip install -e .`
3. Install package: `pip install -e .`


## Dialog Audio


```python

from datasets_turntaking.dialog_audio_dm import DialogAudioDM
from vap_turn_taking import VAP

conf = DialogAudioDM.load_config()
conf["dataset"]["waveform"] = False
conf["dataset"]["vad"] = True
conf["dataset"]["vad_history"] = False
conf["dataset"]["datasets"] = ["fisher", "switchboard"]
conf["dataset"]["audio_duration"] = 20
conf["dataset"]["vad_hz"] = 50
conf["dataset"]["vad_horizon"] = 2
conf["dataset"]["flip_channels"] = False
dm = DialogAudioDM(
    datasets=conf["dataset"]["datasets"],
    type=conf["dataset"]["type"],
    sample_rate=conf["dataset"]["sample_rate"],
    waveform=conf["dataset"]["waveform"],
    audio_duration=conf["dataset"]["audio_duration"],
    audio_normalize=conf["dataset"]["audio_normalize"],
    audio_overlap=conf["dataset"]["audio_overlap"],
    vad_hz=conf["dataset"]["vad_hz"],
    vad_horizon=conf["dataset"]["vad_horizon"],
    vad_history=conf["dataset"]["vad_history"],
    vad_history_times=conf["dataset"]["vad_history_times"],
    flip_channels=conf["dataset"]["flip_channels"],
    batch_size=32,
    num_workers=4,
    pin_memory=False,
)
dm.prepare_data()
dm.setup(None)
print(dm)
print(f"Number of samples for sliding window {dm.train_dset.audio_step_time}s step")
print("Train: ", len(dm.train_dset))
print("Val: ", len(dm.val_dset))
print("Test: ", len(dm.test_dset))

# VAP objective
vapper = VAP(type="discrete", frame_hz=conf["dataset"]["vad_hz"])
print("Hz: ", vapper.frame_hz)
print("n_classes: ", vapper.n_classes)


for batch in dm.train_dataloader():
    y = vapper.extract_label(batch["vad"])
    print("Y: ", tuple(y.shape))
    print('batch: ', list(batch.keys()))
    break
```

* [dialog_audio_dm.py](./dialog_audio_dm.py)
* [x] Switchboard
* [x] Fisher
* [ ] Candor
* [ ] Callhome
* [ ] Maptask
* [ ] Spotify


## Mono Speech Audio

* [mono_speech_dm.py](./mono_speech_dm.py)
* [x] Librispeech
* [x] LJ-speech
* [x] VCTK
* [x] Librilight
* [ ] Blizzard
* ..., etc


## Conversational Text

* [dialog_text_dm.py](./dialog_text_dm.py)
* [x] `curiosity_dialogs`
* [x] `daily_dialog`
* [x] `multi_woz_v22`
* [x] `meta_woz`
* [x] `taskmaster1`
* [x] `taskmaster2`
* [x] `taskmaster3`
