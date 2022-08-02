# Datasets for Turn-Taking

## Installation

* conda env `conda create -n datasets_turntaking python=3.9`
* Dependencies:
  * `pip install -r requirements`
  * [vap_turn_taking]()
* Install: `pip install -e .`

## WIP

* Dialog Audio
  * [dialog_audio_dm.py](./dialog_audio_dm.py)
  * [x] Switchboard
  * [x] Fisher
  * [ ] Candor
  * [ ] Callhome
  * [ ] Maptask
  * [ ] Spotify
* Mono Speech Audio
  * [mono_speech_dm.py](./mono_speech_dm.py)
  * [x] Librispeech
  * [x] LJ-speech
  * [x] VCTK
  * [x] Librilight
  * [ ] Blizzard
* Conversational Text
  * [dialog_text_dm.py](./dialog_text_dm.py)
  * [x] `curiosity_dialogs`
  * [x] `daily_dialog`
  * [x] `multi_woz_v22`
  * [x] `meta_woz`
  * [x] `taskmaster1`
  * [x] `taskmaster2`
  * [x] `taskmaster3`
