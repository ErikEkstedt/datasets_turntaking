# Vacation Interview


The data collected for [Projection of Turn Completion in Incremental Spoken Dialogue Systems](https://sigdial.org/sites/default/files/workshops/conference22/Proceedings/pdf/2021.sigdial-1.45.pdf).

The data consists of Human-Robot interactions, HRI, where the robot interviews
the human about past vacations. The robot follows an apriori set of questions
and utilizes turn-taking behavior in two forms, a baseline (google asr
finish-flag) and turnGPT-projection.


## structure

```bash
 projection_dialogs
├──  audio
│  ├──  session_0_baseline.wav
│  ├──  session_0_prediction.wav
│  ├── ...
│  └──  session_9_prediction.wav
├──  dialogs
│  ├──  session_0_baseline.json
│  ├──  session_0_prediction.json
│  ├── ...
│  └──  session_9_prediction.json
├──  vad
│  ├──  session_0_baseline.json
│  ├──  session_0_prediction.json
│  ├── ...
│  └──  session_9_prediction.json
└──  README.md
```

