from datasets_turntaking.utils import repo_root
from datasets import Value, Sequence

DIALOG_AUDIO_FEATURES = {
    "dataset": Value("string"),
    "session": Value("string"),
    "audio_path": Value("string"),
    "vad": [[Sequence(Value("float"))]],
    "dialog": [
        Sequence(
            {
                "start": Value("float"),
                "end": Value("float"),
                "text": Value("string"),
            }
        )
    ],
}
