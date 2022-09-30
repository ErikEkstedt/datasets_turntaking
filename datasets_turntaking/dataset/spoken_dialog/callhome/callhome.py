import os
from os.path import join, exists, basename
from typing import List

import datasets
from datasets import Value, Sequence

from datasets_turntaking.callhome.utils import load_utterances, extract_vad

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = """ CALLHOME """
_CITATION = """ 
Canavan, Alexandra, David Graff, and George Zipperlen. 
CALLHOME American English Speech LDC97S42. 
Web Download. Philadelphia: Linguistic Data Consortium, 1997.
"""
_HOMEPAGE = "https://catalog.ldc.upenn.edu/LDC97S42"
_URL = "https://catalog.ldc.upenn.edu/LDC97S42"


FEATURES = {
    "session": Value("string"),
    "audio_path": Value("string"),
    "vad": [
        [Sequence(Value("float"))],
    ],
    "dialog": Sequence(
        {
            "text": Value("string"),
            "speaker": Value("int32"),
            "start": Value("float"),
            "end": Value("float"),
        }
    ),
}


class CallHomeConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class CallHome(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [CallHomeConfig(name="default", description="CALLHOME")]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            features=datasets.Features(FEATURES),
            supervised_keys=None,
        )

    def _load_english(self):
        audio_path = join(self.config.data_dir, "callhome_eng", "data")
        text_path = join(
            self.config.data_dir, "callhome_english_trans_970711", "transcrpt"
        )

        if not exists(text_path):
            raise FileNotFoundError(f"text_path not found: {text_path}")

        splits = {}
        for split, folder in zip(
            ["train", "validation", "test"], ["train", "devtest", "evltest"]
        ):
            splits[split] = []
            tmp_audio_path = join(audio_path, folder)
            tmp_text_path = join(text_path, folder.replace("evltest", "evaltest"))
            for file in os.listdir(tmp_audio_path):
                if file.endswith(".wav"):
                    sample = {"audio_path": join(tmp_audio_path, file)}
                    txt = join(tmp_text_path, file.replace(".wav", ".txt"))
                    if exists(txt):
                        sample["text"] = txt
                    splits[split].append(sample)
        return splits

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        if not exists(self.config.data_dir):
            raise FileExistsError(
                f"data_dir: {self.config.data_dir} does not exist! Provide a valid `data_dir`."
            )

        splits = self._load_english()
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepaths": splits["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepaths": splits["validation"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepaths": splits["test"]},
            ),
        ]

    def _generate_examples(self, filepaths):
        logger.info("generating examples from = %s", filepaths)

        # process transcripts with callhome specific regexp
        clean = True

        for sample in filepaths:
            id = basename(sample["audio_path"]).replace(".sph", "")
            utterances = load_utterances(sample["text"], clean)
            vad = extract_vad(utterances)
            yield id, {
                "session": id,
                "vad": vad,
                "audio_path": sample["audio_path"],
                "dialog": utterances,
            }
