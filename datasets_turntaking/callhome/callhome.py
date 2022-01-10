import os
from os.path import join, exists, basename
from typing import List

import datasets
from datasets import Value, Sequence

from datasets_turntaking.callhome.utils import load_utterances

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = """ CALLHOME """
_CITATION = """ citation """
_HOMEPAGE = """ Homepage """
_URL = "url"


class CallHomeConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class CallHome(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [CallHomeConfig(name="default", description="CALLHOME")]

    def _info(self):
        features = {
            "id": Value("string"),
            "file": Value("string"),
            "dialog": Sequence(
                {
                    "text": Value("string"),
                    "speaker": Value("int32"),
                    "start": Value("float"),
                    "end": Value("float"),
                }
            ),
        }
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            features=datasets.Features(features),
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
                if file.endswith(".sph"):
                    sample = {"audio": join(tmp_audio_path, file)}
                    txt = join(tmp_text_path, file.replace(".sph", ".txt"))
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
            id = basename(sample["audio"]).replace(".sph", "")
            dialog = load_utterances(sample["text"], clean)
            yield id, {
                "id": id,
                "file": sample["audio"],
                "dialog": dialog,
            }
