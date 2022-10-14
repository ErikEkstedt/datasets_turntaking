import datasets
from datasets import Value, Sequence
from typing import List
from os.path import expanduser, exists, join
from .utils import (
    extract_vad_list,
    extract_vad_list_from_words,
    get_data_paths,
    load_transcript,
)
from datasets_turntaking.dataset import DIALOG_AUDIO_FEATURES

logger = datasets.logging.get_logger(__name__)

_HOMEPAGE = "https://catalog.ldc.upenn.edu/LDC2004S13"
_CITATION = """Fischer"""
_DESCRIPTION = """
Fisher English Training Speech Part 1 Speech represents the first half of a
collection of conversational telephone speech (CTS) that was created at the LDC
during 2003. It contains 5,850 audio files, each one containing a full
conversation of up to 10 minutes. Additional information regarding the speakers
involved and types of telephones used can be found in the companion text corpus of transcripts, Fisher English Training Speech Part 1, Transcripts
(LDC2004T19).
"""


TOTAL_FILES = 5850


class FisherConfig(datasets.BuilderConfig):
    def __init__(
        self,
        root=join(expanduser("~"), "projects/data/Fisher"),
        word_level_transcripts=join(
            expanduser("~"), "projects/data/Fisher/fisher_transcripts_word_level"
        ),
        min_word_vad_diff: float = 0.05,
        apply_regexp: bool = True,
        remove_restarts: bool = False,  # "h-" -> "" if True
        train_sessions: List[str] = [str(i) for i in range(1, 5100)],
        val_sessions: List[str] = [str(i) for i in range(5100, 5500)],
        test_sessions: List[str] = [str(i) for i in range(5500, TOTAL_FILES + 1)],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.root = root
        self.word_level_transcripts = word_level_transcripts
        self.apply_regexp = apply_regexp
        self.remove_restarts = remove_restarts
        self.min_word_vad_diff = min_word_vad_diff
        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions


class Fisher(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.1")
    DEFAULT_CONFIG_NAME = "default"
    BUILDER_CONFIGS = [
        FisherConfig(
            name="default",
            description="Fisher speech dataset",
        )
    ]

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            features=datasets.Features(DIALOG_AUDIO_FEATURES),
            supervised_keys=None,
        )

    def _split_generators(self, *args, **kwargs) -> List[datasets.SplitGenerator]:
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"sessions": self.config.train_sessions},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"sessions": self.config.val_sessions},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"sessions": self.config.test_sessions},
            ),
        ]

    def generate(self, sessions):
        for n in sessions:
            nnn = str(n).zfill(5)
            paths = get_data_paths(nnn=nnn, root=self.config.root)
            dialog = load_transcript(
                paths["utterance"],
                apply_regexp=self.config.apply_regexp,
                remove_restarts=self.config.remove_restarts,
            )
            vad_list = extract_vad_list_from_words(
                nnn=nnn,
                root=self.config.root,
                min_word_vad_diff=self.config.min_word_vad_diff,
            )
            if vad_list is None:
                vad_list = extract_vad_list(dialog)

            yield f"{nnn}", {
                "session": nnn,
                "dataset": "fisher",
                "audio_path": paths["audio"],
                "vad_list": vad_list,
                "dialog": dialog,
            }

    def _generate_examples(self, sessions):
        logger.info("generating examples from = %s", sessions)
        return self.generate(sessions)
