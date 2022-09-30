import datasets
from datasets import Value, Sequence
from typing import List
from os.path import expanduser, join
from .utils import extract_vad_list, get_paths, load_transcript
from datasets_turntaking.dataset import DIALOG_AUDIO_FEATURES

logger = datasets.logging.get_logger(__name__)

_HOMEPAGE = "https://catalog.ldc.upenn.edu/LDC2004S13"
_CITATION = """Fischer"""
_DESCRIPTION = """
Fisher English Training Speech Part 1 Speech represents the first half of a
collection of conversational telephone speech (CTS) that was created at the LDC
during 2003. It contains 5,850 audio files, each one containing a full
conversation of up to 10 minutes. Additional information regarding the speakers
involved and types of telephones used can be found in the companion text corpus
of transcripts, Fisher English Training Speech Part 1, Transcripts
(LDC2004T19).
"""


TOTAL_FILES = 5850


class FisherConfig(datasets.BuilderConfig):
    def __init__(
        self,
        root=join(expanduser("~"), "projects/data/Fisher"),
        train_sessions=[str(i) for i in range(1, 5100)],
        val_sessions=[str(i) for i in range(5100, 5500)],
        test_sessions=[str(i) for i in range(5500, TOTAL_FILES + 1)],
        apply_regexp=True,
        remove_restarts=False,  # "h-" -> "" if True
        ext=".wav",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.root = root
        self.ext = ext
        self.apply_regexp = apply_regexp
        self.remove_restarts = remove_restarts
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
            session = str(n).zfill(5)
            trans_path, audio_path = get_paths(
                session, self.config.root, ext=self.config.ext
            )
            dialog = load_transcript(
                trans_path,
                apply_regexp=self.config.apply_regexp,
                remove_restarts=self.config.remove_restarts,
            )
            vad = extract_vad_list(dialog)
            yield f"{session}", {
                "session": session,
                "dataset": "fisher",
                "audio_path": audio_path,
                "vad": vad,
                "dialog": dialog,
            }

    def _generate_examples(self, sessions):
        logger.info("generating examples from = %s", sessions)
        return self.generate(sessions)
