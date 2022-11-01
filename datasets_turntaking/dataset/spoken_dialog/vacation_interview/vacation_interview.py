from typing import List
import datasets
from datasets import Value, Sequence

from datasets_turntaking.utils import read_txt, read_json

from .utils import (
    extract_vad_list,
    get_sessions,
    get_paths,
    get_vad_path,
    load_transcript,
)

logger = datasets.logging.get_logger(__name__)

_HOMEPAGE = "https://sigdial.org/sites/default/files/workshops/conference22/Proceedings/pdf/2021.sigdial-1.45.pdf"
_CITATION = ""
_DESCRIPTION = "TurnGPT Projection: Vacation Interviews"

TOTAL_FILES = 20
FEATURES = {
    "session": Value("string"),
    "audio_path": Value("string"),
    "vad": [
        [Sequence(Value("float"))],
    ],
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


class VacIntConfig(datasets.BuilderConfig):
    def __init__(self, root, ext=".wav", **kwargs):
        super().__init__(**kwargs)
        self.root = root
        self.ext = ext


class VacationInterviews(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.1")
    DEFAULT_CONFIG_NAME = "default"
    BUILDER_CONFIGS = [
        VacIntConfig(
            root="/home/erik/projects/data/projection_dialogs",
            name="default",
            description="",
        )
    ]

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            features=datasets.Features(FEATURES),
            supervised_keys=None,
        )

    def _split_generators(self, *args, **kwargs) -> List[datasets.SplitGenerator]:
        sessions = get_sessions(self.config.root)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"sessions": sessions},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"sessions": sessions},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"sessions": sessions},
            ),
        ]

    def generate(self, sessions):
        for session in sessions:
            trans_path, audio_path, vad_path = get_paths(session, self.config.root)
            anno = load_transcript(trans_path)
            vad = read_json(vad_path)
            yield f"{session}", {
                "session": session,
                "audio_path": audio_path,
                "vad": vad,
                "dialog": anno,
            }

    def _generate_examples(self, sessions):
        logger.info("generating examples from = %s", sessions)
        return self.generate(sessions)
