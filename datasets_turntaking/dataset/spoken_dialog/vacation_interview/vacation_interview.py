from os.path import join, expanduser
from typing import List
import datasets
import os

from datasets_turntaking.dataset import DIALOG_AUDIO_FEATURES

from .utils import (
    extract_vad_list,
    get_sessions,
    get_paths,
    load_transcript,
)

logger = datasets.logging.get_logger(__name__)

_HOMEPAGE = "https://sigdial.org/sites/default/files/workshops/conference22/Proceedings/pdf/2021.sigdial-1.45.pdf"
_CITATION = ""
_DESCRIPTION = "TurnGPT Projection: Vacation Interviews"

TOTAL_FILES = 20


class VacationInterviewConfig(datasets.BuilderConfig):
    def __init__(self, root, ext=".wav", **kwargs):
        super().__init__(**kwargs)
        self.root = root
        self.ext = ext


class VacationInterviews(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.1")
    DEFAULT_CONFIG_NAME = "default"
    BUILDER_CONFIG_CLASS = VacationInterviewConfig
    BUILDER_CONFIGS = [
        VacationInterviewConfig(
            name="default",
            root=join(expanduser("~"), "projects/data/vacation_interview"),
            description="Vacation Interview Dataset",
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
        sessions = get_sessions(self.config.root)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"sessions": sessions},
            )
        ]

    def generate(self, sessions):
        for session in sessions:
            audio_path, dialog_path = get_paths(self.config.root, session)
            dialog = load_transcript(dialog_path)
            vad_list = extract_vad_list(dialog)
            yield f"{session}", {
                "session": session,
                "audio_path": audio_path,
                "dialog": dialog,
                "vad_list": vad_list,
                "dataset": "vacation_interview",
            }

    def _generate_examples(self, sessions):
        logger.info("generating examples from = %s", sessions)
        return self.generate(sessions)
