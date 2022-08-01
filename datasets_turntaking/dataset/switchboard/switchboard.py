from os.path import join, expanduser
import os
from typing import List

import datasets
from datasets import Value, Sequence

from datasets_turntaking.dataset.switchboard.utils import (
    load_transcript,
    extract_vad_list_from_words,
    remove_words_from_dialog,
)
from datasets_turntaking.utils import (
    read_txt,
    read_json,
    repo_root,
)
from datasets_turntaking.dataset import DIALOG_AUDIO_FEATURES

logger = datasets.logging.get_logger(__name__)


REL_AUDIO_PATH = join(
    repo_root(),
    "datasets_turntaking/dataset/switchboard/files/relative_audio_path.json",
)
SPLIT_PATH = os.path.join(repo_root(), "datasets_turntaking/dataset/switchboard/files")


_HOMEPAGE = "https://catalog.ldc.upenn.edu/LDC97S62"
_URL = "https://www.isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz"
_DESCRIPTION = """
Switchboard annotations (swb_ms98_transcriptions) in a convenient format
TODO
"""
_CITATION = """
@inproceedings{Godfrey92,
    author = {Godfrey, John J. and Holliman, Edward C. and McDaniel, Jane},
    title = {SWITCHBOARD: Telephone Speech Corpus for Research and Development},
    year = {1992},
    isbn = {0780305329},
    publisher = {IEEE Computer Society},
    address = {USA},
    booktitle = {Proceedings of the 1992 IEEE International Conference on Acoustics, Speech and Signal Processing - Volume 1},
    pages = {517–520},
    numpages = {4},
    location = {San Francisco, California},
    series = {ICASSP’92}
}
"""


class SwitchboardConfig(datasets.BuilderConfig):
    def __init__(
        self,
        root=join(expanduser("~"), "projects/data/switchboard/audio"),
        train_sessions=None,
        val_sessions=None,
        test_sessions=None,
        min_word_vad_diff=0.05,
        ext=".wav",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ext = ext
        self.root = root
        self.min_word_vad_diff = min_word_vad_diff
        self.train_sessions = (
            read_txt(os.path.join(SPLIT_PATH, "train.txt"))
            if train_sessions is None
            else train_sessions
        )
        self.val_sessions = (
            read_txt(os.path.join(SPLIT_PATH, "val.txt"))
            if val_sessions is None
            else val_sessions
        )
        self.test_sessions = (
            read_txt(os.path.join(SPLIT_PATH, "test.txt"))
            if test_sessions is None
            else test_sessions
        )


class Swithchboard(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.1")
    DEFAULT_CONFIG_NAME = "default"
    BUILDER_CONFIGS = [
        SwitchboardConfig(
            name="default",
            description="Switchboard speech dataset",
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

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        self.extracted_path = dl_manager.download_and_extract(_URL)  # hash
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
        sess_2_rel_path = read_json(REL_AUDIO_PATH)
        for session in sessions:
            session = str(session)
            audio_path = join(
                self.config.root, sess_2_rel_path[session] + self.config.ext
            )
            session_dir = join(
                self.extracted_path, "swb_ms98_transcriptions", session[:2], session
            )

            dialog = load_transcript(session, session_dir)
            vad = extract_vad_list_from_words(dialog, self.config.min_word_vad_diff)
            # omit words
            dialog = remove_words_from_dialog(dialog)
            yield f"{session}", {
                "session": session,
                "dataset": "switchboard",
                "audio_path": audio_path,
                "vad": vad,
                "dialog": dialog,
            }

    def _generate_examples(self, sessions):
        # logger.info("generating examples from = %s", sessions)
        return self.generate(sessions)
