from typing import List
import datasets
from datasets import Value, Sequence

from datasets_turntaking.utils import read_txt

from .utils import extract_vad_list, get_paths, load_transcript

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
FEATURES = {
    "session": Value("string"),
    "audio_path": Value("string"),
    "vad": [
        [Sequence(Value("float"))],
    ],
    "dialog": {
        "A": Sequence(
            {
                "start": Value("float"),
                "end": Value("float"),
                "text": Value("string"),
            }
        ),
        "B": Sequence(
            {
                "start": Value("float"),
                "end": Value("float"),
                "text": Value("string"),
            }
        ),
    },
}


class FisherConfig(datasets.BuilderConfig):
    def __init__(
        self,
        root,
        train_ids=[str(i) for i in range(1, 5100)],
        val_ids=[str(i) for i in range(5100, 5500)],
        test_ids=[str(i) for i in range(5500, TOTAL_FILES + 1)],
        ext=".wav",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.root = root
        self.ext = ext
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids

        # FULL
        # self.test_ids = [str(i) for i in range(150, 200)]
        # self.val_ids = [str(i) for i in range(100, 150)]
        # self.train_ids = [str(i) for i in range(1, 100)]


class Fisher(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.1")
    DEFAULT_CONFIG_NAME = "default"
    BUILDER_CONFIGS = [
        FisherConfig(
            root="/home/erik/projects/data/Fisher",
            name="default",
            description="Fisher speech dataset",
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
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"ids": self.config.train_ids},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"ids": self.config.val_ids},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"ids": self.config.test_ids},
            ),
        ]

    def generate(self, all_nnns):
        for n in all_nnns:
            nnn = str(n).zfill(5)
            trans_path, audio_path = get_paths(
                nnn, self.config.root, ext=self.config.ext
            )
            anno = load_transcript(trans_path)
            vad = extract_vad_list(anno)
            yield f"{nnn}", {
                "session": nnn,
                "audio_path": audio_path,
                "vad": vad,
                "dialog": anno,
            }

    def _generate_examples(self, ids):
        logger.info("generating examples from = %s", ids)
        return self.generate(ids)
