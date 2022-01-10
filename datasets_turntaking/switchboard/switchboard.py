import os
from tqdm import tqdm
from typing import List

import datasets
from datasets import Value, Sequence

from datasets_turntaking.switchboard.utils import (
    Classification,
    SegmentIPU,
    SwitchboardUtils,
    TextFocusDialog,
    REL_AUDIO_PATH,
    get_audio_relpath,
)
from datasets_turntaking.utils import (
    read_json,
    read_txt,
    repo_root,
    write_json,
)

logger = datasets.logging.get_logger(__name__)


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

_URL = "https://www.isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz"

SPLIT_ROOT = os.path.join(repo_root(), "datasets_turntaking/switchboard/files")

INCLUDES_ASIDES = [
    "2264",
    "2203",
    "2277",
    "2299",
    "2260",
    "2220",
    "2689",
    "2662",
    "2640",
    "3081",
    "3046",
    "3028",
    "3051",
    "3063",
    "3045",
    "3068",
    "4003",
    "4086",
    "4031",
    "4070",
    "4004",
    "3490",
    "3443",
    "3400",
    "2701",
    "2744",
    "2709",
    "2960",
    "2924",
    "2984",
    "2913",
    "2963",
    "2988",
    "2999",
    "2953",
    "3712",
    "2420",
    "2401",
    "2491",
    "2432",
    "2473",
    "2480",
    "2447",
    "3549",
    "3539",
    "2894",
    "2856",
    "2819",
    "2879",
    "2892",
    "2867",
    "4702",
    "3220",
    "3265",
    "3285",
    "3245",
    "3230",
    "2164",
    "2149",
    "2585",
    "2584",
    "2569",
    "2566",
    "2517",
    "2550",
    "3165",
    "3109",
    "3122",
    "4905",
    "4812",
    "2054",
    "2057",
    "2022",
    "2067",
    "2078",
    "3634",
    "3657",
    "3629",
    "2347",
    "2334",
    "2381",
    "2386",
    "3376",
    "3396",
    "3357",
    "3324",
    "3350",
    "3373",
    "3338",
    "3364",
    "3835",
    "4288",
    "4219",
    "4265",
    "3997",
    "3968",
    "3978",
    "3995",
]


features_raw = {
    "session": Value("string"),
    "audio_path": Value("string"),
    "vad": [
        [Sequence(Value("float"))],
    ],
    "dialog": Sequence(
        {
            "id": Value("string"),
            "text": Value("string"),
            "speaker": Value("int32"),
            "start": Value("float"),
            "end": Value("float"),
            "words": Sequence(
                {
                    "text": Value("string"),
                    "start": Value("float"),
                    "end": Value("float"),
                }
            ),
        }
    ),
}

features_clean = {
    "session": Value("string"),
    "audio_path": Value("string"),
    "vad": [
        [Sequence(Value("float"))],
    ],
    "dialog": Sequence(
        {
            "id": Value("string"),
            "text": Value("string"),
            "speaker": Value("int32"),
            "start": Value("float"),
            "end": Value("float"),
            "words": Sequence(
                {
                    "text": Value("string"),
                    "start": Value("float"),
                    "end": Value("float"),
                }
            ),
        }
    ),
}

features_default = {
    "session": Value("string"),
    "audio_path": Value("string"),
    "vad": [
        [Sequence(Value("float"))],
    ],
    "dialog": Sequence(
        {
            "id": Value("string"),
            "text": Value("string"),
            "speaker": Value("int32"),
            "start": Value("float"),
            "end": Value("float"),
            "backchannel": Sequence(
                {
                    "text": Value("string"),
                    "start": Value("float"),
                    "end": Value("float"),
                }
            ),
            "within": Sequence(
                {
                    "text": Value("string"),
                    "start": Value("float"),
                    "end": Value("float"),
                }
            ),
            "words": Sequence(
                {
                    "text": Value("string"),
                    "start": Value("float"),
                    "end": Value("float"),
                }
            ),
        }
    ),
}

features_ipu = {
    "session": Value("string"),
    "audio_path": Value("string"),
    "label": Value("string"),
    "context": {
        "turns": Sequence(Value("string")),
        "start": Value("float"),
        "end": Value("float"),
        "speaker": Value("int32"),
        "join_with_turns": Value("int32"),
        "overlap": Value("int32"),
        "vad": [
            [Sequence(Value("float"))],
        ],
        "text": Value("string"),
        "words": datasets.features.Sequence(
            {
                "text": Value("string"),
                "start": Value("float"),
                "end": Value("float"),
            }
        ),
    },
    "response": {
        "text": Value("string"),
        "speaker": Value("int32"),
        "start": Value("float"),
        "end": Value("float"),
        "words": Sequence(
            {
                "text": Value("string"),
                "start": Value("float"),
                "end": Value("float"),
            }
        ),
    },
}

features_classification = {
    "session": Value("string"),
    "audio_path": Value("string"),
    "label": Value("string"),
    "context": {
        "turns": Sequence(Value("string")),
        "start": Value("float"),
        "end": Value("float"),
        "speaker": Value("int32"),
        "vad": [
            [Sequence(Value("float"))],
        ],
        "words": Sequence(
            {
                "text": Value("string"),
                "start": Value("float"),
                "end": Value("float"),
            }
        ),
    },
    "response": {
        "id": Value("string"),
        "text": Value("string"),
        "speaker": Value("int32"),
        "start": Value("float"),
        "end": Value("float"),
        "words": Sequence(
            {
                "text": Value("string"),
                "start": Value("float"),
                "end": Value("float"),
            }
        ),
    },
}


# TODO: Rewrite with `data_dir` flag like callhome
class SwitchboardConfig(datasets.BuilderConfig):
    def __init__(self, custom_overwrite=False, omit_post_words=False, **kwargs):
        super().__init__(**kwargs)
        self.custom_overwrite = custom_overwrite
        self.omit_post_words = omit_post_words


class SwitchboardIPUConfig(datasets.BuilderConfig):
    def __init__(
        self,
        ipu_threshold=0.2,
        min_context_ipu_time=1.5,
        lookahead_duration=2,
        custom_overwrite=False,
        omit_post_words=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.custom_overwrite = custom_overwrite
        self.omit_post_words = omit_post_words
        self.min_context_ipu_time = min_context_ipu_time
        self.ipu_threshold = ipu_threshold
        self.lookahead_duration = lookahead_duration


class Switchboard(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.1")
    DEFAULT_CONFIG_NAME = "default"
    BUILDER_CONFIGS = [
        SwitchboardConfig(
            name="default",
            description="Switchboard dataset in a convenient format with annotated word times",
        ),
        SwitchboardConfig(
            name="raw",
            description="Switchboard raw dataset (without processing) in a convenient format with annotated word times",
        ),
        SwitchboardConfig(
            name="clean",
            description="Switcboard dataset like raw but processing of text (omit special annotation text)",
        ),
        SwitchboardConfig(
            name="classification",
            description="Switchboard dataset that finds Shifts, Backchannels and holds (with full context) used to train turntaking classifier",
        ),
        SwitchboardIPUConfig(
            name="ipu",
            ipu_threshold=0.2,
            description="Switchboard dataset that finds suitable IPUs (with full context) used to train ipu classifier",
        ),
    ]

    def _info(self) -> datasets.DatasetInfo:
        if self.config.name == "default":
            features = features_default
        elif self.config.name == "classification":
            features = features_classification
        elif self.config.name == "ipu":
            features = features_ipu
        elif self.config.name == "raw":
            features = features_raw
        elif self.config.name == "clean":
            features = features_raw
        else:
            raise NotADirectoryError(f"{self.config.name} is not implemented")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            homepage="https://www.isip.piconepress.com/projects/switchboard/",
            citation=_CITATION,
            features=datasets.Features(features),
            supervised_keys=None,
        )

    def _process_dialogs(self, extracted_path):
        """
        Iterate over each dialog in the annotation directory.
        Combine utterance-/word- level annotations for each speaker.
        Join speaker data into a single dialog.

        Write the combined annotations to disc (in the annotation directory).

        - Default: combine speakers, utterances, word-timing but don't do any extra processing
        - strict_times_clean:   strict-times only use word-level timing information (no padding around utterances),
                                clean refers to switchboard (annotation) specific RegExp.
        """
        filepaths = []
        for root, dirs, _ in tqdm(
            os.walk(extracted_path), desc="Process Switchboard", total=2438
        ):
            if len(dirs) == 0:

                # Get the session and create a filename
                session = os.path.split(root)[-1]  # e.g. 2284
                if self.config.name != "raw":
                    filename = session + "_default" + ".json"
                else:
                    filename = session + "_" + self.config.name + ".json"
                filepath = os.path.join(root, filename)

                # Using a manual flag to overwrite already processed files
                # Mostly useful while development
                if os.path.exists(filepath) and not self.config.custom_overwrite:
                    filepaths.append(filepath)
                    continue

                # Don't include dialogs with "aside" annotations for all configs except `default`
                raw = True
                if self.config.name != "raw":
                    raw = False
                    if session in INCLUDES_ASIDES:
                        continue

                # A sorted (by start of utterance) list of utterances
                dialog = SwitchboardUtils.extract_dialog(session, root, raw)

                # save as json
                write_json(dialog, filepath)
                filepaths.append(filepath)
        return filepaths

    def generate_default(self, filepaths):
        """omits backchannels and utterance totally 'within' another speakers utterance."""
        text_focus_refiner = TextFocusDialog()
        return text_focus_refiner.generate_refined_dialogs(filepaths)

    def generate_raw(self, filepaths):
        RelAudioMap = read_json(REL_AUDIO_PATH)
        for filepath in filepaths:
            utterances = read_json(filepath)
            session = os.path.basename(filepath).split("_")[0]
            audio_path = get_audio_relpath(session, RelAudioMap)
            vad = SwitchboardUtils.extract_vad(utterances)
            yield f"{session}", {
                "dialog": utterances,
                "session": session,
                "audio_path": audio_path,
                "vad": vad,
            }

    def generate_classification(self, filepaths):
        """extracts classification samples from after same processing as for 'default'."""
        classification = Classification(omit_post_words=self.config.omit_post_words)
        return classification.generate_classification(filepaths)

    def generate_ipus(self, filepaths):
        ipus = SegmentIPU()
        return ipus.generate_ipus(filepaths)

    def get_splits(self, processed_files):
        train_files = read_txt(os.path.join(SPLIT_ROOT, "train.txt"))
        val_files = read_txt(os.path.join(SPLIT_ROOT, "val.txt"))
        test_files = read_txt(os.path.join(SPLIT_ROOT, "test.txt"))

        train, val, test = [], [], []
        for f in processed_files:
            # /path/to/xxxx.json
            # /path/to/xxxx_ipu_clean.json
            # /path/to/xxxx_strict_times.json
            # -> xxxx
            session = os.path.basename(f)[:4]
            if session in train_files:
                train.append(f)
            elif session in val_files:
                val.append(f)
            elif session in test_files:
                test.append(f)
            else:
                raise KeyError(
                    f"Did not find split for:\n{f}\n{session}\n. Should not happen."
                )

        return {"train": train, "val": val, "test": test}

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        extracted_path = dl_manager.download_and_extract(_URL)  # hash
        processed_files = self._process_dialogs(extracted_path)
        splits = self.get_splits(processed_files)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepaths": splits["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepaths": splits["val"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepaths": splits["test"]},
            ),
        ]

    def _generate_examples(self, filepaths):
        logger.info("generating examples from = %s", filepaths)
        if self.config.name == "default":
            return self.generate_default(filepaths)
        elif self.config.name == "clean":
            return self.generate_raw(filepaths)
        elif self.config.name == "classification":
            return self.generate_classification(filepaths)
        elif self.config.name == "ipu":
            return self.generate_ipus(filepaths)
        else:
            return self.generate_raw(filepaths)
