from os.path import join, basename, dirname
from os import makedirs, listdir
from glob import glob

import datasets
import torchaudio
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
)
from typing import List
from datasets_turntaking.utils import read_txt

logger = datasets.logging.get_logger(__name__)

"""
Heavily based on torchaudio.datasets.vctk:
    https://pytorch.org/audio/stable/_modules/torchaudio/datasets/vctk.html#VCTK_092
"""

_URL = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"

_CHECKSUMS = {
    "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip": "8a6ba2946b36fcbef0212cad601f4bfa"
}
_DESCRIPTION = "VCTK"
_HOMEPAGE = "https://datashare.is.ed.ac.uk/handle/10283/3443"
_CITATION = """Yamagishi, Junichi; Veaux, Christophe; MacDonald, Kirsten. (2019). 
CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit 
(version 0.92), [sound]. University of Edinburgh. The Centre for Speech
Technology Research (CSTR). 
https://doi.org/10.7488/ds/2645.
"""

from datasets.utils.file_utils import (
    get_from_cache,
    hash_url_to_filename,
    url_or_path_join,
)


class VCTKConfig(datasets.BuilderConfig):
    def __init__(self, mic_id="mic2", **kwargs):
        super().__init__(**kwargs)
        self.mic_id = mic_id


class VCTK(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [VCTKConfig(name="default", description="VCTK-0.92")]

    def _info(self):
        features = {
            "id": datasets.Value("string"),
            "file": datasets.Value("string"),
            "text": datasets.Value("string"),
            "speaker_id": datasets.Value("string"),
            "utterance_id": datasets.Value("string"),
        }
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            features=datasets.Features(features),
            supervised_keys=None,
        )

    def custom_download(self, dl_manager):
        cache_dir = dl_manager._download_config.cache_dir
        max_retries = dl_manager._download_config.max_retries

        def url_to_downloaded_path(url):
            return join(cache_dir, hash_url_to_filename(url))

        downloaded_path_or_paths = url_to_downloaded_path(_URL)

        gfc = downloaded_path_or_paths
        try:
            gfc = get_from_cache(
                _URL,
                cache_dir=cache_dir,
                local_files_only=True,
                use_etag=False,
                max_retries=max_retries,
            )
            cached = True
        except FileNotFoundError:
            cached = False

        if not cached or dl_manager._download_config.force_download:
            checksum = _CHECKSUMS.get(_URL, None)
            download_url(
                _URL,
                download_folder=cache_dir,
                filename=basename(downloaded_path_or_paths),
                hash_value=checksum,
                hash_type="md5",
            )

        return gfc

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        zip_path = self.custom_download(dl_manager)
        extracted_path = dl_manager.extract(zip_path)

        # Splits
        # simple split over speakers
        text_root = join(extracted_path, "txt")
        speakers = listdir(text_root)
        speakers.pop(speakers.index("p280"))  # bad? doing as torchaudio
        speakers.sort()

        n_speakers = len(speakers)
        tr = int(0.85 * n_speakers)
        v = int(0.1 * n_speakers)
        speakers_train = speakers[:tr]
        speakers_val = speakers[tr : tr + v + 1]
        speakers_test = speakers[tr + v + 1 :]

        # get waveform-path and txt-path pairs
        filepaths = {"train": [], "val": [], "test": []}
        for file in glob(
            join(
                extracted_path,
                "wav48_silence_trimmed",
                f"**/*{self.config.mic_id}.flac",
            )
        ):
            pNNN = basename(dirname(file))

            if pNNN == "p280" and self.config.mic_id == "mic2":
                continue
            session = basename(file).split("_")[:2]
            session = "_".join(session)
            txt = join(text_root, pNNN, session + ".txt")

            if pNNN in speakers_val:
                filepaths["val"].append((file, txt))
            if pNNN in speakers_test:
                filepaths["test"].append((file, txt))
            else:
                filepaths["train"].append((file, txt))

        # sort for consistency
        for k, v in filepaths.items():
            v.sort(key=lambda x: x[0])
            filepaths[k] = v
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepaths": filepaths["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepaths": filepaths["val"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepaths": filepaths["test"]},
            ),
        ]

    def _generate_examples(self, filepaths):
        logger.info("generating examples from = %s", filepaths)
        for wav_path, txt_path in filepaths:
            text = read_txt(txt_path)[0]
            id = basename(txt_path).replace(".txt", "")
            speaker_id, utterance_id = id.split("_")
            yield id, {
                "id": id,
                "file": wav_path,
                "text": text,
                "speaker_id": speaker_id,
                "utterance_id": utterance_id,
            }
