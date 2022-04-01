from argparse import ArgumentParser
from os.path import join, expanduser, exists
from os import cpu_count
from typing import Optional

from datasets import load_dataset, load_from_disk
import pytorch_lightning as pl
import torch
import torchaudio

from torch.utils.data import DataLoader
from datasets_turntaking.utils import repo_root, read_json
from datasets_turntaking.features.f0 import pYAAPT

from vap_turn_taking.utils import get_vad_condensed_history, get_current_vad_onehot

DATASET_SCRIPT = join(repo_root(), "datasets_turntaking/switchboard/switchboard.py")
F0_MEAN_PATH = join(repo_root(), "datasets_turntaking/switchboard/f0_means.json")
CACHE_PATH = join(
    expanduser("~"), ".cache/datasets_turntaking/switchboard/classification"
)
AUDIO_ROOT = join(expanduser("~"), "projects/data/switchboard/audio")


class UtilsNotUsed:
    def get_token_start_end(self, examples):
        """
        map start/end times of words to the corresponding tokens of the `keep_n_word_info`
        last words in context.
        Used to create negative samples:
          which tokens to omit if a negative sample, 1 second prior to response, is created
          the waveform is simply shifted.
          but which tokens to omit would otherwise be impossible
        """
        tokens, starts, ends = [], [], []
        for text, start, end in zip(
            examples["context.words.text"],
            examples["context.words.start"],
            examples["context.words.end"],
        ):
            # Choose only the last, keep_n_word_info, words.
            keep_start = start[-self.keep_n_word_info :]
            keep_end = end[-self.keep_n_word_info :]

            # WARNING!
            # Very important to insert whitspace/space before tokenizing
            # words, otherwise they will get very different values.
            sn = [
                " " + self.tokenizer.normalize_string(t)
                for t in text[-self.keep_n_word_info :]
            ]
            # raw tokenization without `SpokenDialogTokenizer` features
            _tokens = self.tokenizer._tokenizer(sn)["input_ids"]
            tmp_tokens, tmp_starts, tmp_ends = [], [], []
            for i, toks in enumerate(_tokens):
                for t in toks:
                    tmp_starts.append(keep_start[i])
                    tmp_ends.append(keep_end[i])
                    tmp_tokens.append(t)
            starts.append(tmp_starts)
            ends.append(tmp_ends)
            tokens.append(tmp_tokens)
        return tokens, starts, ends

    def _encode(self, examples):
        ret = self.tokenizer(examples["context.turns"], include_end_ts=False)
        _ = ret.pop("attention_mask")
        ret["frame_offset"], ret["num_frames"] = self.get_frames(
            examples["response.start"]
        )

        tokens, starts, ends = self.get_token_start_end(examples)
        ret["word.token"] = tokens
        ret["word.start"] = starts
        ret["word.end"] = ends
        return ret


class ClassificationDataModule(pl.LightningDataModule):
    label_to_idx = {"backchannel": 0, "shift": 1, "hold": 2}
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    FEATURES = ["token", "waveform", "vad_history", "vad", "f0"]

    def __init__(
        self,
        tokenizer=None,
        audio_duration=5,
        lookahead_time=2,
        response_diff_time=0.1,
        sample_rate=8000,
        max_length=100,
        omit_post_words=True,
        dataset_path=None,
        vad_frame_time=0.1,
        f0_mean_path=F0_MEAN_PATH,
        f0_frame_time=0.05,
        f0_hop_time=0.02,
        features=["waveform", "vad_history", "vad", "f0"],
        batch_size=8,
        num_workers=4,
        pin_memory=False,
        num_proc=4,
        load_from_cache_file=True,
        batched=True,
        audio_root=None,
    ):
        super().__init__()
        self.audio_root = audio_root
        self.tokenizer = tokenizer

        assert all(
            [f in self.FEATURES for f in features]
        ), f"Features not accepted: {features}. Choose {self.FEATURES}"

        if tokenizer is None and "token" in features:
            features.pop(features.index("token"))
            print('No tokenizer -> omitting "token" features')
        self.features = features
        print(self.features)

        # Collate fn
        self.audio_duration = audio_duration  # audio clip duration
        self.lookahead_time = lookahead_time
        self.sample_rate = sample_rate
        self.n_samples = int(audio_duration * self.sample_rate)
        self.response_diff_time = response_diff_time  # padding prior to response start
        self.response_diff_samples = int(sample_rate * response_diff_time)
        self.max_length = max_length

        # VAD
        self.vad_frame_time = vad_frame_time

        # F0
        self.frame_time = f0_frame_time
        self.hop_time = f0_hop_time
        self.hop_length = int(sample_rate * self.hop_time)
        self.frame_length = int(sample_rate * self.frame_time)
        self.f0_mean_path = f0_mean_path
        self.extract_f0 = False
        if "f0" in self.features:
            self.extract_f0 = True
            self.f0_means = read_json(self.f0_mean_path)

        # datasets: dataset.map(encode, ...)
        self.dataset_path = self._dataset_path(dataset_path)
        self.omit_post_words = omit_post_words
        self.num_proc = num_proc
        self.load_from_cache_file = load_from_cache_file
        self.batched = batched

        # DataLoaders
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def _dataset_path(self, dataset_path):
        if dataset_path is None:
            dataset_path = self.CACHE

        if self.tokenizer is not None:
            dataset_path = join(dataset_path, self.tokenizer.name_or_path)

        # dataset_path += f"_hop{self.hop_time}_frame{self.frame_time}"
        return dataset_path

    def _encode(self, examples):
        """
        we set `include_end_ts=False` b/c we don't want the last <ts> for this task
        """
        ret = self.tokenizer(examples["context.turns"], include_end_ts=False)
        _ = ret.pop("attention_mask")
        return ret

    def _encode_f0(self, examples):
        """
        we set `include_end_ts=False` b/c we don't want the last <ts> for this task
        """

        ap = join(self.audio_root, examples["audio_path"] + ".wav")
        frame_offset, num_frames = self.get_audio_samples(
            end=examples["response.start"]
        )
        x, _ = torchaudio.load(ap, frame_offset=frame_offset, num_frames=num_frames)
        x = x[examples["context.speaker"]].unsqueeze(0)
        f0, _ = pYAAPT(
            x,
            sample_rate=self.sample_rate,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )
        return {"f0": f0}

    def _split_path(self, split):
        return join(self.dataset_path, split)

    def prepare_data(self):
        for split in ["train", "validation", "test"]:
            split_path = self._split_path(split)
            if not exists(split_path) or not self.load_from_cache_file:
                assert (
                    self.tokenizer is not None
                ), f"Dataset requires processing with tokenizer!\n{self.dataset_path}"

                dataset = load_dataset(
                    DATASET_SCRIPT,
                    split=split,
                    name="classification",
                    omit_post_words=self.omit_post_words,
                )
                dataset = dataset.flatten()
                dataset = dataset.remove_columns(
                    [
                        # "session",
                        # "audio_path",
                        # "label",
                        # "context.turns",
                        "context.start",
                        "context.end",
                        # "context.speaker",
                        "context.words.text",
                        "context.words.start",
                        "context.words.end",
                        "response.id",
                        "response.text",
                        "response.speaker",
                        # "response.start",
                        "response.end",
                        "response.words.text",
                        "response.words.start",
                        "response.words.end",
                    ]
                )
                dataset = dataset.map(
                    self._encode,
                    batched=self.batched,
                    load_from_cache_file=self.load_from_cache_file,
                    num_proc=self.num_proc,
                )
                # if self.extract_f0:
                #     dataset = dataset.map(
                #         self._encode_f0,
                #         load_from_cache_file=self.load_from_cache_file,
                #         batched=False,
                #         num_proc=1,
                #     )
                dataset.set_format(type="torch")
                dataset.save_to_disk(split_path)

    def setup(self, stage: Optional[str] = None):
        if stage == "test":
            self.test_dset = load_from_disk(self._split_path("test"))
        else:
            self.train_dset = load_from_disk(self._split_path("train"))
            self.val_dset = load_from_disk(self._split_path("validation"))

            if stage == "dev":
                self.train_dset = self.train_dset[:100]
                self.val_dset = self.val_dset[:40]

    def get_audio_samples(self, end):
        num_frames = int(self.sample_rate * self.audio_duration)
        frame_end = int(self.sample_rate * end) - self.response_diff_samples
        frame_offset = frame_end - num_frames

        if frame_offset < 0:
            frame_offset = 0
            num_frames = frame_end
        return frame_offset, num_frames

    def collate_fn(self, batch):
        ret = {
            "label": [],
            "speaker": [],
            "session": [],
            "response.start": [],
        }
        if "token" in self.features:
            ret["input_ids"] = []
            ret["speaker_ids"] = []

        if "waveform" in self.features:
            ret["waveform"] = []
            ret["n_samples"] = []

        if "vad" in self.features:
            ret["vad"] = []

        if "vad_history" in self.features:
            ret["vad_history"] = []

        if "f0" in self.features:
            ret["f0_mean"] = []

        for b in batch:
            # General
            ret["speaker"].append(b["context.speaker"])
            ret["session"].append(b["session"])
            ret["label"].append(self.label_to_idx[b["label"]])
            ret["response.start"].append(b["response.start"])

            # TODO: Keep track of which words occur in the current features
            if "token" in self.features:
                ret["input_ids"].append(b["input_ids"][-self.max_length :])
                ret["speaker_ids"].append(b["speaker_ids"][-self.max_length :])

            if "vad" in self.features:
                # Vad for the current features
                vad = get_current_vad_onehot(
                    b["context.vad"],
                    end=b["response.start"] + self.lookahead_time,
                    speaker=b["context.speaker"],
                    frame_size=self.vad_frame_time,
                    duration=self.audio_duration + self.lookahead_time,
                )
                ret["vad"].append(vad)

            if "vad_history" in self.features:
                # history up until the current features arrive
                vad_hist = get_vad_condensed_history(
                    b["context.vad"],
                    b["response.start"] - self.audio_duration,
                    b["context.speaker"],
                )
                ret["vad_history"].append(vad_hist)

            if "f0" in self.features:
                ret["f0_mean"].append(self.f0_means[b["session"]][b["context.speaker"]])

            if "waveform" in self.features:
                ap = join(self.audio_root, b["audio_path"] + ".wav")
                frame_offset, num_frames = self.get_audio_samples(
                    end=b["response.start"]
                )
                x, _ = torchaudio.load(
                    ap, frame_offset=frame_offset, num_frames=num_frames
                )

                # Add silence for consistent waveform shape
                if x.shape[-1] != self.n_samples:
                    diff = self.n_samples - x.shape[-1]
                    x = torch.cat((torch.zeros((x.shape[0], diff)), x), dim=-1)
                ret["waveform"].append(x[b["context.speaker"]])
                ret["n_samples"].append(ret["waveform"][-1].shape[-1])

        ret["response.start"] = torch.tensor(ret["response.start"])
        ret["speaker"] = torch.tensor(ret["speaker"])
        ret["label"] = torch.tensor(ret["label"])

        if "token" in self.features:
            r = self.tokenizer.pad({"input_ids": ret["input_ids"]}, return_tensors="pt")
            ret["input_ids"] = r["input_ids"]
            ret["attention_mask"] = r["attention_mask"]
            ret["speaker_ids"] = self.tokenizer.pad(
                {"input_ids": ret["speaker_ids"]}, return_tensors="pt"
            )["input_ids"]

        if "waveform" in self.features:
            ret["waveform"] = torch.stack(ret["waveform"])
            ret["n_samples"] = torch.tensor(ret["n_samples"])

        if "vad" in self.features:
            ret["vad"] = torch.stack(ret["vad"])

        if "vad_history" in self.features:
            ret["vad_history"] = torch.stack(ret["vad_history"])

        if "f0" in self.features:
            ret["f0_mean"] = torch.tensor(ret["f0_mean"])

        return ret

    def train_dataloader(self):
        return DataLoader(
            self.train_dset,
            collate_fn=self.collate_fn,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            collate_fn=self.collate_fn,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dset,
            collate_fn=self.collate_fn,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # general
        parser.add_argument("--audio_root", default=None, type=str)
        parser.add_argument("--dataset_path", default=None, type=str)
        parser.add_argument("--overwrite", default=False, type=bool)

        # Features
        parser.add_argument("--audio_duration", default=5, type=int)
        parser.add_argument("--lookahead_time", default=2, type=int)
        parser.add_argument("--vad_frame_time", default=0.1, type=float)
        parser.add_argument("--f0_frame_time", default=0.05, type=float)
        parser.add_argument("--f0_hop_time", default=0.02, type=float)
        parser.add_argument("--response_diff_time", default=0.1, type=float)
        parser.add_argument("--sample_rate", default=8000, type=int)
        parser.add_argument("--omit_post_words", default=True, type=bool)
        parser.add_argument("--f0_mean_path", default=None, type=str)
        parser.add_argument(
            "--features",
            action="store",
            type=str,
            nargs="*",
            default=["token", "waveform", "vad_history", "vad", "f0"],
            help="Examples: -i waveform token, -i item3",
        )
        parser.add_argument(
            "--max_length",
            default=500,
            type=int,
            help="maximum length of sequences (applied in `collate_fn`)",
        )

        # arguments for `datasets` library
        n_cpus = cpu_count()
        parser.add_argument("--load_from_cache_file", default=True, type=bool)
        parser.add_argument("--num_proc", default=n_cpus, type=int)
        parser.add_argument("--batched", default=True, type=bool)

        # DataLoaders
        parser.add_argument("--batch_size", default=4, type=int)
        parser.add_argument("--num_workers", default=n_cpus, type=int)
        parser.add_argument("--pin_memory", default=True, type=bool)
        return parser


def debug_load_dm(batch_size=4, lookahead_time=2):
    from argparse import ArgumentParser
    from convlm.turngpt.tokenizer import SpokenDialogTokenizer

    parser = ArgumentParser()
    parser = ClassificationDataModule.add_data_specific_args(parser)
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    print("Loading tokenizer...")
    tokenizer = SpokenDialogTokenizer("gpt2")
    print("Done")

    args.audio_root = AUDIO_ROOT
    args.f0_mean_path = F0_MEAN_PATH
    args.batch_size = batch_size
    args.lookahead_time = lookahead_time
    dm = ClassificationDataModule(
        tokenizer=tokenizer,
        audio_duration=args.audio_duration,
        lookahead_time=args.lookahead_time,
        vad_frame_time=args.vad_frame_time,
        sample_rate=args.sample_rate,
        response_diff_time=args.response_diff_time,
        max_length=args.max_length,
        omit_post_words=args.omit_post_words,
        dataset_path=args.dataset_path,
        f0_mean_path=args.f0_mean_path,
        features=args.features,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        num_proc=args.num_proc,
        load_from_cache_file=args.load_from_cache_file,
        batched=args.batched,
        audio_root=args.audio_root,
    )
    dm.prepare_data()
    dm.setup("fit")
    return dm


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from convlm.turngpt.tokenizer import SpokenDialogTokenizer
    from datasets_turntaking.features.plot_utils import plot_vad_oh
    from datasets_turntaking.features.f0 import mean_ratio_f0, mean_subtracted_f0

    parser = ArgumentParser()
    parser = ClassificationDataModule.add_data_specific_args(parser)
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    print("Loading tokenizer...")
    tokenizer = None
    tokenizer = SpokenDialogTokenizer("gpt2")
    print("Done")

    args.audio_root = "/home/erik/projects/data/switchboard/audio"
    args.f0_mean_path = F0_MEAN_PATH
    args.features = ["token", "waveform", "vad_history", "vad", "f0"]
    args.batch_size = 32
    dm = ClassificationDataModule(
        tokenizer=tokenizer,
        audio_duration=args.audio_duration,
        lookahead_time=args.lookahead_time,
        sample_rate=args.sample_rate,
        response_diff_time=args.response_diff_time,
        max_length=args.max_length,
        omit_post_words=args.omit_post_words,
        dataset_path=args.dataset_path,
        f0_mean_path=args.f0_mean_path,
        features=args.features,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        num_proc=args.num_proc,
        load_from_cache_file=args.load_from_cache_file,
        batched=args.batched,
        audio_root=args.audio_root,
    )
    dm.prepare_data()
    dm.setup("fit")
    print("train: ", len(dm.train_dset))
    print("val: ", len(dm.val_dset))
    dloader = dm.train_dataloader()
    batch = next(iter(dloader))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")

    # sample_rate = 8000
    # frame_time = 0.05
    # hop_time = 0.02
    # frame_length = int(sample_rate * frame_time)
    # hop_length = int(sample_rate * hop_time)
    # # print("hop_length: ", hop_length, hop_time)
    # # print("frame_lenth: ", frame_length, frame_time)
    # f0, _ = pYAAPT(
    #     batch["waveform"],
    #     sample_rate=8000,
    #     frame_length=frame_length,
    #     hop_length=hop_length,
    # )
    # f = mean_ratio_f0(f0, batch["f0_mean"])
    # # f = mean_subtracted_f0(f0, batch['f0_mean'])
    # fig, ax = plt.subplots(1, 1)
    # for x in f:
    #     ax.plot(x)
    # plt.pause(0.1)

    target = int(
        batch["vad"].shape[-1]
        * (args.audio_duration / (args.audio_duration + args.lookahead_time))
    )
    for i in range(batch["vad"].shape[0]):
        fig, ax = plot_vad_oh(vad_oh=batch["vad"][i], plot=False)
        ax.set_title(f"label: {dm.idx_to_label[batch['label'][i]]}")
        ax.vlines(x=target, ymin=-1, ymax=1, color="r", linewidth=2)
        plt.show()
