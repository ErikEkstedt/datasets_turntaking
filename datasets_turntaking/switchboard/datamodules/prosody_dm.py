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

DATASET_SCRIPT = join(
    repo_root(), "datasets_turntaking/dataset/switchboard/switchboard.py"
)
F0_MEAN_PATH = join(
    repo_root(), "datasets_turntaking/dataset/switchboard/f0_means.json"
)
AUDIO_ROOT = join(expanduser("~"), "projects/data/switchboard/audio")
CACHE_PATH = join(
    expanduser("~"), ".cache/datasets_turntaking/switchboard/classification/prosody"
)


class ClassiProsodyDataModule(pl.LightningDataModule):
    label_to_idx = {"backchannel": 0, "shift": 1, "hold": 2}
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    def __init__(
        self,
        audio_duration=4,
        audio_root=AUDIO_ROOT,
        f0_mean_path=F0_MEAN_PATH,
        dataset_path=None,
        max_length=100,
        omit_post_words=True,
        batch_size=8,
        num_workers=4,
        pin_memory=False,
        num_proc=4,
        load_from_cache_file=True,
        batched=True,
    ):
        super().__init__()
        self.dataset_path = self._dataset_path(dataset_path)

        self.f0_mean_path = f0_mean_path
        self.f0_means = read_json(self.f0_mean_path)

        # datasets: dataset.map(encode, ...)
        self.omit_post_words = omit_post_words
        self.num_proc = num_proc
        self.load_from_cache_file = load_from_cache_file
        self.batched = batched

        # Collate fn
        self.audio_root = audio_root
        self.max_length = max_length

        # audio frame count
        self.audio_duration = audio_duration
        self.sample_rate = 8000
        self.n_samples = int(audio_duration * self.sample_rate)

        # DataLoaders
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def _dataset_path(self, dataset_path):
        if dataset_path is not None:
            dataset_path = dataset_path
        else:
            dataset_path = CACHE_PATH
        return dataset_path

    def _split_path(self, split):
        return join(self.dataset_path, split)

    def get_frames(self, start):
        num_frames = int(self.sample_rate * self.audio_duration)
        frame_end = int(self.sample_rate * start)  # end at response start
        frame_offset = frame_end - num_frames

        if frame_offset < 0:
            frame_offset = 0
            num_frames = frame_end
        return frame_offset, num_frames

    def _encode(self, examples):
        """
        we set `include_end_ts=False` b/c we don't want the last <ts> for this task
        """
        ret = {}
        # ret = self.tokenizer(examples["context.turns"], include_end_ts=False)
        # _ = ret.pop("attention_mask")

        # Extract Prosody
        ap = join(self.audio_root, examples["audio_path"] + ".wav")
        frame_offset, num_frames = self.get_frames(examples["response.start"])
        x, _ = torchaudio.load(ap, frame_offset=frame_offset, num_frames=num_frames)

        speaker = examples["context.speaker"]
        x = x[speaker].unsqueeze(0)
        f0_mean = self.f0_means[examples["session"]][speaker]
        f0_mean = torch.tensor(f0_mean).view(-1, 1)
        ret["prosody"] = self.prosody_encoder(x, f0_mean)
        return ret

    def prepare_data(self):
        self.prosody_encoder = Prosody(sample_rate=8000)
        # for split in ["train", "validation", "test"]:
        for split in ["test"]:
            split_path = self._split_path(split)
            if not exists(split_path) or not self.load_from_cache_file:
                # assert (
                #     self.tokenizer is not None
                # ), "Dataset requires processing with tokenizer!"

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
                    # batched=self.batched,
                    batched=False,
                    load_from_cache_file=self.load_from_cache_file,
                    num_proc=self.num_proc,
                )
                dataset.set_format(type="torch")
                makedirs(dirname(split_path), exist_ok=True)
                dataset.save_to_disk(split_path)

    def setup(self, stage: Optional[str] = None):
        if stage == "test":
            self.test_dset = load_from_disk(self._split_path("test"))
        else:
            self.train_dset = load_from_disk(self._split_path("train"))
            self.val_dset = load_from_disk(self._split_path("validation"))

    def collate_fn(self, batch):
        labels, speaker = [], []
        prosody = []
        waveforms = []
        for b in batch:
            labels.append(self.label_to_idx[b["label"]])
            speaker.append(b["context.speaker"])
            prosody.append(torch.tensor(b["prosody"]))

            ap = join(self.audio_root, b["audio_path"] + ".wav")
            frame_offset, num_frames = self.get_frames(b["response.start"])
            x, _ = torchaudio.load(ap, frame_offset=frame_offset, num_frames=num_frames)

            # Add silence for consistent waveform shape
            if x.shape[-1] != self.n_samples:
                diff = self.n_samples - x.shape[-1]
                x = torch.cat((x, torch.zeros((x.shape[0], diff))), dim=-1)
            waveforms.append(x[b["context.speaker"]])

        ret = {
            "label": torch.tensor(labels),
            "speaker": torch.tensor(speaker),
            "waveform": torch.stack(waveforms),
            "prosody": torch.cat(prosody),
        }

        for k, v in ret.items():
            if not isinstance(v, torch.Tensor):
                ret[k] = torch.tensor(v)
        return ret

    def _dataloader(self, dset, shuffle=True):
        return DataLoader(
            dset,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self._dataloader(self.train_dset, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dset, shuffle=False)

    def test_dataloader(self):
        return self._dataloader(self.test_dset, shuffle=False)

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--audio_root", default=AUDIO_ROOT, type=str)
        parser.add_argument("--f0_mean_path", default=F0_MEAN_PATH, type=str)
        parser.add_argument("--dataset_path", default=None, type=str)
        parser.add_argument("--overwrite", default=False, type=bool)
        parser.add_argument("--audio_duration", default=3, type=int)

        n_cpus = cpu_count()
        parser.add_argument("--batch_size", default=4, type=int)
        parser.add_argument("--num_workers", default=n_cpus, type=int)
        parser.add_argument("--pin_memory", default=True, type=bool)
        parser.add_argument(
            "--max_length",
            default=200,
            type=int,
            help="maximum length of sequences (applied in `collate_fn`)",
        )

        # arguments for `datasets` library
        parser.add_argument("--load_from_cache_file", default=True, type=bool)
        parser.add_argument("--num_proc", default=n_cpus, type=int)
        return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = ClassiProsodyDataModule.add_data_specific_args(parser)
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    dm = ClassiProsodyDataModule(
        audio_duration=args.audio_duration,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        audio_root=args.audio_root,
        max_length=args.max_length,
        num_proc=args.num_proc,
        # load_from_cache_file=False,
    )
    dm.prepare_data()
    dm.setup("test")
    # print("train: ", len(dm.train_dset))
    # print("val: ", len(dm.val_dset))
    # dloader = dm.train_dataloader()

    dloader = dm.test_dataloader()
    batch = next(iter(dloader))
    for k, v in batch.items():
        if k == "session":
            print(v)
        else:
            print(k, type(v), v.shape)

    from datasets_turntaking.features.plot_utils import plot_prosody

    x = batch["waveform"]
    p = batch["prosody"]

    # plot_prosody(p, x, channel=0, sample_rate=8000, hop_length=200, frame_length=400)
    for i in range(x.shape[0]):
        plot_prosody(
            p, x, channel=i, sample_rate=8000, hop_length=200, frame_length=400
        )
