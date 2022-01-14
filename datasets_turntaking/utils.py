from os.path import basename, dirname
from omegaconf import OmegaConf
import json

import torch
import torchaudio
import torchaudio.functional as AF
from torchaudio.backend.sox_io_backend import info as info_sox


def time2frames(t, sample_rate):
    return int(t * sample_rate)


def time2sample(t, sr):
    return int(t * sr)


def get_audio_info(audio_path):
    info = info_sox(audio_path)
    return {
        "name": basename(audio_path),
        "duration": info.num_frames / info.sample_rate,
        "sample_rate": info.sample_rate,
        "num_frames": info.num_frames,
        "bits_per_sample": info.bits_per_sample,
        "num_channels": info.bits_per_sample,
    }


def load_waveform(
    path, sample_rate=None, start_time=None, end_time=None, normalize=False, mono=False
):
    if start_time:
        info = get_audio_info(path)
        frame_offset = time2sample(start_time, info["sample_rate"])
        num_frames = info["num_frames"]
        if end_time:
            num_frames = time2sample(end_time, info["sample_rate"]) - frame_offset
        else:
            num_frames = num_frames - frame_offset
        x, sr = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
    else:
        x, sr = torchaudio.load(path)

    if normalize:
        if x.shape[0] > 1:
            x[0] /= x[0].abs().max()
            x[1] /= x[1].abs().max()
        else:
            x[0] /= x[0].abs().max()

    if mono and x.shape[0] > 1:
        x = x.mean(dim=0).unsqueeze(0)

    if sample_rate:
        if sr != sample_rate:
            x = AF.resample(x, orig_freq=sr, new_freq=sample_rate)
            sr = sample_rate
    return x, sr


def repo_root():
    """
    Returns the absolute path to the git repository
    """
    root = dirname(__file__)
    root = dirname(root)
    return root


def write_json(data, filename):
    with open(filename, "w", encoding="utf-8") as jsonfile:
        json.dump(data, jsonfile, ensure_ascii=False)


def read_json(path, encoding="utf8"):
    with open(path, "r", encoding=encoding) as f:
        data = json.loads(f.read())
    return data


def read_txt(path, encoding="utf-8"):
    data = []
    with open(path, "r", encoding=encoding) as f:
        for line in f.readlines():
            data.append(line.strip())
    return data


def find_island_idx_len(x):
    """
    Finds patches of the same value.

    starts_idx, duration, values = find_island_idx_len(x)

    e.g:
        ends = starts_idx + duration

        s_n = starts_idx[values==n]
        ends_n = s_n + duration[values==n]  # find all patches with N value

    """
    assert x.ndim == 1
    n = len(x)
    y = x[1:] != x[:-1]  # pairwise unequal (string safe)
    i = torch.cat(
        (torch.where(y)[0], torch.tensor(n - 1, device=x.device).unsqueeze(0))
    ).long()
    it = torch.cat((torch.tensor(-1, device=x.device).unsqueeze(0), i))
    dur = it[1:] - it[:-1]
    idx = torch.cumsum(
        torch.cat((torch.tensor([0], device=x.device, dtype=torch.long), dur)), dim=0
    )[
        :-1
    ]  # positions
    return idx, dur, x[i]


def load_config(path=None, args=None, format="dict"):
    conf = OmegaConf.load(path)
    if args is not None:
        conf = OmegaConfArgs.update_conf_with_args(conf, args)

    if format == "dict":
        conf = OmegaConf.to_object(conf)
    return conf


class OmegaConfArgs:
    """
    This is annoying... And there is probably a SUPER easy way to do this... But...

    Desiderata:
        * Define the model completely by an OmegaConf (yaml file)
            - OmegaConf argument syntax  ( '+segments.c1=10' )
        * run `sweeps` with WandB
            - requires "normal" argparse arguments (i.e. '--batch_size' etc)

    This class is a helper to define
    - argparse from config (yaml)
    - update config (loaded yaml) with argparse arguments


    See ./config/sosi.yaml for reference yaml
    """

    @staticmethod
    def add_argparse_args(parser, conf, omit_fields=None):
        for field, settings in conf.items():
            if omit_fields is None:
                for setting, value in settings.items():
                    name = f"--{field}.{setting}"
                    parser.add_argument(name, default=None, type=type(value))
            else:
                if not any([field == f for f in omit_fields]):
                    for setting, value in settings.items():
                        name = f"--{field}.{setting}"
                        parser.add_argument(name, default=None, type=type(value))
        return parser

    @staticmethod
    def update_conf_with_args(conf, args, omit_fields=None):
        if not isinstance(args, dict):
            args = vars(args)

        for field, settings in conf.items():
            if omit_fields is None:
                for setting in settings:
                    argname = f"{field}.{setting}"
                    if argname in args and args[argname] is not None:
                        conf[field][setting] = args[argname]
            else:
                if not any([field == f for f in omit_fields]):
                    for setting in settings:
                        argname = f"{field}.{setting}"
                        if argname in args:
                            conf[field][setting] = args[argname]
        return conf
