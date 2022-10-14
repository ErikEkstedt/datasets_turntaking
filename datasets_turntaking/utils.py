from os.path import basename, dirname
from os import remove
from omegaconf import OmegaConf
import json
import subprocess
from typing import Any, Dict, Union, Optional, Tuple

import torch
import torchaudio
import torchaudio.functional as AF
from torchaudio.backend.sox_io_backend import info as info_sox


def samples_to_frames(s, hop_len):
    return int(s / hop_len)


def sample_to_time(n_samples, sample_rate):
    return n_samples / sample_rate


def frames_to_time(f, hop_time):
    return f * hop_time


def time_to_frames(t, hop_time):
    return int(t / hop_time)


def time_to_frames_samples(t: float, sample_rate: int, hop_length: int) -> int:
    return int(t * sample_rate / hop_length)


def time_to_samples(t: float, sample_rate: int) -> float:
    return int(t * sample_rate)


def get_audio_info(audio_path: str) -> Dict[str, Any]:
    info = info_sox(audio_path)
    return {
        "name": basename(audio_path),
        "duration": sample_to_time(info.num_frames, info.sample_rate),
        "sample_rate": info.sample_rate,
        "num_frames": info.num_frames,
        "bits_per_sample": info.bits_per_sample,
        "num_channels": info.bits_per_sample,
    }


def load_waveform(
    path: str,
    sample_rate: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    normalize: bool = False,
    mono: bool = False,
    audio_normalize_threshold: float = 0.05,
) -> Tuple[torch.Tensor, int]:
    if start_time is None:
        x, sr = torchaudio.load(path, normalize=False)
    else:
        info = get_audio_info(path)
        frame_offset = time_to_samples(start_time, info["sample_rate"])
        num_frames = info["num_frames"]
        if end_time is not None:
            num_frames = time_to_samples(end_time, info["sample_rate"]) - frame_offset
        else:
            num_frames = num_frames - frame_offset
        x, sr = torchaudio.load(
            path, frame_offset=frame_offset, num_frames=num_frames, normalize=False
        )

    # if normalize:
    #     if x.shape[0] > 1:
    #         if x[0].abs().max() > audio_normalize_threshold:
    #             x[0] /= x[0].abs().max()
    #         if x[1].abs().max() > audio_normalize_threshold:
    #             x[1] /= x[1].abs().max()
    #     else:
    #         if x.abs().max() > audio_normalize_threshold:
    #             x /= x.abs().max()

    if mono and x.shape[0] > 1:
        x = x.mean(dim=0).unsqueeze(0)
        # if normalize:
        #     if x.abs().max() > audio_normalize_threshold:
        #         x /= x.abs().max()

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


def write_txt(txt, name):
    """
    Argument:
        txt:    list of strings
        name:   filename
    """
    with open(name, "w") as f:
        f.write("\n".join(txt))


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


def load_config(
    path=None, args=None, format="dict"
) -> Union[Dict[str, Any],]:
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


def delete_path(filepath):
    remove(filepath)


def sph2pipe_to_wav(sph_file):
    wav_file = sph_file.replace(".sph", ".wav")
    subprocess.check_call(["sph2pipe", sph_file, wav_file])
    return wav_file
