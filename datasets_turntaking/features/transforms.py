import torch
import torch.nn as nn
import torchaudio.transforms as AT
import einops
from typing import Dict

from datasets_turntaking.utils import time_to_samples, time_to_frames
import datasets_turntaking.features.functional as DF


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        n_mels: int = 80,
        window_time: float = 0.025,
        hop_time: float = 0.05,
        f_min: int = 55,
        f_max: int = 4000,
        sample_rate: int = 16_000,
    ):
        super().__init__()
        self.n_fft = time_to_samples(window_time, sample_rate)
        self.hop_length = time_to_samples(hop_time, sample_rate)
        self.f_min = f_min
        self.f_max = f_max

        self.mel_spectrogram = AT.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            normalized=True,
        )

    def __repr__(self) -> str:
        s = "LogMelSpectrogram(\n"
        s += str(self.mel_spectrogram)
        return s

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mel_spec = self.mel_spectrogram(waveform)
        log_mel_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_mel_spec = torch.maximum(log_mel_spec, log_mel_spec.max() - 8.0)
        log_mel_spec = (log_mel_spec + 4.0) / 4.0
        return log_mel_spec


class ProsodyTorch(nn.Module):
    """
    Computes the RootMeanSquare (rms) of a waveform

    Loosely based on (inspired by) the 'emobase' config in Opensmile:
        https://github.com/audeering/opensmile-python/blob/master/opensmile/core/config/emobase/emobase.conf

    Feature names 'emobase' :
        ['pcm_intensity_sma',
         'pcm_loudness_sma',
         'mfcc_sma[1]',
         'mfcc_sma[2]',
         'mfcc_sma[3]',
         'mfcc_sma[4]',
         'mfcc_sma[5]',
         'mfcc_sma[6]',
         'mfcc_sma[7]',
         'mfcc_sma[8]',
         'mfcc_sma[9]',
         'mfcc_sma[10]',
         'mfcc_sma[11]',
         'mfcc_sma[12]',
         'lspFreq_sma[0]',
         'lspFreq_sma[1]',
         'lspFreq_sma[2]',
         'lspFreq_sma[3]',
         'lspFreq_sma[4]',
         'lspFreq_sma[5]',
         'lspFreq_sma[6]',
         'lspFreq_sma[7]',
         'pcm_zcr_sma',
         'voiceProb_sma',
         'F0_sma',
         'F0env_sma']
    """

    def __init__(
        self,
        sample_rate=16000,
        frame_time=0.04,  # 40ms
        hop_time=0.01,  # 10ms
        n_mfcc=13,
        n_mels=26,
        lpc_order=7,
        center=True,
        mode="reflect",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_time = frame_time
        self.hop_time = hop_time
        self.frame_length = time_to_samples(frame_time, sample_rate)
        self.hop_length = time_to_samples(hop_time, sample_rate)
        self.center = center
        self.mode = mode

        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self._mfcc = AT.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": self.frame_length,
                "n_mels": n_mels,
                "hop_length": self.hop_length,
                "mel_scale": "htk",
            },
        )

        self.lpc_order = lpc_order
        self.n_features = 2 + (n_mfcc - 1) + (lpc_order + 1)

    @property
    def feature_names(self):
        names = ["rms", "zcr"]
        names += [f"mfcc_{i}" for i in range(1, self.n_mfcc)]
        names += [f"lpc_{i}" for i in range(self.lpc_order + 1)]
        return names

    def __repr__(self):
        s = "ProsodyTorch(\n"
        s += f"\tframe_time={self.frame_time}\n"
        s += f"\thop_time={self.hop_time}\n"
        s += f"\tframe_length={self.frame_length}\n"
        s += f"\thop_length={self.hop_length}\n"
        s += f"\tcenter={self.center}\n"
        s += f"\tmode={self.mode}\n"
        s += ")\n"
        s += f"{self.feature_names}"
        return s

    def mfcc(self, waveform):
        """MFCC extraction
        Opensmile 'emobase' uses 26 melspec bins and 1-12 mfcc so we omit the mfcc 0
        """

        mfcc = self._mfcc(waveform)
        mfcc = einops.rearrange(mfcc, "... d t -> ... t d")  # feature dim last
        return mfcc[..., 1:]

    def lpc(self, waveform):
        return DF.lpc(
            waveform,
            order=self.lpc_order,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            padding=True,
        )

    @torch.no_grad()
    def forward(self, waveform):
        features = []
        # RMS (Intensity/Loudness ?)
        features.append(
            DF.rms_torch(
                waveform,
                frame_length=self.frame_length,
                hop_length=self.hop_length,
                center=self.center,
                mode=self.mode,
            ).unsqueeze(-1)
        )
        # Zero Crossing Rate
        features.append(
            DF.zero_crossing_rate(
                waveform,
                frame_length=self.frame_length,
                hop_length=self.hop_length,
                center=self.center,
            ).unsqueeze(-1)
        )
        # MFCC
        features.append(self.mfcc(waveform))

        # LPC
        features.append(self.lpc(waveform))

        features = torch.cat(features, dim=-1)
        return features


class VadMaskScale(nn.Module):
    def __init__(
        self,
        scale: float = 0.1,
        vad_hz: int = 50,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.scale = scale
        self.vad_hz = vad_hz
        self.sample_rate = sample_rate

    def forward(self, x: torch.Tensor, vad: torch.Tensor):
        return DF.mask_around_vad(
            waveform=x,
            vad=vad,
            vad_hz=self.vad_hz,
            sample_rate=self.sample_rate,
            scale=self.scale,
        )


class FlipBatch(nn.Module):
    flippable = ["waveform", "vad", "vad_history"]

    def __init__(self):
        super().__init__()

    def flip_vad(self, vad: torch.Tensor) -> torch.Tensor:
        return torch.stack((vad[..., 1], vad[..., 0]), dim=-1)

    def flip_vad_history(self, vad_history: torch.Tensor) -> torch.Tensor:
        return 1 - vad_history

    def flip_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        assert (
            waveform.ndim == 3
        ), f"Expected waveform (B, C, n_samples) but got {waveform.shape}"

        if waveform.shape[1] == 2:
            waveform = torch.stack((waveform[:, 1], waveform[:, 0]), dim=1)
        return waveform

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Flips the channels/speakers (for effected fields)"""
        if "vad" in batch:
            batch["vad"] = self.flip_vad(batch["vad"])

        if "vad_history" in batch:
            batch["vad_history"] = self.flip_vad_history(batch["vad_history"])

        if "waveform" in batch:
            batch["waveform"] = self.flip_waveform(batch["waveform"])

        return batch


if __name__ == "__main__":
    import librosa

    extractor = ProsodyTorch()
    print(extractor)

    waveform, sr = librosa.load(librosa.ex("trumpet"), sr=16000, duration=10)
    waveform = torch.from_numpy(waveform).unsqueeze(0)

    features = extractor(waveform)
    print("features: ", tuple(features.shape))
