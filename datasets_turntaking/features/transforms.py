import torch
import torch.nn as nn
import torchaudio.transforms as AT
import einops

from datasets_turntaking.utils import time_to_samples
from datasets_turntaking.features.functional import rms_torch, zero_crossing_rate, lpc


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
        return lpc(
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
            rms_torch(
                waveform,
                frame_length=self.frame_length,
                hop_length=self.hop_length,
                center=self.center,
                mode=self.mode,
            ).unsqueeze(-1)
        )
        # Zero Crossing Rate
        features.append(
            zero_crossing_rate(
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


if __name__ == "__main__":
    import librosa

    extractor = ProsodyTorch()
    print(extractor)

    waveform, sr = librosa.load(librosa.ex("trumpet"), sr=16000, duration=10)
    waveform = torch.from_numpy(waveform).unsqueeze(0)

    features = extractor(waveform)
    print("features: ", tuple(features.shape))
