import time
import random
import torch
import torchaudio.functional as AF
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets_turntaking.dialog_audio_dataset import (
    DialogAudioDataset,
    load_spoken_dialog_audio_dataset,
)
from datasets_turntaking.features.transforms import LogMelSpectrogram
from datasets_turntaking.features.plot_utils import plot_mel_spec, plot_stereo_mel_spec


def check_both_active(vad, min_active_frames=0):
    # assert vad.ndim == 3, f"expexts vad of shape (B, N_frames, 2) got {vad.shape}"
    # sum over frame dim
    v = vad.sum(dim=1)
    a_is_active = v[:, 0] > min_active_frames
    b_is_active = v[:, 1] > min_active_frames
    ok = torch.logical_and(a_is_active, b_is_active)
    return ok


if __name__ == "__main__":

    mel_transform = LogMelSpectrogram(hop_time=0.02)
    dset_hf = load_spoken_dialog_audio_dataset(
        ["switchboard"], split="val", min_word_vad_diff=0.1
    )
    dset = DialogAudioDataset(
        dataset=dset_hf,
        # type="sliding",
        type="events",
        vad_history=False,
        vad_hz=50,
        audio_mono=False,
        mask_vad=True,
        mask_vad_probability=1.,
        mask_vad_scale=0.01,
        transforms=mel_transform,
    )
    vad_hz = 50
    mel_hz = mel_transform.frame_hz
    # d = dset[0]
    # mel_spec = d["waveform"][0]
    # vad = d['vad'][0, :-dset.vad_horizon]
    # print("mel_spec: ", tuple(mel_spec.shape))
    # n_channel, n_mels, n_frames = mel_spec.shape

    d=dset[0]
    # vad = torch.cat([d['vad'], d['vad']])
    vad = d['vad']

    ok_vad = check_both_active(vad)

    N = 3000
    indices = random.sample(list(range(len(dset))), k=N)
    not_ok = 0
    for idx in tqdm(indices, total=N):
        d = dset[idx]
        ok_vad = check_both_active(d['vad'], 30)[0]
        if not ok_vad:
            not_ok += 1
    p_bad = round(100 * not_ok / N, 2)
    print(f"{not_ok}/{N} -> {p_bad}%")
    # 207/1000


    waveform = d['waveform']

    while True:
        idx = torch.randint(0, len(dset), (1,)).item()
        d = dset[idx]
        mel_spec = d["waveform"][0]
        n_chanels, n_mels, n_frames = mel_spec.shape
        vad = d['vad'][0, :-dset.vad_horizon]
        # print("mel_spec: ", tuple(mel_spec.shape))
        plt.close('all')
        fig, ax = plt.subplots(2, 1, figsize=(12, 4))
        # Upsample vad
        if vad_hz != mel_hz:
            vad_mel_frames = AF.resample(
                vad.permute(1, 0), orig_freq=vad_hz, new_freq=mel_hz
            ).permute(1, 0)
            vad_mel_frames.clamp_(min=0, max=1)
        else:
            vad_mel_frames = vad.clone()
        plot_stereo_mel_spec(mel_spec, ax=ax)
        ax[0].plot(vad_mel_frames[:n_frames, 0] * (n_mels - 1), alpha=0.9, linewidth=2, color="b")
        ax[1].plot(vad_mel_frames[:n_frames, 1] * (n_mels - 1), alpha=0.9, linewidth=2, color="orange")
        plt.tight_layout()
        plt.pause(0.1)
        time.sleep(1)
        # input()

    mins = []
    maxs = []
    N = 1000
    for ii, d in tqdm(enumerate(dset), total=N):
        mel_spec = d["waveform"]
        tmp_min = mel_spec.min()
        tmp_max = mel_spec.max()
        mins.append(tmp_min)
        maxs.append(tmp_max
        if ii == N:
            break

    mins = torch.stack(mins)
    minimum = mins.min()
    maxs = torch.stack(maxs)
    maximum = mins.max()
    print("Min")
    print("\tmean: ", mins.mean())
    print("\tstd: ", mins.std())
    print("Max")
    print("\tmean: ", maxs.mean())
    print("\tstd: ", maxs.std())
    print(minimum, maximum)

    # print('min: ', )
    # print('max: ', mel_spec.max())
    # fig, [ax1, ax2, ax3] = plt.subplots(3, 1)
    # plot_stereo_mel_spec(mel_spec[0], ax=[ax1, ax2])
    # plt.show()
