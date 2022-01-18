import torch
import torch.nn.functional as F
import opensmile

from datasets_turntaking.features.utils import z_norm, z_norm_non_zero


class OpenSmile:
    FEATURE_SETS = ["egemapsv02", "emobase"]

    def __init__(
        self,
        feature_set="egemapsv02",
        sample_rate=16000,
        normalize=False,
    ):
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.feature_set_name = feature_set
        feature_set = self.get_feature_set(feature_set)
        self.smile = opensmile.Smile(
            feature_set=feature_set,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
        self._settings()

    @property
    def feat2idx(self):
        return {k: idx for idx, k in enumerate(self.smile.feature_names)}

    @property
    def idx2feat(self):
        return {idx: k for idx, k in enumerate(self.smile.feature_names)}

    @property
    def feature_names(self):
        return self.smile.feature_names

    def _settings(self):
        if self.feature_set_name == "egemapsv02":
            self.pad_samples = int(self.sample_rate * 0.02)
            self.pad_frames = 2
            self.f0_idx = 10
            self.idx_special = [10, 13, 14, 15, 18, 21, 24]
            self.idx_reg = list(range(25))
            for ii in self.idx_special:
                self.idx_reg.pop(self.idx_reg.index(ii))

        elif self.feature_set_name == "emobase":
            self.pad_samples = int(self.sample_rate * 0.01)
            self.pad_frames = 1
            self.f0_idx = 24
            self.idx_special = [24, 25]
            self.idx_reg = list(range(26))
            for ii in self.idx_special:
                self.idx_reg.pop(self.idx_reg.index(ii))
        else:
            raise NotImplementedError()

        self.feat_reg = [self.idx2feat[idx] for idx in self.idx_reg]
        self.feat_special = [self.idx2feat[idx] for idx in self.idx_special]
        self.idx_reg = torch.tensor(self.idx_reg)
        self.idx_special = torch.tensor(self.idx_special)

    def get_feature_set(self, feature_set):
        feature_set = feature_set.lower()
        assert (
            feature_set in self.FEATURE_SETS
        ), f"{feature_set} not found. Try {self.FEATURE_SETS}"

        if feature_set == "egemapsv02":
            return opensmile.FeatureSet.eGeMAPSv02
        elif feature_set == "emobase":
            return opensmile.FeatureSet.emobase
        else:
            raise NotImplementedError()

    def __repr__(self):
        return str(self.smile)

    def __call__(self, waveform):
        # waveform = F.pad(waveform, (self.pad_samples, self.pad_samples))
        f = torch.from_numpy(
            self.smile.process_signal(waveform, self.sample_rate).to_numpy()
        )

        # pad
        print("f: ", tuple(f.shape))
        pre_pad = f[0].repeat(self.pad_frames, 1)
        print("pre_pad: ", tuple(pre_pad.shape))
        post_pad = f[-1].repeat(self.pad_frames, 1)
        print("post_pad: ", tuple(post_pad.shape))
        f = torch.cat((pre_pad, f, post_pad))

        if self.normalize:
            fr = z_norm(f[..., self.idx_reg])
            fs = z_norm_non_zero(f[..., self.idx_special])
            f[..., self.idx_reg] = fr
            f[..., self.idx_special] = fs

        if waveform.ndim == 2:
            f = f.unsqueeze(0)
        return f


def plot_features(features, names=None, figsize=(8, 16), plot=False):
    import matplotlib.pyplot as plt

    n_feats = features.shape[-1]

    if names is not None:
        assert len(names) == n_feats, "names must be same lenth as number of features"

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for i in range(n_feats):
        label = None
        if names is not None:
            label = names[i]
        ax.plot(features[..., i], label=label)

    if names is not None:
        ax.legend(loc="upper left")

    if plot:
        plt.pause(0.1)

    return fig, ax


def imshow_features(features, names=None, figsize=(8, 16), plot=False):
    import matplotlib.pyplot as plt

    n_feats = features.shape[-1]

    if names is not None:
        assert len(names) == n_feats, "names must be same lenth as number of features"

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.imshow(features.t(), aspect="auto", origin="lower", interpolation="none")
    if names is not None:
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)

    if plot:
        plt.pause(0.1)

    return fig, ax


def debug():
    import matplotlib.pyplot as plt
    from datasets_turntaking.dm_dialog_audio import quick_load_dm

    dm = quick_load_dm()
    diter = iter(dm.val_dataloader())
    batch = next(diter)
    smile = OpenSmile(feature_set="emobase", normalize=True)
    # smile = OpenSmile(feature_set="egemapsv02", normalize=True)
    # _ = [print(f) for f in smile.feature_names]
    features = smile(batch["waveform"][0])
    print("features: ", tuple(features.shape))
    fig, ax = imshow_features(features, names=smile.feature_names, plot=True)

    fig, ax = plot_features(features, names=smile.feature_names, plot=True)

    fig, ax = plot_features(
        features[..., :-2], names=smile.feature_names[:-2], plot=True
    )


if __name__ == "__main__":
    from tqdm import tqdm
    from datasets_turntaking.dm_dialog_audio import quick_load_dm

    dm = quick_load_dm()
    diter = iter(dm.val_dataloader())
    batch = next(diter)

    batch["waveform"].shape
    waveform = batch["waveform"]
    sample_rate = 16000

    ###################################################################

    smile = OpenSmile("emobase")

    x = torch.cat((waveform, waveform)).permute(1, 0)
    a = smile.smile.process_signal(x, sample_rate).to_numpy()

    # 12 - 62 for 10,000 samples (less individual dialogs)
    f0_min = 999
    f0_max = -f0_min
    N = 10000
    for i, batch in enumerate(tqdm(dm.train_dataloader(), total=N)):
        f = smile(batch["waveform"])
        f0s = f[..., 10].round()
        fmin = f0s[f0s != 0].min().item()
        fmax = f0s.max().item()
        if fmin < f0_min:
            f0_min = fmin
        if fmax > f0_max:
            f0_max = fmax
        if i == N:
            break
    print("min: ", f0_min)
    print("max: ", f0_max)

    ###################################################################
    f = smile(batch["waveform"], sample_rate, norm=False)
    fig, ax = plt.subplots(1, 1)
    ax.plot(f[..., 10])
    plt.pause(0.1)

    n = f.shape[-1]
    fig, ax = plt.subplots(n, 1, sharex=True, figsize=(8, 16))
    for i in range(n):
        ax[i].plot(f[:, i], label=smile.feature_names[i])
        ax[i].legend(loc="upper left")
    plt.pause(0.1)

    # Same plot
    n = f.shape[-1]
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 16))
    for i in range(n):
        ax.plot(f[:, i], label=smile.feature_names[i])
    ax.legend(loc="upper left")
    plt.pause(0.1)

    ###################################################################
    f0_idx = smile.feat2idx["F0semitoneFrom27.5Hz_sma3nz"]
