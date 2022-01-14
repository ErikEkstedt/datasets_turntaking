import torch


def z_norm(x):
    m = x.mean(dim=-2)
    s = x.std(dim=-2)
    return (x - m) / s


def z_norm_non_zero(x):
    fnorm = []
    for i in range(x.shape[-1]):
        tmp_f = x[..., i]
        nz = tmp_f != 0
        m = tmp_f[nz].mean()
        s = tmp_f[nz].std()
        tmp_f[nz] = (tmp_f[nz] - m) / s
        fnorm.append(tmp_f)
    return torch.stack(fnorm, dim=-1)
