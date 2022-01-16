from datasets_turntaking.callhome import load_callhome


if __name__ == "__main__":
    from datasets_turntaking.utils import load_waveform
    from datasets_turntaking.features.vad import VAD
    from datasets_turntaking.features.plot_utils import plot_vad_list, plot_vad_oh
    import matplotlib.pyplot as plt

    dset = load_callhome("train")
    print("Callhome: ", len(dset))
    print("-" * 30)
    d = dset[1]
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for kk, vv in v.items():
                print(f"\t{kk}: {len(vv)}")
        elif k == "vad":
            print("vad: ", len(v[0]), len(v[1]))
        else:
            print(f"{k}: {v}")
    x, sr = load_waveform(d["audio_path"])
    duration = x.shape[-1] / sr
    print("waveform: ", x.shape)
    print("duration: ", round(duration, 2))
    print("sample_rate: ", sr)

    vad_frames = VAD.vad_list_to_onehot(
        d["vad"],
        sample_rate=sr,
        hop_length=80,
        duration=duration,
        channel_last=True,
    )
    plot_vad_list(d["vad"], end_time=70, target_time=40)

    x, sr = load_waveform(d["audio_path"], start_time=10, end_time=30, normalize=False)

    fig, ax = plt.subplots(1, 1)
    ax.plot(x[0, : int(8e4)])
    ax.plot(x[1, : int(8e4)])
    ax.set_ylim([-1, 1])
    plt.show()
