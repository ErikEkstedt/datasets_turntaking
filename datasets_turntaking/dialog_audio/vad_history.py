import torch
import torch.nn.functional as F
from einops import rearrange



@torch.no_grad()
def get_activity_history2(vad_frames, bin_end_frames, channel_last=True):
    """

    Uses convolutions to sum the activity over each segment of interest.

    The kernel size is set to be the number of frames of any particular segment i.e.

    ---------------------------------------------------


    ```
    ... h0       | h1 | h2 | h3 | h4 +
    distant past |    |    |    |    +
    -inf -> -t0  |    |    |    |    +

    ```

    ---------------------------------------------------

    Arguments:
        vad_frames:         torch.tensor: (Channels, N_Frames) or (N_Frames, Channels)
        bin_end_frames:     list: boundaries for the activity history windows i.e. [6000, 3000, 1000, 500]
        channel_last:       bool: if true we expect `vad_frames` to be (N_Frames, Channels)

    Returns:
        ratios:             torch.tensor: (Channels, N_frames, bins) or (N_frames, bins, Channels) (dependent on `channel_last`)
        history_bins:       torch.tesnor: same size as ratio but contains the number of active frames, over each segment, for both speakers.
    """

    N = vad_frames.shape[0]
    if channel_last:
        vad_frames = rearrange(vad_frames, "n c -> c n")

    # container for the activity of the defined bins
    hist_bins = []

    # Distance past activity history/ratio
    # The segment from negative infinity to the first bin_end_frames
    if vad_frames.shape[0] > bin_end_frames[0]:
        h0 = vad_frames[:, : -bin_end_frames[0]].cumsum(dim=-1)
        diff_pad = torch.ones(2, bin_end_frames[0]) * -1
        h0 = torch.cat((diff_pad, h0), dim=-1)
    else:
        # there is not enough duration to get any long time information
        # -> set to prior of equal speech
        # negative values for debugging to see where we provide prior
        # (not seen outside of this after r0/r1 further down)
        h0 = torch.ones(2, N) * -1
    hist_bins.append(h0)

    # Activity of segments defined by the the `bin_end_frames`

    # If 0 is not included in the window (i.e. the current frame)
    # we append it for consistency in loop below
    if bin_end_frames[-1] != 0:
        bin_end_frames = bin_end_frames + [0]

    # Loop over each segment window, construct conv1d (summation: all weights are 1.)
    # Omit end-frames which are not used for the current bin
    # concatenate activity sum with pad (= -1) at the start where the bin values are
    # not defined.
    for start, end in zip(bin_end_frames[:-1], bin_end_frames[1:]):
        ks = start - end
        if end > 0:
            vf = vad_frames[:, :-end]
        else:
            vf = vad_frames
        if vf.shape[1] > 0:
            filters = torch.ones((1, 1, ks), dtype=torch.float)
            vf = F.pad(vf, [ks - 1, 0]).unsqueeze(1)  # add channel dim
            o = F.conv1d(vf, weight=filters).squeeze(1)  # remove channel dim
            if end > 0:
                # print('diffpad: ', end)
                diff_pad = torch.ones(2, end) * -1
                o = torch.cat((diff_pad, o), dim=-1)
        else:
            # there is not enough duration to get any long time information
            # -> set to prior of equal speech
            # negative values for debugging to see where we provide prior
            # (not seen outside of this after r0/r1 further down)
            o = torch.ones(2, N) * -1
        hist_bins.append(o)

    # stack together -> (2, N, len(bin_end_frames) + 1) default: (2, N, 5)
    hist_bins = torch.stack(hist_bins, dim=-1)

    # find the ratios for each speaker
    r0 = hist_bins[0] / hist_bins.sum(dim=0)
    r1 = hist_bins[1] / hist_bins.sum(dim=0)

    # segments where both speakers are silent (i.e. [0, 0] activation)
    # are not defined (i.e. hist_bins / hist_bins.sum = 0 / 0 ).
    # Where both speakers are silent they have equal amount of
    nan_inds = torch.where(r0.isnan())
    r0[nan_inds] = 0.5
    r1[nan_inds] = 0.5

    # Consistent input/output with `channel_last` VAD
    if channel_last:
        ratio = torch.stack((r0, r1), dim=-1)
    else:
        ratio = torch.stack((r0, r1))
    return ratio, hist_bins


if __name__ == "__main__":

    from datasets_turntaking.dialog_audio.dataset import DialogAudioDataset
    from datasets_turntaking.dialog_audio.dm_dialog_audio import get_dialog_audio_datasets
    import matplotlib.pyplot as plt

    
    hfdset = get_dialog_audio_datasets(datasets=['switchboard'], split="train")

    dset = DialogAudioDataset(dataset=hfdset, sample_rate=8000, vad_hz=20)

    b = hfdset[0]
    d = dset.get_full_sample(b)

    v = d['vad']
    print("v: ", tuple(v.shape))

    a = v[0, :, 0].cumsum(0)
    b = v[0, :, 1].cumsum(0)

    fig, ax = plt.subplots(2, 1)
    ax[0].pcolormesh(a.unsqueeze(0))
    ax[1].pcolormesh(b.unsqueeze(0))
    plt.pause(0.1)

    vh, bins = get_activity_history2(v[0], bin_end_frames=[60, 30, 10, 5])
    print("vh: ", tuple(vh.shape))
