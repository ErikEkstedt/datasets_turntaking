import torch
from copy import deepcopy

from datasets_turntaking.utils import find_island_idx_len


def get_channel_ipus(
    vad_channel, ipu_pause_frames, ipu_min_frames, audio_context_frames=-1
):
    ipu = deepcopy(vad_channel)
    starts, dur, v = find_island_idx_len(ipu)

    # Pause silences below threshold (ipu_pause_frames)
    # are filled to join vad-segments to IPU
    pause_starts = starts[v == 0]
    pause_dur = dur[v == 0]
    fill_starts = pause_starts[pause_dur < ipu_pause_frames]
    fill_dur = pause_dur[pause_dur < ipu_pause_frames]

    # Fill silences below `ipu_pause_frames`
    for s, d in zip(fill_starts, fill_dur):
        ipu[s : s + d] = 1

    # get new values for the filled "ipus"
    starts, dur, v = find_island_idx_len(ipu)
    # focus on the active segments (vadvalue => 1)
    starts = starts[v == 1]
    dur = dur[v == 1]

    # check which IPU segments are above the threshold
    keep = dur >= ipu_min_frames
    starts = starts[keep]
    dur = dur[keep]
    ends = starts + dur

    # check that the end is not before the required audio context
    if audio_context_frames > 0:
        keep = ends >= audio_context_frames
        ends = ends[keep]
    return ends, ipu


def get_ipu_ends(vad, ipu_pause_frames, ipu_min_frames, audio_context_frames=-1):
    ends0, _ = get_channel_ipus(
        vad[:, 0], ipu_pause_frames, ipu_min_frames, audio_context_frames
    )
    ends1, _ = get_channel_ipus(
        vad[:, 1], ipu_pause_frames, ipu_min_frames, audio_context_frames
    )

    # ipu = torch.stack((ipu0, ipu1), dim=-1) # _, ipu0 = get_channel_ipus...
    v = torch.cat((ends0, ends1))
    s = torch.cat((torch.zeros_like(ends0), torch.ones_like(ends1)))
    ipu_ends, perm = v.sort()
    speakers = s[perm]
    return ipu_ends, speakers
