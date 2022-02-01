from typing import Dict, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from datasets_turntaking.utils import find_island_idx_len


def frame2time(f, frame_time):
    return f * frame_time


def time2frames(t, sample_rate, hop_length):
    return int(t * sample_rate / hop_length)


class VAD:
    @staticmethod
    def vad_to_dialog_vad_states(vad) -> torch.Tensor:
        """Vad to the full state of a 2 person vad dialog
        0: only speaker 0
        1: none
        2: both
        3: only speaker 1
        """
        assert vad.ndim >= 1
        return (2 * vad[..., 1] - vad[..., 0]).long() + 1

    @staticmethod
    def vad_list_to_onehot(vad, sample_rate, hop_length, duration, channel_last=False):
        n_frames = time2frames(duration, sample_rate, hop_length) + 1

        if isinstance(vad[0][0], list):
            vad_tensor = torch.zeros((len(vad), n_frames))
            for ch, ch_vad in enumerate(vad):
                for v in ch_vad:
                    s = time2frames(v[0], sample_rate, hop_length)
                    e = time2frames(v[1], sample_rate, hop_length)
                    vad_tensor[ch, s:e] = 1.0
        else:
            vad_tensor = torch.zeros((1, n_frames))
            for v in vad:
                s = time2frames(v[0], sample_rate, hop_length)
                e = time2frames(v[1], sample_rate, hop_length)
                vad_tensor[:, s:e] = 1.0

        if channel_last:
            vad_tensor = vad_tensor.permute(1, 0)

        return vad_tensor

    @staticmethod
    def get_current_vad_onehot(vad, end, duration, speaker, frame_size):
        """frame_size in seconds"""
        start = end - duration
        n_frames = int(duration / frame_size)
        vad_oh = torch.zeros((2, n_frames))

        for ch, ch_vad in enumerate(vad):
            for s, e in ch_vad:
                if start <= s <= end:
                    rel_start = s - start
                    v_start_frame = round(rel_start / frame_size)
                    if start <= e <= end:  # vad segment completely in chunk
                        rel_end = e - start
                        v_end_frame = round(rel_end / frame_size)
                        vad_oh[ch, v_start_frame : v_end_frame + 1] = 1.0
                    else:  # only start in chunk -> fill until end
                        vad_oh[ch, v_start_frame:] = 1.0
                elif start <= e <= end:  # only end in chunk
                    rel_end = e - start
                    v_end_frame = round(rel_end / frame_size)
                    vad_oh[ch, : v_end_frame + 1] = 1.0
                elif s > end:
                    break

        # current speaker is always channel 0
        if speaker == 1:
            vad_oh = torch.stack((vad_oh[1], vad_oh[0]))

        return vad_oh

    @staticmethod
    def get_last_speaker(vad):
        def last_speaker_single(vad):
            s = VAD.vad_to_dialog_vad_states(vad)
            start, _, val = find_island_idx_len(s)

            # exlude silences (does not effect last_speaker)
            # silences should be the value of the previous speaker
            sil_idx = torch.where(val == 1)[0]
            if len(sil_idx) > 0:
                if sil_idx[0] == 0:
                    val[0] = 2  # 2 is both we don't know if its a shift or hold
                    sil_idx = sil_idx[1:]
                val[sil_idx] = val[sil_idx - 1]
            # map speaker B state (=3) to 1
            val[val == 3] = 1
            # get repetition lengths
            repeat = start[1:] - start[:-1]
            # Find difference between original and repeated
            # and use diff to repeat the last speaker until the end of segment
            diff = len(s) - repeat.sum(0)
            repeat = torch.cat((repeat, diff.unsqueeze(0)))
            # repeat values to create last speaker over entire segment
            last_speaker = torch.repeat_interleave(val, repeat)
            return last_speaker

        assert (
            vad.ndim > 1
        ), "must provide vad of size: (N, channels) or (B, N, channels)"
        # get last active speaker (for turn shift/hold)
        if vad.ndim < 3:
            last_speaker = last_speaker_single(vad)
        else:  # (B, N, Channels) = (B, N, n_speakers)
            last_speaker = []
            for batch_vad in vad:
                last_speaker.append(last_speaker_single(batch_vad))
            last_speaker = torch.stack(last_speaker)
        return last_speaker

    @staticmethod
    def get_next_speaker(vad):
        """Doing `get_next_speaker` in reverse"""
        # Reverse Vad
        vad_reversed = vad.flip(dims=(1,))
        # get "last speaker"
        next_speaker = VAD.get_last_speaker(vad_reversed)
        # reverse back
        next_speaker = next_speaker.flip(dims=(1,))
        return next_speaker

    @staticmethod
    def get_hold_shift_onehot(vad):
        prev_speaker = VAD.get_last_speaker(vad)
        next_speaker = VAD.get_next_speaker(vad)
        silence_ids = torch.where(vad.sum(-1) == 0)

        hold_one_hot = torch.zeros_like(prev_speaker)
        shift_one_hot = torch.zeros_like(prev_speaker)

        hold = prev_speaker[silence_ids] == next_speaker[silence_ids]
        hold_one_hot[silence_ids] = hold.long()
        shift_one_hot[silence_ids] = torch.logical_not(hold).long()
        return hold_one_hot, shift_one_hot

    # vad context history
    @staticmethod
    def get_vad_condensed_history(vad, t, speaker, bin_end_times=[60, 30, 15, 5, 0]):
        """
        get the vad-condensed-history over the history of the dialog.

        the amount of active seconds are calculated for each speaker in the segments defined by `bin_end_times`
        (starting from 0).
        The idea is to represent the past further away from the current moment in time more granularly.

        for example:
            bin_end_times=[60, 30, 10, 5, 0] extracts activity for each speaker in the intervals:

                [-inf, t-60]
                [t-60, t-30]
                [t-30, t-10]
                [t-10, t-5]
                [t-50, t]

            The final representation is then the ratio of activity for the
            relevant `speaker` over the total activity, for each bin. if there
            is no activity, that is the segments span before the dialog started
            or (unusually) both are silent, then we set the ratio to 0.5, to
            indicate equal participation.

        Argument:
            - vad:      list: [[(0, 3), (4, 6), ...], [...]] list of list of channel start and end time
        """
        n_bins = len(bin_end_times)
        T = t - torch.tensor(bin_end_times)
        bin_times = [0] + T.tolist()

        bins = torch.zeros(2, n_bins)
        for ch, ch_vad in enumerate(vad):  # iterate over each channel
            s = bin_times[0]
            for i, e in enumerate(bin_times[1:]):  # iterate over bin segments
                if e < 0:  # skip if before dialog start
                    s = e  # update
                    continue
                for vs, ve in ch_vad:  # iterate over channel VAD
                    if vs >= s:  # start inside bin time
                        if vs < e and ve <= e:  # both vad_start/end occurs in segment
                            bins[ch][i] += ve - vs
                        elif vs < e:  # only start occurs in segment
                            bins[ch][i] += e - vs
                    elif (
                        vs > e
                    ):  # all starts occus after bin-end -> no need to process further
                        break
                    else:  # vs is before segment
                        if s <= ve <= e:  # ending occurs in segment
                            bins[ch][i] += ve - s
                # update bin start
                s = e
        # Avoid nan -> for loop
        # get the ratio of the relevant speaker
        # if there is no information (bins are before dialog start) we use an equal prior (=.5)
        ratios = torch.zeros(n_bins)
        for b in range(n_bins):
            binsum = bins[:, b].sum()
            if binsum > 0:
                ratios[b] = bins[speaker, b] / binsum
            else:
                ratios[b] = 0.5  # equal prior for segments with no one speaking
        return ratios

    @staticmethod
    @torch.no_grad()
    def get_activity_history(vad_frames, bin_end_frames, channel_last=True):
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


class DialogEvents:
    @staticmethod
    def mutual_silences(vad):
        ds = VAD.vad_to_dialog_vad_states(vad)
        return ds == 1

    @staticmethod
    def single_speaker(vad):
        ds = VAD.vad_to_dialog_vad_states(vad)
        return torch.logical_or(ds == 0, ds == 3)

    @staticmethod
    def fill_pauses(vad, prev_speaker, next_speaker, ds):
        fill_hold = vad.clone()
        silence = ds == 1
        same_next_prev = prev_speaker == next_speaker
        holds_oh = torch.logical_and(silence, same_next_prev)
        for speaker in [0, 1]:
            fill_oh = torch.logical_and(holds_oh, next_speaker == speaker)
            fill = torch.where(fill_oh)
            fill_hold[(*fill, [speaker] * len(fill[0]))] = 1
        return fill_hold

    @staticmethod
    def find_valid_silences(
        vad, horizon=150, min_context=0, min_duration=0, start_pad=0, target_frames=-1
    ):
        max_frames = vad.shape[1] - horizon

        # Fill pauses where appropriate
        ###############################################
        prev_speaker = VAD.get_last_speaker(vad)
        next_speaker = VAD.get_next_speaker(vad)
        ds = VAD.vad_to_dialog_vad_states(vad)

        ###############################################
        fill_hold = DialogEvents.fill_pauses(vad, prev_speaker, next_speaker, ds)
        # ds = ds.cpu()

        ###############################################
        valid = torch.zeros(vad.shape[:-1], device=vad.device)
        for nb in range(ds.shape[0]):
            s, d, v = find_island_idx_len(ds[nb])

            if v[-1] == 1:
                # if segment ends in mutual silence we can't
                # lookahead what happens after
                # thus we omit the last entry
                s = s[:-1]
                d = d[:-1]
                v = v[:-1]

            if len(s) < 1:
                continue

            sil = torch.where(v == 1)[0]
            sil_start = s[sil]
            sil_dur = d[sil]
            after_sil = s[sil + 1]
            for ii, start in enumerate(after_sil):
                if start <= min_context:
                    continue
                if sil_start[ii] <= min_context:
                    continue
                if start >= max_frames:
                    break

                total_activity_window = fill_hold[nb, start : start + horizon].sum(
                    dim=0
                )
                # a single channel has no activity
                if (total_activity_window == 0).sum() == 1:
                    if sil_dur[ii] < min_duration:
                        continue

                    vs = sil_start[ii]
                    vs += start_pad  # pad to get silence away from last activity
                    end = vs + sil_dur[ii]
                    if target_frames < 0:
                        ve = end
                    else:
                        ve = vs + target_frames
                        if ve > end:
                            continue
                    valid[nb, vs:ve] = 1
        return valid

    @staticmethod
    def find_hold_shifts(vad):
        prev_speaker = VAD.get_last_speaker(vad)
        next_speaker = VAD.get_next_speaker(vad)
        silence = DialogEvents.mutual_silences(vad)

        ab = torch.logical_and(prev_speaker == 0, next_speaker == 1)
        ab = torch.logical_and(ab, silence)
        ba = torch.logical_and(prev_speaker == 1, next_speaker == 0)
        ba = torch.logical_and(ba, silence)
        aa = torch.logical_and(prev_speaker == 0, next_speaker == 0)
        aa = torch.logical_and(aa, silence)
        bb = torch.logical_and(prev_speaker == 1, next_speaker == 1)
        bb = torch.logical_and(bb, silence)

        # we order by NEXT Speaker
        shifts = torch.stack((ba, ab), dim=-1)
        holds = torch.stack((aa, bb), dim=-1)
        return holds, shifts


class ProjectionCodebook(nn.Module):
    def __init__(
        self, bin_times=[0.20, 0.40, 0.60, 0.80], frame_hz=100, threshold_ratio=0.5
    ):
        super().__init__()
        self.frame_hz = frame_hz
        self.bin_sizes = self.time_to_frames(bin_times, frame_hz)
        self.n_bins = len(self.bin_sizes) * 2
        self.n_classes = 2 ** self.n_bins
        self.horizon = sum(self.bin_sizes)
        self.threshold_ratio = threshold_ratio

        self.codebook = self.init_codebook()
        self.requires_grad_(False)

    def time_to_frames(self, time, frame_hz) -> Union[List, int]:
        if isinstance(time, list):
            time = torch.tensor(time)

        frame = time * frame_hz

        if isinstance(frame, torch.Tensor):
            frame = frame.long().tolist()
        else:
            frame = int(frame)

        return frame

    def init_codebook(self) -> nn.Module:
        """
        Initializes the codebook for the vad-projection horizon labels.

        Map all vectors of binary digits of length `n_bins` to their corresponding decimal value.

        This allows a VAD future of shape (*, 4, 2) to be flatten to (*, 8) and mapped to a number
        corresponding to the class index.
        """

        def single_idx_to_onehot(idx, d=8):
            assert idx < 2 ** d, "must be possible with {d} binary digits"
            z = torch.zeros(d)
            b = bin(idx).replace("0b", "")
            for i, v in enumerate(b[::-1]):
                z[i] = float(v)
            return z

        def create_code_vectors(n_bins):
            """
            Create a matrix of all one-hot encodings representing a binary sequence of `self.n_bins` places
            Useful for usage in `nn.Embedding` like module.
            """
            n_codes = 2 ** n_bins
            embs = torch.zeros((n_codes, n_bins))
            for i in range(2 ** n_bins):
                embs[i] = single_idx_to_onehot(i, d=n_bins)
            return embs

        codebook = nn.Embedding(
            num_embeddings=self.n_classes, embedding_dim=self.n_bins
        )
        codebook.weight.data = create_code_vectors(self.n_bins)
        codebook.weight.requires_grad_(False)
        return codebook

    def horizon_to_onehot(self, vad_projections):
        """
        Iterate over the bin boundaries and sum the activity
        for each channel/speaker.
        divide by the number of frames to get activity ratio.
        If ratio is greater than or equal to the threshold_ratio
        the bin is considered active
        """
        start = 0
        v_bins = []
        for b in self.bin_sizes:
            end = start + b
            m = vad_projections[..., start:end].sum(dim=-1) / b
            m = (m >= self.threshold_ratio).float()
            v_bins.append(m)
            start = end
        v_bins = torch.stack(v_bins, dim=-1)  # (*, t, c, n_bins)
        # Treat the 2-channel activity as a single binary sequence
        v_bins = v_bins.flatten(-2)  # (*, t, c, n_bins) -> (*, t, (c n_bins))
        return rearrange(v_bins, "... (c d) -> ... c d", c=2)

    def vad_to_label_oh(self, vad) -> torch.Tensor:
        """
        Given a sequence of binary VAD information (two channels) we extract a prediction horizon
        (frame length = the sum of all bin_sizes).

        ! WARNING ! VAD is expected to be shifted one step to get the 'next frame horizon'

        ```python
        # vad: (B, N, 2)
        # DO THIS
        vad_label_idx = VadProjection.vad_to_idx(vad[:, 1:])
        ```

        Arguments:
            vad:        torch.Tensor, (b, n, c) or (n, c)
        """
        # (b, n, c) -> (b, N, c, M), M=horizon window size, N=valid frames
        vad_projections = vad.unfold(dimension=-2, size=sum(self.bin_sizes), step=1)

        # (b, N, c, M) -> (B, N, 2, len(self.bin_sizes))
        v_bins = self.horizon_to_onehot(vad_projections)
        return v_bins

    def vad_to_label_idx(self, vad) -> torch.Tensor:
        """
        Given a sequence of binary VAD information (two channels) we extract a prediction horizon
        (frame length = the sum of all bin_sizes).

        ! WARNING ! VAD is shifted one step to get the 'next frame horizon'

        ```python
        # vad: (B, N, 2)
        # DONT DO THIS
        vad_label_idx = VadProjection.vad_to_idx(vad[:, 1:])
        ```

        Arguments:
            vad:        torch.Tensor, (b, n, c) or (n, c)

        Returns:
            classes:    torch.Tensor (b, t) or (t,)
        """
        v_bins = self.vad_to_label_oh(vad)
        return self.onehot_to_idx(v_bins)

    def onehot_to_idx(self, x) -> torch.Tensor:
        """
        The inverse of the 'forward' function.

        Arguments:
            x:          torch.Tensor (*, 2, 4)

        Inspiration for distance calculation:
            https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/vector_quantize_pytorch.py
        """
        assert x.shape[-2:] == (2, self.n_bins // 2)

        # compare with codebook and get closest idx
        shape = x.shape
        flatten = rearrange(x, "... c bpp -> (...) (c bpp)", c=2, bpp=self.n_bins // 2)
        embed = self.codebook.weight.t()

        dist = -(
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        embed_ind = dist.max(dim=-1).indices
        embed_ind = embed_ind.view(*shape[:-2])
        return embed_ind

    def idx_to_onehot(self, idx):
        v = self.codebook(idx)
        return rearrange(v, "... (c b) -> ... c b", c=2)

    def forward(self, idx):
        return self.idx_to_onehot(idx)


class VadProjection(ProjectionCodebook):
    def __init__(
        self,
        bin_times=[0.2, 0.4, 0.6, 0.8],
        vad_threshold=0.5,
        pred_threshold=0.5,
        event_min_context=100,
        event_min_duration=20,
        event_horizon=100,
        event_start_pad=10,
        event_target_duration=10,
        frame_hz=100,
    ):
        super().__init__(bin_times, frame_hz, vad_threshold)
        # Minimum amount of context frame in dialog-segment
        self.event_min_context = self.time_to_frames(event_min_context, frame_hz)

        # The shift/hold must be at least this many frames
        self.event_min_duration = self.time_to_frames(event_min_duration, frame_hz)

        # The future horizon which defines VALID events to measure
        self.event_horizon = self.time_to_frames(event_horizon, frame_hz)

        # Offset the start frame after last speaker is finished
        self.event_start_pad = self.time_to_frames(event_start_pad, frame_hz)

        # Minimum amount of valid frames in each dialog event
        self.event_target_duration = self.time_to_frames(
            event_target_duration, frame_hz
        )

        # indices for extracting turn-taking metrics
        self.on_silent_shift, self.on_silent_hold = self.init_on_silent_shift()
        self.on_active_shift, self.on_active_hold = self.init_on_activity_shift()

        # Shift/Hold Threshold
        # The probability of a shift/hold must be over this threshold
        # to be considered a positive prediction
        self.pred_threshold = pred_threshold

    def __repr__(self):
        s = "VadProjection\n"
        s += f"\tframe_hz: {self.frame_hz}\n"
        s += f"\tbin_sizes: {self.bin_sizes}\n"
        s += f"\tevent_min_context: {self.event_min_context}\n"
        s += f"\tevent_min_duration: {self.event_min_duration}\n"
        s += f"\tevent_horizon: {self.event_horizon}\n"
        s += f"\tevent_start_pad: {self.event_start_pad}\n"
        s += f"\tevent_target_duration: {self.event_target_duration}\n"
        s += f"\tpred_threshold: {self.pred_threshold}\n"
        return s

    ############# MONO ######################################
    def _all_permutations_mono(self, n, start=0):
        vectors = []
        for i in range(start, 2 ** n):
            i = bin(i).replace("0b", "").zfill(n)
            tmp = torch.zeros(n)
            for j, val in enumerate(i):
                tmp[j] = float(val)
            vectors.append(tmp)
        return torch.stack(vectors)

    def _end_of_segment_mono(self, n, max=3):
        """
        # 0, 0, 0, 0
        # 1, 0, 0, 0
        # 1, 1, 0, 0
        # 1, 1, 1, 0
        """
        v = torch.zeros((max + 1, n))
        for i in range(max):
            v[i + 1, : i + 1] = 1
        return v

    def _on_activity_change_mono(self, n=4, min_active=2):
        """

        Used where a single speaker is active. This vector (single speaker) represents
        the classes we use to infer that the current speaker will end their activity
        and the other take over.

        the `min_active` variable corresponds to the minimum amount of frames that must
        be active AT THE END of the projection window (for the next active speaker).
        This used to not include classes where the activity may correspond to a short backchannel.
        e.g. if only the last bin is active it may be part of just a short backchannel, if we require 2 bins to
        be active we know that the model predicts that the continuation will be at least 2 bins long and thus
        removes the ambiguouty (to some extent) about the prediction.
        """

        base = torch.zeros(n)
        # force activity at the end
        if min_active > 0:
            base[-min_active:] = 1

        # get all permutations for the remaining bins
        permutable = n - min_active
        if permutable > 0:
            perms = self._all_permutations_mono(permutable)
            base = base.repeat(perms.shape[0], 1)
            base[:, :permutable] = perms
        return base

    def _combine_speakers(self, x1, x2, mirror=False):
        if x1.ndim == 1:
            x1 = x1.unsqueeze(0)
        if x2.ndim == 1:
            x2 = x2.unsqueeze(0)
        vad = []
        for a in x1:
            for b in x2:
                vad.append(torch.stack((a, b), dim=0))

        vad = torch.stack(vad)
        if mirror:
            vad = torch.stack((vad, torch.stack((vad[:, 1], vad[:, 0]), dim=1)))
        return vad

    def _sort_idx(self, x):
        if x.ndim == 1:
            x, _ = x.sort()
        elif x.ndim == 2:
            if x.shape[0] == 2:
                a, _ = x[0].sort()
                b, _ = x[1].sort()
                x = torch.stack((a, b))
            else:
                x, _ = x[0].sort()
                x = x.unsqueeze(0)
        return x

    ############# MONO ######################################
    def init_on_silent_shift(self):
        """
        During mutual silences we wish to infer which speaker the model deems most likely.

        We focus on classes where only a single speaker is active in the projection window,
        renormalize the probabilities on this subset, and determine which speaker is the most
        likely next speaker.
        """

        n = len(self.bin_sizes)

        # active channel: At least 1 bin is active -> all permutations (all except the no-activity)
        # active = self._all_permutations_mono(n, start=1)  # at least 1 active
        # active channel: At least 1 bin is active -> all permutations (all except the no-activity)
        active = self._on_activity_change_mono(n, min_active=2)
        # non-active channel: zeros
        non_active = torch.zeros((1, active.shape[-1]))
        # combine
        shift_oh = self._combine_speakers(active, non_active, mirror=True)
        shift = self.onehot_to_idx(shift_oh)
        shift = self._sort_idx(shift)

        # symmetric, this is strictly unneccessary but done for convenience and to be similar
        # to 'get_on_activity_shift' where we actually have asymmetric classes for hold/shift
        hold = shift.flip(0)
        return shift, hold

    def init_on_activity_shift(self):
        n = len(self.bin_sizes)

        # Shift subset
        eos = self._end_of_segment_mono(n, max=2)
        nav = self._on_activity_change_mono(n, min_active=2)
        shift_oh = self._combine_speakers(nav, eos, mirror=True)
        shift = self.onehot_to_idx(shift_oh)
        shift = self._sort_idx(shift)

        # Don't shift subset
        eos = self._on_activity_change_mono(n, min_active=2)
        zero = torch.zeros((1, n))
        hold_oh = self._combine_speakers(zero, eos, mirror=True)
        hold = self.onehot_to_idx(hold_oh)
        hold = self._sort_idx(hold)
        return shift, hold

    #############################################################
    def get_marginal_probs(self, probs, pos_idx, neg_idx):
        p = []
        for next_speaker in [0, 1]:
            joint = torch.cat((pos_idx[next_speaker], neg_idx[next_speaker]), dim=-1)
            p_sum = probs[..., joint].sum(dim=-1)
            p.append(probs[..., pos_idx[next_speaker]].sum(dim=-1) / p_sum)
        return torch.stack(p, dim=-1)

    def get_silence_shift_probs(self, probs):
        return self.get_marginal_probs(probs, self.on_silent_shift, self.on_silent_hold)

    def get_active_shift_probs(self, probs):
        return self.get_marginal_probs(probs, self.on_active_shift, self.on_active_hold)

    def get_next_speaker_probs(self, probs, vad):
        sil_probs = self.get_silence_shift_probs(probs)
        act_probs = self.get_active_shift_probs(probs)

        p_a = torch.zeros_like(sil_probs[..., 0])
        p_b = torch.zeros_like(sil_probs[..., 0])

        # dialog states
        ds = VAD.vad_to_dialog_vad_states(vad)
        silence = ds == 1
        a_current = ds == 0
        b_current = ds == 3
        both = ds == 2

        # silence
        w = torch.where(silence)
        p_a[w] = sil_probs[w][..., 0]
        p_b[w] = sil_probs[w][..., 1]

        # A current speaker
        w = torch.where(a_current)
        p_b[w] = act_probs[w][..., 1]
        p_a[w] = 1 - act_probs[w][..., 1]

        # B current speaker
        w = torch.where(b_current)
        p_a[w] = act_probs[w][..., 0]
        p_b[w] = 1 - act_probs[w][..., 0]

        # Both
        w = torch.where(both)
        # Re-Normalize and compare next-active
        sum = act_probs[w][..., 0] + act_probs[w][..., 1]
        p_a[w] = act_probs[w][..., 0] / sum
        p_b[w] = act_probs[w][..., 1] / sum
        return torch.stack((p_a, p_b), dim=-1)

    def speaker_prob_to_shift(self, probs, vad):
        assert probs.ndim == 3, "Assumes probs.shape = (B, N, 2)"

        shift_probs = torch.zeros(probs.shape[:-1])

        # dialog states
        ds = VAD.vad_to_dialog_vad_states(vad)
        silence = ds == 1
        a_current = ds == 0
        b_current = ds == 3
        prev_speaker = VAD.get_last_speaker(vad)

        # A active -> B = 1 is next_speaker
        w = torch.where(a_current)
        shift_probs[w] = p_next[w][..., 1]
        # B active -> A = 0 is next_speaker
        w = torch.where(b_current)
        shift_probs[w] = p_next[w][..., 0]
        # silence and A was previous speaker -> B = 1 is next_speaker
        w = torch.where(torch.logical_and(silence, prev_speaker == 0))
        shift_probs[w] = p_next[w][..., 1]
        # silence and B was previous speaker -> A = 0 is next_speaker
        w = torch.where(torch.logical_and(silence, prev_speaker == 1))
        shift_probs[w] = p_next[w][..., 0]
        return shift_probs

    def extract_acc(self, p_next, shift, hold):
        ret = {
            "shift": {"correct": 0.0, "n": 0.0},
            "hold": {"correct": 0.0, "n": 0.0},
        }
        # shifts
        next_speaker = 0
        w = torch.where(shift[..., next_speaker])
        if len(w[0]) > 0:
            sa = (p_next[w][..., next_speaker] > self.pred_threshold).sum().item()
            ret["shift"]["correct"] += sa
            ret["shift"]["n"] += len(w[0])
        next_speaker = 1
        w = torch.where(shift[..., next_speaker])
        if len(w[0]) > 0:
            sb = (p_next[w][..., next_speaker] > self.pred_threshold).sum().item()
            ret["shift"]["correct"] += sb
            ret["shift"]["n"] += len(w[0])
        # holds
        next_speaker = 0
        w = torch.where(hold[..., next_speaker])
        if len(w[0]) > 0:
            ha = (p_next[w][..., next_speaker] > self.pred_threshold).sum().item()
            ret["hold"]["correct"] += ha
            ret["hold"]["n"] += len(w[0])
        next_speaker = 1
        w = torch.where(hold[..., next_speaker])
        if len(w[0]) > 0:
            hb = (p_next[w][..., next_speaker] > self.pred_threshold).sum().item()
            ret["hold"]["correct"] += hb
            ret["hold"]["n"] += len(w[0])
        return ret

    def forward(self, logits, vad):
        probs = logits.softmax(dim=-1)
        p_next = self.get_next_speaker_probs(probs, vad)

        ret = {}

        # TEST PLACES
        # Valid shift/hold
        valid = DialogEvents.find_valid_silences(
            vad,
            horizon=self.event_horizon,
            min_context=self.event_min_context,
            min_duration=self.event_min_duration,
            start_pad=self.event_start_pad,
            target_frames=self.event_target_duration,
        )
        hold, shift = DialogEvents.find_hold_shifts(vad)
        hold, shift = torch.logical_and(hold, valid.unsqueeze(-1)), torch.logical_and(
            shift, valid.unsqueeze(-1)
        )
        ret["event"] = {"hold": hold, "shift": shift}

        # Hold/Shift Acc
        res_shift_hold = self.extract_acc(p_next, shift, hold)
        ret.update(res_shift_hold)
        return ret


if __name__ == "__main__":
    from datasets_turntaking.utils import get_audio_info
    from datasets_turntaking.dialog_audio.dataset import DialogAudioDataset
    from datasets_turntaking.dialog_audio.dm_dialog_audio import (
        get_dialog_audio_datasets,
    )

    dset_hf = get_dialog_audio_datasets(datasets=["switchboard"], split="val")

    dset = DialogAudioDataset(
        dataset=dset_hf, type="sliding", vad_history=True, vad_hz=50
    )
    # dset = DialogAudioDataset(dataset=dset_hf, type='ipu', vad_history=True, vad_hz=50)
    print(dset)
    print("N: ", len(dset))
    idx = 299
    d = dset[idx]
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")

    # # Extract VAD frames
    # channel_last = True
    # vad_frames = VAD.vad_list_to_onehot(
    #     d["vad"],
    #     sample_rate=16000,
    #     hop_length=160,
    #     duration=duration,
    #     channel_last=channel_last,
    # )
    # print("vad_frames: ", tuple(vad_frames.shape))  # (N, 2)
    #
    # bin_end_frames = [6000, 3000, 1000, 500]
    # # Extract VAD History
    # vad_history, hist = VAD.get_activity_history(
    #     vad_frames, bin_end_frames=bin_end_frames, channel_last=channel_last
    # )
    # print("vad_history: ", tuple(vad_history.shape))

    # Extract VAD prediction labels
    # codebook_vad = VadProjection(n_bins=8)
    # vad_labels = codebook_vad.vad_to_idx(vad_frames[1:])
    # vad_labels_oh = codebook_vad(vad_labels)

    vad_projection = VadProjection(
        bin_times=[0.2, 0.4, 0.6, 0.8],
        vad_threshold=0.5,
        pred_threshold=0.5,
        event_min_context=1.0,
        event_min_duration=0.15,
        event_horizon=1.0,
        event_start_pad=0.05,
        event_target_duration=0.10,
        frame_hz=100,
    )
