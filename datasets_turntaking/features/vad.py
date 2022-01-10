from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import F1
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


class VadProjection:
    def __init__(self, n_bins, bin_sizes=[20, 40, 60, 80], threshold_ratio=0.5):
        super().__init__()
        assert n_bins % 2 == 0, "Must be divisble by two (number of speakers)"
        assert len(bin_sizes) * 2 == n_bins
        self.n_bins = n_bins  # the total number of bins n_speaker * bins_per_speaker
        self.n_classes = 2 ** n_bins
        self.bin_sizes = bin_sizes
        self.threshold_ratio = threshold_ratio

        # Onehot-representation vectors
        self.codebook = self.init_codebook()

        # next speaker: ns
        self.next_speaker_emb, self.ns2idx = self.init_first_speaker_mapping()

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

    def next_speaker_from_vad_oh(self, x):
        """
        Calculates the next speaker in the label.
        0: speaker 0
        1: speaker 1
        2: equal (both active at same time or no activity)
        Args:
            x:  torch.Tensor: (2, n_bins)
        """

        def single(x):
            first = 2
            for i in range(x.shape[-1]):
                tmp_vad = x[:, i]
                if tmp_vad.sum() == 2:
                    first = 2
                    break
                elif tmp_vad[0] > 0:
                    first = 0
                    break
                elif tmp_vad[1] > 0:
                    first = 1
                    break
            return first

        if x.ndim == 3:  # (N, 2, window)
            first = []
            for xxx in x:
                first.append(single(xxx))
            first = torch.stack(first)
        elif x.ndim == 4:  # (B, N, 2, window)
            first = []
            for batch_x in x:
                tmp_first = []
                for seq_x in batch_x:
                    tmp_first.append(single(seq_x))
                first.append(torch.tensor(tmp_first))
            first = torch.stack(first)
        else:  # (2, window)
            first = single(x)
        return first

    def init_first_speaker_mapping(self) -> Tuple[nn.Module, Dict[int, torch.Tensor]]:
        """
        Map all classes and corresponding one-hot representation to a small set of labels
        which encodes which speaker the first non-zero activity belongs to.

        Used in order to take turns based on future window prediction and whether it would
        be considered a Shift or a Hold.
        """

        # 0:A, 1:B, 2:equal
        ns2idx = {0: [], 1: [], 2: []}  # next-speaker 2 index
        next_speaker_emb = nn.Embedding(num_embeddings=self.n_classes, embedding_dim=1)

        idx = torch.arange(self.n_classes)
        vad_labels_oh = self(idx)
        for i, vl in enumerate(vad_labels_oh):
            n = self.next_speaker_from_vad_oh(vl)
            ns2idx[n].append(i)
            next_speaker_emb.weight.data[i] = n

        # List -> tensors
        for i, v in ns2idx.items():
            ns2idx[i] = torch.tensor(v)

        next_speaker_emb.weight.requires_grad_(False)
        return next_speaker_emb, ns2idx

    def vad_to_projection_window(self, vad):
        if vad.shape[-2:] == (2, len(self.bin_sizes) * 2):
            print("vad: ", tuple(vad.shape))
            vad = rearrange(vad, "... c n -> ... n c")
            print("vad: ", tuple(vad.shape))

        # extract the horizon, h, segments.
        # v: (b, t, c, h) or (n, t, h)
        v = vad.unfold(dimension=-2, size=sum(self.bin_sizes), step=1)
        return v

    def vad_to_idx(self, vad) -> torch.LongTensor:
        """
        Given a sequence of binary VAD information (two channels) we extract a prediction horizon
        (frame length = the sum of all bin_sizes).

        ! WARNING ! VAD should be shifted one step to get the 'next frame horizon'

        ```python
        # vad: (B, N, 2)
        vad_label_idx = VadProjection.vad_to_idx(vad[:, 1:])
        ```

        Arguments:
            vad:        torch.Tensor, (b, n, c) or (n, c)

        Returns:
            classes:    torch.Tensor (b, t) or (t,)
        """
        vad_projections = self.vad_to_projection_window(vad)

        # Iterate over the bin boundaries and sum the activity
        # for each channel/speaker.
        # divide by the number of frames to get activity ratio.
        # If ratio is greater than or equal to the threshold_ratio
        # the bin is considered active
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

        # How to map d binary digits to classes?
        # We treat them as a binary sequences and their
        # Decimal number is the corresponding class index

        # Extract the decimal value according to position
        # (the order does not matter and we may start fram small->large)
        # which is our "query" vector.
        # i.e.  [1, 2, 4, 8, 16, ...]
        d = v_bins.shape[-1]
        q = torch.tensor([2.0 ** i for i in range(d)])

        # We calculate the dot product (multiply and sum)
        # given the binary values in `v_bins` and the query vector
        # which result in the Decimal value and class index
        if v_bins.ndim == 2:  # single vad
            binary_val = torch.einsum("t d, d -> t", v_bins, q)
        else:  # batched vad
            binary_val = torch.einsum("b t d, d -> b t", v_bins, q)
        return binary_val.long()

    def first_speaker_probs(self, logits, probs=None) -> torch.Tensor:
        if probs is None:
            probs = logits.softmax(dim=-1)
        a_idx = self.ns2idx[0]
        b_idx = self.ns2idx[1]
        c_idx = self.ns2idx[2]
        a_probs = probs[..., a_idx].sum(dim=-1)
        b_probs = probs[..., b_idx].sum(dim=-1)
        c_probs = probs[..., c_idx].sum(dim=-1)
        return torch.stack([a_probs, b_probs, c_probs], dim=-1)

    def get_hold_shift_probs(
        self, logits=None, vad=None, probs=None
    ) -> Dict[str, torch.Tensor]:
        """
        Extracts the speaker probs organized into Hold/Shifts.

        From the prediction side we only need to know who the previous speaker
        was and, for holds, map the probability associated with that speaker being
        the same as the predicted. i.e.

        Hold:
            if last speaker was 0 then the probabilities associated with 0 being
            the next speaker is a HOLD prediction.
            And the opposite is true for Shift.
        """
        speaker_probs = self.first_speaker_probs(logits, probs)
        last_speaker = VAD.get_last_speaker(vad)

        hold = torch.zeros_like(speaker_probs[..., 0])
        shift = torch.zeros_like(hold)

        a_is_previous_speaker = last_speaker == 0
        b_is_previous_speaker = last_speaker == 1

        hold[a_is_previous_speaker] = speaker_probs[..., 0][a_is_previous_speaker]
        hold[b_is_previous_speaker] = speaker_probs[..., 1][b_is_previous_speaker]

        shift[a_is_previous_speaker] = speaker_probs[..., 1][a_is_previous_speaker]
        shift[b_is_previous_speaker] = speaker_probs[..., 0][b_is_previous_speaker]
        return {
            "hold": hold,
            "shift": shift,
            "speaker": speaker_probs,
            "last_speaker": last_speaker,
        }

    def get_next_speaker(self, idx) -> torch.Tensor:
        self.next_speaker_emb.to(idx.device)
        return self.next_speaker_emb(idx).squeeze(-1).long()

    @torch.no_grad()
    def onehot_to_idx(self, x) -> torch.Tensor:
        """
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

    @torch.no_grad()
    def __call__(self, idx) -> torch.Tensor:
        self.codebook.to(idx.device)
        vector_1d = self.codebook(idx)
        return vector_1d.view(
            (*vector_1d.shape[:-1], 2, self.n_bins // 2)
        )  # (..., 2, n_bins)

    # Metrics & Evaluation
    def get_topk_acc(self, topk_idx, label):
        if topk_idx.ndim > label.ndim:
            label = label.unsqueeze(-1)
        K = topk_idx.shape[-1]
        correct = (topk_idx == label).float()
        acc = []
        for i in range(1, K + 1):
            s = (correct[..., :i].sum(dim=-1) > 0).float().mean()
            acc.append(s)
        acc = torch.stack(acc)
        return acc, label.nelement()

    def average_prob(self, probs, where_onehot):
        if where_onehot.sum() == 0:
            return None, None
        ids = torch.where(where_onehot)
        return probs[ids].mean().cpu(), where_onehot.sum()

    def topk_acc_specific_frames(self, topk_ns, label_ns, where_onehot):
        """
        separate hold/shift topk accuracy using the `next_speaker` labels and predictions.

        If the model predicts the correct next-speaker for the `HOLD` frames (`where_onehot`) then
        the hold guess is correct. Symmetrically this is true for shifts as well.
        """

        # Check if relevant segments exists
        n = where_onehot.sum()
        if n == 0:
            return None, None

        # k provided
        K = topk_ns.shape[-1]

        # where are frames for shift/hold
        ids = torch.where(where_onehot)
        y = label_ns[ids]  # next_speaker labels
        y_top = topk_ns[ids]  # predicted next speaker topk
        correct_ns = y.unsqueeze(-1) == y_top  # compare

        # Loop over the k to find if the model prediction is correct
        # in prediction 0 -> k
        # If the correct speaker is in the topk then we get a correct prediction
        # for that given k
        topk_acc = []
        for i in range(1, K + 1):
            s = (correct_ns[..., :i].sum(dim=-1) > 0).float().mean()
            topk_acc.append(s)
        return torch.stack(topk_acc), n

    def get_turn_labels_and_predictions(
        self, topk_ns, label_ns, hold_one_hot, shift_one_hot
    ):
        """
        From 'next speaker' prediction/labels we extract the hold/shift predictions based
        on `hold_one_hot`/`shift_one_hot`
        """
        turn_label = []
        pred = []
        if hold_one_hot.sum() > 0:
            ids = torch.where(hold_one_hot)
            hold_lab = label_ns[ids]
            hold_pred = topk_ns[ids]
            # all correct preds (for hold) is set to 0
            # and incorrect to 1.
            hold_pred = (hold_pred != hold_lab.unsqueeze(-1)).float()
            pred.append(hold_pred)
            # Hold -> class = 0
            turn_label.append(torch.zeros(hold_one_hot.sum(), dtype=torch.long))

        if shift_one_hot.sum() > 0:
            ids = torch.where(shift_one_hot)
            shift_lab = label_ns[ids]
            shift_pred = topk_ns[ids]
            # all correct preds (for shift) is set to 1
            # and incorrect to 0.
            shift_pred = (shift_pred == shift_lab.unsqueeze(-1)).float()
            pred.append(shift_pred)
            # Shift -> class = 1
            turn_label.append(torch.ones(shift_one_hot.sum(), dtype=torch.long))

        if len(pred) == 0:
            return None, None

        pred = torch.cat(pred)
        turn_label = torch.cat(turn_label).long()
        return pred, turn_label

    def prepare_class_metrics(self, out, batch, min_context_frames=0, k=5, cpu=False):
        vad = batch["vad"]
        vad_label = batch["vad_label"]
        probs_vp = out["logits_vp"].softmax(dim=-1)

        topk_probs, topk_idx = probs_vp.topk(k)

        hold_one_hot, shift_one_hot = VAD.get_hold_shift_onehot(vad)
        topk_ns = self.get_next_speaker(topk_idx)
        label_ns = self.get_next_speaker(vad_label)
        turn_probs = self.get_hold_shift_probs(logits=out["logits_vp"], vad=vad)

        ######################################################################
        if min_context_frames > 0:
            topk_idx = topk_idx[:, min_context_frames:]
            label_ns = label_ns[:, min_context_frames:]
            topk_ns = topk_ns[:, min_context_frames:]
            topk_probs = topk_probs[:, min_context_frames:]
            # Hold/Shift
            hold_one_hot = hold_one_hot[:, min_context_frames:]
            shift_one_hot = shift_one_hot[:, min_context_frames:]
            turn_probs["shift"] = turn_probs["shift"][:, min_context_frames:]
            turn_probs["hold"] = turn_probs["hold"][:, min_context_frames:]
            turn_probs["speaker"] = turn_probs["speaker"][:, min_context_frames:]
            # VAD
            vad_label = vad_label[:, min_context_frames:]
            vad = vad[:, min_context_frames:]

        if cpu:
            topk_idx = topk_idx.cpu()
            label_ns = label_ns.cpu()
            topk_ns = topk_ns.cpu()
            topk_probs = topk_probs.cpu()
            # Hold/Shift
            hold_one_hot = hold_one_hot.cpu()
            shift_one_hot = shift_one_hot.cpu()
            turn_probs["shift"] = turn_probs["shift"].cpu()
            turn_probs["hold"] = turn_probs["hold"].cpu()
            turn_probs["speaker"] = turn_probs["speaker"].cpu()
            # VAD
            vad_label = vad_label.cpu()
            vad = vad.cpu()

        ######################################################################
        # find "nucleus" subset
        # p = 0.8
        # tp = topk_probs.cumsum(dim=-1)
        # tp = tp[tp <= 0.8]
        # tpk = topk_idx[: len(tp)]
        # top_p_speaker_probs = self.get_hold_shift_probs(vad=vad, probs=tp)

        ######################################################################
        silence_onehot = (vad.sum(dim=-1) == 0).float()

        # TopK label classification
        class_topk, n = self.get_topk_acc(topk_idx, vad_label)
        class_sil_topk, class_sil_n = self.topk_acc_specific_frames(
            topk_idx, vad_label, silence_onehot
        )

        # Next Speaker topK
        ns_topk, n = self.get_topk_acc(topk_ns, label_ns)
        ns_sil_topk, ns_sil_n = self.topk_acc_specific_frames(
            topk_ns, label_ns, silence_onehot
        )

        ######################################################################
        # TURN: Shift/Hold
        # Average probability of shift/hold during shift/hold frames
        hold_prob, n_hold = self.average_prob(turn_probs["hold"], hold_one_hot)
        shift_prob, n_shift = self.average_prob(turn_probs["shift"], shift_one_hot)
        # n_hold = hold_one_hot.sum()
        # n_shift = shift_one_hot.sum()

        # TopK hold/shift acc
        hold_topk, _ = self.topk_acc_specific_frames(topk_ns, label_ns, hold_one_hot)
        shift_topk, _ = self.topk_acc_specific_frames(topk_ns, label_ns, shift_one_hot)

        # Prediction/Label classes for F1 hold/shift metric
        turn_pred, turn_label = self.get_turn_labels_and_predictions(
            topk_ns=topk_ns,
            label_ns=label_ns,
            hold_one_hot=hold_one_hot,
            shift_one_hot=shift_one_hot,
        )
        return {
            "class": {"topk": class_topk, "n": n},
            "class_silence": {"topk": class_sil_topk, "n": class_sil_n},
            "next_speaker": {"topk": ns_topk, "n": n},
            "next_speaker_silence": {"topk": ns_sil_topk, "n": ns_sil_n},
            "hold": {"topk": hold_topk, "prob": hold_prob, "n": n_hold},
            "shift": {"topk": shift_topk, "prob": shift_prob, "n": n_shift},
            "turn": {"prediction": turn_pred, "label": turn_label},
        }

    # Metrics & Evaluation
    def prepare_regression_metrics(
        self, out, batch, min_context_frames=0, k=5, cpu=False
    ):
        vad = batch["vad"]
        vad_label = batch["vad_label"]
        vp = out["logits_vp"].sigmoid().round()
        topk_idx = self.onehot_to_idx(vp).unsqueeze(-1)
        # topk_probs, topk_idx = probs_vp.topk(k)

        hold_one_hot, shift_one_hot = VAD.get_hold_shift_onehot(vad)
        topk_ns = self.get_next_speaker(topk_idx)
        label_ns = self.get_next_speaker(vad_label)

        ######################################################################
        if min_context_frames > 0:
            topk_idx = topk_idx[:, min_context_frames:]
            label_ns = label_ns[:, min_context_frames:]
            topk_ns = topk_ns[:, min_context_frames:]
            # Hold/Shift
            hold_one_hot = hold_one_hot[:, min_context_frames:]
            shift_one_hot = shift_one_hot[:, min_context_frames:]
            # VAD
            vad_label = vad_label[:, min_context_frames:]
            vad = vad[:, min_context_frames:]

        if cpu:
            topk_idx = topk_idx.cpu()
            label_ns = label_ns.cpu()
            topk_ns = topk_ns.cpu()
            # Hold/Shift
            hold_one_hot = hold_one_hot.cpu()
            shift_one_hot = shift_one_hot.cpu()
            # VAD
            vad_label = vad_label.cpu()
            vad = vad.cpu()

        ######################################################################
        # find "nucleus" subset
        # p = 0.8
        # tp = topk_probs.cumsum(dim=-1)
        # tp = tp[tp <= 0.8]
        # tpk = topk_idx[: len(tp)]
        # top_p_speaker_probs = self.get_hold_shift_probs(vad=vad, probs=tp)

        ######################################################################
        silence_onehot = (vad.sum(dim=-1) == 0).float()

        # TopK label classification
        class_topk, n = self.get_topk_acc(topk_idx, vad_label)
        class_sil_topk, class_sil_n = self.topk_acc_specific_frames(
            topk_idx, vad_label, silence_onehot
        )

        # Next Speaker topK
        ns_topk, n = self.get_topk_acc(topk_ns, label_ns)
        ns_sil_topk, ns_sil_n = self.topk_acc_specific_frames(
            topk_ns, label_ns, silence_onehot
        )

        ######################################################################
        # TURN: Shift/Hold
        # Average probability of shift/hold during shift/hold frames
        n_hold = hold_one_hot.sum()
        n_shift = shift_one_hot.sum()

        # TopK hold/shift acc
        hold_topk, _ = self.topk_acc_specific_frames(topk_ns, label_ns, hold_one_hot)
        shift_topk, _ = self.topk_acc_specific_frames(topk_ns, label_ns, shift_one_hot)

        # Prediction/Label classes for F1 hold/shift metric
        turn_pred, turn_label = self.get_turn_labels_and_predictions(
            topk_ns=topk_ns,
            label_ns=label_ns,
            hold_one_hot=hold_one_hot,
            shift_one_hot=shift_one_hot,
        )
        return {
            "class": {"topk": class_topk, "n": n},
            "class_silence": {"topk": class_sil_topk, "n": class_sil_n},
            "next_speaker": {"topk": ns_topk, "n": n},
            "next_speaker_silence": {"topk": ns_sil_topk, "n": ns_sil_n},
            "hold": {"topk": hold_topk, "n": n_hold},
            "shift": {"topk": shift_topk, "n": n_shift},
            "turn": {"prediction": turn_pred, "label": turn_label},
        }

    def prepare_metrics(self, out, batch, min_context_frames=0, k=5, cpu=False):
        if out["logits_vp"].ndim == 4:
            return self.prepare_regression_metrics(
                out, batch, min_context_frames, k=k, cpu=cpu
            )
        else:
            return self.prepare_class_metrics(out, batch, min_context_frames, k, cpu)


if __name__ == "__main__":
    from datasets_turntaking.dm_dialog_audio import get_dialog_audio_datasets
    from datasets_turntaking.utils import get_audio_info

    # dloader = quick_load_dataloader()
    # batch = next(iter(dloader))
    # vad = batch["vad"]
    # vad_labels = batch["vad_label"]
    # vad_labels_oh = codebook_vad(vad_labels)
    # print("vad_labels: ", tuple(vad_labels.shape), vad_labels.dtype)
    # print("vad_labels_oh: ", tuple(vad_labels_oh.shape), vad_labels_oh.dtype)

    # Load dataset
    dset_hf = get_dialog_audio_datasets(datasets=["switchboard"], split="val")
    d = dset_hf[0]
    print("d: ", d.keys())
    vad = d["vad"]
    info = get_audio_info(d["audio_path"])
    duration = info["duration"]

    # Extract VAD frames
    channel_last = True
    vad_frames = VAD.vad_list_to_onehot(
        d["vad"],
        sample_rate=16000,
        hop_length=160,
        duration=duration,
        channel_last=channel_last,
    )
    print("vad_frames: ", tuple(vad_frames.shape))  # (N, 2)
    bin_end_frames = [6000, 3000, 1000, 500]
    # Extract VAD History
    vad_history, hist = VAD.get_activity_history(
        vad_frames, bin_end_frames=bin_end_frames, channel_last=channel_last
    )
    print("vad_history: ", tuple(vad_history.shape))

    # Extract VAD prediction labels
    codebook_vad = VadProjection(n_bins=8)
    vad_labels = codebook_vad.vad_to_idx(vad_frames[1:])
    vad_labels_oh = codebook_vad(vad_labels)
