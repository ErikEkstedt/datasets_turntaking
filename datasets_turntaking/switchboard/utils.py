import re
from os import walk
from os.path import join, basename
from copy import deepcopy

from datasets_turntaking.features.vad import VAD, frame2time
from datasets_turntaking.utils import (
    find_island_idx_len,
    read_json,
    read_txt,
    repo_root,
)

REL_AUDIO_PATH = join(
    repo_root(), "datasets_turntaking/switchboard/files/relative_audio_path.json"
)

OmitText = [
    "[silence]",
    "[noise]",
    "[vocalized-noise]",
]

BACKCHANNELS = [
    "huh-uh",
    "hum-um",
    "um",
    "oh really",
    "oh uh-huh",
    "oh yeah",
    "oh",
    "right",
    "uh-huh uh-huh",
    "uh-huh yeah",
    "um-hum um-hum",
    "uh-huh",
    "uh-hum",
    "uh-hums",
    "uh-oh",
    "um-hum",
    "yeah yeah",
    "yeah",
]

BACKCHANNEL_MAP = {
    "uh-huh": "uhuh",
    "huh-uh": "uhuh",
    "uh-hum": "mhm",
    "uh-hums": "mhm",
    "um-hum": "mhm",
    "hum-um": "mhm",
    "uh-oh": "uhoh",
}


def swb_regexp(s):
    """
    Switchboard annotation specific regexp.

    See:
        - `datasets_turntaking/features/dataset/switchboard.md`
        - https://www.isip.piconepress.com/projects/switchboard/doc/transcription_guidelines/transcription_guidelines.pdf

    """
    # Noise
    s = re.sub(r"\[noise\]", "", s)
    s = re.sub(r"\[vocalized-noise\]", "", s)

    # laughter
    s = re.sub(r"\[laughter\]", "", s)
    # laughing and speech e.g. [laughter-yeah] -> yeah
    s = re.sub(r"\[laughter-(\w*)\]", r"\1", s)
    s = re.sub(r"\[laughter-(\w*\'*\w*)\]", r"\1", s)

    # Partial words: w[ent] -> went
    s = re.sub(r"(\w+)\[(\w*\'*\w*)\]", r"\1\2", s)
    # Partial words: -[th]at -> that
    s = re.sub(r"-\[(\w*\'*\w*)\](\w+)", r"\1\2", s)

    # restarts
    s = re.sub(r"(\w+)-\s", r"\1 ", s)
    s = re.sub(r"(\w+)-$", r"\1", s)

    # Pronounciation variants
    s = re.sub(r"(\w+)\_\d", r"\1", s)

    # Mispronounciation [splace/space] -> space
    s = re.sub(r"\[\w+\/(\w+)\]", r"\1", s)

    # Coinage. remove curly brackets... keep word
    s = re.sub(r"\{(\w*)\}", r"\1", s)

    # remove double spacing on last
    s = re.sub(r"\s\s+", " ", s)
    return s.strip()  # remove whitespace start/end


def find_asides(s):
    """
    Asides: If one of the speakers involved in the conversation talks to someone in the
    background and the words can be understood, then transcribe it as an aside enclosed in
    the markers, <b_aside> and <e_aside>. This only applies if one of the conversation
    speakers is involved in the background conversation.

    Example: "yeah i know what you <b_aside> honey i can’t play with you right
    now i’m on the phone <e_aside> sorry you know kids always want
    mommy all to themselves"
    """
    found = False
    if re.search(r"\<b_aside\>", s) or re.search(r"\<e_aside\>", s):
        found = True
    return found


def extract_audio_mapping(audio_root):
    """
    Used to create `relative_audio_path.json`.

    The audio files requires manual download and the extracted files are
    organized in a "non-straight-forward" manner. This function maps the
    session ids to their relative audio_path.

    This should only be run once and the mapping should be "shipped" with the
    dataset. For now it is included in the git-repo but should be downloaded
    along with the transcripts.

    Then given the path to the audio-root (extracted audio files) the audio
    path is reconstructed.

    (I have changed the format from .sph -> .wav)
    ```
      AUDIO_ROOT
    ├──  swb1_d1
    │  └──  data
    |       |-- sw02285.{wav,sph}
    |       └-- ..
    ├──  swb1_d2
    │  └──  data
    |       └-- ...
    ├──  swb1_d3
    │  └──  data
    |       └-- ...
    └──  swb1_d4
       └──  data
            └-- ...
    ```

    ```python
    # Construct relative-audio-path-mappings:
    audio_root = "/Path/To/Extracted/Audio"
    map = extract_audio_mapping(audio_root)
    write_json(map, "swb_session_to_audio_map.json")
    ```

    RETURN:
        map:    dict, i.e. map['3002'] -> 'swb1_d1/data/sw03002'
    """
    map = {}
    for root, _, files in walk(audio_root):
        if len(files) > 0:
            rel_path = root.replace(audio_root, "")
            for f in files:
                if f.endswith(".wav") or f.endswith(".sph"):
                    # sw03815.{wav,sph} -> 3815
                    session = basename(f)
                    session = re.sub(r"^sw0(\w*).*", r"\1", session)
                    # sw03815.{wav,sph} ->sw03815
                    f_no_ext = re.sub(r"^(.*)\.\w*", r"\1", f)
                    map[session] = join(rel_path, f_no_ext)
    return map


def get_audio_relpath(session, map):
    """
    Given a precomputed mapping return the relative audio path
    """
    if not isinstance(session, str):
        session = str(session)
    return map[session]


class SwitchboardUtils:
    @staticmethod
    def extract_word_level_annotations(session_id, speaker, root, apply_regexp=False):
        def remove_multiple_whitespace(s):
            s = re.sub(r"\t", " ", s)
            return re.sub(r"\s\s+", " ", s)

        # Load word-level annotations
        words_filename = "sw" + session_id + speaker + "-ms98-a-word.text"
        words_list = read_txt(join(root, words_filename))

        # process word-level annotation
        word_dict = {}
        for word_row in words_list:
            word_row = remove_multiple_whitespace(word_row).strip()
            try:
                idx, wstart, wend, word = word_row.split(" ")
            except Exception as e:
                print("word_row: ", word_row)
                print("word_split: ", word_row.split(" "))
                print(e)
                input()
                assert False

            if apply_regexp:
                word = swb_regexp(word)

            if not (word in OmitText or word == ""):
                if idx in word_dict:
                    word_dict[idx].append(
                        {"text": word, "start": float(wstart), "end": float(wend)}
                    )
                else:
                    word_dict[idx] = [
                        {"text": word, "start": float(wstart), "end": float(wend)}
                    ]
        return word_dict

    @staticmethod
    def combine_speaker_utterance_and_words(
        session_id, speaker, root, apply_regexp=False, strict_times=False
    ):
        """Combines word- and utterance-level annotations"""
        # Read word-level annotation and format appropriately
        word_dict = SwitchboardUtils.extract_word_level_annotations(
            session_id, speaker, root, apply_regexp=apply_regexp
        )

        # Read utterance-level annotation
        trans_filename = "sw" + session_id + speaker + "-ms98-a-trans.text"
        trans_list = read_txt(join(root, trans_filename))

        # correct channels for wavefiles
        speaker = 0 if speaker == "A" else 1

        # Combine word-/utterance- level annotations
        utterances = []
        for row in trans_list:
            utt_idx, start, end, *words = row.split(" ")
            if not (words[0] in OmitText and len(words) == 1):  # only noise/silence
                wd = word_dict.get(utt_idx, None)
                if wd is None:
                    continue

                words = " ".join(words)
                if apply_regexp:
                    words = swb_regexp(words)

                tmp_utt = {
                    "id": utt_idx,
                    "speaker": speaker,
                    "text": words,
                    "words": wd,
                }

                # if self.config.name == "default":
                if strict_times:
                    # Otherwise use the more exact word-level times
                    # use word start/end times for utterance
                    tmp_utt["start"] = tmp_utt["words"][0]["start"]
                    tmp_utt["end"] = tmp_utt["words"][-1]["end"]
                else:
                    # By default use the utterance level timings (with padding)
                    # use annotated start/end times for utterance
                    tmp_utt["start"] = float(start)
                    tmp_utt["end"] = float(end)
                utterances.append(tmp_utt)
        return utterances

    @staticmethod
    def extract_dialog(session_id, root, raw=False):
        """Extract the annotated dialogs based on config `name`"""
        # Config settings
        apply_regexp = False
        strict_times = False
        if not raw:
            apply_regexp = True
            strict_times = True

        # Speaker A: original name in annotations
        a_utterances = SwitchboardUtils.combine_speaker_utterance_and_words(
            session_id,
            speaker="A",
            root=root,
            apply_regexp=apply_regexp,
            strict_times=strict_times,
        )

        # Speaker B: original name in annotations
        b_utterances = SwitchboardUtils.combine_speaker_utterance_and_words(
            session_id,
            speaker="B",
            root=root,
            apply_regexp=apply_regexp,
            strict_times=strict_times,
        )

        # Combine speaker utterances and sort by start-time
        dialog = a_utterances + b_utterances
        dialog.sort(key=lambda x: x["start"])
        return dialog

    @staticmethod
    def extract_vad(utterances):
        vad = [[], []]
        for utt in utterances:
            channel = utt["speaker"]
            s, e = utt["words"][0]["start"], utt["words"][0]["end"]
            for w in utt["words"][1:]:
                if w["start"] - e < 0.05:
                    e = w["end"]
                    # print('joint')
                else:
                    vad[channel].append((s, e))
                    # update
                    s = w["start"]
                    e = w["end"]
            vad[channel].append((s, e))
        return vad


class TextFocusDialog:
    RelAudioMap = read_json(REL_AUDIO_PATH)

    def __init__(self, backchannel_list=BACKCHANNELS, lookahead_duration=2):
        self.backchannel_list = backchannel_list
        self.lookahead_duration = lookahead_duration

    def extract_context_vad(self, vad, end_time):
        """Extract vad up until `end_time`."""
        vad_end = end_time + self.lookahead_duration
        cvad = [[], []]
        for ch, ch_vad in enumerate(vad):
            for start, end in ch_vad:
                if start < vad_end:
                    if end < vad_end:
                        cvad[ch].append([start, end])
                    else:
                        cvad[ch].append([start, vad_end])
        return cvad

    def is_backchannel(self, utt):
        return utt["text"] in self.backchannel_list

    def join_utterances(self, utt1, utt2):
        utt = deepcopy(utt1)
        utt["text"] += " " + utt2["text"]
        utt["words"] += utt2["words"]
        utt["end"] = utt2["end"]
        return utt

    def is_overlap_within(self, current, prev):
        start_within = prev["start"] <= current["start"] <= prev["end"]
        end_within = prev["start"] <= current["end"] <= prev["end"]
        return start_within and end_within

    def refine_dialog(self, utterances, vad=None):
        """

        Refine the dialog by omitting `overlap_within` and `backchannel`
        speech, both of which are added to the current/major utterance. Keeps
        the original fields for text, words, start, end, speaker.

        i.e:
            refined[i] = {'id', 'speaker', 'text', 'words', 'start', 'end', 'backchannel', 'within'}
        """
        first = utterances[0]
        first["backchannel"] = []
        first["within"] = []
        if vad is not None:
            first["vad"] = self.extract_context_vad(vad, first["end"])
        refined = [first]
        last_speaker = first["speaker"]
        for current in utterances[1:]:
            if self.is_backchannel(current):
                refined[-1]["backchannel"] += current["words"]
            elif self.is_overlap_within(current, refined[-1]):
                refined[-1]["within"] += current["words"]
            else:
                if current["speaker"] == last_speaker:
                    refined[-1] = self.join_utterances(refined[-1], current)
                else:
                    current["backchannel"] = []
                    current["within"] = []
                    if vad is not None:
                        current["vad"] = self.extract_context_vad(vad, current["end"])
                    refined.append(current)
                    last_speaker = current["speaker"]
        return refined

    def generate_refined_dialogs(self, filepaths):
        for filepath in filepaths:
            utterances = read_json(filepath)
            vad = SwitchboardUtils.extract_vad(utterances)
            session = basename(filepath).split("_")[0]
            audio_path = get_audio_relpath(session, self.RelAudioMap)
            refined = self.refine_dialog(utterances)  # , vad)
            sample = {
                "session": session,
                "audio_path": audio_path,
                "vad": vad,
                "dialog": refined,
            }

            yield f"{session}", sample


class Classification(TextFocusDialog):
    def __init__(
        self,
        backchannel_list=BACKCHANNELS,
        lookahead_duration=2,
        min_hold_segment_time=3,
        omit_post_words=True,
    ):
        super().__init__(
            backchannel_list=backchannel_list, lookahead_duration=lookahead_duration
        )
        self.omit_post_words = omit_post_words
        self.min_hold_segment_time = min_hold_segment_time

        # Negative samples
        self.min_dist_start = 1  # min time distance from current utterance start
        self.min_dist_end = 2  # min time distance from actual turn-shift

    def omit_post_response_words(self, sample):
        """
        only keep the words that are fully spoken before the turn-shift actually occurs.
        """
        # check if response occurs before end of context
        if sample["context"]["end"] > sample["response"]["start"]:
            words = []
            # check which words to be included in context
            # for i in range(len(sample['context']["words"])):
            for word in sample["context"]["words"]:
                if sample["response"]["start"] > word["end"]:
                    words.append(word)
                else:
                    break

            if len(words) == 0:
                return None
            num_removed = len(sample["context"]["words"]) - len(words)
            # update omitted words from turns
            sample["context"]["turns"][-1] = " ".join(
                sample["context"]["turns"][-1].split(" ")[:-num_removed]
            )
            # update words
            sample["context"]["words"] = words
            sample["context"]["end"] = words[-1]["end"]
        return sample

    def extract_hold(self, sample, vad):
        """
        Check if there is room to add a negative sample
              |-----------------I----------------|
        last_start         neg-sample      next_turn_start
               <---------------> <--------------->
                 min_dist_start     min_dist_end
        """
        hold = None
        min_neg_start = sample["context"]["start"] + self.min_dist_start
        min_neg_end = sample["response"]["start"] - self.min_dist_end
        if min_neg_start < min_neg_end:
            # Check which is the last word to be included in the 'hold' sample
            # go backwards from last word and stop when the end of a word is prior to
            # the "hold" start time
            hold_start_time = None
            for word in sample["context"]["words"][::-1]:
                if word["end"] < min_neg_end:
                    hold_start_time = word["end"] + 0.02  # add time after word
                    break

            if hold_start_time is not None:
                hold = deepcopy(sample)
                # TODO: find end time and extract vad -> 0:end_time+self.lookahead_duration
                hold["context"]["vad"] = self.extract_context_vad(vad, hold_start_time)
                hold["label"] = "hold"
                hold["response"] = {
                    "id": "hold",
                    "speaker": 0 if sample["context"]["speaker"] == 1 else 1,
                    "text": "HOLD",
                    "words": [
                        {
                            "text": "HOLD",
                            "start": -1.0,
                            "end": -1.0,
                        }
                    ],
                    "start": hold_start_time,
                    "end": hold_start_time,
                }
        return hold

    def extract_hold_times(self, vad):
        end = max(vad[0][-1][-1], vad[1][-1][-1])
        # extract longer time segments with only one speaker
        hold_times = []
        frame_time = 0.1
        vad_oh = VAD.get_current_vad_onehot(
            vad, end=end, duration=end, speaker=0, frame_size=frame_time
        )
        states = VAD.vad_to_dialog_vad_states(vad_oh)
        idx, dur, val = find_island_idx_len(states)

        # channel 0
        starts1 = frame2time(idx[val == 0], frame_time)
        dur1 = frame2time(dur[val == 0], frame_time)
        # not at start of dialog
        dur1 = dur1[starts1 > 3]
        starts1 = starts1[starts1 > 3]

        # get mid point time
        starts1 = starts1[dur1 > self.min_hold_segment_time]
        dur1 = dur1[dur1 > self.min_hold_segment_time]
        mid1 = starts1 + (dur1 / 2)
        hold_times.append(mid1)

        # channel 1
        starts2 = frame2time(idx[val == 3], frame_time)
        dur2 = frame2time(dur[val == 3], frame_time)
        # not at start of dialog
        dur2 = dur2[starts2 > 3]
        starts2 = starts2[starts2 > 3]
        starts2 = starts2[dur2 > self.min_hold_segment_time]
        dur2 = dur2[dur2 > self.min_hold_segment_time]
        # get mid point time
        mid2 = starts2 + (dur2 / 2)
        hold_times.append(mid2)
        return hold_times

    def get_hold_sample(self, i, hold_time, current, refined, vad, session, audio_path):
        """given a predefined time for a hold, the relevant turn is found `current` and we extract the sample from this"""

        sample = {
            "session": session,
            "audio_path": audio_path,
            "label": "hold",
            "context": {
                "turns": [
                    t["text"] for t in refined[:i]
                ],  # history up to and including last turn
                "words": current["words"],  # prev turn
                "start": current["start"],  # prev turn
                "end": current["end"],  # prev turn
                "speaker": current["speaker"],  # prev turn
                "vad": self.extract_context_vad(vad, hold_time),
            },
            "response": {
                "id": "hold",
                "speaker": 0 if current["speaker"] == 1 else 1,
                "text": "HOLD",
                "words": [
                    {
                        "text": "HOLD",
                        "start": hold_time,
                        "end": hold_time,
                    }
                ],
                "start": hold_time,
                "end": hold_time,
            },
        }
        if self.omit_post_words:
            sample = self.omit_post_response_words(sample)
        return sample

    def generate_classification(self, filepaths):
        key = 0
        N = {"hold": 0, "backchannel": 0, "shift": 0}
        for filepath in filepaths:
            session = basename(filepath).split("_")[0]
            utterances = read_json(filepath)
            vad = SwitchboardUtils.extract_vad(utterances)
            refined = self.refine_dialog(utterances)

            tmp_n = {"hold": 0, "backchannel": 0, "shift": 0}
            audio_path = get_audio_relpath(session, self.RelAudioMap)
            for i, current in enumerate(refined):
                if len(current["backchannel"]) > 0:
                    # vad_context = self.extract_context_vad(refined, i)
                    for bc in current["backchannel"]:
                        # find end time and extract vad -> 0:end_time+self.lookahead_duration
                        vad_context = self.extract_context_vad(
                            vad, end_time=bc["start"]
                        )
                        bc_sample = {
                            "session": session,
                            "audio_path": audio_path,
                            "label": "backchannel",
                            "context": {
                                "turns": [
                                    t["text"] for t in refined[: i + 1]
                                ],  # include current
                                "words": current["words"],
                                "start": current["start"],
                                "end": current["end"],
                                "speaker": current["speaker"],
                                "vad": vad_context,
                            },
                            "response": {
                                "id": "backchannel",
                                "speaker": 0 if current["speaker"] == 1 else 1,
                                "text": bc["text"],
                                "start": bc["start"],
                                "end": bc["end"],
                                "words": [bc],
                            },
                        }

                        if self.omit_post_words:
                            bc_sample = self.omit_post_response_words(bc_sample)

                        if bc_sample is not None:
                            # Create Backchannel sample
                            N["backchannel"] += 1
                            tmp_n["backchannel"] += 1
                            yield f"{key}_bc", bc_sample
                            key += 1

                # If not the first turn we extract a turn-shift
                if i > 0:  # check turn-shift
                    if refined[i - 1]["speaker"] != current["speaker"]:
                        response = deepcopy(current)
                        response.pop("backchannel")
                        response.pop("within")
                        # TODO: find end time and extract vad -> 0:end_time+self.lookahead_duration
                        vad_context = self.extract_context_vad(vad, response["start"])
                        shift_sample = {
                            "session": session,
                            "audio_path": audio_path,
                            "label": "shift",
                            "context": {
                                "turns": [
                                    t["text"] for t in refined[:i]
                                ],  # history up to and including last turn
                                "words": refined[i - 1]["words"],  # prev turn
                                "start": refined[i - 1]["start"],  # prev turn
                                "end": refined[i - 1]["end"],  # prev turn
                                "speaker": refined[i - 1]["speaker"],  # prev turn
                                "vad": vad_context,
                            },
                            "response": response,
                        }

                        if self.omit_post_words:
                            shift_sample = self.omit_post_response_words(
                                sample=shift_sample
                            )

                        if shift_sample is not None:
                            N["shift"] += 1
                            tmp_n["shift"] += 1
                            yield f"{key}_ts", shift_sample
                            key += 1

            #####################################################33
            # Extract holds (maximum amount that matches turn-shifts)
            hold_times = self.extract_hold_times(vad)
            n = 0
            stop = False
            for i, turn in enumerate(refined[1:], start=1):
                if turn["speaker"] == 0:
                    for t in hold_times[0]:
                        if turn["start"] < t < turn["end"]:
                            hold_sample = self.get_hold_sample(
                                i,
                                hold_time=t,
                                current=turn,
                                refined=refined,
                                vad=vad,
                                session=session,
                                audio_path=audio_path,
                            )
                            if hold_sample is not None:
                                n += 1
                                N["hold"] += 1
                                yield f"{key}_ho", hold_sample
                                key += 1
                                if n >= tmp_n["shift"]:
                                    stop = True
                                    break
                        if stop:
                            break
                else:
                    for t in hold_times[1]:
                        if turn["start"] < t < turn["end"]:
                            hold_sample = self.get_hold_sample(
                                i,
                                hold_time=t,
                                current=turn,
                                refined=refined,
                                vad=vad,
                                session=session,
                                audio_path=audio_path,
                            )
                            if hold_sample is not None:
                                n += 1
                                N["hold"] += 1
                                yield f"{key}_ho", hold_sample
                                key += 1
                                if n >= tmp_n["shift"]:
                                    break
                        if stop:
                            break
                if stop:
                    break

        print("\nExtracted Samples")
        for label, n in N.items():
            print(f"{label}: ", n)


class SegmentIPU(TextFocusDialog):
    def __init__(
        self,
        ipu_threshold=0.2,
        min_context_ipu_time=1.5,
        lookahead_duration=2,
        backchannel_list=BACKCHANNELS,
    ):
        super().__init__(
            backchannel_list=backchannel_list, lookahead_duration=lookahead_duration
        )
        self.ipu_threshold = ipu_threshold
        self.min_context_ipu_time = min_context_ipu_time

    def ipu_is_overlap_within(self, context, response):
        if response["start"] < context["end"] and response["end"] <= context["end"]:
            return True
        return False

    def get_history(self, ipus, i):
        """Extracts the IPU history. Does not include the current iput."""
        history = [ipus[0]["text"]]
        last_speaker = ipus[0]["speaker"]
        for j, ipu in enumerate(ipus[1:i], start=1):
            if ipu["text"] in self.backchannel_list:
                continue
            if self.ipu_is_overlap_within(ipus[j - 1], ipu):
                continue
            if ipu["speaker"] == last_speaker:
                history[-1] += " " + ipu["text"]
            else:
                history.append(ipu["text"])
                last_speaker = ipu["speaker"]
        join_with_turns = 0
        if ipus[i]["speaker"] == last_speaker:
            join_with_turns = 1
        return history, join_with_turns

    def get_sorted_ipus(self, utterances, ipu_threshold):
        words = [[], []]
        for utt in utterances:
            words[utt["speaker"]] += utt["words"]

        all_ipus = []
        for ch in range(2):
            ch_words = words[ch]
            last_end = ch_words[0]["end"]
            tmp_ipu = {"words": [ch_words[0]], "speaker": ch}
            for w in ch_words[1:]:
                if w["start"] - last_end <= ipu_threshold:
                    tmp_ipu["words"].append(w)
                else:
                    tmp_ipu["start"] = tmp_ipu["words"][0]["start"]
                    tmp_ipu["end"] = tmp_ipu["words"][-1]["end"]
                    tmp_ipu["text"] = " ".join([w["text"] for w in tmp_ipu["words"]])
                    all_ipus.append(tmp_ipu)
                    tmp_ipu = {"words": [w], "speaker": ch}
                last_end = w["end"]
            tmp_ipu["start"] = tmp_ipu["words"][0]["start"]
            tmp_ipu["end"] = tmp_ipu["words"][-1]["end"]
            tmp_ipu["text"] = " ".join([w["text"] for w in tmp_ipu["words"]])
            all_ipus.append(tmp_ipu)

        all_ipus.sort(key=lambda x: x["words"][0]["start"])
        return all_ipus

    def generate_ipus(self, filepaths):
        key = 0
        n_shift, n_bc, n_hold = 0, 0, 0
        for filepath in filepaths:
            session = basename(filepath).split("_")[0]
            audio_path = get_audio_relpath(session, self.RelAudioMap)
            utterances = read_json(filepath)
            vad = SwitchboardUtils.extract_vad(utterances)
            ipus = self.get_sorted_ipus(utterances, self.ipu_threshold)

            n_overlaps = 0
            for i, context in enumerate(ipus[1:-1], start=1):
                # we don't need backchannels as context only as responses
                if context["text"] in self.backchannel_list:
                    continue

                # do not include overlaps totally inside the context
                # look for the next utterance which is not totally inside the context
                response = ipus[i + 1]
                if self.ipu_is_overlap_within(context, response):
                    for r in ipus[i + 2 :]:
                        if not self.ipu_is_overlap_within(context, r):
                            response = r
                            break

                if context["end"] - context["start"] < self.min_context_ipu_time:
                    continue

                # Find the label and add info about overlaps
                overlap = False
                if response["speaker"] == context["speaker"]:
                    label = "hold"
                    n_hold += 1
                    if response["start"] < context["end"]:
                        overlap = True
                        n_overlaps += 1
                else:
                    label = "shift"
                    if response["start"] < context["end"]:
                        overlap = True
                        n_overlaps += 1
                    if response["text"] in self.backchannel_list:
                        label = "backchannel"
                        n_bc += 1
                    else:
                        n_shift += 1

                # get turns/vad up until context
                # join with turns indicates if the last turn in history is also the current speaker
                history, join_with_turns = self.get_history(ipus, i)
                vad_context = self.extract_context_vad(
                    vad, end_time=context["end"] + self.lookahead_duration
                )
                context["vad"] = vad_context
                context["turns"] = history
                context["join_with_turns"] = join_with_turns
                context["overlap"] = overlap
                sample = {
                    "session": session,
                    "audio_path": audio_path,
                    "label": label,
                    "context": context,
                    "response": response,
                }
                yield key, sample
                key += 1

        print("-" * 30)
        print("Extracted")
        print("Shifts: ", n_shift)
        print("BC: ", n_bc)
        print("Shift + BC: ", n_bc + n_shift)
        print("Holds: ", n_hold)
        print("-" * 30)


class Debuggin:
    @staticmethod
    def test_text_focus_dialog(filepaths):
        T = TextFocusDialog()
        dialog = list(T.generate_refined_dialogs(filepaths))[0]

        for turn in dialog[1]["dialog"]:
            print("Speaker: ", turn["speaker"])
            print("text: ", turn["text"])
            print("Start: ", turn["start"])
            print("End: ", turn["end"])
            # print("vad: ", turn["vad"])
            print("bc: ", len(turn["backchannel"]))
            print("within: ", len(turn["within"]))
            # print(turn.keys())
            input()

    @staticmethod
    def test_classification(filepaths):
        C = Classification(omit_post_words=True, backchannel_list=BACKCHANNELS)
        samples = list(C.generate_classification(filepaths))
        print(len(samples))

        i = 14
        end_time = samples[i][1]["response"]["start"]
        pred_end_time = end_time + C.lookahead_duration

        d = samples[i][1]
        vad = d["context"]["vad"]
        vad_oh = VAD.vad_list_to_onehot(
            vad, sample_rate=8000, hop_length=0.02, duration=pred_end_time
        )
        target_time = d["response"]["start"]
        pred_end_time = target_time + C.lookahead_duration

        fig, _ = plot_vad_list(vad, end_time=pred_end_time, target_time=target_time)

    @staticmethod
    def test_ipu(filepaths):
        from os.path import expanduser
        import sounddevice as sd
        import matplotlib.pyplot as plt
        from datasets_turntaking.utils import load_waveform
        from datasets_turntaking.features.plot_utils import plot_vad_list
        from datasets_turntaking.features.vad import VAD

        audio_root = join(expanduser("~"), "projects/data/switchboard/audio")
        ipuer = SegmentIPU(lookahead_duration=3, backchannel_list=BACKCHANNELS)
        samples = list(ipuer.generate_ipus(filepaths[1:]))
        _, sample = samples[10]

        print("sample: ", sample.keys())
        print("context: ", sample["context"].keys())

        for key, sample in samples:
            print(sample["label"])
            if sample["context"]["join_with_turns"]:
                for t in sample["context"]["turns"][:-1]:
                    print("-" * 20)
                    print(t)
                print("=" * 40)
                print(
                    "CONTEXT:\n",
                    sample["context"]["turns"][-1],
                    sample["context"]["text"],
                )
            else:
                for t in sample["context"]["turns"]:
                    print("-" * 20)
                    print(t)
                print("=" * 40)
                print("CONTEXT:\n", sample["context"]["text"])
            print("=" * 40)
            print("Response:\n", sample["response"]["text"])
            print("#" * 50)
            input()

        _, sample = samples[50]
        ap = join(audio_root, sample["audio_path"] + ".wav")
        x, sr = load_waveform(
            ap,
            start_time=sample["context"]["start"],
            end_time=sample["response"]["end"],
        )
        # x = x.mean(dim=0)
        print("LABEL: ", sample["label"])
        print("context: ", sample["context"]["text"])
        end_time = sample["context"]["end"] + ipuer.lookahead_duration
        vad = VAD.vad_list_to_onehot(
            sample["context"]["vad"],
            sample_rate=8000,
            hop_length=0.05,
            duration=end_time,
        )
        a_time = sample["context"]["end"] - sample["context"]["start"]
        a = int(sr * a_time)
        fig, ax = plt.subplots(2, 1, figsize=(12, 6))
        ax[0].plot(x[0], alpha=0.6)
        ax[0].plot(x[1], color="y", alpha=0.6)
        ax[0].vlines(a, ymin=-1, ymax=1, color="k", linewidth=2)
        ax[0].set_xlim([0, x.shape[-1]])
        _, ax[1] = plot_vad_list(sample["context"]["vad"], end_time=end_time, ax=ax[1])
        ax[1].set_xlim([sample["context"]["start"], sample["response"]["end"]])
        sd.play(x.mean(dim=0), samplerate=8000)


if __name__ == "__main__":

    from os.path import expanduser
    from datasets_turntaking.features.vad import VAD
    from datasets_turntaking.features.plot_utils import plot_vad_list

    # for filepath in filepaths:
    filepaths = []
    root = join(expanduser("~"), ".cache/huggingface/datasets/downloads/extracted/")
    root += "3bb5f33eb413284d4ef4098cadaccfa92b81653428c64ea8f954fe77a21c687c/"
    filepath = root + "swb_ms98_transcriptions/20/2001/2001_default.json"
    filepaths.append(filepath)
    filepath = root + "swb_ms98_transcriptions/22/2284/2284_default.json"
    filepaths.append(filepath)

    Debuggin.test_text_focus_dialog(filepaths)

# if __name__ == "__main__":
#     from datasets import load_dataset
#     from datasets_turntaking.utils import repo_root
#
#     dset_path = join(repo_root(), "datasets_turntaking/switchboard/switchboard.py")
#     dset = load_dataset(dset_path, split="train", name="strict_times")
#
#     # d = dset[0]
#     # text = d["utterances"]["text"]
#     # n_utts = len(d["utterances"]["text"])
#
#     s = "okay_1 th[ere's]- i[t's]- [vocalized-noise] hello [noise] there [laughter-right] [laughter-that's] on [laughter] and {wheatherwise} in [splace/space] where I w[ent] -[th]at"
#     w = swb_regexp(s)
#     print(s)
#     print("-" * 20)
#     print(w)
#
#     a = find_asides(s)
#     print(a)
#     s = "hello <b_aside> I can't talk now <e_aside>"
#     a = find_asides(s)
#     print(a)
