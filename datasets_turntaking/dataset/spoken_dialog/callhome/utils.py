import re
from datasets_turntaking.utils import read_txt

"""
Transcript

&Tufts
%um/uh: non-lexemes
{laugh}
{lipsmack}

see


Special symbols:
    Noises, conversational phenomena, foreign words, etc. are marked
    with special symbols.  In the table below, "text" represents any
    word or descriptive phrase.

    {text}              sound made by the talker {laugh} {cough} {sneeze} {breath}
    [text]              sound not made by the talker (background or channel)
                        [distortion]    [background noise]      [buzz]

    [/text]             end of continuous or intermittent sound not made by
                        the talker (beginning marked with previous [text])

    [[text]]            comment; most often used to describe unusual
                        characteristics of immediately preceding or following
                        speech (as opposed to separate noise event)

                        [[previous word lengthened]]    [[speaker is singing]]

    ((text))            unintelligible; text is best guess at transcription
                        ((coffee klatch))

    (( ))               unintelligible; can't even guess text
                        (( ))


    <language text>     speech in another language
                        <English going to California>

    <? (( ))>           ? indicates unrecognized language; 
                        (( )) indicates untranscribable speech
                        <? ayo canoli>  <? (( ))>

    -text		partial word
    text-               -tion absolu- 

    #text#              simultaneous speech on the same channel
                        (simultaneous speech on different channels is not
                        explicitly marked, but is identifiable as such by
                        reference to time marks)

    //text//            aside (talker addressing someone in background)
                        //quit it, I'm talking to your sister!//

    +text+              mispronounced word (spell it in usual orthography) +probably+

   **text**             idiosyncratic word, not in common use
                        **poodle-ish**

    %text               This symbol flags non-lexemes, which are
			general hesitation sounds.  See the section on
			non-lexemes below to see a complete list for
			each language.  
			%mm %uh 

    &text               used to mark proper names and place names
                        &Mary &Jones    &Arizona        &Harper's
                        &Fiat           &Joe's &Grill

    text --             marks end of interrupted turn and continuation
    -- text             of same turn after interruption, e.g.
                        A: I saw &Joe yesterday coming out of --
                        B: You saw &Joe?!
                        A: -- the music store on &Seventeenth and &Chestnut.
"""


def callhome_regexp(s):
    """callhome specific regexp"""
    # {text}:  sound made by the talker {laugh} {cough} {sneeze} {breath}
    s = re.sub(r"\{.*\}", "", s)

    # [[text]]: comment; most often used to describe unusual
    s = re.sub(r"\[\[.*\]\]", "", s)

    # [text]: sound not made by the talker (background or channel) [distortion]    [background noise]      [buzz]
    s = re.sub(r"\[.*\]", "", s)

    # (( )): unintelligible; can't even guess text
    s = re.sub(r"\(\(\s*\)\)", "", s)

    # single word ((taiwa))
    # ((text)): unintelligible; text is best guess at transcription ((coffee klatch))
    s = re.sub(r"\(\((\w*)\)\)", r"\1", s)

    # multi word ((taiwa and other))
    # s = re.sub(r"\(\((\w*)\)\)", r"\1", s)
    s = re.sub(r"\(\(", "", s)
    s = re.sub(r"\)\)", "", s)

    # -text / text-: partial word = "-tion absolu-"
    s = re.sub(r"\-(\w+)", r"\1", s)
    s = re.sub(r"(\w+)\-", r"\1", s)

    # +text+: mispronounced word (spell it in usual orthography) +probably+
    s = re.sub(r"\+(\w*)\+", r"\1", s)

    # **text**: idiosyncratic word, not in common use
    s = re.sub(r"\*\*(\w*)\*\*", r"\1", s)

    # remove proper names symbol
    s = re.sub(r"\&(\w+)", r"\1", s)

    # remove non-lexemes symbol
    s = re.sub(r"\%(\w+)", r"\1", s)

    # text --             marks end of interrupted turn and continuation
    # -- text             of same turn after interruption, e.g.
    s = re.sub(r"\-\-", "", s)

    # <language text>: speech in another language
    s = re.sub(r"\<\w*\s(\w*\s*\w*)\>", r"\1", s)

    # remove double spacing on last
    s = re.sub(r"\s\s+", " ", s)
    s = re.sub(r"^\s", "", s)
    return s


def preprocess_utterance(filepath):
    """
    Load filepath and preprocess the annotations

    * Omit empty rows
    * join utterances spanning multiple lines
    """
    data = []
    for row in read_txt(filepath):
        # omit empty rows and rows starting with '#' (global file info)
        if row == "" or row.startswith("#"):
            continue

        # Some utterances span multiple rows:
        # i.e. evaltest/en_6467.txt
        #
        # 462.58 468.59 B: That's were, that's where I guess, they would, %um, they
        # get involved in initial public offering, stock offerings
        if row[0].isdigit():
            data.append(row)
        else:
            data[-1] += " " + row
    return data


def load_utterances(filepath, clean=True):
    data = preprocess_utterance(filepath)

    last_speaker = None
    utterances = []
    try:
        for row in data:
            split = row.split(" ")
            start, end, = (
                float(split[0]),
                float(split[1]),
            )
            speaker = split[2].replace(":", "")
            speaker = 0 if speaker == "A" else 1
            text = " ".join(split[3:])
            if clean:
                text = callhome_regexp(text)
            if last_speaker is None:
                utterances.append(
                    {"start": start, "end": end, "speaker": speaker, "text": text}
                )
            elif last_speaker == speaker:
                utterances[-1]["end"] = end
                utterances[-1]["text"] += " " + text
            else:
                utterances.append(
                    {"start": start, "end": end, "speaker": speaker, "text": text}
                )
            last_speaker = speaker
    except:
        print(f"CALLHOME UTILS load_utterance")
        print(f"ERROR on split {filepath}")
    return utterances


def extract_vad(utterances):
    vad = [[], []]
    for utt in utterances:
        vad[utt["speaker"]].append((utt["start"], utt["end"]))
    return vad


if __name__ == "__main__":

    filepath = "/home/erik/projects/data/callhome/callhome_english_trans_970711/transcrpt/train/en_4065.txt"
    filepath = "/home/erik/projects/data/callhome/callhome_english_trans_970711/transcrpt/train/en_4926.txt"
    utterances = load_utterances(filepath)
    vad = extract_vad(utterances)
    print("vad 0: ", len(vad[0]))
    print("vad 1: ", len(vad[1]))
