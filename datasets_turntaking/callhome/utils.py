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


def load_utterances(filepath, clean=True):
    data = read_txt(filepath)

    last_speaker = None
    utterances = []
    for row in data:
        # TODO: get this right
        # example evaltest/en_6467.txt
        # file start with hashtag+information
        # #Language: eng
        # #File id: 6467
        # #Starting at 383 Ending at 983
        # # 327 393 #BEGIN
        # # 973 983 #END
        m = re.match(r"^\#", row)
        if row == "" or m:
            continue
        # example spanning multiple rows
        # 462.58 468.59 B: That's were, that's where I guess, they would, %um, they
        # get involved in initial public offering, stock offerings
        split = row.split(" ")
        try:
            start, end, = (
                float(split[0]),
                float(split[1]),
            )
        except:
            print(filepath)
            print(row)
            input()
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
    return utterances


def some_test_later():
    s = "hello &ibm {laugh} so %um ((unintelligible)) (( )) and call <German Feldmessen>. --"
    s = "hello <German Feldmessen>. yes indeed <German Feldmessen>."
    s = "I ((sent her to)) ((Taiwa)) she was "
    print(s)
    s = callhome_regexp(s)
    print(s)

    s = "#Lanugage"
    s = "# Lanugage"
    s = "Lanugage and #"
    m = re.match(r"^\#", s)
    if m:
        print(m)


if __name__ == "__main__":

    filepath = "/home/erik/projects/data/callhome/callhome_english_trans_970711/transcrpt/train/en_4065.txt"
    utterances = load_utterances(filepath)
    for utt in utterances:
        print(utt["text"])
