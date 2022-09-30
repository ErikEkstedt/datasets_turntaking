# Callhome

* [URL](https://catalog.ldc.upenn.edu/LDC97S42)
* [Documentation](https://catalog.ldc.upenn.edu/docs/LDC97S42/)


The Callhome dataset must first be manually downloaded from
[URL](https://catalog.ldc.upenn.edu/LDC97S42). The audio files are of format
`ulaw,embedded-shorten-v2.00` and can't be loaded by torchaudio
([issue](https://github.com/pytorch/audio/issues/989)).

Before using the dataset/dataloaders please format all `.sph` files to `.wav` which can be done
with: 

```bash
python datasets_turntaking/callhome/process_audio_files.py --data_dir $CALLHOME_ROOT
```

which copies the `.sph` files to `.wav`. Add the `--delete_sph` flag to also
delete the original files (but you may want to check that everything went
smoothly).

```python
from datasets_turntaking.callhome import load_callhome
from datasets_turntaking.utils import load_waveform

dset = load_callhome(split="train")
print("Callhome: ", len(dset))

d = dset[0]
for k, v in d.items():
    if isinstance(v, dict):
        print(f"{k}:")
        for kk, vv in v.items():
            print(f"\t{kk}: {len(vv)}")
    else:
        print(f"{k}: {v}")
x, sr = load_waveform(d["audio_path"])
print("waveform: ", x.shape)
print("duration: ", round(x.shape[-1] / sr, 2))
print("sample_rate: ", sr)

# ---------------------------------------
# id: en_4431.wav
# audio_path: .../callhome/callhome_eng/data/train/en_4431.wav
# dialog:
#         text: 139
#         speaker: 139
#         start: 139
#         end: 139
# dataset_name: callhome
# waveform:  torch.Size([2, 8293558])
# duration:  1036.69
# sample_rate:  8000
```


### Special annotations

```
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
```

### Reference

Given on [LDC](https://catalog.ldc.upenn.edu/LDC97S42)

```
Canavan, Alexandra, David Graff, and George Zipperlen. 
CALLHOME American English Speech LDC97S42. 
Web Download. Philadelphia: Linguistic Data Consortium, 1997.
```

-------------------

This may not be correct.... use with caution

```latex
@misc{callhome,
  author = {Canavan, Alexandra, David Graff, and George Zipperlen},
  title = {CALLHOME American English Speech LDC97S42.},
  year = {1997},
  url = {https://catalog.ldc.upenn.edu/LDC97S42}
}
```

