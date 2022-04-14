from datasets import load_dataset


def load_lj_speech(split="train"):
    """ """

    def process_and_add_name(examples):
        examples["dataset_name"] = "lj_speech"
        return examples

    if split == "val":
        split = "validation"

    if split == "validation":
        dset = load_dataset("lj_speech", split="train[85%:95%]")
    elif split == "test":
        dset = load_dataset("lj_speech", split="train[95%:]")
    else:
        dset = load_dataset("lj_speech", split="train[:85%]")

    # all_names = ['one_person_dialogs', 'woz_dialogs']
    # only 'one_person_dialogs' does not contain duplicate keys!!
    # dset = load_dataset("lj_speech", split=split)
    dset = dset.remove_columns("normalized_text")
    dset = dset.map(process_and_add_name)
    return dset


if __name__ == "__main__":

    import sounddevice as sd
    from datasets_turntaking.utils import load_waveform

    sd.default.samplerate = 8000

    dset2 = load_lj_speech("test")
    d = dset2[0]
    x, sr = load_waveform(d["file"], sample_rate=8000)

    print("X: ", tuple(x.shape))
    print("SR: ", sr)
    print("TEXT: ", d["text"])

    sd.play(x[0])
    print(d["text"])
