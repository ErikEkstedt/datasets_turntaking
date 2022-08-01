from datasets_turntaking.dataset.switchboard import load_switchboard

if __name__ == "__main__":
    # for split in ["train", "val", "test"]:
    #     dset = load_switchboard(split=split)

    dset = load_switchboard(split="train")

    d = dset[0]

    a = d["dialog"][0]
    b = d["dialog"][1]

    for utt in a["text"]:
        print(utt)
        input()
