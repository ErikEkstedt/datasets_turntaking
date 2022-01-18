from datasets_turntaking.switchboard import load_switchboard

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        dset = load_switchboard(split=split)
