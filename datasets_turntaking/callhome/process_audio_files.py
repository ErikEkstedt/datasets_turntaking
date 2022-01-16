from os.path import join
from os import remove
from glob import glob
from tqdm import tqdm


try:
    from sphfile import SPHFile
except ModuleNotFoundError as e:
    print("Requires sphfile `pip install sphfile` to process Callhome sph-files.")
    raise e


def sph_to_wav(filepath):
    wavpath = filepath.replace(".sph", ".wav")
    sph = SPHFile(filepath)
    # write out a wav file with content from 111.29 to 123.57 seconds
    # sph.write_wav(wavpath, start=111.29, end=123.57)
    return sph.write_wav(wavpath, start=None, stop=None)


def delete_path(filepath):
    remove(filepath)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from datasets_turntaking.callhome import DATA_DIR

    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--delete_sph", action="store_true")
    args = parser.parse_args()

    sph_files = glob(join(args.data_dir, "**/*.sph"), recursive=True)
    for filepath in tqdm(sph_files):
        _ = sph_to_wav(filepath)

    if args.delete_sph:
        print("#" * 50)
        print("WARNING!")
        print("#" * 50)
        print("WARNING!")
        print("#" * 50)
        print()
        print("Deleting all CALLHOME `.sph` files")
        inp = input("Continue? (y/n)")
        if inp.lower() in ["y", "yes"]:
            for filepath in tqdm(sph_files):
                _ = delete_path(filepath)
            print("Done!")
