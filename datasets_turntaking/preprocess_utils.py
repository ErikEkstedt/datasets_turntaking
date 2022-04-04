from os import remove

try:
    from sphfile import SPHFile
except ModuleNotFoundError as e:
    print(
        "Requires sphfile `pip install sphfile` to process Callhome/Fisher sph-files."
    )
    raise e


def sph_to_wav(filepath):
    wavpath = filepath.replace(".sph", ".wav")
    sph = SPHFile(filepath)
    # write out a wav file with content from 111.29 to 123.57 seconds
    # sph.write_wav(wavpath, start=111.29, end=123.57)
    return sph.write_wav(wavpath, start=None, stop=None)


def delete_path(filepath):
    remove(filepath)
