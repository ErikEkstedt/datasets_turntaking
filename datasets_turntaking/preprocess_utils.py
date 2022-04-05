from os import remove
import subprocess

try:
    from sphfile import SPHFile
except ModuleNotFoundError as e:
    print(
        "Requires sphfile `pip install sphfile` to process Callhome/Fisher sph-files."
    )
    raise e


def sph2pipe_to_wav(sph_file):
    wav_file = sph_file.replace(".sph", ".wav")
    subprocess.check_call(["sph2pipe", sph_file, wav_file])
    return wav_file


def sph_to_wav(filepath):
    sph = SPHFile(filepath)
    wavpath = filepath.replace(".sph", ".wav")
    return sph.write_wav(wavpath, start=None, stop=None)


def delete_path(filepath):
    remove(filepath)
