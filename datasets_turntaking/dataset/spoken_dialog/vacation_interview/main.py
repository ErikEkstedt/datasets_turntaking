from datasets_turntaking.dataset.spoken_dialog.vacation_interview import (
    load_vacation_interview,
)
from datasets_turntaking.utils import load_waveform


if __name__ == "__main__":

    dset_hf = load_vacation_interview()
    d = dset_hf[0]
