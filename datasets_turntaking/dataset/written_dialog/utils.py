from .daily_dialog import load_daily_dialog
from .curiosity_dialogs import load_curiosity_dialogs
from .multiwoz_v22 import load_multiwoz_v22
from .metawoz import load_metawoz
from .taskmaster import load_taskmaster1, load_taskmaster2, load_taskmaster3

from ..spoken_dialog import load_spoken_dataset


def load_multiple_datasets(datasets, split):
    dsets = []
    for d in datasets:
        if d == "curiosity_dialogs":
            dsets.append(load_curiosity_dialogs(split))
        elif d == "daily_dialog":
            dsets.append(load_daily_dialog(split))
        elif d == "multi_woz_v22":
            dsets.append(load_multiwoz_v22(split))
        elif d == "meta_woz":
            dsets.append(load_metawoz(split))
        elif d == "taskmaster1":
            dsets.append(load_taskmaster1(split))
        elif d == "taskmaster2":
            dsets.append(load_taskmaster2(split))
        elif d == "taskmaster3":
            dsets.append(load_taskmaster3(split))
        elif d in ["fisher", "switchboard", "callhome"]:
            dsets.append(load_spoken_dataset(datasets=[d], split=split))
    return dsets
