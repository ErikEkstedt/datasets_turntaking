from os.path import join
from sklearn.model_selection import KFold

from datasets_turntaking.utils import read_txt, write_txt
from datasets_turntaking.dataset.switchboard.switchboard import INCLUDES_ASIDES


KFOLD_ROOT = "datasets_turntaking/dataset/switchboard/files/kfolds"


def write_kfolds(train_val, path=None):
    if path is None:
        path = KFOLD_ROOT
    kf = KFold(n_splits=11)
    for ii, (train_index, test_index) in enumerate(kf.split(train_val)):
        tmp_train = []
        for i in train_index:
            tmp_train.append(train_val[i])
        tmp_val = []
        for i in test_index:
            tmp_val.append(train_val[i])
        print("val: ", len(tmp_val))
        print("train: ", len(tmp_train))
        write_txt(tmp_val, join(path, f"{ii}_fold_val.txt"))
        write_txt(tmp_train, join(path, f"{ii}_fold_train.txt"))


if __name__ == "__main__":

    # Load all files/dialogs-numbers
    train = read_txt("datasets_turntaking/dataset/switchboard/files/train.txt")
    val = read_txt("datasets_turntaking/dataset/switchboard/files/val.txt")
    test = read_txt("datasets_turntaking/dataset/switchboard/files/test.txt")

    # combine train+val but omit those which INCLUDES_ASIDES
    train_val = []
    for n_dialog in train + val:
        if not n_dialog in INCLUDES_ASIDES:
            train_val.append(n_dialog)

    # save kfolds
    write_kfolds(train_val)
