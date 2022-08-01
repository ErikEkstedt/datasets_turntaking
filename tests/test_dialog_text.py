import pytest

from datasets_turntaking.dataset.switchboard import load_switchboard
from datasets_turntaking.dataset.fisher import load_fisher


@pytest.mark.dialog_text
@pytest.mark.fisher
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_only_fisher(split):
    dset = load_fisher(split)
    for d in dset:
        for speaker in d["dialog"]:
            for utt in speaker["text"]:
                assert len(utt) > 0, "empty string"
                assert "((" not in utt, f"double parenthesis: {utt} {d['session']}"


@pytest.mark.dialog_text
@pytest.mark.switchboard
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_only_swb(split):
    dset = load_switchboard(split)
    for d in dset:
        for speaker in d["dialog"]:
            for utt in speaker["text"]:
                assert len(utt) > 0, "empty string"
                assert "((" not in utt, f"double parenthesis: {utt} {d['session']}"
