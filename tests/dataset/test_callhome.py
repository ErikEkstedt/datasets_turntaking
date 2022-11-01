import pytest

from datasets_turntaking.dataset.spoken_dialog.callhome.utils import callhome_regexp


@pytest.mark.callhome
def test_regexp():

    # ((unintelligble to annotator but best attempt))
    s = "hello &ibm {laugh} so %um ((i think)) (( )) and call <German Feldmessen>."
    s = callhome_regexp(s)
    ans = "hello ibm so um i think and call Feldmessen."
    assert s == ans, "unintelligible fail ((...))"

    s = "I ((sent her to)) ((Taiwan)) she was"
    s = callhome_regexp(s)
    ans = "I sent her to Taiwan she was"
    assert s == ans, "unintelligible fail ((...))"

    s = "hello <German Feldmessen>. yes indeed <German Feldmessen>."
    s = callhome_regexp(s)
    ans = "hello Feldmessen. yes indeed Feldmessen."
    assert s == ans, "other language fail <...>"
