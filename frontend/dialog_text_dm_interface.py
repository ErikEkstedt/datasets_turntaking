import streamlit as st
import html
from htbuilder import H, styles
from htbuilder.units import unit

from datasets_turntaking import ConversationalDM
from turngpt.tokenizer import SpokenDialogTokenizer


def get_html_string(text):

    html_string = ""
    for ii, utt in enumerate(text.split("<ts>")):
        color = "blue"
        if ii % 2 == 0:
            color = "red"

        utt = utt + "[ts]"
        html_string += f'<p style="color:{color}; margin:0px">{utt}</p>'

    return H.div(
        html_string,
        style=styles(
            background="white",
            border_radius=unit.rem(0.33),
            border="solid",
            font_weight="bold",
            padding=(unit.rem(0.125), unit.rem(0.5)),
            # margin=(unit.rem(0.1)),
            # overflow="hidden",
        ),
    )


if __name__ == "__main__":

    if "tokenizer" not in st.session_state:
        st.session_state.tokenizer = SpokenDialogTokenizer()

    if "dm" not in st.session_state:
        dm = ConversationalDM(
            st.session_state.tokenizer,
            datasets=["switchboard", "fisher"],  # "curiosity_dialogs", "daily_dialog"]
            max_length=256,
            # overwrite=True,
            batch_size=20,
        )
        dm.prepare_data()
        dm.setup("all")

        st.session_state.dsets = {
            "all": {"train": dm.train_dset, "val": dm.val_dset, "test": dm.test_dset},
            "fisher": {
                "train": dm.train_dset.filter(lambda x: x["dataset"] == "fisher"),
                "val": dm.val_dset.filter(lambda x: x["dataset"] == "fisher"),
                "test": dm.test_dset.filter(lambda x: x["dataset"] == "fisher"),
            },
            "switchboard": {
                "train": dm.train_dset.filter(lambda x: x["dataset"] == "switchboard"),
                "val": dm.val_dset.filter(lambda x: x["dataset"] == "switchboard"),
                "test": dm.test_dset.filter(lambda x: x["dataset"] == "switchboard"),
            },
        }
        st.session_state.dm = dm

    with st.container():
        st.title("Dialog Text DM")

    dataset = st.selectbox(
        "Corpus: ", ["all"] + list(set(st.session_state.dm.train_dset["dataset"]))
    )
    split = st.selectbox("Split: ", ["train", "val", "test"])

    dset = st.session_state.dsets[dataset][split]
    n = len(dset)
    ii = st.slider("Pick a number", 0, n, step=1)

    d = st.session_state.dm.train_dset[ii]
    t = st.session_state.tokenizer.decode(d["input_ids"])
    html_string = get_html_string(t)
    st.markdown(html_string, unsafe_allow_html=True)
    # st.write(tt)
