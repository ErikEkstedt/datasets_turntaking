import streamlit as st
from htbuilder import H, styles
from htbuilder.units import unit
import numpy as np
import matplotlib.pyplot as plt

import tokenizers
from turngpt.tokenizer import SpokenDialogTokenizer
from datasets_turntaking.dataset.spoken_dialog import load_fisher, load_switchboard


@st.cache
def load_dset(dataset, split, omit_bc, omit_ov):
    if dataset == "fisher":
        dset = load_fisher(
            split=split,
            omit_backchannels=omit_bc,
            omit_overlap_within=omit_ov,
            format_turns=True,
            num_proc=1,
        )
    else:
        dset = load_switchboard(
            split=split,
            omit_backchannels=omit_bc,
            omit_overlap_within=omit_ov,
            format_turns=True,
            num_proc=1,
        )
    return dset


@st.cache(
    hash_funcs={
        tokenizers.normalizers.Sequence: lambda _: None,
        tokenizers.Tokenizer: lambda _: None,
    }
)
def load_tokenizer():
    return SpokenDialogTokenizer("gpt2")


def dialog_to_html(dialog):
    html_string = ""
    for ii, utt in enumerate(dialog):
        color = "blue"
        if ii % 2 == 0:
            color = "red"
        html_string += f'<p style="color:{color}; margin:0px">' + utt + "</p>"
    return H.div(
        html_string,
        style=styles(
            background="white",
            border_radius=unit.rem(0.33),
            border="solid",
            font_weight="bold",
            padding=(unit.rem(0.125), unit.rem(0.5)),
            margin=(unit.rem(0.1)),
            overflow="hidden",
        ),
    )


def vad_statistics(vad):
    dur = []
    for ch in range(2):
        ch_dur = []
        for start, end in vad[ch]:
            ch_dur.append(end - start)
        dur.append(np.array(ch_dur))

    stats = {
        "A": {
            "total": dur[0].sum().round(2),
            "mean": dur[0].mean().round(2),
            "std": dur[0].std().round(2),
        },
        "B": {
            "total": dur[1].sum().round(2),
            "mean": dur[1].mean().round(2),
            "std": dur[1].std().round(2),
        },
    }
    total = stats["A"]["total"] + stats["B"]["total"]
    stats["A"]["ratio"] = round(100 * stats["A"]["total"] / total, 1)
    stats["B"]["ratio"] = round(100 - stats["A"]["ratio"], 1)

    # Histogram
    fig, ax = plt.subplots(1, 1, figsize=(9, 3))
    # A

    n_bins = 20
    max_range = 6
    _ = ax.hist(
        dur[0],
        bins=20,
        range=(0, max_range),
        facecolor="b",
        alpha=0.5,
        label=f'A: {stats["A"]["ratio"]}%, m={stats["A"]["mean"]}, u={stats["A"]["std"]}',
    )
    _ = ax.hist(
        dur[1],
        bins=20,
        range=(0, max_range),
        facecolor="r",
        alpha=0.5,
        label=f'B: {stats["B"]["ratio"]}%, m={stats["B"]["mean"]}, u={stats["B"]["std"]}',
    )
    ax.set_xlabel("VA segments (s)")
    ax.set_ylabel("N")
    ax.set_title("VA segment histogram")
    ax.set_xlim((0, max_range))
    ax.legend()

    stats["vad_histogram"] = fig
    return stats


if __name__ == "__main__":
    if "tokenizer" not in st.session_state:
        st.session_state.tokenizer = load_tokenizer()

    with st.sidebar:
        st.title("Datasets Turn-taking")
        st.session_state.dataset = st.selectbox("Dataset", ["fisher", "switchboard"])
        st.session_state.split = st.selectbox("Split", ["train", "val", "test"])
        st.session_state.omit_bc = st.checkbox("Omit backchannel")
        st.session_state.omit_ov = st.checkbox("Omit overlap within", value=True)

    st.session_state.dset = load_dset(
        st.session_state.dataset,
        st.session_state.split,
        st.session_state.omit_bc,
        st.session_state.omit_ov,
    )

    with st.container():
        dataset = st.session_state.dataset
        split = st.session_state.split
        n = len(st.session_state.dset)
        st.title(f"{dataset[0].upper() + dataset[1:]} ({split})")
        st.session_state.sample_idx = st.number_input(
            f"Sample index {n}",
            min_value=0,
            max_value=n,
            value=0,
        )

        d = st.session_state.dset[st.session_state.sample_idx]
        st.text(f"Session: {d['session']}   |   Turns: {len(d['dialog'])}")

        st.header("Voice Activity")
        va_stats = vad_statistics(d["vad"])
        st.pyplot(va_stats["vad_histogram"])

        st.header("Dialog")
        html_string = dialog_to_html(d["dialog"])
        st.markdown(html_string, unsafe_allow_html=True)
