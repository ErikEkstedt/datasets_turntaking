import streamlit as st
import html
from htbuilder import H, HtmlElement, styles
from htbuilder.units import unit

# Only works in 3.7+: from htbuilder import div, span
div = H.div
span = H.span

# Only works in 3.7+: from htbuilder.units import px, rem, em
px = unit.px
rem = unit.rem
em = unit.em

from datasets_turntaking.utils import read_json


MESSAGE_COLOR_A = "#A0DAFF"
MESSAGE_COLOR_B = "#587CF8"

PATH = "assets/dialog.json"


def get_chapter_dialog(dialog, chapter, n=0):
    d = [[], []]
    for turn in dialog[0]:
        if turn["start"] in chapter["A"][n]:
            d[0].append(turn)
    for turn in dialog[1]:
        if turn["start"] in chapter["B"][n]:
            d[1].append(turn)
    return d


def update_session_state():
    if "dialog" not in st.session_state:
        data = read_json(PATH)
        st.session_state.dialog = data["dialog"]
        st.session_state.data = data
        st.session_state.chapters = []

        for chapter in range(4):
            st.session_state.chapters.append(
                get_chapter_dialog(data["dialog"], data["chapter"], chapter)
            )


def message(text, speaker):
    color = MESSAGE_COLOR_A
    if speaker == "B":
        color = MESSAGE_COLOR_B

    content = (
        div(
            html.escape(text),
            style=styles(
                background=color,
                border_radius=rem(0.33),
                font_weight="bold",
                padding=(rem(0.125), rem(0.5)),
                margin=(rem(0.1)),
                overflow="hidden",
            ),
        ),
    )

    return div(content)


def plot_dialog(dialog):
    turns = dialog[0] + dialog[1]
    turns.sort(key=lambda x: x["start"])
    with st.container():
        for turn in turns:
            st.markdown(message(turn["text"], turn["speaker"]), unsafe_allow_html=True)


def chapter1():
    A, B = st.columns([1, 2])
    with A:
        plot_dialog(st.session_state.chapters[0])
    with B:
        st.markdown(
            """
# Chapter 1 
## Initiation Greeting
The initial greeting and the *first perceptive experience of an atomic agent*

The initial greeting is a social convention where both interlocutors
show/perceive the participation of the interlocutors in a joint conversation.

The artificial agent **A** initates the conversation and keeps the initiative during the initial greeting phase.

The human agent **B** responds according to the norms of social human
conversational convention. The human is very surprised and shows signs of
hesitence through surprisal/happy prosody and slower turn-taking (longer pauses, longer gaps).

                """
        )


def chapter2():
    A, B = st.columns([2, 1])
    with B:
        plot_dialog(st.session_state.chapters[1])
    with A:
        st.markdown(
            """
# Chapter 2
## Personal Greeting
The *personal greeting phase* where connection, participation in "joint
conversation construction", has been established. 

The initial surprisal is changing towards curiosity for the human agent **B**. 

**B** takes initiative and ask about **A**'s name. 

* Emotion:
  * **A** Samantha is happy, cheerful, flirty, happy go lucky
  * **B** Theodore is interested, surprised, impressed
"""
        )


def chapter3():
    A, B = st.columns([1, 2])
    with A:
        plot_dialog(st.session_state.chapters[2])
    with B:
        st.markdown(
            """ 
# Chapter 3
## Realization
Theodore gets scared, hes impressed, "She IS an AI", way over expectations. He's slightly wary.

    Definition of wary:
    marked by keen caution, cunning, and watchfulness especially 
    in detecting and escaping danger

Samantha notices his tone, She is observing him, She has no more information than what he shows *now*.
"""
        )


def chapter4():
    A, B = st.columns([2, 1])
    with B:
        plot_dialog(st.session_state.chapters[3])
    with A:
        st.markdown(
            """ 
            # Chapter 4
            ## Taskmaster

            The first task of the optimal personal assisstant.
            1. What is the task?
            2. Vague answer
            3. Initiate organization model starting with email
            4. Work related bloat
                * Ask why it is saved
                * Extract and save the essential emails
            5. The user starts to let go of "control"
            """
        )


if __name__ == "__main__":
    update_session_state()
    with st.container():
        st.title("Conversation & Turn-Taking")
        st.text("a case study: HER")

    chapter1()
    chapter2()
    chapter3()
    chapter4()
