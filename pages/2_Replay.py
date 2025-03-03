# [2502] n8
#

import streamlit as st
import pandas as pd
from time import sleep


testing_path = os.environ.get('MOUNT_PATH', '/c3pa-app/testing')    # gcs
# testing_path='./c3pa-app/testing'

if 'pause' not in st.session_state:
    st.session_state.pause = 5


def c3pa_replay(name_json, sleep_sec, number_loops):
    txt = st.empty()
    with txt:
        st.text('QUESTION?\n\nAnswer')
    sleep(sleep_sec)

    in_path = os.path.join(testing_path, name_json)
    df = pd.read_json(in_path)

    for j in range(number_loops):
        for i, q in enumerate(df['question']):
            with txt:
                st.text(q.upper() + "\n\n" + df['response'][i])

            sleep(sleep_sec)


st.title("Replay")

st.write("Replay previously collected test results.")

logs = (
    "log_testing-c3pa-2025-02-26-18:14:07.441878.json",
    "log_testing-c3pa-2025-02-26-18:32:10.432825.json",
    "log_testing-c3pa-2025-02-26-19:45:24.814108.json",
    "log_testing-c3pa-2025-02-27-18:19:10.793607.json"
)

col1, col2 = st.columns([0.5, 0.5])

with col1:
    option = st.selectbox(
        "Select a log :",
        logs
    )
with col2:
    _ = st.slider("Pause in seconds : ", 1, 15, 5, key='pause')

if st.button("Start Replay"):
    c3pa_replay(option, st.session.pause, 5)

st.sidebar.success("Select an option above.")

