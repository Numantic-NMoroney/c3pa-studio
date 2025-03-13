# [2502] n8
#

import streamlit as st
import os, sys

sys.path.insert(0, "utils")

from utility_functions import get_count, configure_app


mount_path, app_path, cfg = configure_app()

st.image("data/Numantic_Solutions_Logotype_light.png", width=200)

st.title("Feedback")

st.markdown("If you have tried the [ccc-bot](https://ccc-polasst.numanticsolutions.com), please consider providing us with some feedback.")

st.write("All of the fields below are optional, fill in as many or as few as you prefer.")

st.divider()

positive_topics = st.text_input("Topics or areas that were good or helpful : ", "Add a topic")

negative_topics = st.text_input("Topics or areas that were poor or not helpful : ", "Add a topic")

data_sources = st.text_input("The tool improves with data, are there websites or sources you would recommend? : ", "Add a website")

other_input = st.text_input("Any other questions or input you might have? : ", "Add a question or input")

st.divider()

st.write("One goal for this tool is to help people learn something new about the California Community Colleges.")

learned_something = st.radio(
    "Did you learn anything new about California Community Colleges using this tool?",
    ["Yes", "No"], index=None
)

st.write("Thank you for your time")

st.sidebar.success("Select an option above.")

