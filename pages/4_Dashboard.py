# [2503] n8
#

import streamlit as st
import os, sys

sys.path.insert(0, "data")
sys.path.insert(0, "utils")

from utility_functions import get_count, configure_app


mount_path, app_path, cfg = configure_app()

st.image("data/Numantic_Solutions_Logotype_light.png", width=200)

st.title("Dashboard")

st.write("C3PA Studio version : " + cfg['version'])

test_count = get_count(app_path, "test_count.txt")
st.write("Number of test questions logged : " + test_count)

user_question_count = get_count(app_path, "user_question_count.txt")
st.write("Number of user questions answered : " + user_question_count)

st.divider()
st.write("March 2025")
st.image("data/2503-cccbot-gemini_2.jpg", width=500)

st.divider()
st.write("February 2025")
st.image("data/2502-cccbot-phi3.jpg", width=500)

st.sidebar.success("Select an option above.")

