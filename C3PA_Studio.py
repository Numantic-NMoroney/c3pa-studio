# [2502] [2503] n8
#

import streamlit as st
import os, sys

sys.path.insert(0, "utils")

from utility_functions import get_count, configure_app


mount_path, app_path, cfg = configure_app()

st.image("data/Numantic_Solutions_Logotype_light.png", width=400)

st.title("C3PA Studio")

st.write("Tools to work with C3PA data assets.")
st.write("Select an option to the left to drill down to a specific topic.")
st.write("Link to CCC-Bot")
st.write("Link to the corresponding repo")

st.write("C3PA Studio version : " + cfg['version'])

test_count = get_count(app_path, "test_count.txt")
st.write("Number of test questions logged : " + test_count)

user_question_count = get_count(app_path, "user_question_count.txt")
st.write("Number of user questions answered : " + user_question_count)

term = os.environ.get('TERM')
st.write("term : " + str(term))

st.sidebar.success("Select an option above.")

