# [2502] n8
#

import streamlit as st
import os, sys

sys.path.insert(0, "utils")

from utility_functions import get_count, configure_app


mount_path, app_path, cfg = configure_app()

st.image("data/Numantic_Solutions_Logotype_light.png", width=200)

st.title("Feedback")

st.write("Draft a user feedback page for the main ccc-bot.")

st.sidebar.success("Select an option above.")

