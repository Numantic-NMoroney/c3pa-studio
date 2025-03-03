# [2502] n8
#

import streamlit as st

st.title("Test")

st.write("Conduct tests over collections of test questions.")

qs = (
    "questions-250226-q10-a.txt",
    "questions-250226-q10-b.txt",
    "questions-250226-q10-c.txt",
    "questions-250226-q10-d.txt"
)

option = st.selectbox(
    "Select a collection :",
    qs
)

if st.button("Start Testing"):
    st.write("Testing started.")


st.sidebar.success("Select an option above.")

