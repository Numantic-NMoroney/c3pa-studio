# [2502] n8
#

import streamlit as st
import os

testing_path = os.environ.get('MOUNT_PATH', '/c3pa-app/testing')    # gcs
# testing_path='./c3pa-app/testing'


def get_count(name_txt):
    count_path = os.path.join(testing_path, name_txt)
    test_count = ''
    with open(count_path, "r") as file:
        test_count = file.readline()
    return test_count        


st.title("C3PA Studio")

st.write("Tools to work with C3PA data assets.")
st.write("Select an option to the left to drill down to a specific topic.")
st.write("Link to CCC-Bot")
st.write("Link to the corresponding repo")

st.write("Number of test questions logged : " + get_count("test_count.txt"))
st.write("Number of user questions answered : " + get_count("user_question_count.txt"))

st.sidebar.success("Select an option above.")

