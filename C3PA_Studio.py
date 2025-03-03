# [2502] n8
#

import streamlit as st
import os

testing_path = os.environ.get('MOUNT_PATH', '/c3pa-app/testing')    # gcs
# testing_path='./c3pa-app/testing'


def get_test_count():
    count_path = os.path.join(testing_path, "test_count.txt")
    test_count = ''
    with open(count_path, "r") as file:
        test_count = file.readline()
    return test_count        


st.title("C3PA Studio")

st.write("Tools to work with C3PA data assets.")
st.write("Select an option to the left to drill down to a specific topic.")
st.write("Link to CCC-Bot")
st.write("Link to the corresponding repo")

st.write("Number of test questions logged : " + get_test_count())

st.sidebar.success("Select an option above.")

