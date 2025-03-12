# [2502] [2503] n8
#

import streamlit as st
import json, os, sys


def configure_app():
    sys.path.insert(0, "utils")
    with open("utils/c3pa_studio_path.json") as file:
        mount_path = json.load(file)['mount_path']

    has_mp = os.environ.get("MOUNT_PATH") != None
    if has_mp:
        app_path = os.environ.get('MOUNT_PATH', mount_path)
    else :
        app_path = './c3pa-app/testing'

    with open(app_path + "/c3pa_studio_config.json") as file:
        cfg = json.load(file)

    return mount_path, app_path, cfg


def get_count(name_txt):
    count_path = os.path.join(app_path, name_txt)
    test_count = ''
    with open(count_path, "r") as file:
        test_count = file.readline()
    return test_count        


mount_path, app_path, cfg = configure_app()

st.title("C3PA Studio")

st.write("Tools to work with C3PA data assets.")
st.write("Select an option to the left to drill down to a specific topic.")
st.write("Link to CCC-Bot")
st.write("Link to the corresponding repo")

st.write("C3PA Studio version : " + cfg['version'])
st.write("Number of test questions logged : " + get_count("test_count.txt"))
st.write("Number of user questions answered : " + get_count("user_question_count.txt"))

st.sidebar.success("Select an option above.")

