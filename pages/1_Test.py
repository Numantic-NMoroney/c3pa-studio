# [2502] n8
#

import streamlit as st
import sys, os
import json
from datetime import datetime
import time
import hashlib

sys.path.insert(0, "rag")
sys.path.insert(0, "utils")

from utility_functions import get_count, configure_app
from rag_bot import CCCPolicyAssistant


mount_path, app_path, cfg = configure_app()

# sys.path.insert(0, "rag")
# sys.path.insert(0, "utils")
# from rag_bot import CCCPolicyAssistant
# 
# testing_path = os.environ.get('MOUNT_PATH', '/c3pa-app/testing')    # gcs
# testing_path = 'c3pa-app/testing'


def portable_hash(string):
    encoded_string = string.encode('utf-8')
    hash_object = hashlib.sha256(encoded_string)
    return hash_object.hexdigest()

def update_count(name_txt, n):
    # count_path = os.path.join(testing_path, name_txt)
    count_path = os.path.join(app_path, name_txt)
    count = ''
    with open(count_path, "r") as file:
        count = file.readline()
    with open(count_path, "w") as file:
        cn = int(count) + n
        file.write(str(cn) + "\n")


st.image("data/Numantic_Solutions_Logotype_light.png", width=200)

st.title("Test")

st.write("Conduct tests over sets of test questions.")

qs = (
    "questions-250226-q10-a.txt",
    "questions-250226-q10-b.txt",
    "questions-250226-q10-c.txt",
    "questions-250226-q10-d.txt"
)

name_qs = st.selectbox(
    "Select a set of questions :",
    qs
)
in_path = os.path.join(app_path, name_qs)
# in_path = os.path.join(testing_path, name_qs)

if st.button("Start Testing"):
    st.write("Testing started.")

    bot = CCCPolicyAssistant(chroma_collection_name = "crawl_docs-vai-2",
                             chat_bot_verbose=False,
                             dot_env_path = "")

    questions = []
    with open(in_path, "r") as file:
        for line in file:
            questions.append(line.strip())

    qn = str(len(questions))
    dt = str(datetime.now()).replace(" ", "-")

    initial_context = " You are a chat bot that answers policy questions about California Community Colleges. "

    data = []
    for i, question in enumerate(questions) :

        st.text(question.upper())

        h1 = portable_hash(question)

        user_input = initial_context + question
        # prompt = question

        t1 = str(datetime.now()).replace(" ", "_")

        time.sleep(15)
        #
        # response = bot.show_conversation(input_message=prompt, chat_bot_verbose=False)
        # context_urls = bot.source_urls
        # ai_response = bot.ai_response
        #
        # if len(context_urls) > 0:
        #    ai_response += " Context URLs : "
        #    for url in context_urls:
        #        ai_response += (" " + url + " ")

        # Get the AI assistant's response
        bot.show_conversation(input_message=user_input)

        # Get retrieved urls
        retrieved_urls = ["- [{}]({})\n".format(up[0], up[1]) for up in bot.retrieved_urls]
        retrieved_urls = list(set(retrieved_urls))

        # Create a single string of retrieved URLs
        res_phrase = ""
        if len(retrieved_urls) > 0:
            res_phrase = "\n\nThese references might be useful: \n{}".format(" ".join(retrieved_urls))

        # Combine into a single response
        ai_response = "{} {}".format(bot.ai_response, res_phrase)

        st.text("\n")
        st.text(ai_response)
        st.divider()

        t2 = str(datetime.now()).replace(" ", "_")

        dict_ = {
            "n" : str(i+1),
            "qn" : qn,
            "question" : question,
            "user_input" : user_input,
            # "prompt" : prompt,
            "response" : ai_response,
            "start" : t1,
            "stop" : t2,
            "hash" : h1,
            "version" : bot.version
        }
        data.append(dict_)

    json_str = json.dumps(data)

    name_json = "log_testing-c3pa-" + dt + ".json"
    out_path = os.path.join(app_path, name_json)
    # out_path = os.path.join(testing_path, name_json)
    with open(out_path, "w") as f:
        f.write(json_str)

    update_count("test_count.txt", len(questions))


st.sidebar.success("Select an option above.")

