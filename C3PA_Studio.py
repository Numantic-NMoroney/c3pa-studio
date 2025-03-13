# [2502] [2503] n8
#

import streamlit as st
import os, sys

sys.path.insert(0, "utils")

from utility_functions import get_count, configure_app


mount_path, app_path, cfg = configure_app()

st.image("data/Numantic_Solutions_Logotype_light.png", width=400)

st.title("C3PA Studio")

st.markdown("Supplmental tools for the [California Community Colleges Policy-Assistant](https://ccc-polasst.numanticsolutions.com).")
st.write("This includes : ")

st.markdown("* Minimalist small batch testing, with logging")
st.markdown("* Replay previous testing results")
st.markdown("* Collect optional user feedback")
st.markdown("* Summary dashboard view")

st.write("Select an option to the left to drill down on that topic.")

st.divider()

st.markdown("[CCC-Bot](https://ccc-polasst.numanticsolutions.com) | " +
    " [GitHub](https://github.com/NumanticSolutions/ccc-policy_assistant) | " +
    " [Numantic Solutions](https://numanticsolutions.com)")

# term = os.environ.get('TERM')
# st.write("term : " + str(term))

st.sidebar.success("Select an option above.")

