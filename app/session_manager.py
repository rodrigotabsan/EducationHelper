import streamlit as st

def init_session():
    defaults = {
        'vectorstore': None,
        'qa_chain': None,
        'chat_history': [],
        'document_processed': False
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
