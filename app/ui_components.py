import streamlit as st

def render_css():
    st.markdown("""<style>... (tu CSS actual aquí) ...</style>""", unsafe_allow_html=True)
