import streamlit as st
from MCQ import test
from summarizer import Summarizer
import torch
from flashtext import KeywordProcessor
st.set_page_config(page_title='MCQ', layout='wide')

st.title("Multiple Choice Question Generator")

full_text = st.text_area("Enter your text here:-")


button2 = st.button("Generate Questions")

if full_text and button2:
    test(full_text)

