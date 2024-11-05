import streamlit as st
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("whatthemahad/lamini-test")

st.title = "Llama3 Demo"

@st.cache
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("whatthemahad/llama3")
    model = AutoModelForCausalLM.from_pretrained("whatthemahad/llama3")
    return tokenizer, model

tokenizer, model = load_model()

