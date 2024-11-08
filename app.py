import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from huggingface_hub import login
import os

hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

st.title("LaMini App")

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model_name = "MBZUAI/LaMini-T5-738M"  # Update with correct model path if necessary
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)  # Use AutoModelForSeq2SeqLM
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

tokenizer, model = load_model()

# Generate text function
def generate_text(prompt):
    if tokenizer and model:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        return "Model failed to load."

# Add a text input for user to enter a prompt
prompt = st.text_area("Enter your prompt:", "")

# If the user enters a prompt, generate and display the response
if prompt:
    with st.spinner("Generating response..."):
        response = generate_text(prompt)
    st.subheader("Generated Text:")
    st.write(response)
