import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Set the title of the app
st.title("LaMini Demo")

# Load the model and tokenizer
@st.cache_resource
def load_model():
    # Replace with the correct model path for LaMini
    model_name = "whatthemahad/lamini"  # Update this if the model name is different
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Generate text function
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Add a text input for user to enter a prompt
prompt = st.text_area("Enter your prompt:", "")

# If the user enters a prompt, generate and display the response
if prompt:
    with st.spinner("Generating response..."):
        response = generate_text(prompt)
    st.subheader("Generated Text:")
    st.write(response)
