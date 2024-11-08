import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import random
import time

st.title("Welcome to the Tutela")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_name = "whatthemahad/lamini-test"  # Update with correct model path if necessary
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

tokenizer, model = load_model()

# Function to generate text with context from previous conversation
def generate_text_with_context(prompt, context):
    # Combine the prompt with the conversation history to give more context
    conversation = "\n".join([message["content"] for message in context]) + "\nUser: " + prompt
    inputs = tokenizer(conversation, return_tensors="pt", max_length=512, truncation=True)  # Ensure we don't exceed token limit
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Display previous chat history (user and assistant messages)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display assistant response with context
    with st.chat_message("assistant"):
        st.markdown("Thinking...")  # Show initial "thinking" message
        
        # Generate response using the model with context
        response = generate_text_with_context(prompt, st.session_state.messages)
        
        # Display the assistant's response
        st.markdown(response)
        
        # Add assistant's response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
