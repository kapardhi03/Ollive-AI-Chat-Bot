"""
OSS Assistant Streamlit Application
Main interface for the open-source AI assistant
"""

import streamlit as st
import sys
import os
import traceback
from model import OSSModel
from memory import ConversationMemory
from prompts import PromptTemplates

# Model mapping
MODEL_MAPPING = {
    "Llama-3.2-1B-Instruct": "unsloth/Llama-3.2-1B-Instruct",
    "Phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct"
}

@st.cache_resource
def load_model(model_name: str):
    """Load and cache the model"""
    try:
        return OSSModel(model_name)
    except Exception as e:
        st.error(f"Failed to load model {model_name}: {str(e)}")
        return None

def initialize_session():
    """Initialize session state variables"""
    if 'oss_memory' not in st.session_state:
        st.session_state.oss_memory = ConversationMemory()
    if 'oss_messages' not in st.session_state:
        st.session_state.oss_messages = []
    if 'current_model' not in st.session_state:
        st.session_state.current_model = None

def main():
    st.set_page_config(
        page_title="OSS AI Assistant",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Open Source AI Assistant")
    st.markdown("*Powered by open-source models*")

    initialize_session()

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        # Model selection
        model_options = list(MODEL_MAPPING.keys())
        selected_model_name = st.selectbox("Select Model", model_options, index=0)
        selected_model_id = MODEL_MAPPING[selected_model_name]

        # Load model if changed
        if st.session_state.current_model != selected_model_id:
            with st.spinner(f"Loading {selected_model_name}..."):
                st.session_state.current_model = selected_model_id
                st.session_state.oss_model = load_model(selected_model_id)

        # Temperature slider
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)

        # Max tokens
        max_tokens = st.slider("Max Tokens", 50, 500, 200, 50)

        # Model info
        if hasattr(st.session_state, 'oss_model') and st.session_state.oss_model:
            with st.expander("Model Info"):
                model_info = st.session_state.oss_model.get_model_info()
                st.json(model_info)

        # Export conversation
        if st.button("Export Conversation"):
            if st.session_state.oss_memory.conversation_history:
                exported = st.session_state.oss_memory.export_conversation()
                st.download_button(
                    label="Download as JSON",
                    data=exported,
                    file_name="conversation.json",
                    mime="application/json"
                )

        # Clear conversation button
        if st.button("Clear Conversation"):
            st.session_state.oss_messages = []
            st.session_state.oss_memory.clear()
            st.rerun()

    # Check if model is loaded
    if not hasattr(st.session_state, 'oss_model') or st.session_state.oss_model is None:
        st.error("Model not loaded. Please check the sidebar for any error messages.")
        return

    # Display conversation history
    for message in st.session_state.oss_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What can I help you with?"):
        # Add user message to chat
        st.session_state.oss_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.oss_model.generate_response(
                        prompt,
                        st.session_state.oss_memory.get_context(),
                        temperature=temperature,
                        max_tokens=max_tokens
                    )

                    st.markdown(response)

                    # Update memory and messages
                    st.session_state.oss_memory.add_interaction(prompt, response)
                    st.session_state.oss_messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.oss_messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()