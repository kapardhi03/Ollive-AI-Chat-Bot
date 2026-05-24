"""
Frontier Assistant Streamlit Application
Main interface for the frontier model AI assistant (GPT-4, Claude, etc.)
"""

import streamlit as st
import os
try:
    from .api_client import FrontierModelClient
    from .memory import ConversationMemory
    from .prompts import PromptTemplates
except ImportError:
    # Fallback for direct execution
    from api_client import FrontierModelClient
    from memory import ConversationMemory
    from prompts import PromptTemplates

def initialize_session():
    """Initialize session state variables"""
    if 'frontier_client' not in st.session_state:
        st.session_state.frontier_client = FrontierModelClient()
    if 'frontier_memory' not in st.session_state:
        st.session_state.frontier_memory = ConversationMemory()
    if 'frontier_messages' not in st.session_state:
        st.session_state.frontier_messages = []
    if 'api_keys_configured' not in st.session_state:
        st.session_state.api_keys_configured = set()

def load_api_keys_from_env():
    """Load API keys from environment variables"""
    env_keys = {}
    if os.getenv("OPENAI_API_KEY"):
        env_keys["OpenAI"] = os.getenv("OPENAI_API_KEY")
    if os.getenv("ANTHROPIC_API_KEY"):
        env_keys["Anthropic"] = os.getenv("ANTHROPIC_API_KEY")
    if os.getenv("GOOGLE_API_KEY"):
        env_keys["Google"] = os.getenv("GOOGLE_API_KEY")
    if os.getenv("DEEPSEEK_API_KEY"):
        env_keys["DeepSeek"] = os.getenv("DEEPSEEK_API_KEY")
    return env_keys

def main():
    st.set_page_config(
        page_title="Frontier AI Assistant",
        page_icon="🚀",
        layout="wide"
    )

    st.title("🚀 Frontier Model AI Assistant")
    st.markdown("*Powered by frontier models (GPT-4, Claude, Gemini)*")

    initialize_session()

    # Load environment keys
    env_keys = load_api_keys_from_env()

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        # Model selection (confirmed working model first)
        model_options = [
            "gemini-3.1-flash-lite-preview",  # ✅ CONFIRMED WORKING
            "gemini-2.0-flash",
            "gemini-3.1-pro-preview",
            "gpt-4-turbo-preview",
            "gpt-3.5-turbo",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "deepseek-chat"
        ]
        selected_model = st.selectbox("Select Model", model_options,
                                     help="gemini-3.1-flash-lite-preview is confirmed working")

        # Show model status
        if selected_model == "gemini-3.1-flash-lite-preview":
            st.success("✅ This model is confirmed working with your setup")

        # API Key configuration
        st.subheader("API Configuration")

        # Auto-configure from environment if available
        if env_keys:
            st.info(f"Found API keys in environment: {', '.join(env_keys.keys())}")
            for provider, key in env_keys.items():
                if provider not in st.session_state.api_keys_configured:
                    try:
                        # Map provider to model for initialization
                        if provider == "OpenAI" and selected_model.startswith("gpt"):
                            st.session_state.frontier_client.set_api_key(selected_model, key)
                        elif provider == "Anthropic" and selected_model.startswith("claude"):
                            st.session_state.frontier_client.set_api_key(selected_model, key)
                        elif provider == "Google" and selected_model.startswith("gemini"):
                            st.session_state.frontier_client.set_api_key(selected_model, key)
                        elif provider == "DeepSeek" and selected_model.startswith("deepseek"):
                            st.session_state.frontier_client.set_api_key(selected_model, key)
                        st.session_state.api_keys_configured.add(provider)
                    except Exception as e:
                        st.error(f"Failed to configure {provider}: {str(e)}")

        # Manual API key input
        api_key = st.text_input(
            "API Key (if not in environment)",
            type="password",
            help="Enter your API key for the selected model"
        )

        if api_key:
            try:
                st.session_state.frontier_client.set_api_key(selected_model, api_key)
                st.success(f"API key configured for {selected_model}")
            except Exception as e:
                st.error(f"Failed to configure API key: {str(e)}")

        # Temperature slider
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)

        # Max tokens
        max_tokens = st.slider("Max Tokens", 50, 1000, 500, 50)

        # System prompt selection
        system_prompt_options = ["default", "concise", "detailed", "safe"]
        system_prompt_type = st.selectbox("System Prompt", system_prompt_options)

        # Usage stats
        if hasattr(st.session_state.frontier_client, 'get_usage_stats'):
            with st.expander("Usage Statistics"):
                usage_stats = st.session_state.frontier_client.get_usage_stats()
                st.json(usage_stats)

        # Export conversation
        if st.button("Export Conversation"):
            if st.session_state.frontier_memory.conversation_history:
                exported = st.session_state.frontier_memory.export_conversation()
                st.download_button(
                    label="Download as JSON",
                    data=exported,
                    file_name="frontier_conversation.json",
                    mime="application/json"
                )

        # Clear conversation button
        if st.button("Clear Conversation"):
            st.session_state.frontier_messages = []
            st.session_state.frontier_memory.clear()
            st.rerun()

        # Model info
        with st.expander("Model Information"):
            model_info = st.session_state.frontier_client.get_model_info()
            st.json(model_info)

    # Display conversation history
    for message in st.session_state.frontier_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What can I help you with?"):
        # Add user message to chat
        st.session_state.frontier_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.frontier_client.generate_response(
                        model=selected_model,
                        user_input=prompt,
                        conversation_context=st.session_state.frontier_memory.get_context(),
                        temperature=temperature,
                        max_tokens=max_tokens,
                        system_prompt_type=system_prompt_type
                    )

                    st.markdown(response)

                    # Update memory and messages
                    st.session_state.frontier_memory.add_interaction(prompt, response)
                    st.session_state.frontier_messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    st.error(error_message)
                    st.session_state.frontier_messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()