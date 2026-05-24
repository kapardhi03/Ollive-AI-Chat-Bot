#!/usr/bin/env python3
"""
Simple Qwen2.5-0.5B Gradio Deployment
Minimal version for Hugging Face Spaces
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# Global model variables
model = None
tokenizer = None

def load_model():
    """Load the Qwen2.5-0.5B model"""
    global model, tokenizer

    if model is None:
        print("Loading Qwen2.5-0.5B-Instruct model...")
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("Model loaded successfully!")

def chat_with_qwen(message, history):
    """Chat function for Gradio interface"""
    # Load model on first use
    if model is None:
        load_model()

    # Prepare conversation
    messages = []
    for human_msg, ai_msg in history:
        messages.append({"role": "user", "content": human_msg})
        messages.append({"role": "assistant", "content": ai_msg})
    messages.append({"role": "user", "content": message})

    # Generate response
    try:
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode response
        response = tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )

        return response.strip()

    except Exception as e:
        return f"Error generating response: {str(e)}"

# Create Gradio interface
def create_interface():
    """Create the Gradio chat interface"""

    with gr.Blocks(title="Qwen2.5-0.5B Chat", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🤖 Qwen2.5-0.5B Chat Assistant")
        gr.Markdown("Chat with the Qwen2.5-0.5B-Instruct model")

        chatbot = gr.Chatbot(
            height=500,
            show_label=False,
            container=False
        )

        with gr.Row():
            msg = gr.Textbox(
                placeholder="Type your message here...",
                container=False,
                scale=7
            )
            submit = gr.Button("Send", scale=1, variant="primary")

        # Chat functionality
        def respond(message, chat_history):
            if not message.strip():
                return "", chat_history

            # Get bot response
            bot_message = chat_with_qwen(message, chat_history)

            # Update chat history
            chat_history.append([message, bot_message])

            return "", chat_history

        # Event handlers
        submit.click(respond, [msg, chatbot], [msg, chatbot])
        msg.submit(respond, [msg, chatbot], [msg, chatbot])

        # Examples
        gr.Examples(
            examples=[
                "Hello! How are you?",
                "What is machine learning?",
                "Write a short poem about AI",
                "Explain quantum computing in simple terms"
            ],
            inputs=msg
        )

    return interface

if __name__ == "__main__":
    # Create and launch interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )