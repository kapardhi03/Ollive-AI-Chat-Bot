#!/usr/bin/env python3
"""
Minimal Qwen2.5-0.5B Chat for HF Spaces
Python 3.11 compatible version
"""

import sys
print(f"Python version: {sys.version}")

try:
    import gradio as gr
    print("✅ Gradio imported successfully")
except Exception as e:
    print(f"❌ Gradio import error: {e}")
    sys.exit(1)

try:
    import torch
    print("✅ PyTorch imported successfully")
except Exception as e:
    print(f"❌ PyTorch import error: {e}")
    sys.exit(1)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("✅ Transformers imported successfully")
except Exception as e:
    print(f"❌ Transformers import error: {e}")
    sys.exit(1)

# Global variables
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    try:
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"Loading {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print("✅ Tokenizer loaded")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def chat_response(message, history):
    """Generate chat response"""
    global model, tokenizer

    if model is None:
        if not load_model():
            return "❌ Model failed to load. Please try again."

    try:
        # Build conversation
        messages = []
        for user_msg, bot_msg in history:
            if user_msg and bot_msg:  # Skip empty messages
                messages.append({"role": "user", "content": str(user_msg)})
                messages.append({"role": "assistant", "content": str(bot_msg)})

        messages.append({"role": "user", "content": str(message)})

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode response
        response = tokenizer.decode(
            outputs[0][len(inputs['input_ids'][0]):],
            skip_special_tokens=True
        ).strip()

        return response if response else "I'm sorry, I couldn't generate a response."

    except Exception as e:
        print(f"Generation error: {e}")
        return f"❌ Error: {str(e)[:100]}..."

# Create Gradio interface
print("Creating Gradio interface...")

with gr.Blocks(
    title="Qwen2.5-0.5B Chat",
    theme=gr.themes.Soft()
) as demo:
    gr.Markdown("# 🤖 Qwen2.5-0.5B Chat Assistant")
    gr.Markdown("*Chat with the Qwen2.5-0.5B-Instruct model*")

    chatbot = gr.Chatbot(
        value=[],
        height=400,
        show_label=False
    )

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="Type your message here...",
            container=False,
            scale=4
        )
        send_btn = gr.Button("Send", scale=1, variant="primary")

    def user_chat(message, chat_history):
        if not message.strip():
            return "", chat_history

        # Add user message to history
        chat_history = chat_history + [[message, ""]]
        return "", chat_history

    def bot_chat(chat_history):
        if not chat_history:
            return chat_history

        # Get user message
        user_message = chat_history[-1][0]

        # Generate bot response
        bot_response = chat_response(user_message, chat_history[:-1])

        # Update chat history
        chat_history[-1][1] = bot_response
        return chat_history

    # Event handlers
    msg_box.submit(user_chat, [msg_box, chatbot], [msg_box, chatbot], queue=False).then(
        bot_chat, chatbot, chatbot, queue=True
    )
    send_btn.click(user_chat, [msg_box, chatbot], [msg_box, chatbot], queue=False).then(
        bot_chat, chatbot, chatbot, queue=True
    )

    # Examples
    gr.Examples(
        examples=[
            "Hello! How are you?",
            "What is artificial intelligence?",
            "Tell me a short story",
            "What can you help me with?"
        ],
        inputs=msg_box
    )

if __name__ == "__main__":
    print("Launching Gradio app...")
    demo.queue(max_size=20).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )