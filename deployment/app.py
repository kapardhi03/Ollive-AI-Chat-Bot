import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Starting Qwen2.5-0.5B deployment...")

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print("Model loaded successfully!")

def chat_fn(message, history):
    # Build conversation
    messages = []
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    # Generate response
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return response.strip()

# Create interface
demo = gr.ChatInterface(
    fn=chat_fn,
    title="🤖 Qwen2.5-0.5B Chat",
    description="Chat with Qwen2.5-0.5B-Instruct model",
    examples=["Hello!", "What is AI?", "Write a poem"],
)

if __name__ == "__main__":
    demo.launch()