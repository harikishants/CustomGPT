import warnings
warnings.filterwarnings("ignore")

import torch
import pandas as pd
import streamlit
import gradio as gr

from train import load_gpt
from inference import generate_text
from model import CustomGPT

# Load model once
model, optimizer, epoch, val_loss, hyperparams = load_gpt(
    CustomGPT,
    torch.optim.AdamW,
    file_path='checkpoints/gpt_model_QA_best.pt',
    device='cuda'
)
model.eval()

# History-aware generation function with token slider
def chat(prompt, history, max_tokens):
    # Concatenate history without special tokens
    # full_prompt = ""
    # for user_msg, bot_msg in history:
    #     full_prompt += f"{user_msg}\n{bot_msg}\n"
    # full_prompt += f"{prompt}\n"

    full_prompt = f"<question> {prompt} <answer>"

    # Generate response
    response = generate_text(full_prompt, model, max_new_tokens=max_tokens, temperature=0.9, sampling='prob')

    history.append((prompt, response))
    return "", history

# # Gradio UI
# with gr.Blocks() as demo:
#     gr.Markdown("## 🧠 Chat with Custom GPT")

#     chatbot = gr.Chatbot()
#     msg = gr.Textbox(placeholder="Ask something...")
#     max_tokens_slider = gr.Slider(minimum=10, maximum=300, step=10, value=50, label="Max Tokens to Generate")
#     clear = gr.Button("Clear Chat")

#     state = gr.State([])  # to store conversation history

#     msg.submit(chat, inputs=[msg, state, max_tokens_slider], outputs=[msg, chatbot])
#     clear.click(lambda: ([], ""), outputs=[chatbot, state])

# # Launch the app
# demo.launch()

if __name__ == "__main__":

    task = 'Question-Answering'
    question = 'What is the capital of France?'
    full_prompt = f"<task> {task} <question> {question} <answer>"

    # Generate response
    response = generate_text(full_prompt, model, max_new_tokens=100, temperature=0.9, sampling='prob')

    print(response)
