import warnings
warnings.filterwarnings("ignore")

import torch
import pandas as pd
import streamlit

from train import load_gpt
from inference import generate_text
from model import CustomGPT



if __name__ == "__main__":

    df = pd.read_parquet('data/Wikitext-2_raw/test-00000-of-00001.parquet')

    text = ''.join(df['text'].dropna())
    print(text[:1000])
    exit()

    model, optimizer, epoch, val_loss, hyperparams = load_gpt(CustomGPT, torch.optim.AdamW, file_path='checkpoints/gpt_model_Wiki_best.pt', device='cuda')
    prompt = "Robert Boulter is an English film , television and theatre actor . He had a guest starring role on the television series The Bill in 2000 . This was followed by a starring role in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre ."

    output_text = generate_text(prompt, model, max_new_tokens=100, temperature=0.7, sampling='prob')
    print(output_text)