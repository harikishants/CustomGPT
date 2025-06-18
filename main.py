import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from transformers import get_scheduler
from sklearn.model_selection import train_test_split
import pickle
import time

from tokenizer import BPETokenizer
from dataset import WikiDataset
from model import CustomGPT
from train import train_gpt, load_gpt

if __name__ == "__main__":

    start_time = time.time()

    print('[info] Loading data...')
    tokenizer = BPETokenizer(vocab_file='vocab/vocab_20000')

    # # Salesforce/Wikitext-2
    # train_df = pd.read_parquet('data/Wikitext-2_raw/train-00000-of-00001.parquet')
    # train_df = train_df[train_df['text'].notna()]
    # train_df = train_df[train_df['text'].str.len() >= 100] # removing headings, small sentences

    # val_df = pd.read_parquet('data/Wikitext-2_raw/validation-00000-of-00001.parquet')
    # val_df = val_df[val_df['text'].notna()]
    # val_df = val_df[val_df['text'].str.len() >= 100]

    # train_text = ''.join(train_df['text'].dropna())
    # val_text = ''.join(val_df['text'].dropna())
    
    # # BBC News
    # df = pd.read_csv("data/bbc-news-data.csv", sep='\t')
    # texts = df['content'].tolist()
    # X_train, X_val, y_train, y_val = train_test_split(texts, texts, test_size=0.2, random_state=42)

    # train_text = ''.join(X_train)
    # val_text = ''.join(X_val)

    # tokenization_start_time = time.time()
    # print('[info] Started tokenization...')

    # train_tokens = tokenizer.tokenize(train_text).ids
    # val_tokens = tokenizer.tokenize(val_text).ids

    # with open('data/tokenized/train_tokens_Wiki2.pkl', 'wb') as f:
    #     pickle.dump(train_tokens, f)
    # with open('data/tokenized/val_tokens_Wiki2.pkl', 'wb') as f:
    #     pickle.dump(val_tokens, f)

    # print('[info] Tokenization completed')

    # tokenization_end_time = time.time()
    # print(f"[info] Tokenization completed in {tokenization_end_time-tokenization_start_time:.2f} seconds")

    with open('data/tokenized/train_tokens_Wiki2.pkl', 'rb') as f:
        train_tokens = pickle.load(f)
    with open('data/tokenized/val_tokens_Wiki2.pkl', 'rb') as f:
        val_tokens = pickle.load(f)

    batch_size=10
    context_length = 256
    embed_dim=256
    hidden_dim=1024
    num_heads=4
    num_layers=6
    num_epochs = 50
    scheduler = None
    lr=2e-4
    save_path = 'checkpoints/gpt_model_Wiki2.pt'

    num_chunks = len(train_tokens) // context_length
    train_chunks = [train_tokens[i*context_length:(i+1)*context_length] for i in range(num_chunks)]

    num_chunks = len(val_tokens) // context_length
    val_chunks = [val_tokens[i*context_length:(i+1)*context_length] for i in range(num_chunks)]

    train_dataset = WikiDataset(train_chunks)
    val_dataset = WikiDataset(val_chunks)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    model = CustomGPT(vocab_size = tokenizer.vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers, context_length=context_length)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    
    
 

    # num_training_steps = len(train_loader) * num_epochs
    # # gradually reduction of LR
    # scheduler = get_scheduler(
    #                         name="linear", 
    #                         optimizer=optimizer,
    #                         num_warmup_steps=0,        
    #                         num_training_steps=num_training_steps
    #                         )
    train_start_time = time.time()
    print('[info] Training started...')
    
    train_gpt(model, train_loader, val_loader, optimizer, criterion, num_epochs=num_epochs, device=device, save_path=save_path, start_epoch=1, patience=3, scheduler=scheduler)
    
    

    # model, optimizer, prev_epoch, val_loss, hyperparams = load_gpt(CustomGPT, torch.optim.AdamW, file_path='checkpoints/gpt_model.pt', device=device)
    # train_gpt(model, train_loader, val_loader, optimizer, criterion, num_epochs=num_epochs, device=device, save_path='checkpoints/gpt_model_Wiki.pt', start_epoch=prev_epoch+1, patience=3, scheduler=None)

    train_end_time = time.time()
    print(f"[info] Training completed in {train_end_time-train_start_time:.2f} seconds")
    # print(f"[info] Tokenization completed in {tokenization_end_time-tokenization_start_time:.2f} seconds")
    print(f"[info] Total time taken: {train_end_time-start_time:.2f} seconds")
    print('[info] Done')