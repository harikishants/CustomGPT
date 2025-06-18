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
import json
from tqdm import tqdm

from tokenizer import BPETokenizer
from dataset import WikiDataset, collate_fn
from model import CustomGPT
from train import train_gpt, load_gpt


def load_json_data(filename, context_length):
    
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    tokens = []
    for example in tqdm(data):
        t = example['subset'].strip().replace("\n", " ")
        q = example['input'].strip().replace("\n", " ")
        a = example['output'].strip().replace("\n", " ")

        # Create full QA prompt
        full_prompt = f"<task> {t} <question> {q} <answer> {a}"


        # Tokenize the full prompt
        token_ids = tokenizer.tokenize(full_prompt).ids

        if len(token_ids) >= context_length:
            # Need to truncate — preserve beginning (usually helpful in QA)
            token_ids = token_ids[:context_length-1]

        token_ids.append(tokenizer.get_eos_id()) # Add EOS at the end of answer
        tokens.append(token_ids)
        
    return tokens

if __name__ == "__main__":

    start_time = time.time()
    # torch.set_num_threads(6)

    batch_size=10
    context_length = 256
    embed_dim=256
    hidden_dim=1024
    num_heads=4
    num_layers=6
    num_epochs = 50
    scheduler = None
    lr=1e-5
    save_path = 'checkpoints/gpt_model_QA.pt'

    # print('[info] Loading data...')
    print('[info] Started tokenization...')


    tokenizer = BPETokenizer(vocab_file='vocab/vocab_20000')

    # QA
    train_tokens = load_json_data('data/QA/train.json', context_length)
    val_tokens = load_json_data('data/QA/val.json', context_length)

    with open('data/tokenized/train_tokens_QA.pkl', 'wb') as f:
        pickle.dump(train_tokens, f)
    with open('data/tokenized/val_tokens_QA.pkl', 'wb') as f:
        pickle.dump(val_tokens, f)

    print('[info] Tokenization completed')

    # with open('data/tokenized/train_tokens_QA.pkl', 'rb') as f:
    #     train_tokens = pickle.load(f)
    # with open('data/tokenized/val_tokens_QA.pkl', 'rb') as f:
    #     val_tokens = pickle.load(f)

    # all_ids = [id for chunk in val_tokens for id in chunk]
    # print(f"[sanity check] Max token ID in dataset: {max(all_ids)}")
    # print(f"[sanity check] Vocab size: {tokenizer.vocab_size}")

    # all_ids = [id for chunk in train_tokens for id in chunk]
    # print(f"[sanity check] Max token ID in dataset: {max(all_ids)}")
    # print(f"[sanity check] Vocab size: {tokenizer.vocab_size}")


    # print(len(train_tokens[145]))
    # exit()


    train_dataset = WikiDataset(train_tokens)
    val_dataset = WikiDataset(val_tokens)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # model = CustomGPT(vocab_size = tokenizer.vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers, context_length=context_length)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    
    
    num_epochs = 50
    scheduler = None

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
    
    # train_gpt(model, train_loader, val_loader, optimizer, criterion, num_epochs=num_epochs, device=device, save_path='checkpoints/gpt_model_QA.pt', start_epoch=1, patience=3, scheduler=scheduler)

    model, optimizer, prev_epoch, val_loss, hyperparams = load_gpt(model_class=CustomGPT, optimizer_class=torch.optim.AdamW, file_path='checkpoints/gpt_model_Wiki2_best.pt', device=device)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    train_gpt(model, train_loader, val_loader, optimizer, criterion, num_epochs=num_epochs, device=device, save_path=save_path, start_epoch=prev_epoch+1, patience=3, scheduler=None)

    train_end_time = time.time()
    print(f"[info] Training completed in {train_end_time-train_start_time:.2f} seconds")
    # print(f"[info] Tokenization completed in {tokenization_end_time-tokenization_start_time:.2f} seconds")
    print(f"[info] Total time taken: {train_end_time-start_time:.2f} seconds")
    print('[info] Done')

