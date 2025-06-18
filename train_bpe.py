import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from tokenizer import BPETokenizer

def train_bpe():
    # corpus = ['Hello World', 'Guten Tag']
    train_df = pd.read_parquet('/home/harikishan/HARIKISHAN/Projects/CustomGPT/data/train-00000-of-00001.parquet')
    print(train_df)
    print(train_df['text'].loc[7])
    exit()
    tokenizer = BPETokenizer(vocab_size=100)
    tokenizer.fit(corpus)
    tokenizer.save_vocab('vocab/myvocab.txt')

if __name__ == "__main__":
    train_bpe()
