import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_heads=2, num_layers=2, context_length=256):
        super().__init__()
        
        self.hparams = {
            'vocab_size': vocab_size,
            'embed_dim': embed_dim,
            'hidden_dim': hidden_dim,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'context_length': context_length,
        }

        self.token_embedding = nn.Embedding(vocab_size, embed_dim) # learnable token embedding
        self.position_embedding = nn.Embedding(context_length, embed_dim) # learnable position embedding

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.lm_head = nn.Linear(embed_dim, vocab_size)


    def forward(self, input_ids):

        batch_size, seq_len = input_ids.size()
        
        pos_ids = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)

        token_embed = self.token_embedding(input_ids) 
        pos_embed = self.position_embedding(pos_ids)
        x = token_embed + pos_embed

        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1).bool() # attention mask
        memory = torch.zeros_like(x) # keep encoder output as zero, as GPT is decoder only model

        output = self.decoder(
            tgt=x, memory=memory, tgt_mask=tgt_mask
        )

        logits = self.lm_head(output)

        return logits
