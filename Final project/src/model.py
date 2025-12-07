import torch
import torch.nn as nn
from torch.nn import functional as F

class TinyModel(nn.Module):

    def __init__(self, vocab_size, n_embd, n_heads, n_layers, dropout=0.1):
        super().__init__()
        # token -> vector conversion
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(64, n_embd)
        # Simplified network ("brain") for faster training
        self.net = nn.Sequential(nn.Linear(n_embd, n_embd), nn.ReLU(), nn.Linear(n_embd, n_embd), nn.Dropout(dropout))
        # Output Vector --> probability of the next word
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,n_embd)
        x = tok_emb + pos_emb  # (B,T,n_embd)
        x = self.net(x)  # (B,T,n_embd)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss