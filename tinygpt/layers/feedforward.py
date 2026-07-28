"""Standard two-layer FFN with GELU."""

import torch.nn as nn


class FeedForward(nn.Module):
    """Standard two-layer FFN with GELU."""

    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.w1   = nn.Linear(embed_dim, ffn_dim)
        self.w2   = nn.Linear(ffn_dim, embed_dim)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w2(self.act(self.w1(x))))
