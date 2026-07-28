"""
Shared model definitions for Tiny-GPT.
Single source of truth for TinyGPT. All building blocks live in tinygpt.*.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from tinygpt.attention import CausalSelfAttention, LinearAttention
from tinygpt.layers import FeedForward, TransformerBlock


class TinyGPT(nn.Module):
    """Standard Dense GPT model."""

    def __init__(self, vocab_size, block_size, embed_dim, num_heads,
                 num_layers, ffn_dim, dropout=0.1, attention_cls=None,
                 use_manual_attention=False):
        super().__init__()
        self._embed_dim   = embed_dim
        self._vocab_size  = vocab_size
        self._num_layers  = num_layers

        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_dim, dropout,
                             attention_cls=attention_cls,
                             use_manual_attention=use_manual_attention,
                             block_size=block_size)
            for _ in range(num_layers)
        ])
        self.ln_f    = nn.LayerNorm(embed_dim)
        self.head    = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)
        scale = (2 * self._num_layers) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=0.02 * scale)
            nn.init.normal_(block.ffn.w2.weight, mean=0.0, std=0.02 * scale)

    def forward(self, idx, targets=None, use_activation_checkpoint=False):
        B, T = idx.shape
        x = self.drop(
            self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        )

        for block in self.blocks:
            if self.training and use_activation_checkpoint:
                x = grad_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self._vocab_size), targets.view(-1))

        return logits, loss
