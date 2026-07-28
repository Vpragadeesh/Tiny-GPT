"""Pre-norm Transformer block: Attention + Dense FFN, with residuals."""

import torch.nn as nn
from tinygpt.attention import CausalSelfAttention
from tinygpt.layers.feedforward import FeedForward


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: Attention + Dense FFN, with residuals."""

    def __init__(self, embed_dim, num_heads, ffn_dim, dropout,
                 attention_cls=None, use_manual_attention=False, block_size=512):
        super().__init__()
        self.ln1  = nn.LayerNorm(embed_dim)
        if attention_cls is not None:
            self.attn = attention_cls(embed_dim, num_heads, dropout)
        else:
            self.attn = CausalSelfAttention(embed_dim, num_heads, dropout,
                                            use_manual_attention=use_manual_attention,
                                            block_size=block_size)
        self.ln2  = nn.LayerNorm(embed_dim)
        self.ffn  = FeedForward(embed_dim, ffn_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
