"""Causal self-attention with fused QKV."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with fused QKV and FlashAttention."""

    def __init__(self, embed_dim, num_heads, dropout=0.1,
                 use_manual_attention=False, block_size=512):
        super().__init__()
        self.n_heads  = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv      = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj      = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_drop = nn.Dropout(dropout)
        self._dropout  = dropout
        self._use_manual = use_manual_attention

        if use_manual_attention:
            self.attn_drop = nn.Dropout(dropout)
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(block_size, block_size)).view(
                    1, 1, block_size, block_size
                ),
                persistent=True,
            )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)          # each (B, H, T, D)

        if self._use_manual:
            att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att.float(), dim=-1).to(x.dtype)
            att = self.attn_drop(att)
            out = (att @ v).transpose(1, 2).reshape(B, T, C)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self._dropout if self.training else 0.0,
                is_causal=True,
            )
            out = out.transpose(1, 2).reshape(B, T, C)

        return self.proj_drop(self.proj(out))
