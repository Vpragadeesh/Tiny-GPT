"""Linear attention with elu+1 kernel for O(N) complexity."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttention(nn.Module):
    """Multi-head linear attention with elu+1 kernel."""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.n_heads  = num_heads
        self.head_dim = embed_dim // num_heads

        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_drop = nn.Dropout(dropout)

    @staticmethod
    def feature_map(x):
        return F.elu(x) + 1.0

    def forward(self, x):
        B, T, C = x.shape

        q = self.feature_map(self.q(x).reshape(B, T, self.n_heads, self.head_dim))
        k = self.feature_map(self.k(x).reshape(B, T, self.n_heads, self.head_dim))
        v = self.v(x).reshape(B, T, self.n_heads, self.head_dim)

        kv = torch.zeros(B, self.n_heads, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)
        state_k = torch.zeros(B, self.n_heads, self.head_dim, device=x.device, dtype=x.dtype)
        out = torch.zeros_like(v)

        for t in range(T):
            kv = kv + torch.einsum("bhd,bhe->bhde", k[:, t], v[:, t])
            state_k = state_k + k[:, t]
            num = torch.einsum("bhde,bhd->bhe", kv, q[:, t])
            den = (q[:, t] * state_k).sum(dim=-1, keepdim=True)
            out[:, t] = num / (den + 1e-6)

        out = out.reshape(B, T, C)
        return self.proj_drop(self.proj(out))
