from turtle import forward
import torch
import torch.nn as nn
from torch.nn import functional as F

from config.config import BLOCK_SIZE, N_EMBED


class Head(nn.Module):
    """A Single Head of the Attention Mechanism

    Args:
        nn (_type_): _description_
    """

    def __init__(self, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Compute Attention Scores i.e. The Affinity
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # Perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """ Multiple Heads of Self Attention in parallel"""

    def __init__(self, num_heads, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
