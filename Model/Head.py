import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """A Single Head of the Attention Mechanism

    Args:
        nn (_type_): _description_
    """

    def __init__(self, head_size, n_embed, block_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Compute Attention Scores i.e. The Affinity
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        # Perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out
