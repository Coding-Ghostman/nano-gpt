import torch
import torch.nn as nn
from torch.nn import functional as F
from Model.Head import MultiHeadAttention

from config.config import BLOCK_SIZE, DEVICE
torch.manual_seed(42)


class FeedForward(nn.Module):
    """A Simple Linear layer followed by a non-linearity ReLU"""

    def __init__(self, n_embed, dropout) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: Communication followed by a computation """

    def __init__(self, n_embed, n_head, dropout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        head_size = n_embed // n_head
        self.sa_head = MultiHeadAttention(n_head, head_size, dropout=dropout)
        self.ffwd = FeedForward(n_embed, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, block_size, n_embed, n_head, n_layer, dropout=0.5):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_head=n_head, dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embed = self.token_embedding_table(idx)  # (B T C)
        position_embed = self.position_embedding_table(
            torch.arange(T, device=DEVICE))

        x = token_embed+position_embed
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B T vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # F.crossentropy expects(minibatch, channels) and minibatch = B*T
            logits = logits.view(B*T, C)
            # F.crossentropy expects(minibatch, channels) and minibatch = B*T
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            # Get Predictions
            logits, loss = self(idx_cond)

            # Focus only on the last time step
            logits = logits[:, -1, :]  # becomes B,C

            # Applt Softmax
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
        return idx
