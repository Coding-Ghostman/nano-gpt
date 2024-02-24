import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(42)


class BigramLanguageModel(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()

    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, idx, targets=None):
    logits = self.token_embedding_table(idx)  # (BTC)

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
      # Get Predictions
      logits, loss = self(idx)

      # Focus only on the last time step
      logits = logits[:, -1, :]  # becomes B,C

      # Applt Softmax
      probs = F.softmax(logits, dim=-1)

      # Sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)

      # Append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
    return idx
