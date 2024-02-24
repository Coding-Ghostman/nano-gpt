import torch
from config.config import BATCH_SIZE, BLOCK_SIZE, EVAL_ITERS


def setup_utils():
    from data.data import get_chars
    chars = get_chars()
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s): return [stoi[c] for c in s]  # string to integers
    def decode(l): return ''.join(itos[i] for i in l)  # integers to string

    return encode, decode


encode, decode = setup_utils()


def get_batch(split, train_data, val_data):
    data = train_data if split == "train" else val_data

    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)

        for k in range(EVAL_ITERS):
            X, Y = get_batch(split, train_data, val_data)

            _, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()
    model.train()
    return out
