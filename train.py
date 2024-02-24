import torch
import argparse
from Model.BigramLM import BigramLanguageModel
from config.config import DEVICE, EVAL_INTERVAL, LEARNING_RATE, MAX_ITERS
from data.data import get_data, get_chars
from utils.utils import decode, estimate_loss, get_batch

def train_and_test(train_mode):
    data = get_data()
    VOCAB_SIZE = len(get_chars())

    n = int(0.9 * len(data))  # 90%
    train_data = data[:n].to(device=DEVICE)
    val_data = data[n:].to(device=DEVICE)

    model = BigramLanguageModel(vocab_size=VOCAB_SIZE)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for iter in range(MAX_ITERS):
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data)
            losses_train = losses['train']
            losses_val = losses['val']
            print(
                f"Step {iter}\ntrain loss: {losses_train:.4f}\nval loss: {losses_val:.4f}")

        xb, yb = get_batch('train', train_data, val_data)

        logits, loss = model.forward(idx=xb, targets=yb)
        if (loss != None):
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    if train_mode == 'test':
        context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
        print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test the Bigram Language Model.')
    parser.add_argument('--mode', choices=['train', 'test'], help='Select mode: train or test', required=True)
    args = parser.parse_args()

    train_and_test(args.mode)
