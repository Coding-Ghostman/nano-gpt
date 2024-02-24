import torch


def get_chars():
    with open('data/tiny-shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    return chars


def get_data():
    from utils.utils import encode
    with open('data/tiny-shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    data = torch.tensor(encode(text), dtype=torch.long)
    return data
