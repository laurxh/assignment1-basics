from typing import Any
import torch.nn as nn

import torch


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        w = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(w, std=std, a=-3 * std, b=3 * std)
        self.weights = torch.nn.Parameter(w)

    def forward(self, x):
        return x @ self.weights.T


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, device=None, dtype=None):
        super().__init__()
        w = torch.empty(vocab_size, d_model, device=device, dtype=dtype)
        nn.init.trunc_normal_(w, a=-3, b=3)
        self.weights = torch.nn.Parameter(w)

    def forward(self, x):
        return self.weights[x]


def softmax(x: torch.Tensor, dim: int):
    max_val = torch.max(x, dim=dim, keepdim=True).values
    x = torch.exp(x - max_val)
    sum = torch.sum(x, dim=dim, keepdim=True)
    return x / sum
