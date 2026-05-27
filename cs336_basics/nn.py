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


class RmsNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        w = torch.ones(d_model, device=device, dtype=dtype)
        self.weights = torch.nn.Parameter(w)

    def forward(self, x):
        sum = torch.mean(x.to(torch.float32) ** 2, dim=-1, keepdim=True)
        std = torch.sqrt(sum + self.eps)
        return x / std * self.weights


def silu(x):
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_ff, d_model, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x):
        y = self.w1(x)
        return self.w2(self.w3(x) * silu(y))
