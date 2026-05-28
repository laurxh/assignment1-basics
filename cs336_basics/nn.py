from typing import Any
import torch.nn as nn

import torch


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        w = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(w, std=std, a=-3 * std, b=3 * std)
        self.weight = torch.nn.Parameter(w)

    @property
    def weights(self):
        return self.weight

    def forward(self, x):
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, device=None, dtype=None):
        super().__init__()
        w = torch.empty(vocab_size, d_model, device=device, dtype=dtype)
        nn.init.trunc_normal_(w, a=-3, b=3)
        self.weight = torch.nn.Parameter(w)

    @property
    def weights(self):
        return self.weight

    def forward(self, x):
        return self.weight[x]


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
        self.weight = torch.nn.Parameter(w)

    @property
    def weights(self):
        return self.weight

    def forward(self, x):
        sum = torch.mean(x.to(torch.float32) ** 2, dim=-1, keepdim=True)
        std = torch.sqrt(sum + self.eps)
        return x / std * self.weight


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


class Rope(nn.Module):
    def __init__(self, theta, d_k, device=None, max_seq_len=0):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.device = device
        self.max_seq_len = max_seq_len

    def forward(self, x, token_positions):
        device = x.device if self.device is None else self.device
        beta = self.theta ** (
            (2 * torch.arange(self.d_k // 2, device=device)) / self.d_k
        )
        theta = token_positions[..., None] / beta[None, ...]
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        odd = x[..., 0::2]
        even = x[..., 1::2]
        result = torch.empty_like(x)
        result[..., 0::2] = odd * cos_theta - even * sin_theta
        result[..., 1::2] = odd * sin_theta + even * cos_theta
        return result


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scores = q @ k.transpose(-1, -2) * d_k**-0.5
    if mask is not None:
        scores = torch.masked_fill(scores, ~mask, -float("inf"))
    scores = softmax(scores, dim=-1)
    return scores @ v


class Multihead_Self_Attention(nn.Module):
    def __init__(self, d_model, n_head, apply_rope=False, theta=0, max_seq_len=0):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        assert d_model % n_head == 0
        self.d_head = d_model // n_head
        self.apply_rope = apply_rope
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        if apply_rope:
            self.rope = Rope(theta, self.d_head, max_seq_len=max_seq_len)

    @property
    def proj_q(self):
        return self.q_proj

    @property
    def proj_k(self):
        return self.k_proj

    @property
    def proj_v(self):
        return self.v_proj

    @property
    def proj_o(self):
        return self.output_proj

    def forward(self, x: torch.Tensor, token_position=None):
        B, S, _ = x.shape
        n = self.n_head
        d = self.d_head
        q = self.q_proj(x).view(B, S, n, d).transpose(1, 2)
        k = self.k_proj(x).view(B, S, n, d).transpose(1, 2)
        v = self.v_proj(x).view(B, S, n, d).transpose(1, 2)

        if self.apply_rope:
            if token_position is None:
                token_position = torch.arange(S, device=x.device)
            token_position = token_position.unsqueeze(-2)
            q = self.rope(q, token_position)
            k = self.rope(k, token_position)
        mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=x.device))
        res = (
            scaled_dot_product_attention(q, k, v, mask)
            .transpose(1, 2)
            .contiguous()
            .view(B, S, self.d_model)
        )
        return self.output_proj(res)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, layer_idx=None):
        super().__init__()
        self.attn = Multihead_Self_Attention(
            d_model, num_heads, True, theta, max_seq_len
        )
        self.ln1 = RmsNorm(d_model)
        self.ln2 = RmsNorm(d_model)
        self.ffn = SwiGLU(d_ff, d_model)
        self.layer_idx = layer_idx

    def load_assignment_weights(self, weights):
        self.attn.q_proj.weight.data = weights["attn.q_proj.weight"]
        self.attn.k_proj.weight.data = weights["attn.k_proj.weight"]
        self.attn.v_proj.weight.data = weights["attn.v_proj.weight"]
        self.attn.output_proj.weight.data = weights["attn.output_proj.weight"]
        self.ln1.weight.data = weights["ln1.weight"]
        self.ln2.weight.data = weights["ln2.weight"]
        self.ffn.w1.weight.data = weights["ffn.w1.weight"]
        self.ffn.w2.weight.data = weights["ffn.w2.weight"]
        self.ffn.w3.weight.data = weights["ffn.w3.weight"]
        return self

    def forward(self, x, token_positions=None):
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


Transfomer_Block = TransformerBlock


class Transfoermer_LM(nn.Module):

    def __init__(
        self, vocab_size, context_len, d_model, num_layers, num_heads, d_ff, rope_theta
    ):
        super().__init__()
        self.num_layer = num_layers
        self.embedding = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    max_seq_len=context_len,
                    theta=rope_theta,
                    layer_idx=_,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RmsNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def load_assignment_weights(self, weights):
        self.embedding.weight.data = weights["token_embeddings.weight"]
        layer_weights = [dict() for _ in range(self.num_layer)]
        for key, value in weights.items():
            if key.startswith("layers."):
                parts = key.split(".")
                layer_idx = int(parts[1])
                block_key = ".".join(parts[2:])
                layer_weights[layer_idx][block_key] = value
        for i, layer in enumerate(self.layers):
            layer.load_assignment_weights(layer_weights[i])

        self.ln_final.weight.data = weights["ln_final.weight"]
        self.lm_head.weight.data = weights["lm_head.weight"]
        return self

    def forward(self, x, token_position):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, token_position)
        x = self.ln_final(x)
        return self.lm_head(x)
