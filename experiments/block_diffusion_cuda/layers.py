# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_f * rms).to(x.dtype) * self.weight


class TimestepEmbedding(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.hidden_dim // 2
        device = timesteps.device
        freqs = torch.exp(-math.log(10_000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half, 1))
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if emb.shape[-1] < self.hidden_dim:
            emb = F.pad(emb, (0, self.hidden_dim - emb.shape[-1]))
        return self.proj(emb)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float, dropout: float):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaLN(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 6 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, ...]:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.proj(cond).chunk(6, dim=-1)
        return shift1, scale1, gate1, shift2, scale2, gate2


class AttentionMixer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, *, attn_mask: torch.Tensor | None) -> torch.Tensor:
        batch, seq_len, width = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).reshape(batch, seq_len, width)
        return self.out(y)


class DeltaRuleMixer(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.alpha_proj = nn.Linear(d_model, n_heads, bias=True)
        self.beta_proj = nn.Linear(d_model, n_heads, bias=True)
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, *, reverse: bool = False) -> torch.Tensor:
        if x.numel() == 0:
            return x
        residual_input = x
        if reverse:
            x = torch.flip(x, dims=(1,))
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q = F.normalize(q.float(), dim=-1).to(v.dtype)
        k = F.normalize(k.float(), dim=-1).to(v.dtype)
        alpha = torch.sigmoid(self.alpha_proj(x)).transpose(1, 2).to(v.dtype)
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2).to(v.dtype)

        state = x.new_zeros(batch, self.n_heads, self.head_dim, self.head_dim)
        outputs: list[torch.Tensor] = []
        for index in range(seq_len):
            k_t = k[:, :, index, :]
            q_t = q[:, :, index, :]
            v_t = v[:, :, index, :]
            alpha_t = alpha[:, :, index].unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, index].unsqueeze(-1)
            pred = torch.einsum("bhde,bhd->bhe", state, k_t)
            error = v_t - pred
            update = torch.einsum("bhe,bhd->bhde", beta_t * error, k_t)
            state = alpha_t * state + update
            outputs.append(torch.einsum("bhde,bhd->bhe", state, q_t))

        y = torch.stack(outputs, dim=2).transpose(1, 2).reshape(batch, seq_len, -1)
        if reverse:
            y = torch.flip(y, dims=(1,))
        gate = F.silu(self.gate_proj(residual_input))
        y = self.out_norm(y) * gate
        return self.out_proj(y)


class BidirectionalBlockGatedDeltaMixer(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model={d_model} must be even for bidirectional split GDN")
        half = d_model // 2
        half_heads = max(1, n_heads // 2)
        if half % half_heads != 0:
            raise ValueError(f"half width {half} must be divisible by half head count {half_heads}")
        self.forward_mixer = DeltaRuleMixer(half, half_heads)
        self.backward_mixer = DeltaRuleMixer(half, half_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = x.chunk(2, dim=-1)
        return torch.cat(
            [self.forward_mixer(left, reverse=False), self.backward_mixer(right, reverse=True)],
            dim=-1,
        )


class HybridSequenceMixer(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.context_mixer = DeltaRuleMixer(d_model, n_heads)
        self.block_mixer = BidirectionalBlockGatedDeltaMixer(d_model, n_heads)

    def forward(self, x: torch.Tensor, *, context_len: int) -> torch.Tensor:
        context = x[:, :context_len]
        block = x[:, context_len:]
        outputs = []
        if context_len > 0:
            outputs.append(self.context_mixer(context, reverse=False))
        outputs.append(self.block_mixer(block))
        return torch.cat(outputs, dim=1)
