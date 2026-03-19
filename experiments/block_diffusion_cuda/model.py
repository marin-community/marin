from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig
from .layers import AdaLN, AttentionMixer, FeedForward, HybridSequenceMixer, RMSNorm, TimestepEmbedding


def build_attention_mask(*, context_len: int, block_len: int, device: torch.device) -> torch.Tensor:
    total = context_len + block_len
    q_idx = torch.arange(total, device=device)[:, None]
    k_idx = torch.arange(total, device=device)[None, :]
    mask = torch.ones(total, total, dtype=torch.bool, device=device)
    if context_len > 0:
        context_rows = q_idx < context_len
        context_causal = k_idx <= q_idx
        mask = torch.where(context_rows, context_causal, mask)
    return mask.unsqueeze(0).unsqueeze(0)


class DenoiserLayer(nn.Module):
    def __init__(self, config: ModelConfig, *, layer_idx: int):
        super().__init__()
        self.variant = config.variant
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        self.ada_ln = AdaLN(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.ffn = FeedForward(config.d_model, config.mlp_ratio, config.dropout)
        self.use_attention = config.variant == "baseline" or ((layer_idx + 1) % config.attention_period == 0)
        if self.use_attention:
            self.mixer = AttentionMixer(config.d_model, config.n_heads, config.dropout)
        else:
            self.mixer = HybridSequenceMixer(config.d_model, config.gdn_heads)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cond: torch.Tensor,
        attn_mask: torch.Tensor | None,
        context_len: int,
    ) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada_ln(cond)
        gate1 = gate1.unsqueeze(1)
        gate2 = gate2.unsqueeze(1)
        h = self.norm1(x)
        h = h * (1.0 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        if self.use_attention:
            mixed = self.mixer(h, attn_mask=attn_mask)
        else:
            mixed = self.mixer(h, context_len=context_len)
        x = x + self.dropout(gate1 * mixed)
        h = self.norm2(x)
        h = h * (1.0 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = x + self.dropout(gate2 * self.ffn(h))
        return x


class BlockDiffusionDenoiser(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.mask_token_id + 1, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.time_emb = TimestepEmbedding(config.d_model, config.timestep_embed_dim)
        self.layers = nn.ModuleList([DenoiserLayer(config, layer_idx=i) for i in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        context_tokens: torch.Tensor,
        noisy_tokens: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        context_len = context_tokens.shape[1]
        block_len = noisy_tokens.shape[1]
        tokens = torch.cat([context_tokens, noisy_tokens], dim=1)
        if tokens.shape[1] > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {tokens.shape[1]} exceeds configured max_seq_len {self.config.max_seq_len}"
            )
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        x = self.token_emb(tokens) + self.pos_emb(positions)
        cond = self.time_emb(timesteps)
        attn_mask = build_attention_mask(context_len=context_len, block_len=block_len, device=tokens.device)
        for layer in self.layers:
            x = layer(x, cond=cond, attn_mask=attn_mask, context_len=context_len)
        x = self.final_norm(x[:, context_len:])
        return self.output_proj(x)
