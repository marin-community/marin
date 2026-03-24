# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from einops import rearrange
from haliax.jax_utils import named_call
from jax import random
from jaxtyping import Array, Float, Int, PRNGKeyArray

from levanter.grug.attention import AttentionMask, apply_rotary_embedding, attention

from experiments.grug.base.model import GrugModelConfig

IsingLmConfig = GrugModelConfig

ISING_TOKENIZER_V0_MODEL = IsingLmConfig(
    vocab_size=256,
    hidden_dim=96,
    intermediate_dim=256,
    num_layers=3,
    num_heads=4,
    num_kv_heads=4,
    max_seq_len=256,
)


class TemperatureConditionedTransformer(eqx.Module):
    """Small grug-style transformer with scalar temperature conditioning."""

    token_embed: jax.Array
    output_proj: jax.Array
    temperature_direction: jax.Array
    blocks: tuple[Block, ...]
    final_norm: RMSNorm
    config: IsingLmConfig = eqx.field(static=True)

    def _resolve_attention_mask(self, mask: AttentionMask | jax.Array | None) -> AttentionMask | jax.Array:
        if mask is None:
            return AttentionMask.causal(sliding_window=self.config.sliding_window)
        if isinstance(mask, AttentionMask) and self.config.sliding_window is not None:
            return mask.with_sliding_window(self.config.sliding_window)
        return mask

    @staticmethod
    def init(cfg: IsingLmConfig, *, key: PRNGKeyArray) -> TemperatureConditionedTransformer:
        embed_key, out_key, temperature_key, *block_keys = random.split(key, cfg.num_layers + 3)
        token_embed = _init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std)
        output_proj = _init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std)
        temperature_direction = _init_weight(temperature_key, (cfg.hidden_dim,), cfg.initializer_std)
        blocks = tuple(Block.init(cfg, key=layer_key) for layer_key in block_keys)
        final_norm = RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps)
        return TemperatureConditionedTransformer(
            token_embed=token_embed,
            output_proj=output_proj,
            temperature_direction=temperature_direction,
            blocks=blocks,
            final_norm=final_norm,
            config=cfg,
        )

    def __call__(
        self,
        token_ids: Int[Array, "B S"],
        temperature: jax.Array,
        mask: AttentionMask | jax.Array | None = None,
    ) -> Float[Array, "B S D"]:
        mask = self._resolve_attention_mask(mask)
        hidden = self.token_embed[token_ids]
        temperature = jnp.asarray(temperature, dtype=hidden.dtype)
        hidden = hidden + temperature[:, None, None] * self.temperature_direction[None, None, :]
        for block in self.blocks:
            hidden = block(hidden, mask)
        return self.final_norm(hidden)

    def logits(
        self,
        token_ids: Int[Array, "B S"],
        temperature: jax.Array,
        mask: AttentionMask | jax.Array | None = None,
    ) -> Float[Array, "B S V"]:
        hidden = self(token_ids, temperature, mask=mask)
        return jnp.einsum("bsh,hd->bsd", hidden, self.output_proj)

    def next_token_loss(
        self,
        token_ids: Int[Array, "B S"],
        loss_weight: Float[Array, "B S"],
        *,
        temperature: jax.Array,
        mask: AttentionMask | jax.Array | None = None,
        reduction: str = "mean",
        logsumexp_weight: float | None = None,
        loss_dtype: jnp.dtype = jnp.float32,
    ) -> jax.Array:
        labels = jnp.concatenate([token_ids[:, 1:], token_ids[:, :1] * 0], axis=1).astype(jnp.int32)
        loss_weight = loss_weight.astype(loss_dtype)
        logits = self.logits(token_ids, temperature, mask=mask).astype(loss_dtype)
        per_token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        if logsumexp_weight is not None:
            log_z = jax.nn.logsumexp(logits, axis=-1)
            per_token_loss = per_token_loss + logsumexp_weight * jnp.square(log_z)
        weighted_loss = per_token_loss * loss_weight
        if reduction == "none":
            return weighted_loss
        total_weight = jnp.sum(loss_weight)
        total_loss = jnp.sum(weighted_loss)
        if reduction == "sum":
            return total_loss
        if reduction == "mean":
            return jnp.where(total_weight > 0, total_loss / total_weight, jnp.zeros_like(total_loss))
        raise ValueError(f"Unknown reduction: {reduction}")


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Float[Array, ...]:
    return std * random.truncated_normal(key, -3, 3, shape)


class CausalSelfAttention(eqx.Module):
    w_q: jax.Array
    w_k: jax.Array
    w_v: jax.Array
    w_o: jax.Array
    cfg: IsingLmConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: IsingLmConfig, *, key: PRNGKeyArray) -> CausalSelfAttention:
        k_q, k_k, k_v, k_o = random.split(key, 4)
        d_model = cfg.hidden_dim
        head_dim = cfg.inferred_head_dim
        return CausalSelfAttention(
            w_q=_init_weight(k_q, (d_model, cfg.num_heads * head_dim), cfg.initializer_std),
            w_k=_init_weight(k_k, (d_model, cfg.num_kv_heads * head_dim), cfg.initializer_std),
            w_v=_init_weight(k_v, (d_model, cfg.num_kv_heads * head_dim), cfg.initializer_std),
            w_o=_init_weight(k_o, (cfg.num_heads * head_dim, d_model), cfg.initializer_std),
            cfg=cfg,
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"], mask: AttentionMask | jax.Array) -> Float[Array, "B S D"]:
        head_dim = self.cfg.inferred_head_dim
        seq_len = x.shape[1]
        q = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_q), "... (n d) -> ... n d", d=head_dim)
        k = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_k), "... (n d) -> ... n d", d=head_dim)
        v = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_v), "... (n d) -> ... n d", d=head_dim)
        q, k = apply_rotary_embedding(q, k, seq_len=seq_len, head_dim=head_dim, rope=self.cfg.rope)
        attn_out = attention(q, k, v, mask)
        attn_out = rearrange(attn_out, "... n d -> ... (n d)")
        return jnp.einsum("bsh,hd->bsd", attn_out, self.w_o)


class MLP(eqx.Module):
    mlp_up: jax.Array
    mlp_down: jax.Array

    @staticmethod
    def init(cfg: IsingLmConfig, *, key: PRNGKeyArray) -> MLP:
        k_up, k_down = random.split(key, 2)
        return MLP(
            mlp_up=_init_weight(k_up, (cfg.hidden_dim, cfg.intermediate_dim), cfg.initializer_std),
            mlp_down=_init_weight(k_down, (cfg.intermediate_dim, cfg.hidden_dim), cfg.initializer_std),
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
        up = jnp.einsum("bsh,hm->bsm", x, self.mlp_up)
        activated = jax.nn.relu(up)
        return jnp.einsum("bsm,mh->bsh", activated, self.mlp_down)


class RMSNorm(eqx.Module):
    weight: jax.Array
    eps: float = eqx.field(static=True)

    @staticmethod
    def init(dim: int, eps: float) -> RMSNorm:
        return RMSNorm(weight=jnp.ones((dim,), dtype=jnp.float32), eps=eps)

    @named_call
    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        dtype = x.dtype
        x = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.eps)
        return (normed * self.weight).astype(dtype)


class Block(eqx.Module):
    rms_attn: RMSNorm
    attn: CausalSelfAttention
    rms_mlp: RMSNorm
    mlp: MLP

    @staticmethod
    def init(cfg: IsingLmConfig, *, key: PRNGKeyArray) -> Block:
        attn_key, mlp_key = random.split(key, 2)
        return Block(
            rms_attn=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            attn=CausalSelfAttention.init(cfg, key=attn_key),
            rms_mlp=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            mlp=MLP.init(cfg, key=mlp_key),
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"], mask: AttentionMask | jax.Array) -> Float[Array, "B S D"]:
        x = x + self.attn(self.rms_attn(x), mask)
        x = x + self.mlp(self.rms_mlp(x))
        return x


__all__ = [
    "ISING_TOKENIZER_V0_MODEL",
    "IsingLmConfig",
    "TemperatureConditionedTransformer",
]
