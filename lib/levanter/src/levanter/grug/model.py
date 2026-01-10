# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from einops import rearrange
from jax import random
from jax.sharding import PartitionSpec as P, reshard
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, Int, PRNGKeyArray

from .attention import AttentionMask, apply_rotary_embedding, attention
from .config import GrugModelConfig
from .loss import linear_softmax_cross_entropy_loss_and_logz


#### Conventions

# Mesh meanings:
# - "replica_dcn": replica for multislice or similar training setups. We replicate parameters over this axis.
# - "replica": standard data parallel replica axis. We replicate parameters over this axis.
# - "data": data parallel sharding axis. We also shard parameters over this axis.
# - "model": model parallel sharding axis. TP

# Dim names:
# - B = batch
# - D = embedding / hidden dim
# - S = sequence length
# - N = num heads
# - M = num kv heads
# - H = head dim
# - I = intermediate dim

#### PartitionSpecs and sharding

# convenience shorthand for batch sharding.
# if this were Haliax, we'd say {"batch": ("replica_dcn", "replica", "data")}
Pbatch = P(
    ("replica_dcn", "replica", "data"),
)
Pvocab = P(None, None)


def unshard(x: jax.Array) -> jax.Array:
    return reshard(x, P((None,)))


@register_dataclass
@dataclass(frozen=True)
class GrugAttentionParams:
    w_q: jax.Array
    w_k: jax.Array
    w_v: jax.Array
    w_o: jax.Array


@register_dataclass
@dataclass(frozen=True)
class GrugBlockParams:
    attn: GrugAttentionParams
    rms_attn: jax.Array
    rms_mlp: jax.Array
    mlp_gate: jax.Array
    mlp_up: jax.Array
    mlp_down: jax.Array


@register_dataclass
@dataclass(frozen=True)
class GrugModelParameters:
    token_embed: jax.Array
    output_proj: jax.Array
    blocks: tuple[GrugBlockParams, ...]
    final_norm: jax.Array


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Float[Array, "..."]:
    return std * random.truncated_normal(key, -3, 3, shape)


@partial(jax.jit, static_argnames=("cfg",))
def init_parameters(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> GrugModelParameters:
    head_dim = cfg.inferred_head_dim
    keys = random.split(key, 3 + 7 * cfg.num_layers)
    embed_key, out_key, final_norm_key, *rest = keys
    layer_keys = [rest[i * 7 : (i + 1) * 7] for i in range(cfg.num_layers)]

    token_embed = reshard(_init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), Pvocab)
    output_proj = reshard(_init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std), Pvocab)
    final_norm = reshard(jnp.ones((cfg.hidden_dim,), dtype=jnp.float32), P(None))

    blocks: list[GrugBlockParams] = []
    for i in range(cfg.num_layers):
        k_q, k_k, k_v, k_o, k_gate, k_up, k_down = layer_keys[i]
        # extract shape sizes for brevity and consistency
        D, N, M, H, I = cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, head_dim, cfg.intermediate_dim

        attn = GrugAttentionParams(
            w_q=reshard(_init_weight(k_q, (D, N * H), cfg.initializer_std), P("data", "model")),
            w_k=reshard(_init_weight(k_k, (D, M * H), cfg.initializer_std), P("data", "model")),
            w_v=reshard(_init_weight(k_v, (D, M * H), cfg.initializer_std), P("data", "model")),
            w_o=reshard(_init_weight(k_o, (N * H, D), cfg.initializer_std), P("model", "data")),
        )
        mlp_gate = reshard(_init_weight(k_gate, (D, I), cfg.initializer_std), P("data", "model"))
        mlp_up = reshard(_init_weight(k_up, (D, I), cfg.initializer_std), P("data", "model"))
        mlp_down = reshard(_init_weight(k_down, (I, D), cfg.initializer_std), P("model", "data"))
        # keep rms replicated
        rms_attn = jnp.ones((D,), dtype=jnp.float32)
        rms_mlp = jnp.ones((D,), dtype=jnp.float32)

        blocks.append(
            GrugBlockParams(
                attn=attn,
                rms_attn=rms_attn,
                rms_mlp=rms_mlp,
                mlp_gate=mlp_gate,
                mlp_up=mlp_up,
                mlp_down=mlp_down,
            )
        )

    return GrugModelParameters(
        token_embed=token_embed,
        output_proj=output_proj,
        blocks=tuple(blocks),
        final_norm=jnp.ones_like(final_norm),
    )


def rms_norm(x: Float[Array, "... D"], weight: Float[Array, "D"], eps: float) -> Float[Array, "... D"]:
    weight = unshard(weight)
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed = x * jax.lax.rsqrt(variance + eps)
    return normed * weight


def mlp(block: GrugBlockParams, x: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
    gate = jnp.einsum("bsh,hm->bsm", x, block.mlp_gate)
    up = jnp.einsum("bsh,hm->bsm", x, block.mlp_up)
    activated = jax.nn.silu(gate) * up
    return jnp.einsum("bsm,mh->bsh", activated, block.mlp_down, out_sharding=Pbatch)


def _transformer_hidden(
    params: GrugModelParameters,
    token_ids: Int[Array, "B S"],
    cfg: GrugModelConfig,
    *,
    mask: AttentionMask | jax.Array | None,
) -> Float[Array, "B S D"]:
    head_dim = cfg.inferred_head_dim
    seq_len = token_ids.shape[1]

    if mask is None:
        mask = AttentionMask.causal()
    elif isinstance(mask, AttentionMask) and not mask.is_causal:
        mask = dataclasses.replace(mask, is_causal=True)

    hidden = params.token_embed.at[token_ids].get(out_sharding=Pbatch)

    for block in params.blocks:
        attn_in = rms_norm(hidden, block.rms_attn, cfg.layer_norm_eps)
        q = rearrange(jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_q), "... (n d) -> ... n d", d=head_dim)
        k = rearrange(jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_k), "... (m d) -> ... m d", d=head_dim)
        v = rearrange(jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_v), "... (m d) -> ... m d", d=head_dim)
        q, k = apply_rotary_embedding(q, k, seq_len=seq_len, head_dim=head_dim, rope=cfg.rope)
        attn_out = attention(q, k, v, mask)
        attn_out = rearrange(attn_out, "... n d -> ... (n d)")
        attn_out = jnp.einsum("bsh,hd->bsd", attn_out, block.attn.w_o, out_sharding=Pbatch)

        hidden = hidden + attn_out
        mlp_in = rms_norm(hidden, block.rms_mlp, cfg.layer_norm_eps)
        mlp_out = mlp(block, mlp_in)
        hidden = hidden + mlp_out

    hidden = rms_norm(hidden, params.final_norm, cfg.layer_norm_eps)
    return hidden


def forward(
    params: GrugModelParameters,
    token_ids: Int[Array, "B S"],
    cfg: GrugModelConfig,
    *,
    mask: AttentionMask | jax.Array | None = None,
) -> Float[Array, "B S V"]:
    hidden = _transformer_hidden(params, token_ids, cfg, mask=mask)
    logits = jnp.einsum("bsh,hd->bsd", hidden, params.output_proj, out_sharding=Pbatch)
    return logits


def activations(
    params: GrugModelParameters,
    token_ids: Int[Array, "B S"],
    cfg: GrugModelConfig,
    *,
    mask: AttentionMask | jax.Array | None = None,
) -> Float[Array, "B S D"]:
    """Return final hidden states with shape (batch, seq, hidden_dim)."""
    return _transformer_hidden(params, token_ids, cfg, mask=mask)


def loss_fn(
    params: GrugModelParameters,
    token_ids: Int[Array, "B S"],
    loss_weight: Float[Array, "B S"],
    cfg: GrugModelConfig,
    *,
    mask: AttentionMask | jax.Array | None = None,
    reduction: str = "mean",
    logsumexp_weight: float | None = None,
    loss_dtype: jnp.dtype = jnp.float32,
    logit_soft_cap: float | None = None,
) -> jax.Array:
    """Compute next-token cross-entropy loss for a batch.

    This is the "activations vs lm_head" friendly path: it avoids materializing full logits.

    Args:
        params: Model parameters.
        token_ids: Integer array with shape (batch, seq).
        loss_weight: Float array with shape (batch, seq), typically 1 except last position (0).
        cfg: Model config (uses `cfg.cross_entropy_block_size`).
        mask: Optional attention mask spec.
        reduction: One of {"mean", "sum", "none"}.
        logsumexp_weight: Optional z-loss weight (logsumexp^2 term).
        loss_dtype: Accumulator dtype for logsumexp / loss.
        logit_soft_cap: Optional tanh soft cap for logits (applied before exp).

    Returns:
        If reduction=="none": array with shape (batch, seq).
        Else: scalar array.
    """
    hidden = _transformer_hidden(params, token_ids, cfg, mask=mask)
    labels = jnp.concatenate([token_ids[:, 1:], token_ids[:, :1] * 0], axis=1).astype(jnp.int32)
    loss_weight = loss_weight.astype(loss_dtype)

    # NOTE: `block_size=None` corresponds to a single full-vocab block. On the 125M speedrun,
    # disabling blockwise chunking doubled observed MFU (~20 -> ~40). We'll likely need a better
    # large-vocab loss kernel eventually (esp. for sharded vocab / padding weights), but this is
    # good enough for now.
    block_size = cfg.cross_entropy_block_size

    per_pos_loss, logz = linear_softmax_cross_entropy_loss_and_logz(
        hidden,
        params.output_proj,
        labels,
        block_size=block_size,
        dtype=loss_dtype,
        logit_soft_cap=logit_soft_cap,
    )
    per_pos_loss = per_pos_loss.astype(loss_dtype) * loss_weight
    if logsumexp_weight is not None and logsumexp_weight != 0.0:
        per_pos_loss = per_pos_loss + logsumexp_weight * (logz.astype(loss_dtype) ** 2) * loss_weight

    if reduction == "none":
        return per_pos_loss
    if reduction == "sum":
        return jnp.sum(per_pos_loss)
    if reduction == "mean":
        denom = jnp.sum(loss_weight)
        return jnp.sum(per_pos_loss) / jnp.maximum(denom, jnp.array(1.0, dtype=loss_dtype))
    raise ValueError(f"Unknown reduction: {reduction}")


__all__ = [
    "GrugAttentionParams",
    "GrugBlockParams",
    "GrugModelParameters",
    "init_parameters",
    "activations",
    "forward",
    "loss_fn",
]
