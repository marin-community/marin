# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A reshard-free, plain-JAX grug-MoE-flavored transformer for pipeline parity.

This is a deliberately legible re-implementation of the grug MoE architecture
(``experiments/grug/moe/model.py``) with every mesh-coupled construct removed:
no ``jax.sharding.reshard``, no ``shard_map``, no abstract-mesh lookups. The
production model's routing path uses ``reshard`` calls whose axes turn ``Manual``
inside the pipeline's ``shard_map``; this module stays pure SPMD-identical JAX so
it can be driven by the zero-bubble pipeline primitive.

Simplifications vs. production grug (intentional, for correctness on CPU):

- Standard softmax-top-k routing with renormalized top-k gates (no QB routing,
  no sharded beta, no dynamic router bias).
- Experts are replicated and dispatched by computing *all* experts and taking a
  gate-weighted sum over the selected ones — exact, no capacity dropping.
- SwiGLU expert MLPs (silu(gate) * up, then down).
- Multi-head attention with rotary embeddings and a manual causal softmax; GQA is
  supported via key/value head expansion.

The module exposes the three SPMD-identical callables the pipeline drives
(:func:`embed_fn`, :func:`stage_fn`, :func:`head_loss_fn`) plus a builder
(:func:`build_pipeline_params`) that stacks per-layer block params into the
``[num_stages, layers_per_stage, ...]`` layout the pipeline expects, and a
non-pipelined :func:`reference_loss` oracle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, Int, PRNGKeyArray

from experiments.grug.moe_zb.pipeline import PipelineModel, PipelineParams

RMS_EPS = 1e-6


@dataclass(frozen=True)
class GrugMoEConfig:
    """Shape/size hyperparameters for the reshard-free grug MoE transformer."""

    vocab_size: int
    hidden_dim: int
    intermediate_dim: int
    num_experts: int
    num_experts_per_token: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    max_seq_len: int
    head_dim: int | None = None
    num_stages: int = 1
    router_z_loss_coef: float = 1e-3
    initializer_std: float = 0.02
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        if self.num_layers % self.num_stages != 0:
            raise ValueError(f"num_layers={self.num_layers} must be divisible by num_stages={self.num_stages}")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads for grouped-query attention")
        if self.num_experts_per_token > self.num_experts:
            raise ValueError("num_experts_per_token must be <= num_experts")

    @property
    def inferred_head_dim(self) -> int:
        """Head dimension, derived from ``hidden_dim / num_heads`` when not set."""
        if self.head_dim is not None:
            return self.head_dim
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim={self.hidden_dim} not divisible by num_heads={self.num_heads}; set head_dim")
        return self.hidden_dim // self.num_heads

    @property
    def layers_per_stage(self) -> int:
        """Number of transformer blocks each pipeline stage owns."""
        return self.num_layers // self.num_stages


# --- Parameter pytrees -------------------------------------------------------
#
# Each is a registered dataclass of plain arrays so jax can flatten it. The
# block params (`BlockParams`) are stacked along a leading `[num_stages,
# layers_per_stage, ...]` axis to form `PipelineParams.stage`.


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AttentionParams:
    """Projection weights for rotary causal multi-head attention."""

    w_q: Float[Array, "D NH"]
    w_k: Float[Array, "D MH"]
    w_v: Float[Array, "D MH"]
    w_o: Float[Array, "NH D"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MoEParams:
    """Router + stacked expert MLP weights (SwiGLU experts)."""

    router: Float[Array, "D E"]
    w_gate: Float[Array, "E D I"]
    w_up: Float[Array, "E D I"]
    w_down: Float[Array, "E I D"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BlockParams:
    """One pre-norm transformer block: attention + MoE FFN, each with a residual."""

    rms_attn: Float[Array, " D"]
    attn: AttentionParams
    rms_mlp: Float[Array, " D"]
    moe: MoEParams


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EmbedParams:
    """Token embedding table plus its post-embed RMSNorm weight."""

    token_embed: Float[Array, "V D"]
    embed_norm: Float[Array, " D"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class HeadParams:
    """Final RMSNorm weight plus the output projection."""

    final_norm: Float[Array, " D"]
    output_proj: Float[Array, "D V"]


# --- Primitive ops -----------------------------------------------------------


def rms_norm(x: Float[Array, "... D"], weight: Float[Array, " D"]) -> Float[Array, "... D"]:
    """Parametric RMS norm over the last axis, computed in float32."""
    dtype = x.dtype
    x = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed = x * jax.lax.rsqrt(variance + RMS_EPS)
    return (normed * weight.astype(jnp.float32)).astype(dtype)


def _rotary_cache(seq_len: int, head_dim: int, theta: float) -> tuple[Float[Array, "S Dh"], Float[Array, "S Dh"]]:
    """Return (cos, sin) of shape ``[seq_len, head_dim/2]`` for rotary embeddings."""
    half_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim))
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    angles = positions[:, None] * inv_freq[None, :]
    return jnp.cos(angles), jnp.sin(angles)


def _apply_rotary(x: Float[Array, "B S H Dh"], cos: Array, sin: Array) -> Float[Array, "B S H Dh"]:
    """Apply split-half rotary position embeddings to ``x`` (grug convention)."""
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def _expand_kv_heads(x: Float[Array, "B S Hkv Dh"], num_q_heads: int) -> Float[Array, "B S Hq Dh"]:
    """Broadcast grouped-query KV heads up to the number of query heads."""
    num_kv_heads = x.shape[2]
    if num_q_heads == num_kv_heads:
        return x
    repeat = num_q_heads // num_kv_heads
    expanded = jnp.expand_dims(x, axis=3)
    tiled = jnp.broadcast_to(expanded, (*x.shape[:3], repeat, x.shape[3]))
    return tiled.reshape(*x.shape[:2], num_q_heads, x.shape[3])


def attention(params: AttentionParams, x: Float[Array, "B S D"], cfg: GrugMoEConfig) -> Float[Array, "B S D"]:
    """Rotary causal multi-head (grouped-query) attention with a manual softmax."""
    head_dim = cfg.inferred_head_dim
    seq_len = x.shape[1]
    n, m = cfg.num_heads, cfg.num_kv_heads

    q = jnp.einsum("bsd,dk->bsk", x, params.w_q).reshape(*x.shape[:2], n, head_dim)
    k = jnp.einsum("bsd,dk->bsk", x, params.w_k).reshape(*x.shape[:2], m, head_dim)
    v = jnp.einsum("bsd,dk->bsk", x, params.w_v).reshape(*x.shape[:2], m, head_dim)

    cos, sin = _rotary_cache(seq_len, head_dim, cfg.rope_theta)
    q = _apply_rotary(q, cos, sin)
    k = _apply_rotary(k, cos, sin)

    k = _expand_kv_heads(k, n)
    v = _expand_kv_heads(v, n)

    scale = 1.0 / math.sqrt(head_dim)
    scores = jnp.einsum("bqhd,bkhd->bhqk", q * scale, k)
    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    scores = jnp.where(causal[None, None, :, :], scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1)
    attn_out = jnp.einsum("bhqk,bkhd->bqhd", weights, v)
    attn_out = attn_out.reshape(*x.shape[:2], n * head_dim)
    return jnp.einsum("bsk,kd->bsd", attn_out, params.w_o)


def moe_ffn(params: MoEParams, x: Float[Array, "B S D"], cfg: GrugMoEConfig) -> tuple[Float[Array, "B S D"], Array]:
    """Softmax-top-k MoE FFN with SwiGLU experts; returns ``(output, z_loss)``.

    All experts are evaluated and combined as a gate-weighted sum over the
    selected top-k (exact dispatch, no capacity dropping). ``z_loss`` is the
    router z-loss ``mean(logsumexp(router_logits)**2)``.
    """
    b, s, d = x.shape
    x_flat = x.reshape(b * s, d)

    router_logits = jnp.einsum("td,de->te", x_flat, params.router).astype(jnp.float32)
    router_probs = jax.nn.softmax(router_logits, axis=-1)

    top_gates, top_idx = jax.lax.top_k(router_probs, cfg.num_experts_per_token)
    top_gates = top_gates / (jnp.sum(top_gates, axis=-1, keepdims=True) + 1e-9)

    # one-hot [T, K, E] of selected experts, weighted by the renormalized gate, then
    # summed over K -> per-token per-expert combine weight [T, E] (0 for unselected).
    select = jax.nn.one_hot(top_idx, cfg.num_experts, dtype=x.dtype)
    combine = jnp.einsum("tk,tke->te", top_gates.astype(x.dtype), select)

    # Evaluate every expert on every token, then mix by `combine`.
    gate = jnp.einsum("td,edi->tei", x_flat, params.w_gate)
    up = jnp.einsum("td,edi->tei", x_flat, params.w_up)
    expert_hidden = jax.nn.silu(gate) * up
    expert_out = jnp.einsum("tei,eid->ted", expert_hidden, params.w_down)
    mixed = jnp.einsum("ted,te->td", expert_out, combine)

    z_loss = jnp.mean(jax.nn.logsumexp(router_logits, axis=-1) ** 2)
    return mixed.reshape(b, s, d), z_loss


def block(params: BlockParams, x: Float[Array, "B S D"], cfg: GrugMoEConfig) -> tuple[Float[Array, "B S D"], Array]:
    """Pre-norm attention residual followed by a pre-norm MoE-FFN residual."""
    x = x + attention(params.attn, rms_norm(x, params.rms_attn), cfg)
    ffn_out, z_loss = moe_ffn(params.moe, rms_norm(x, params.rms_mlp), cfg)
    x = x + ffn_out
    return x, z_loss


# --- Pipeline callables ------------------------------------------------------


def make_embed_fn(cfg: GrugMoEConfig):
    """Build ``embed_fn(embed_params, tokens) -> hidden`` for the pipeline."""

    def embed_fn(embed_params: EmbedParams, tokens: Int[Array, "B S"]) -> Float[Array, "B S D"]:
        hidden = embed_params.token_embed[tokens]
        return rms_norm(hidden, embed_params.embed_norm)

    return embed_fn


def make_stage_fn(cfg: GrugMoEConfig):
    """Build ``stage_fn(stage_params, hidden) -> (hidden, aux)`` for the pipeline.

    ``stage_params`` has a leading ``[layers_per_stage, ...]`` axis (the pipeline
    already squeezed the size-1 stage-shard axis). ``aux`` is this stage's router
    z-loss contribution, scaled so summing it across all stages yields
    ``router_z_loss_coef / num_layers * sum_over_all_layers(z_loss)`` — matching
    the production grug ``next_token_loss`` aux term.
    """
    aux_scale = cfg.router_z_loss_coef / cfg.num_layers

    def stage_fn(stage_params: BlockParams, hidden: Float[Array, "B S D"]) -> tuple[Float[Array, "B S D"], Array]:
        def one_block(h, layer_params):
            h, z_loss = block(layer_params, h, cfg)
            return h, z_loss

        hidden, z_losses = jax.lax.scan(one_block, hidden, stage_params)
        return hidden, aux_scale * jnp.sum(z_losses)

    return stage_fn


def make_head_loss_fn(cfg: GrugMoEConfig):
    """Build ``head_loss_fn(head_params, hidden, tokens) -> scalar`` for the pipeline.

    Computes final RMSNorm + output projection -> logits, then next-token
    cross-entropy. Labels are ``tokens`` shifted left by one; the last position
    has no label and is masked out. The denominator is the fixed constant
    ``(seq_len - 1) * microbatch`` so the reduction matches :func:`reference_loss`
    exactly (a data-dependent count would drift between the two paths).
    """

    def head_loss_fn(
        head_params: HeadParams, hidden: Float[Array, "B S D"], tokens: Int[Array, "B S"]
    ) -> Float[Array, ""]:
        normed = rms_norm(hidden, head_params.final_norm)
        logits = jnp.einsum("bsd,dv->bsv", normed, head_params.output_proj).astype(jnp.float32)

        labels = tokens[:, 1:]
        pred_logits = logits[:, :-1, :]
        log_probs = jax.nn.log_softmax(pred_logits, axis=-1)
        token_ll = jnp.take_along_axis(log_probs, labels[..., None], axis=-1)[..., 0]

        microbatch, seq_len = tokens.shape
        denom = (seq_len - 1) * microbatch
        return -jnp.sum(token_ll) / denom

    return head_loss_fn


def build_model(cfg: GrugMoEConfig) -> PipelineModel:
    """Bundle the three SPMD-identical pipeline callables for ``cfg``."""
    return PipelineModel(
        embed_fn=make_embed_fn(cfg),
        stage_fn=make_stage_fn(cfg),
        head_loss_fn=make_head_loss_fn(cfg),
    )


# --- Initialization ----------------------------------------------------------


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Float[Array, ...]:
    return std * random.truncated_normal(key, -3, 3, shape)


def _init_attention(key: PRNGKeyArray, cfg: GrugMoEConfig) -> AttentionParams:
    k_q, k_k, k_v, k_o = random.split(key, 4)
    d, n, m, h = cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, cfg.inferred_head_dim
    std = cfg.initializer_std
    return AttentionParams(
        w_q=_init_weight(k_q, (d, n * h), std),
        w_k=_init_weight(k_k, (d, m * h), std),
        w_v=_init_weight(k_v, (d, m * h), std),
        w_o=_init_weight(k_o, (n * h, d), std),
    )


def _init_moe(key: PRNGKeyArray, cfg: GrugMoEConfig) -> MoEParams:
    k_router, k_gate, k_up, k_down = random.split(key, 4)
    d, e, i = cfg.hidden_dim, cfg.num_experts, cfg.intermediate_dim
    std = cfg.initializer_std
    return MoEParams(
        router=_init_weight(k_router, (d, e), std),
        w_gate=_init_weight(k_gate, (e, d, i), std),
        w_up=_init_weight(k_up, (e, d, i), std),
        w_down=_init_weight(k_down, (e, i, d), std),
    )


def _init_block(key: PRNGKeyArray, cfg: GrugMoEConfig) -> BlockParams:
    k_attn, k_moe = random.split(key, 2)
    return BlockParams(
        rms_attn=jnp.ones((cfg.hidden_dim,), dtype=jnp.float32),
        attn=_init_attention(k_attn, cfg),
        rms_mlp=jnp.ones((cfg.hidden_dim,), dtype=jnp.float32),
        moe=_init_moe(k_moe, cfg),
    )


def _stack_blocks(blocks: list[BlockParams], num_stages: int, layers_per_stage: int) -> BlockParams:
    """Stack per-layer ``BlockParams`` into a ``[num_stages, layers_per_stage, ...]`` pytree."""

    def stack_leaf(*leaves: Array) -> Array:
        stacked = jnp.stack(leaves, axis=0)
        return stacked.reshape(num_stages, layers_per_stage, *stacked.shape[1:])

    return jax.tree_util.tree_map(stack_leaf, *blocks)


def build_pipeline_params(cfg: GrugMoEConfig, *, key: PRNGKeyArray) -> tuple[PipelineParams, PipelineModel]:
    """Initialize all weights and return ``(PipelineParams, PipelineModel)``.

    ``PipelineParams.stage`` is a ``BlockParams`` pytree whose leaves carry a
    leading ``[num_stages, layers_per_stage, ...]`` axis; ``embed``/``head`` are
    plain replicated pytrees.
    """
    k_embed, k_out, k_blocks = random.split(key, 3)
    block_keys = random.split(k_blocks, cfg.num_layers)
    blocks = [_init_block(block_keys[i], cfg) for i in range(cfg.num_layers)]
    stage = _stack_blocks(blocks, cfg.num_stages, cfg.layers_per_stage)

    embed = EmbedParams(
        token_embed=_init_weight(k_embed, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std),
        embed_norm=jnp.ones((cfg.hidden_dim,), dtype=jnp.float32),
    )
    head = HeadParams(
        final_norm=jnp.ones((cfg.hidden_dim,), dtype=jnp.float32),
        output_proj=_init_weight(k_out, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std),
    )
    params = PipelineParams(embed=embed, stage=stage, head=head)
    return params, build_model(cfg)


def reference_loss(params: PipelineParams, model: PipelineModel, tokens: Int[Array, "B S"], cfg: GrugMoEConfig) -> Array:
    """Non-pipelined ground-truth loss for one microbatch.

    Runs embed -> every block sequentially (summing the per-stage aux exactly as
    ``stage_fn`` does) -> head cross-entropy. Used by the parity check as the
    gradient oracle.
    """
    hidden = model.embed_fn(params.embed, tokens)
    aux_sum = jnp.zeros(())
    for s in range(cfg.num_stages):
        stage_params = jax.tree_util.tree_map(lambda x, s=s: x[s], params.stage)
        hidden, aux = model.stage_fn(stage_params, hidden)
        aux_sum = aux_sum + aux
    ce = model.head_loss_fn(params.head, hidden, tokens)
    return ce + aux_sum
