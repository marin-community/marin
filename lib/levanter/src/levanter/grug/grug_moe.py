# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import math

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from einops import rearrange
from jax import random, shard_map
from jax.sharding import PartitionSpec as P, get_abstract_mesh, reshard
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, Int, PRNGKeyArray

from haliax.nn.linear import gmm_sharded

from .attention import AttentionMask, RotaryConfig, apply_rotary_embedding, attention
from .loss import fused_linear_softmax_cross_entropy_loss
from .sharding import Pvocab, unshard

_DEFAULT_EP_CAPACITY_FACTOR = 1.25


def _mesh_has_axis(mesh: jax.sharding.AbstractMesh | None, axis_name: str) -> bool:
    if mesh is None or mesh.empty:
        return False
    return axis_name in mesh.shape


def _mesh_axis_size(mesh: jax.sharding.AbstractMesh | None, axis_name: str) -> int:
    if mesh is None or mesh.empty:
        return 1
    return int(mesh.shape.get(axis_name, 1))


def _batch_spec(mesh: jax.sharding.AbstractMesh | None) -> P:
    if _mesh_has_axis(mesh, "expert"):
        return P(("data", "expert"))
    return P(("data",))


@dataclass(frozen=True)
class GrugMoeModelConfig:
    """Hyperparameters for the canonical Grug MoE transformer."""

    vocab_size: int
    hidden_dim: int = 2048
    intermediate_dim: int = 5632
    shared_expert_intermediate_dim: int = 5632
    num_experts: int = 8
    num_experts_per_token: int = 2
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 16
    head_dim: int | None = None
    max_seq_len: int = 4096
    layer_norm_eps: float = 1e-5
    initializer_std: float = 0.02
    rope: RotaryConfig = dataclasses.field(default_factory=RotaryConfig)

    def __post_init__(self) -> None:
        _ = self.inferred_head_dim
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads for grouped-query attention")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.num_experts_per_token <= 0:
            raise ValueError("num_experts_per_token must be positive")
        if self.num_experts_per_token > self.num_experts:
            raise ValueError("num_experts_per_token must be <= num_experts")
        if self.shared_expert_intermediate_dim < 0:
            raise ValueError("shared_expert_intermediate_dim must be non-negative")

    @property
    def inferred_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} is not divisible by num_heads={self.num_heads}; set head_dim explicitly"
            )
        return self.hidden_dim // self.num_heads


@register_dataclass
@dataclass(frozen=True)
class GrugMoeAttentionParams:
    w_q: jax.Array
    w_k: jax.Array
    w_v: jax.Array
    w_o: jax.Array


@register_dataclass
@dataclass(frozen=True)
class GrugMoeBlockParams:
    attn: GrugMoeAttentionParams
    rms_attn: jax.Array
    rms_mlp: jax.Array
    moe_router: jax.Array
    moe_w13: jax.Array
    moe_w2: jax.Array
    shared_w13: jax.Array
    shared_w2: jax.Array


@register_dataclass
@dataclass(frozen=True)
class GrugMoeModelParameters:
    token_embed: jax.Array
    output_proj: jax.Array
    blocks: tuple[GrugMoeBlockParams, ...]
    final_norm: jax.Array


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Float[Array, "..."]:
    return std * random.truncated_normal(key, -3, 3, shape)


@partial(jax.jit, static_argnames=("cfg",))
def init_parameters(cfg: GrugMoeModelConfig, *, key: PRNGKeyArray) -> GrugMoeModelParameters:
    """Initialize MoE parameters for Grug."""
    mesh = get_abstract_mesh()
    head_dim = cfg.inferred_head_dim
    key, embed_key, out_key = random.split(key, 3)
    layer_keys = random.split(key, cfg.num_layers)

    expert_axis_size = _mesh_axis_size(mesh, "expert")
    if cfg.num_experts % expert_axis_size != 0:
        raise ValueError(f"num_experts={cfg.num_experts} must be divisible by expert axis size={expert_axis_size}")

    expert_param_spec = P("expert", None, None) if _mesh_has_axis(mesh, "expert") else P(None, None, None)

    token_embed = reshard(_init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), Pvocab)
    output_proj = reshard(_init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std), Pvocab)
    final_norm = reshard(jnp.ones((cfg.hidden_dim,), dtype=jnp.float32), P(None))

    blocks: list[GrugMoeBlockParams] = []
    d, n, m, h = cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, head_dim
    e, i, j = cfg.num_experts, cfg.intermediate_dim, cfg.shared_expert_intermediate_dim
    for layer_key in layer_keys:
        k_q, k_k, k_v, k_o, k_router, k_w13, k_w2, k_shared13, k_shared2 = random.split(layer_key, 9)
        attn = GrugMoeAttentionParams(
            w_q=reshard(_init_weight(k_q, (d, n * h), cfg.initializer_std), P("data", "model")),
            w_k=reshard(_init_weight(k_k, (d, m * h), cfg.initializer_std), P("data", "model")),
            w_v=reshard(_init_weight(k_v, (d, m * h), cfg.initializer_std), P("data", "model")),
            w_o=reshard(_init_weight(k_o, (n * h, d), cfg.initializer_std), P("model", "data")),
        )
        block = GrugMoeBlockParams(
            attn=attn,
            rms_attn=jnp.ones((d,), dtype=jnp.float32),
            rms_mlp=jnp.ones((d,), dtype=jnp.float32),
            moe_router=reshard(_init_weight(k_router, (d, e), cfg.initializer_std), P(None, None)),
            moe_w13=reshard(_init_weight(k_w13, (e, d, 2 * i), cfg.initializer_std), expert_param_spec),
            moe_w2=reshard(_init_weight(k_w2, (e, i, d), cfg.initializer_std), expert_param_spec),
            shared_w13=reshard(_init_weight(k_shared13, (d, 2 * j), cfg.initializer_std), P(None, None)),
            shared_w2=reshard(_init_weight(k_shared2, (j, d), cfg.initializer_std), P(None, None)),
        )
        blocks.append(block)

    return GrugMoeModelParameters(
        token_embed=token_embed,
        output_proj=output_proj,
        blocks=tuple(blocks),
        final_norm=final_norm,
    )


def rms_norm(x: Float[Array, "... D"], weight: Float[Array, "D"], eps: float) -> Float[Array, "... D"]:
    weight = unshard(weight)
    dtype = x.dtype
    x = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed = x * jax.lax.rsqrt(variance + eps)
    out = normed * weight
    return out.astype(dtype)


def _prepare_moe_dispatch(
    x_flat: Float[Array, "T D"],
    topk_idx: Int[Array, "T K"],
    topk_weights: Float[Array, "T K"],
    *,
    num_experts: int,
) -> tuple[
    Float[Array, "TK D"],
    Float[Array, "TK"],
    Int[Array, "TK"],
    Int[Array, "E"],
]:
    """Flatten + argsort by expert into grouped layout for GMM."""
    tokens, topk = topk_idx.shape
    expert_ids = topk_idx.reshape(tokens * topk)
    dispatch_weights = topk_weights.reshape(tokens * topk)

    sort_idx = jnp.argsort(expert_ids, axis=0)
    token_ids = jnp.arange(tokens * topk, dtype=jnp.int32) // topk
    token_ids_sort = token_ids[sort_idx]
    x_sort = x_flat[token_ids_sort]
    w_sort = dispatch_weights[sort_idx].astype(x_flat.dtype)
    group_sizes = jnp.bincount(expert_ids, length=num_experts).astype(jnp.int32)
    return x_sort, w_sort, token_ids_sort, group_sizes


def _moe_mlp_local(
    x: Float[Array, "B S D"],
    moe_router: Float[Array, "D E"],
    moe_w13: Float[Array, "E D I2"],
    moe_w2: Float[Array, "E I D"],
    shared_w13: Float[Array, "D J2"],
    shared_w2: Float[Array, "J D"],
    *,
    num_experts: int,
    num_experts_per_token: int,
) -> Float[Array, "B S D"]:
    """Per-shard non-EP MoE FFN path with argsort routing + grouped matmul."""
    b, s, _ = x.shape
    x_flat = rearrange(x, "b s d -> (b s) d")

    router_logits = jnp.einsum("td,de->te", x_flat, moe_router)
    topk_logits, topk_idx = jax.lax.top_k(router_logits, num_experts_per_token)
    topk_weights = jax.nn.softmax(topk_logits, axis=-1).astype(x.dtype)

    x_dispatch, w_dispatch, token_dispatch, group_sizes = _prepare_moe_dispatch(
        x_flat,
        topk_idx,
        topk_weights,
        num_experts=num_experts,
    )

    w13_out = gmm_sharded(x_dispatch, moe_w13, group_sizes)
    moe_dim = moe_w2.shape[1]
    gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
    out_dispatch = gmm_sharded(jax.nn.silu(gate) * up, moe_w2, group_sizes)

    out_flat = jnp.zeros_like(x_flat).at[token_dispatch].add(out_dispatch * w_dispatch[:, None], mode="drop")

    shared_dim = shared_w2.shape[0]
    if shared_dim > 0:
        shared13 = jnp.einsum("td,dm->tm", x_flat, shared_w13)
        shared_gate, shared_up = jnp.split(shared13, [shared_dim], axis=-1)
        out_flat = out_flat + jnp.einsum("tm,md->td", jax.nn.silu(shared_gate) * shared_up, shared_w2)

    return rearrange(out_flat, "(b s) d -> b s d", b=b, s=s)


def _moe_mlp_ep_ring_local(
    x_local: Float[Array, "B S D"],
    moe_router: Float[Array, "D E"],
    moe_w13_local: Float[Array, "EL D I2"],
    moe_w2_local: Float[Array, "EL I D"],
    *,
    num_experts: int,
    num_experts_per_token: int,
    capacity_factor: float,
) -> Float[Array, "B S D"]:
    """Ring-style EP routed path: all-gather dispatch + psum-scatter collect."""
    b, s, _ = x_local.shape
    x_flat_local = rearrange(x_local, "b s d -> (b s) d")
    x_flat_global = jax.lax.all_gather(x_flat_local, "expert", tiled=True)

    router_logits = jnp.einsum("td,de->te", x_flat_global, moe_router)
    topk_logits, topk_idx = jax.lax.top_k(router_logits, num_experts_per_token)
    topk_weights = jax.nn.softmax(topk_logits, axis=-1).astype(x_local.dtype)

    tokens = x_flat_global.shape[0]
    assignments = tokens * num_experts_per_token
    expert_flat = topk_idx.reshape(assignments)
    weight_flat = topk_weights.reshape(assignments)
    token_flat = jnp.arange(assignments, dtype=jnp.int32) // num_experts_per_token

    sort_idx = jnp.argsort(expert_flat, axis=0)
    expert_sorted = jnp.take(expert_flat, sort_idx, axis=0)
    token_sorted = jnp.take(token_flat, sort_idx, axis=0)
    weight_sorted = jnp.take(weight_flat, sort_idx, axis=0).astype(x_local.dtype)

    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
        )

    ep_size = num_experts // local_experts
    local_capacity = int(math.ceil(capacity_factor * assignments / ep_size))
    local_capacity = max(local_experts, local_capacity)

    expert_axis = jax.lax.axis_index("expert")
    expert_start = expert_axis * local_experts
    expert_end = expert_start + local_experts
    local_mask = jnp.logical_and(expert_sorted >= expert_start, expert_sorted < expert_end)

    local_idx = jnp.nonzero(local_mask, size=local_capacity, fill_value=0)[0]
    local_count = jnp.sum(local_mask, dtype=jnp.int32)
    valid = jnp.arange(local_capacity, dtype=jnp.int32) < local_count
    valid_i32 = valid.astype(jnp.int32)

    token_local = jnp.take(token_sorted, local_idx, axis=0)
    expert_local = jnp.take(expert_sorted, local_idx, axis=0) - expert_start
    weight_local = jnp.take(weight_sorted, local_idx, axis=0)

    x_take = jnp.take(x_flat_global, token_local, axis=0)
    x_dispatch = jnp.where(valid[:, None], x_take, jnp.zeros_like(x_take))
    weight_dispatch = jnp.where(valid, weight_local, jnp.zeros_like(weight_local))
    expert_local = jnp.where(valid, expert_local, 0)

    group_sizes = jnp.bincount(expert_local, weights=valid_i32, length=local_experts).astype(jnp.int32)
    group_sizes = group_sizes.at[0].add(local_capacity - jnp.sum(group_sizes, dtype=jnp.int32))

    w13_out = gmm_sharded(x_dispatch, moe_w13_local, group_sizes)
    moe_dim = moe_w2_local.shape[1]
    gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
    out_dispatch = gmm_sharded(jax.nn.silu(gate) * up, moe_w2_local, group_sizes)

    out_global = (
        jnp.zeros_like(x_flat_global).at[token_local].add(out_dispatch * weight_dispatch[:, None], mode="drop")
    )
    out_flat_local = jax.lax.psum_scatter(out_global, "expert", scatter_dimension=0, tiled=True)
    return rearrange(out_flat_local, "(b s) d -> b s d", b=b, s=s)


def mlp(block: GrugMoeBlockParams, x: Float[Array, "B S D"], cfg: GrugMoeModelConfig) -> Float[Array, "B S D"]:
    """Execute the MoE FFN with EP ring path when an expert mesh axis is available."""
    mesh = get_abstract_mesh()
    has_expert_axis = _mesh_has_axis(mesh, "expert")
    expert_axis_size = _mesh_axis_size(mesh, "expert")
    batch_spec = _batch_spec(mesh)

    if mesh is None or mesh.empty:
        return _moe_mlp_local(
            x,
            block.moe_router,
            block.moe_w13,
            block.moe_w2,
            block.shared_w13,
            block.shared_w2,
            num_experts=cfg.num_experts,
            num_experts_per_token=cfg.num_experts_per_token,
        )

    if has_expert_axis and expert_axis_size > 1:
        if cfg.num_experts % expert_axis_size != 0:
            raise ValueError(f"num_experts={cfg.num_experts} must be divisible by expert axis size={expert_axis_size}")

        shard_fn = shard_map(
            partial(
                _moe_mlp_ep_ring_local,
                num_experts=cfg.num_experts,
                num_experts_per_token=cfg.num_experts_per_token,
                capacity_factor=_DEFAULT_EP_CAPACITY_FACTOR,
            ),
            mesh=mesh,
            in_specs=(
                batch_spec,
                P(None, None),
                P("expert", None, None),
                P("expert", None, None),
            ),
            out_specs=batch_spec,
            check_vma=False,
        )

        routed = shard_fn(x, block.moe_router, block.moe_w13, block.moe_w2)

        shared_dim = block.shared_w2.shape[0]
        if shared_dim > 0:
            x_flat = rearrange(x, "b s d -> (b s) d")
            shared13 = jnp.einsum("td,dm->tm", x_flat, block.shared_w13)
            shared_gate, shared_up = jnp.split(shared13, [shared_dim], axis=-1)
            shared_out_flat = jnp.einsum("tm,md->td", jax.nn.silu(shared_gate) * shared_up, block.shared_w2)
            shared_out = rearrange(shared_out_flat, "(b s) d -> b s d", b=x.shape[0], s=x.shape[1])
            routed = routed + shared_out

        return routed

    shard_fn = shard_map(
        partial(
            _moe_mlp_local,
            num_experts=cfg.num_experts,
            num_experts_per_token=cfg.num_experts_per_token,
        ),
        mesh=mesh,
        in_specs=(
            batch_spec,
            P(None, None),
            P(None, None, None),
            P(None, None, None),
            P(None, None),
            P(None, None),
        ),
        out_specs=batch_spec,
        check_vma=False,
    )

    return shard_fn(
        x,
        block.moe_router,
        block.moe_w13,
        block.moe_w2,
        block.shared_w13,
        block.shared_w2,
    )


def _transformer_hidden(
    params: GrugMoeModelParameters,
    token_ids: Int[Array, "B S"],
    cfg: GrugMoeModelConfig,
    *,
    mask: AttentionMask | jax.Array | None,
) -> Float[Array, "B S D"]:
    mesh = get_abstract_mesh()
    batch_spec = _batch_spec(mesh)
    head_dim = cfg.inferred_head_dim
    seq_len = token_ids.shape[1]

    if mask is None:
        mask = AttentionMask.causal()

    hidden = params.token_embed.at[token_ids].get(out_sharding=batch_spec)

    for block in params.blocks:
        attn_in = rms_norm(hidden, block.rms_attn, cfg.layer_norm_eps)
        q = rearrange(jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_q), "... (n d) -> ... n d", d=head_dim)
        k = rearrange(jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_k), "... (m d) -> ... m d", d=head_dim)
        v = rearrange(jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_v), "... (m d) -> ... m d", d=head_dim)
        q, k = apply_rotary_embedding(q, k, seq_len=seq_len, head_dim=head_dim, rope=cfg.rope)
        attn_out = attention(q, k, v, mask)
        attn_out = rearrange(attn_out, "... n d -> ... (n d)")
        attn_out = jnp.einsum("bsh,hd->bsd", attn_out, block.attn.w_o, out_sharding=batch_spec)

        hidden = hidden + attn_out
        mlp_in = rms_norm(hidden, block.rms_mlp, cfg.layer_norm_eps)
        hidden = hidden + mlp(block, mlp_in, cfg)

    return rms_norm(hidden, params.final_norm, cfg.layer_norm_eps)


def forward(
    params: GrugMoeModelParameters,
    token_ids: Int[Array, "B S"],
    cfg: GrugMoeModelConfig,
    *,
    mask: AttentionMask | jax.Array | None = None,
) -> Float[Array, "B S V"]:
    batch_spec = _batch_spec(get_abstract_mesh())
    hidden = _transformer_hidden(params, token_ids, cfg, mask=mask)
    return jnp.einsum("bsh,hd->bsd", hidden, params.output_proj, out_sharding=batch_spec)


def activations(
    params: GrugMoeModelParameters,
    token_ids: Int[Array, "B S"],
    cfg: GrugMoeModelConfig,
    *,
    mask: AttentionMask | jax.Array | None = None,
) -> Float[Array, "B S D"]:
    return _transformer_hidden(params, token_ids, cfg, mask=mask)


def loss_fn(
    params: GrugMoeModelParameters,
    token_ids: Int[Array, "B S"],
    loss_weight: Float[Array, "B S"],
    cfg: GrugMoeModelConfig,
    *,
    mask: AttentionMask | jax.Array | None = None,
    reduction: str = "mean",
    logsumexp_weight: float | None = None,
    loss_dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    hidden = _transformer_hidden(params, token_ids, cfg, mask=mask)
    labels = jnp.concatenate([token_ids[:, 1:], token_ids[:, :1] * 0], axis=1).astype(jnp.int32)
    loss_weight = loss_weight.astype(loss_dtype)

    return fused_linear_softmax_cross_entropy_loss(
        hidden,
        params.output_proj,
        labels,
        weight=loss_weight,
        reduction=reduction,
        logsumexp_weight=logsumexp_weight,
        dtype=loss_dtype,
    )


__all__ = [
    "GrugMoeAttentionParams",
    "GrugMoeBlockParams",
    "GrugMoeModelConfig",
    "GrugMoeModelParameters",
    "activations",
    "forward",
    "init_parameters",
    "loss_fn",
]
