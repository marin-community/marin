# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical compact Grug MoE block.

Implementation overview:
- Routing keeps the argsort-grouped dispatch path that emerged as the stable
  default from the https://github.com/marin-community/marin/issues/2704 exploration and the implementation in commit
  89318a910 (and its parent).
- Expert parallelism keeps the ring-style strategy from https://github.com/marin-community/marin/issues/2710: token-sharded
  `all_gather` for dispatch, then `psum_scatter` for collection.
- The same block supports both EP and non-EP meshes, and always adds the
  shared dense expert path when configured.

Historical benchmark context for these choices lives in the development notes
that introduced this implementation (for example, `moe_ep_benchmark.md` and
`bench_moe_hillclimb.py` in commit c1de2c1ac).
"""

import dataclasses
import math

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from haliax.jax_utils import named_call
from jax import random, shard_map
from jax.sharding import PartitionSpec as P, get_abstract_mesh, reshard
from jaxtyping import Array, Float, Int, PRNGKeyArray

from haliax.nn.linear import gmm_sharded
from levanter.utils.activation import ActivationFunctionEnum

from .attention import AttentionMask, RotaryConfig, apply_rotary_embedding, attention
from .loss import fused_linear_softmax_cross_entropy_loss
from .sharding import Pvocab, unshard

_DEFAULT_EP_CAPACITY_FACTOR = 1.25
# #2710 used 1.25 as the practical EP ring default to avoid over/under-packing.

MoeActivation: TypeAlias = ActivationFunctionEnum | Callable[[jax.Array], jax.Array]


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


class CausalSelfAttention(eqx.Module):
    w_q: Float[Array, "D NH"]
    w_k: Float[Array, "D MH"]
    w_v: Float[Array, "D MH"]
    w_o: Float[Array, "NH D"]
    cfg: GrugMoeModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugMoeModelConfig, *, key: PRNGKeyArray) -> "CausalSelfAttention":
        k_q, k_k, k_v, k_o = random.split(key, 4)
        d, n, m, h = cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, cfg.inferred_head_dim
        return CausalSelfAttention(
            w_q=reshard(_init_weight(k_q, (d, n * h), cfg.initializer_std), P("data", "model")),
            w_k=reshard(_init_weight(k_k, (d, m * h), cfg.initializer_std), P("data", "model")),
            w_v=reshard(_init_weight(k_v, (d, m * h), cfg.initializer_std), P("data", "model")),
            w_o=reshard(_init_weight(k_o, (n * h, d), cfg.initializer_std), P("model", "data")),
            cfg=cfg,
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"], mask: AttentionMask | jax.Array) -> Float[Array, "B S D"]:
        head_dim = self.cfg.inferred_head_dim
        seq_len = x.shape[1]
        batch_spec = _batch_spec(get_abstract_mesh())

        q = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_q), "... (n d) -> ... n d", d=head_dim)
        k = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_k), "... (m d) -> ... m d", d=head_dim)
        v = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_v), "... (m d) -> ... m d", d=head_dim)
        q, k = apply_rotary_embedding(q, k, seq_len=seq_len, head_dim=head_dim, rope=self.cfg.rope)
        attn_out = attention(q, k, v, mask)
        attn_out = rearrange(attn_out, "... n d -> ... (n d)")
        return jnp.einsum("bsh,hd->bsd", attn_out, self.w_o, out_sharding=batch_spec)


class RMSNorm(eqx.Module):
    weight: jax.Array
    eps: float = eqx.field(static=True)

    @staticmethod
    def init(dim: int, eps: float) -> "RMSNorm":
        return RMSNorm(weight=jnp.ones((dim,), dtype=jnp.float32), eps=eps)

    @named_call
    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        weight = unshard(self.weight)
        dtype = x.dtype
        x = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.eps)
        return (normed * weight).astype(dtype)


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
    # #2704: keep argsort-grouped dispatch as the canonical compact routing
    # strategy, matching the behavior carried forward from 89318a910.
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
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
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
    out_dispatch = gmm_sharded(activation_fn(gate) * up, moe_w2, group_sizes)

    out_flat = jnp.zeros_like(x_flat).at[token_dispatch].add(out_dispatch * w_dispatch[:, None], mode="drop")
    return rearrange(out_flat, "(b s) d -> b s d", b=b, s=s)


def _shared_dense_mlp(
    x: Float[Array, "B S D"],
    shared_w13: Float[Array, "D J2"],
    shared_w2: Float[Array, "J D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> Float[Array, "B S D"]:
    """Dense shared-expert FFN path that can be fused outside routed MoE dispatch."""
    b, s, _ = x.shape
    x_flat = rearrange(x, "b s d -> (b s) d")
    shared_dim = shared_w2.shape[0]
    shared13 = jnp.einsum("td,dm->tm", x_flat, shared_w13)
    shared_gate, shared_up = jnp.split(shared13, [shared_dim], axis=-1)
    out_flat = jnp.einsum("tm,md->td", activation_fn(shared_gate) * shared_up, shared_w2)
    return rearrange(out_flat, "(b s) d -> b s d", b=b, s=s)


def _batch_spec_from_x(x: jax.Array, mesh: jax.sharding.AbstractMesh | None) -> P:
    sharding = getattr(x, "sharding", None)
    spec = getattr(sharding, "spec", None)
    if spec is not None and len(spec) > 0:
        return P(spec[0])
    return _batch_spec(mesh)


def _moe_mlp_ep_ring_local(
    x_local: Float[Array, "B S D"],
    moe_router: Float[Array, "D E"],
    moe_w13_local: Float[Array, "EL D I2"],
    moe_w2_local: Float[Array, "EL I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    num_experts_per_token: int,
    capacity_factor: float,
) -> Float[Array, "B S D"]:
    """Ring-style EP routed path: all-gather dispatch + psum-scatter collect."""
    b, s, _ = x_local.shape
    x_flat_local = rearrange(x_local, "b s d -> (b s) d")
    # #2710 ring EP strategy: each shard routes against global tokens.
    # NB: this means we receive all tokens on the DP axis, best for low EP.
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
    valid_weight = valid.astype(jnp.float32)

    token_local = jnp.take(token_sorted, local_idx, axis=0)
    expert_local = jnp.take(expert_sorted, local_idx, axis=0) - expert_start
    weight_local = jnp.take(weight_sorted, local_idx, axis=0)

    x_take = jnp.take(x_flat_global, token_local, axis=0)
    x_dispatch = jnp.where(valid[:, None], x_take, jnp.zeros_like(x_take))
    weight_dispatch = jnp.where(valid, weight_local, jnp.zeros_like(weight_local))
    expert_local = jnp.where(valid, expert_local, 0)

    group_sizes = jnp.bincount(expert_local, weights=valid_weight, length=local_experts).astype(jnp.int32)
    # `local_idx` pads by appending invalid rows at the end; keep GMM segment
    # boundaries aligned by attributing padding to the final expert segment.
    group_sizes = group_sizes.at[-1].add(local_capacity - jnp.sum(group_sizes, dtype=jnp.int32))

    w13_out = gmm_sharded(x_dispatch, moe_w13_local, group_sizes)
    moe_dim = moe_w2_local.shape[1]
    gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
    out_dispatch = gmm_sharded(activation_fn(gate) * up, moe_w2_local, group_sizes)

    out_global = (
        jnp.zeros_like(x_flat_global).at[token_local].add(out_dispatch * weight_dispatch[:, None], mode="drop")
    )
    # #2710 ring EP strategy: collect only this shard's token slice after
    # reducing contributions from experts across the EP mesh.
    out_flat_local = jax.lax.psum_scatter(out_global, "expert", scatter_dimension=0, tiled=True)
    return rearrange(out_flat_local, "(b s) d -> b s d", b=b, s=s)


@named_call
def moe_mlp(
    x: Float[Array, "B S D"],
    moe_router: Float[Array, "D E"],
    moe_w13: Float[Array, "E D I2"],
    moe_w2: Float[Array, "E I D"],
    *,
    num_experts_per_token: int,
    activation: MoeActivation = ActivationFunctionEnum.silu,
    mesh: jax.sharding.AbstractMesh | None = None,
    capacity_factor: float = _DEFAULT_EP_CAPACITY_FACTOR,
) -> Float[Array, "B S D"]:
    """Functional routed MoE MLP core used by Grug modules and benchmarks.

    This helper handles local and EP execution, but intentionally excludes the
    shared dense expert path so callers can compose/fuse it separately.
    """
    if mesh is None:
        mesh = get_abstract_mesh()

    if isinstance(activation, ActivationFunctionEnum):
        activation_fn = activation.to_jax_fn()
    else:
        activation_fn = activation

    num_experts = int(moe_w13.shape[0])
    if moe_w2.shape[0] != num_experts:
        raise ValueError(
            f"moe_w2 expert dimension ({moe_w2.shape[0]}) must match moe_w13 expert dimension ({num_experts})"
        )
    if moe_router.shape[1] != num_experts:
        raise ValueError(
            f"moe_router expert dimension ({moe_router.shape[1]}) must match moe_w13 expert dimension ({num_experts})"
        )

    has_expert_axis = _mesh_has_axis(mesh, "expert")
    expert_axis_size = _mesh_axis_size(mesh, "expert")

    if mesh is None or mesh.empty:
        return _moe_mlp_local(
            x,
            moe_router,
            moe_w13,
            moe_w2,
            activation_fn=activation_fn,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
        )

    batch_spec = _batch_spec_from_x(x, mesh)
    local_expert_spec = P("expert", None, None) if has_expert_axis else P(None, None, None)

    if has_expert_axis and expert_axis_size > 1:
        if num_experts % expert_axis_size != 0:
            raise ValueError(f"num_experts={num_experts} must be divisible by expert axis size={expert_axis_size}")

        # #2710: prefer ring EP collectives when a real expert mesh is present.
        shard_fn = shard_map(
            partial(
                _moe_mlp_ep_ring_local,
                activation_fn=activation_fn,
                num_experts=num_experts,
                num_experts_per_token=num_experts_per_token,
                capacity_factor=capacity_factor,
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
        return shard_fn(x, moe_router, moe_w13, moe_w2)

    # Fallback path for no expert axis (or expert axis size 1) keeps routing
    # semantics without EP collectives.
    shard_fn = shard_map(
        partial(
            _moe_mlp_local,
            activation_fn=activation_fn,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
        ),
        mesh=mesh,
        in_specs=(
            batch_spec,
            P(None, None),
            local_expert_spec,
            local_expert_spec,
        ),
        out_specs=batch_spec,
        check_vma=False,
    )
    return shard_fn(x, moe_router, moe_w13, moe_w2)


class MoEMLP(eqx.Module):
    moe_router: jax.Array
    moe_w13: jax.Array
    moe_w2: jax.Array
    shared_w13: jax.Array | None
    shared_w2: jax.Array | None
    cfg: GrugMoeModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugMoeModelConfig, *, key: PRNGKeyArray) -> "MoEMLP":
        k_router, k_w13, k_w2, k_shared13, k_shared2 = random.split(key, 5)
        mesh = get_abstract_mesh()

        expert_axis_size = _mesh_axis_size(mesh, "expert")
        if cfg.num_experts % expert_axis_size != 0:
            raise ValueError(f"num_experts={cfg.num_experts} must be divisible by expert axis size={expert_axis_size}")

        expert_param_spec = P("expert", None, None) if _mesh_has_axis(mesh, "expert") else P(None, None, None)

        d, e, i, j = (
            cfg.hidden_dim,
            cfg.num_experts,
            cfg.intermediate_dim,
            cfg.shared_expert_intermediate_dim,
        )

        shared_w13 = None
        shared_w2 = None
        if j > 0:
            shared_w13 = reshard(_init_weight(k_shared13, (d, 2 * j), cfg.initializer_std), P(None, None))
            shared_w2 = reshard(_init_weight(k_shared2, (j, d), cfg.initializer_std), P(None, None))

        return MoEMLP(
            moe_router=reshard(_init_weight(k_router, (d, e), cfg.initializer_std), P(None, None)),
            moe_w13=reshard(_init_weight(k_w13, (e, d, 2 * i), cfg.initializer_std), expert_param_spec),
            moe_w2=reshard(_init_weight(k_w2, (e, i, d), cfg.initializer_std), expert_param_spec),
            shared_w13=shared_w13,
            shared_w2=shared_w2,
            cfg=cfg,
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
        routed = moe_mlp(
            x,
            self.moe_router,
            self.moe_w13,
            self.moe_w2,
            num_experts_per_token=self.cfg.num_experts_per_token,
            activation=ActivationFunctionEnum.silu,
            mesh=get_abstract_mesh(),
            capacity_factor=_DEFAULT_EP_CAPACITY_FACTOR,
        )

        if self.shared_w13 is None or self.shared_w2 is None:
            return routed

        # Keep shared dense expert fusion outside the routed core so `moe_mlp`
        # stays reusable for callers that only need routed experts.
        shared_out = _shared_dense_mlp(
            x,
            self.shared_w13,
            self.shared_w2,
            activation_fn=jax.nn.silu,
        )
        return routed + shared_out


class Block(eqx.Module):
    rms_attn: RMSNorm
    attn: CausalSelfAttention
    rms_mlp: RMSNorm
    mlp: MoEMLP

    @staticmethod
    def init(cfg: GrugMoeModelConfig, *, key: PRNGKeyArray) -> "Block":
        attn_key, mlp_key = random.split(key, 2)
        return Block(
            rms_attn=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            attn=CausalSelfAttention.init(cfg, key=attn_key),
            rms_mlp=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            mlp=MoEMLP.init(cfg, key=mlp_key),
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"], mask: AttentionMask | jax.Array) -> Float[Array, "B S D"]:
        x = x + self.attn(self.rms_attn(x), mask)
        x = x + self.mlp(self.rms_mlp(x))
        return x


class Transformer(eqx.Module):
    token_embed: jax.Array
    output_proj: jax.Array
    blocks: tuple[Block, ...]
    final_norm: RMSNorm
    config: GrugMoeModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugMoeModelConfig, *, key: PRNGKeyArray) -> "Transformer":
        embed_key, out_key, *block_keys = random.split(key, cfg.num_layers + 2)
        token_embed = reshard(_init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), Pvocab)
        output_proj = reshard(_init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std), Pvocab)
        blocks = tuple(Block.init(cfg, key=layer_key) for layer_key in block_keys)
        final_norm = RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps)

        return Transformer(
            token_embed=token_embed,
            output_proj=output_proj,
            blocks=blocks,
            final_norm=final_norm,
            config=cfg,
        )

    @named_call
    def __call__(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> Float[Array, "B S D"]:
        if mask is None:
            mask = AttentionMask.causal()

        batch_spec = _batch_spec(get_abstract_mesh())
        hidden = self.token_embed.at[token_ids].get(out_sharding=batch_spec)
        for block in self.blocks:
            hidden = eqx.filter_checkpoint(block)(hidden, mask)
        return self.final_norm(hidden)

    @named_call
    def logits(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> Float[Array, "B S V"]:
        batch_spec = _batch_spec(get_abstract_mesh())
        hidden = self(token_ids, mask=mask)
        return jnp.einsum("bsh,hd->bsd", hidden, self.output_proj, out_sharding=batch_spec)

    def next_token_loss(
        self,
        token_ids: Int[Array, "B S"],
        loss_weight: Float[Array, "B S"],
        *,
        mask: AttentionMask | jax.Array | None = None,
        reduction: str = "mean",
        logsumexp_weight: float | None = None,
        loss_dtype: jnp.dtype = jnp.float32,
    ) -> jax.Array:
        """Compute next-token cross-entropy loss for a batch."""
        hidden = self(token_ids, mask=mask)
        labels = jnp.concatenate([token_ids[:, 1:], token_ids[:, :1] * 0], axis=1).astype(jnp.int32)
        loss_weight = loss_weight.astype(loss_dtype)

        return fused_linear_softmax_cross_entropy_loss(
            hidden,
            self.output_proj,
            labels,
            weight=loss_weight,
            reduction=reduction,
            logsumexp_weight=logsumexp_weight,
            dtype=loss_dtype,
        )


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Float[Array, "..."]:
    return std * random.truncated_normal(key, -3, 3, shape)


__all__ = [
    "Block",
    "CausalSelfAttention",
    "GrugMoeModelConfig",
    "MoeActivation",
    "MoEMLP",
    "RMSNorm",
    "Transformer",
    "moe_mlp",
]
