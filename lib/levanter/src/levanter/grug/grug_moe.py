# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical compact Grug MoE kernels.

Implementation overview:
- Routing keeps the argsort-grouped dispatch path that emerged as the stable
  default from https://github.com/marin-community/marin/issues/2704 and commit
  89318a910 (and its parent).
- Expert parallelism keeps the ring-style strategy from
  https://github.com/marin-community/marin/issues/2710: token-sharded
  `all_gather` for dispatch, then `psum_scatter` for collection.
- This module intentionally provides functional kernels only; model/module
  wiring lives in the Grug model files.
"""

import math

from collections.abc import Callable
from functools import partial
from typing import TypeAlias

import jax
import jax.numpy as jnp
from haliax.jax_utils import named_call
from jax import shard_map
from jax.sharding import PartitionSpec as P, get_abstract_mesh
from jaxtyping import Array, Float, Int

from haliax.nn.linear import gmm_sharded
from levanter.utils.activation import ActivationFunctionEnum

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


def _prepare_moe_dispatch(
    x: Float[Array, "T D"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
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
    tokens, topk = selected_experts.shape
    expert_ids = selected_experts.reshape(tokens * topk)
    dispatch_weights = combine_weights.reshape(tokens * topk)

    sort_idx = jnp.argsort(expert_ids, axis=0)
    token_ids = jnp.arange(tokens * topk, dtype=jnp.int32) // topk
    token_ids_sort = token_ids[sort_idx]
    x_sort = x[token_ids_sort]
    w_sort = dispatch_weights[sort_idx].astype(x.dtype)
    group_sizes = jnp.bincount(expert_ids, length=num_experts).astype(jnp.int32)
    return x_sort, w_sort, token_ids_sort, group_sizes


def _moe_mlp_local(
    x: Float[Array, "T D"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    moe_w13: Float[Array, "E D I2"],
    moe_w2: Float[Array, "E I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
) -> Float[Array, "T D"]:
    """Per-shard non-EP MoE FFN path with argsort routing + grouped matmul."""
    x_dispatch, w_dispatch, token_dispatch, group_sizes = _prepare_moe_dispatch(
        x,
        selected_experts,
        combine_weights,
        num_experts=num_experts,
    )

    w13_out = gmm_sharded(x_dispatch, moe_w13, group_sizes)
    moe_dim = moe_w2.shape[1]
    gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
    out_dispatch = gmm_sharded(activation_fn(gate) * up, moe_w2, group_sizes)

    return jnp.zeros_like(x).at[token_dispatch].add(out_dispatch * w_dispatch[:, None], mode="drop")


def _batch_spec_from_x(x: jax.Array, mesh: jax.sharding.AbstractMesh | None) -> P:
    sharding = getattr(x, "sharding", None)
    spec = getattr(sharding, "spec", None)
    if spec is not None and len(spec) > 0:
        return P(spec[0])
    return _batch_spec(mesh)


def _moe_mlp_ep_ring_local(
    x_local: Float[Array, "TL D"],
    selected_experts_local: Int[Array, "TL K"],
    combine_weights_local: Float[Array, "TL K"],
    moe_w13_local: Float[Array, "EL D I2"],
    moe_w2_local: Float[Array, "EL I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    report_capacity_overflow: bool,
) -> Float[Array, "TL D"]:
    """Ring-style EP routed path: all-gather dispatch + psum-scatter collect."""
    # #2710 ring EP strategy: gather tokens and their selected-expert routing
    # assignments across expert shards, then psum-scatter back to local tokens.
    x_global = jax.lax.all_gather(x_local, "expert", tiled=True)
    selected_experts_global = jax.lax.all_gather(selected_experts_local, "expert", tiled=True)
    combine_weights_global = jax.lax.all_gather(combine_weights_local, "expert", tiled=True)

    tokens = x_global.shape[0]
    topk = selected_experts_global.shape[1]
    assignments = tokens * topk
    expert_flat = selected_experts_global.reshape(assignments)
    weight_flat = combine_weights_global.reshape(assignments)
    token_flat = jnp.arange(assignments, dtype=jnp.int32) // topk

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
    if report_capacity_overflow:
        dropped = jnp.maximum(local_count - local_capacity, 0)

        def _log_overflow(_):
            jax.debug.print(
                "moe_mlp EP capacity overflow: dropped {dropped} expert assignments "
                "(local_count={local_count}, local_capacity={local_capacity})",
                dropped=dropped,
                local_count=local_count,
                local_capacity=local_capacity,
            )
            return ()

        jax.lax.cond(dropped > 0, _log_overflow, lambda _: (), operand=None)
    valid = jnp.arange(local_capacity, dtype=jnp.int32) < local_count
    valid_weight = valid.astype(jnp.float32)

    token_local = jnp.take(token_sorted, local_idx, axis=0)
    expert_local = jnp.take(expert_sorted, local_idx, axis=0) - expert_start
    weight_local = jnp.take(weight_sorted, local_idx, axis=0)

    x_take = jnp.take(x_global, token_local, axis=0)
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

    out_global = jnp.zeros_like(x_global).at[token_local].add(out_dispatch * weight_dispatch[:, None], mode="drop")
    # #2710 ring EP strategy: collect only this shard's token slice after
    # reducing contributions from experts across the EP mesh.
    return jax.lax.psum_scatter(out_global, "expert", scatter_dimension=0, tiled=True)


@named_call
def moe_mlp(
    x: Float[Array, "T D"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    w_up_gate: Float[Array, "E D I2"],
    w_down: Float[Array, "E I D"],
    *,
    activation: MoeActivation = ActivationFunctionEnum.silu,
    mesh: jax.sharding.AbstractMesh | None = None,
    capacity_factor: float = _DEFAULT_EP_CAPACITY_FACTOR,
    report_capacity_overflow: bool = False,
) -> Float[Array, "T D"]:
    """Functional routed MoE MLP core used by Grug modules and benchmarks.

    This helper handles dispatch/permute/unpermute (+EP collectives) from
    precomputed token-to-expert assignments. Routing logits/top-k selection
    stays in the caller (e.g. model MLP block).

    Set `report_capacity_overflow=True` to emit debug prints when the EP ring
    path drops assignments due to fixed local capacity.
    """
    if mesh is None:
        mesh = get_abstract_mesh()

    if isinstance(activation, ActivationFunctionEnum):
        activation_fn = activation.to_jax_fn()
    else:
        activation_fn = activation

    if x.ndim != 2:
        raise ValueError(f"x must be rank-2 [T, D], got shape={x.shape}")
    if selected_experts.ndim != 2:
        raise ValueError(f"selected_experts must be rank-2 [T, K], got shape={selected_experts.shape}")
    if selected_experts.shape != combine_weights.shape:
        raise ValueError(
            "selected_experts and combine_weights must have identical [T, K] shapes; "
            f"got {selected_experts.shape} vs {combine_weights.shape}"
        )
    if selected_experts.shape[0] != x.shape[0]:
        raise ValueError(
            f"selected_experts/combine_weights token dim ({selected_experts.shape[0]}) must match x token "
            f"dim ({x.shape[0]})"
        )

    num_experts = int(w_up_gate.shape[0])
    if w_down.shape[0] != num_experts:
        raise ValueError(
            f"w_down expert dimension ({w_down.shape[0]}) must match w_up_gate expert dimension ({num_experts})"
        )

    has_expert_axis = _mesh_has_axis(mesh, "expert")
    expert_axis_size = _mesh_axis_size(mesh, "expert")

    if mesh is None or mesh.empty:
        return _moe_mlp_local(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation_fn=activation_fn,
            num_experts=num_experts,
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
                capacity_factor=capacity_factor,
                report_capacity_overflow=report_capacity_overflow,
            ),
            mesh=mesh,
            in_specs=(
                batch_spec,
                batch_spec,
                batch_spec,
                P("expert", None, None),
                P("expert", None, None),
            ),
            out_specs=batch_spec,
            check_vma=False,
        )
        return shard_fn(x, selected_experts, combine_weights, w_up_gate, w_down)

    # Fallback path for no expert axis (or expert axis size 1) keeps routing
    # semantics without EP collectives.
    shard_fn = shard_map(
        partial(
            _moe_mlp_local,
            activation_fn=activation_fn,
            num_experts=num_experts,
        ),
        mesh=mesh,
        in_specs=(
            batch_spec,
            batch_spec,
            batch_spec,
            local_expert_spec,
            local_expert_spec,
        ),
        out_specs=batch_spec,
        check_vma=False,
    )
    return shard_fn(x, selected_experts, combine_weights, w_up_gate, w_down)


__all__ = [
    "MoeActivation",
    "moe_mlp",
]
