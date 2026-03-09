# Copyright The Levanter Authors
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
import os

from collections.abc import Callable
from functools import partial
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
from haliax.jax_utils import named_call
from jax import shard_map
from jax.sharding import PartitionSpec as P, get_abstract_mesh
from jaxtyping import Array, Float, Int

from haliax.nn.ragged_dot import ragged_dot
from levanter.utils.activation import ActivationFunctionEnum

_DEFAULT_EP_CAPACITY_FACTOR = 1.25
# #2710 used 1.25 as the practical EP ring default to avoid over/under-packing.

MoeActivation: TypeAlias = ActivationFunctionEnum | Callable[[jax.Array], jax.Array]
DispatchImplementation: TypeAlias = Literal[
    "auto", "xla", "sparsecore", "sparsecore_pipeline", "sparsecore_expert_pipeline"
]
_SPARSECORE_PIPELINE_TOKEN_BLOCK = 1024
_SPARSECORE_EXPERT_PIPELINE_CHUNK_EXPERTS = 16
_SPARSECORE_EXPERT_PIPELINE_CAPACITY_FACTOR = 1.25
_SPARSECORE_EXPERT_PIPELINE_CAPACITY_PAD = 256


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None else float(value)


def _use_static_sparsecore_expert_pipeline_routing() -> bool:
    return bool(_env_int("GRUG_SPARSECORE_EXPERT_PIPELINE_STATIC_ROUTING", 0))


def _use_dense_single_expert_pipeline() -> bool:
    return bool(_env_int("GRUG_SPARSECORE_EXPERT_PIPELINE_SINGLE_EXPERT_DENSE", 0))


def _single_expert_token_chunk_size() -> int:
    return _env_int("GRUG_SPARSECORE_EXPERT_PIPELINE_TOKEN_CHUNK_SIZE", 0)


def _use_single_expert_pipeline_barrier() -> bool:
    return bool(_env_int("GRUG_SPARSECORE_EXPERT_PIPELINE_BARRIER", 0))


def _ep_return_implementation() -> str:
    return os.environ.get("GRUG_MOE_EP_RETURN_IMPL", "scatter_psum_scatter")


def _ep_return_owner_bucket_factor() -> float:
    return _env_float("GRUG_MOE_EP_OWNER_BUCKET_FACTOR", 1.25)


def _sparsecore_expert_pipeline_bucket_sizes(chunk_capacity: int) -> tuple[int, ...]:
    sizes = {1, chunk_capacity}
    for numerator in (1, 2, 3, 4, 5, 6, 7):
        sizes.add(max(1, math.ceil(chunk_capacity * numerator / 8)))
    return tuple(sorted(size for size in sizes if size <= chunk_capacity))


def _dense_moe_up_down(
    x_chunk: jax.Array,
    moe_w13_expert: jax.Array,
    moe_w2_expert: jax.Array,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    w13_out = jnp.matmul(x_chunk, moe_w13_expert)
    moe_dim = moe_w2_expert.shape[0]
    gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
    return jnp.matmul(activation_fn(gate) * up, moe_w2_expert)


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


@named_call
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


def _prepare_moe_dispatch_impl(
    x: Float[Array, "T D"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    *,
    num_experts: int,
    implementation: DispatchImplementation,
) -> tuple[
    Float[Array, "TK D"],
    Float[Array, "TK"],
    Int[Array, "TK"],
    Int[Array, "E"],
]:
    if implementation == "xla":
        return _prepare_moe_dispatch(x, selected_experts, combine_weights, num_experts=num_experts)

    if implementation == "sparsecore":
        if jax.default_backend() != "tpu":
            raise ValueError("dispatch_implementation='sparsecore' requires TPU backend")
        from .grug_moe_sparsecore import sparsecore_prepare_dispatch

        return sparsecore_prepare_dispatch(
            x,
            selected_experts,
            combine_weights,
            num_experts=num_experts,
        )

    if implementation in ("sparsecore_pipeline", "sparsecore_expert_pipeline"):
        raise ValueError(f"dispatch_implementation={implementation!r} is only valid at the moe_mlp local path")

    if implementation != "auto":
        raise ValueError(f"Unknown dispatch implementation: {implementation}")

    if jax.default_backend() == "tpu":
        try:
            from .grug_moe_sparsecore import sparsecore_prepare_dispatch

            return sparsecore_prepare_dispatch(
                x,
                selected_experts,
                combine_weights,
                num_experts=num_experts,
            )
        except (NotImplementedError, RuntimeError, ValueError):
            pass

    return _prepare_moe_dispatch(x, selected_experts, combine_weights, num_experts=num_experts)


def _take_rows_impl(
    x: Float[Array, "T D"],
    row_indices: Int[Array, "TK"],
    *,
    implementation: DispatchImplementation,
) -> Float[Array, "TK D"]:
    if implementation == "xla":
        return jnp.take(x, row_indices, axis=0)

    if implementation == "sparsecore":
        if jax.default_backend() != "tpu":
            raise ValueError("dispatch_implementation='sparsecore' requires TPU backend")
        from .grug_moe_sparsecore import sparsecore_row_gather

        return sparsecore_row_gather(x, row_indices)

    if implementation in ("sparsecore_pipeline", "sparsecore_expert_pipeline"):
        raise ValueError(f"dispatch_implementation={implementation!r} is only valid at the moe_mlp local path")

    if implementation != "auto":
        raise ValueError(f"Unknown dispatch implementation: {implementation}")

    if jax.default_backend() == "tpu":
        try:
            from .grug_moe_sparsecore import sparsecore_row_gather

            return sparsecore_row_gather(x, row_indices)
        except (NotImplementedError, RuntimeError, ValueError):
            pass

    return jnp.take(x, row_indices, axis=0)


def _moe_mlp_local(
    x: Float[Array, "T D"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    moe_w13: Float[Array, "E D I2"],
    moe_w2: Float[Array, "E I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    dispatch_implementation: DispatchImplementation = "xla",
) -> tuple[Float[Array, "T D"], Int[Array, ""]]:
    """Per-shard non-EP MoE FFN path with argsort routing + grouped matmul."""
    if dispatch_implementation == "sparsecore_pipeline":
        if jax.default_backend() != "tpu":
            raise ValueError("dispatch_implementation='sparsecore_pipeline' requires TPU backend")
        tokens = int(x.shape[0])
        block_tokens = _env_int("GRUG_SPARSECORE_PIPELINE_TOKEN_BLOCK", _SPARSECORE_PIPELINE_TOKEN_BLOCK)
        if block_tokens <= 0:
            raise ValueError(f"GRUG_SPARSECORE_PIPELINE_TOKEN_BLOCK must be positive, got {block_tokens}")
        if tokens <= block_tokens:
            dispatch_implementation = "sparsecore"
        else:
            pad_tokens = (-tokens) % block_tokens
            if pad_tokens:
                x = jnp.pad(x, ((0, pad_tokens), (0, 0)))
                selected_experts = jnp.pad(selected_experts, ((0, pad_tokens), (0, 0)))
                combine_weights = jnp.pad(combine_weights, ((0, pad_tokens), (0, 0)))

            num_blocks = x.shape[0] // block_tokens
            out_blocks = []
            for block_idx in range(num_blocks):
                start = block_idx * block_tokens
                stop = start + block_tokens
                block_out, _ = _moe_mlp_local(
                    x[start:stop],
                    selected_experts[start:stop],
                    combine_weights[start:stop],
                    moe_w13,
                    moe_w2,
                    activation_fn=activation_fn,
                    num_experts=num_experts,
                    dispatch_implementation="sparsecore",
                )
                out_blocks.append(block_out)

            out = jnp.concatenate(out_blocks, axis=0)[:tokens]
            return out, jnp.array(0, dtype=jnp.int32)

    if dispatch_implementation == "sparsecore_expert_pipeline":
        if jax.default_backend() != "tpu":
            raise ValueError("dispatch_implementation='sparsecore_expert_pipeline' requires TPU backend")

        tokens, topk = selected_experts.shape
        assignments = tokens * topk
        expert_ids = selected_experts.reshape(assignments)
        dispatch_weights = combine_weights.reshape(assignments)
        sort_idx = jnp.argsort(expert_ids, axis=0)
        token_ids_sort = (jnp.arange(assignments, dtype=jnp.int32) // topk)[sort_idx]
        dispatch_weights_sort = dispatch_weights[sort_idx].astype(x.dtype)
        group_sizes = jnp.bincount(expert_ids, length=num_experts).astype(jnp.int32)
        expert_offsets = jnp.cumsum(group_sizes, dtype=jnp.int32) - group_sizes

        chunk_experts = min(
            _env_int("GRUG_SPARSECORE_EXPERT_PIPELINE_CHUNK_EXPERTS", _SPARSECORE_EXPERT_PIPELINE_CHUNK_EXPERTS),
            num_experts,
        )
        if chunk_experts <= 0:
            raise ValueError(f"GRUG_SPARSECORE_EXPERT_PIPELINE_CHUNK_EXPERTS must be positive, got {chunk_experts}")
        num_chunks = math.ceil(num_experts / chunk_experts)
        avg_chunk_assignments = math.ceil(assignments * chunk_experts / num_experts)
        chunk_capacity_factor = _env_float(
            "GRUG_SPARSECORE_EXPERT_PIPELINE_CAPACITY_FACTOR",
            _SPARSECORE_EXPERT_PIPELINE_CAPACITY_FACTOR,
        )
        chunk_capacity_pad = _env_int(
            "GRUG_SPARSECORE_EXPERT_PIPELINE_CAPACITY_PAD",
            _SPARSECORE_EXPERT_PIPELINE_CAPACITY_PAD,
        )
        if chunk_capacity_factor <= 0:
            raise ValueError(
                "GRUG_SPARSECORE_EXPERT_PIPELINE_CAPACITY_FACTOR must be positive, " f"got {chunk_capacity_factor}"
            )
        if chunk_capacity_pad < 0:
            raise ValueError(
                f"GRUG_SPARSECORE_EXPERT_PIPELINE_CAPACITY_PAD must be nonnegative, got {chunk_capacity_pad}"
            )
        chunk_capacity = int(math.ceil(avg_chunk_assignments * chunk_capacity_factor)) + chunk_capacity_pad
        single_expert_dense = _use_dense_single_expert_pipeline()
        single_expert_token_chunk = _single_expert_token_chunk_size()
        if single_expert_token_chunk < 0:
            raise ValueError(
                "GRUG_SPARSECORE_EXPERT_PIPELINE_TOKEN_CHUNK_SIZE must be nonnegative, "
                f"got {single_expert_token_chunk}"
            )
        bucket_sizes = _sparsecore_expert_pipeline_bucket_sizes(chunk_capacity)
        bucket_thresholds = jnp.asarray(bucket_sizes[:-1], dtype=jnp.int32)

        pad_assignments = chunk_capacity
        token_ids_sort_padded = jnp.pad(token_ids_sort, ((0, pad_assignments),), constant_values=0)
        weights_sort_padded = jnp.pad(dispatch_weights_sort, ((0, pad_assignments),), constant_values=0)

        if chunk_experts == 1 and (single_expert_dense or single_expert_token_chunk > 0):
            token_chunk = single_expert_token_chunk or chunk_capacity
            num_token_subchunks = math.ceil(chunk_capacity / token_chunk)
            use_single_expert_barrier = _use_single_expert_pipeline_barrier()
            out = jnp.zeros_like(x)

            for expert_idx in range(num_experts):
                expert_group_size = group_sizes[expert_idx]
                expert_offset = expert_offsets[expert_idx]
                moe_w13_expert = moe_w13[expert_idx]
                moe_w2_expert = moe_w2[expert_idx]

                for subchunk_idx in range(num_token_subchunks):
                    subchunk_offset = expert_offset + subchunk_idx * token_chunk
                    remaining = expert_group_size - subchunk_idx * token_chunk
                    valid = jnp.arange(token_chunk, dtype=jnp.int32) < remaining
                    subchunk_token_ids = jax.lax.dynamic_slice(
                        token_ids_sort_padded, (subchunk_offset,), (token_chunk,)
                    )
                    subchunk_weights = jax.lax.dynamic_slice(weights_sort_padded, (subchunk_offset,), (token_chunk,))

                    x_subchunk = _take_rows_impl(x, subchunk_token_ids, implementation="sparsecore")
                    x_subchunk = jnp.where(valid[:, None], x_subchunk, jnp.zeros_like(x_subchunk))
                    subchunk_weights = jnp.where(valid, subchunk_weights, jnp.zeros_like(subchunk_weights))
                    if use_single_expert_barrier:
                        # Benchmark-only knob: break cross-expert fusion so we can
                        # test whether XLA will start dense TC work before later SC
                        # gathers have all completed.
                        x_subchunk = jax.lax.optimization_barrier(x_subchunk)
                        subchunk_weights = jax.lax.optimization_barrier(subchunk_weights)

                    with jax.named_scope(f"moe_single_expert_{expert_idx}_subchunk_{subchunk_idx}"):
                        if single_expert_dense:
                            out_dispatch = _dense_moe_up_down(
                                x_subchunk,
                                moe_w13_expert,
                                moe_w2_expert,
                                activation_fn=activation_fn,
                            )
                        else:
                            subchunk_group_sizes = (
                                jnp.asarray([token_chunk], dtype=jnp.int32).at[0].set(jnp.maximum(remaining, 0))
                            )
                            out_dispatch = ragged_dot(
                                x_subchunk, moe_w13[expert_idx : expert_idx + 1], subchunk_group_sizes
                            )
                            moe_dim = moe_w2_expert.shape[0]
                            gate, up = jnp.split(out_dispatch, [moe_dim], axis=-1)
                            out_dispatch = ragged_dot(
                                activation_fn(gate) * up,
                                moe_w2[expert_idx : expert_idx + 1],
                                subchunk_group_sizes,
                            )
                    if use_single_expert_barrier:
                        out_dispatch = jax.lax.optimization_barrier(out_dispatch)

                    with jax.named_scope(f"scatter_single_expert_{expert_idx}_subchunk_{subchunk_idx}"):
                        out = out.at[subchunk_token_ids].add(out_dispatch * subchunk_weights[:, None], mode="drop")

            return out, jnp.array(0, dtype=jnp.int32)

        if _use_static_sparsecore_expert_pipeline_routing():
            out = jnp.zeros_like(x)

            for chunk_idx in range(num_chunks):
                expert_start = chunk_idx * chunk_experts
                expert_stop = min(expert_start + chunk_experts, num_experts)
                chunk_group_sizes = group_sizes[expert_start:expert_stop]
                chunk_total = jnp.sum(chunk_group_sizes, dtype=jnp.int32)
                chunk_offset = expert_offsets[expert_start]
                moe_w13_chunk = moe_w13[expert_start:expert_stop]
                moe_w2_chunk = moe_w2[expert_start:expert_stop]

                def make_branch(bucket_size: int):
                    def branch(out_in: jax.Array) -> jax.Array:
                        chunk_token_ids = jax.lax.dynamic_slice(token_ids_sort_padded, (chunk_offset,), (bucket_size,))
                        chunk_weights = jax.lax.dynamic_slice(weights_sort_padded, (chunk_offset,), (bucket_size,))
                        valid = jnp.arange(bucket_size, dtype=jnp.int32) < chunk_total

                        x_chunk = _take_rows_impl(x, chunk_token_ids, implementation="sparsecore")
                        x_chunk = jnp.where(valid[:, None], x_chunk, jnp.zeros_like(x_chunk))
                        chunk_weights_masked = jnp.where(valid, chunk_weights, jnp.zeros_like(chunk_weights))
                        padded_group_sizes = chunk_group_sizes.at[-1].add(bucket_size - chunk_total)

                        with jax.named_scope(f"moe_up_down_chunk_{chunk_idx}_bucket_{bucket_size}"):
                            w13_out = ragged_dot(x_chunk, moe_w13_chunk, padded_group_sizes)
                            moe_dim = moe_w2_chunk.shape[1]
                            gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
                            out_dispatch = ragged_dot(activation_fn(gate) * up, moe_w2_chunk, padded_group_sizes)

                        with jax.named_scope(f"scatter_chunk_{chunk_idx}_bucket_{bucket_size}"):
                            return out_in.at[chunk_token_ids].add(
                                out_dispatch * chunk_weights_masked[:, None], mode="drop"
                            )

                    return branch

                bucket_index = (
                    jnp.sum(chunk_total > bucket_thresholds).astype(jnp.int32)
                    if bucket_thresholds.size
                    else jnp.array(0, dtype=jnp.int32)
                )
                branches = tuple(make_branch(bucket_size) for bucket_size in bucket_sizes)
                out = jax.lax.switch(bucket_index, branches, out)

            return out, jnp.array(0, dtype=jnp.int32)

        token_parts = []
        out_parts = []
        valid_parts = []

        for chunk_idx in range(num_chunks):
            expert_start = chunk_idx * chunk_experts
            expert_stop = min(expert_start + chunk_experts, num_experts)
            chunk_group_sizes = group_sizes[expert_start:expert_stop]
            chunk_total = jnp.sum(chunk_group_sizes, dtype=jnp.int32)
            chunk_offset = expert_offsets[expert_start]

            chunk_token_ids = jax.lax.dynamic_slice(token_ids_sort_padded, (chunk_offset,), (chunk_capacity,))
            chunk_weights = jax.lax.dynamic_slice(weights_sort_padded, (chunk_offset,), (chunk_capacity,))
            valid = jnp.arange(chunk_capacity, dtype=jnp.int32) < chunk_total

            x_chunk = _take_rows_impl(x, chunk_token_ids, implementation="sparsecore")
            x_chunk = jnp.where(valid[:, None], x_chunk, jnp.zeros_like(x_chunk))
            chunk_weights = jnp.where(valid, chunk_weights, jnp.zeros_like(chunk_weights))

            padded_group_sizes = chunk_group_sizes.at[-1].add(chunk_capacity - chunk_total)
            moe_w13_chunk = moe_w13[expert_start:expert_stop]
            moe_w2_chunk = moe_w2[expert_start:expert_stop]

            with jax.named_scope(f"moe_up_down_chunk_{chunk_idx}"):
                w13_out = ragged_dot(x_chunk, moe_w13_chunk, padded_group_sizes)
                moe_dim = moe_w2_chunk.shape[1]
                gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
                out_dispatch = ragged_dot(activation_fn(gate) * up, moe_w2_chunk, padded_group_sizes)

            out_parts.append(out_dispatch * chunk_weights[:, None])
            token_parts.append(chunk_token_ids)
            valid_parts.append(valid)

        token_dispatch = jnp.concatenate(token_parts, axis=0)
        out_dispatch = jnp.concatenate(out_parts, axis=0)
        valid_dispatch = jnp.concatenate(valid_parts, axis=0)
        out_dispatch = jnp.where(valid_dispatch[:, None], out_dispatch, jnp.zeros_like(out_dispatch))

        with jax.named_scope("scatter"):
            out = jnp.zeros_like(x).at[token_dispatch].add(out_dispatch, mode="drop")
        return out, jnp.array(0, dtype=jnp.int32)

    x_dispatch, w_dispatch, token_dispatch, group_sizes = _prepare_moe_dispatch_impl(
        x,
        selected_experts,
        combine_weights,
        num_experts=num_experts,
        implementation=dispatch_implementation,
    )

    with jax.named_scope("moe_up_down"):
        w13_out = ragged_dot(x_dispatch, moe_w13, group_sizes)
        moe_dim = moe_w2.shape[1]
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
        out_dispatch = ragged_dot(activation_fn(gate) * up, moe_w2, group_sizes)

    with jax.named_scope("scatter"):
        out = jnp.zeros_like(x).at[token_dispatch].add(out_dispatch * w_dispatch[:, None], mode="drop")
    return out, jnp.array(0, dtype=jnp.int32)


def _batch_spec_from_x(x: jax.Array, mesh: jax.sharding.AbstractMesh | None) -> P:
    sharding = getattr(x, "sharding", None)
    spec = getattr(sharding, "spec", None)
    if spec is not None and len(spec) > 0:
        return P(spec[0])
    return _batch_spec(mesh)


def _prefix_cap_counts(counts: Int[Array, "E"], *, capacity: int) -> Int[Array, "E"]:
    accepted = []
    remaining = jnp.array(capacity, dtype=jnp.int32)
    for expert in range(int(counts.shape[0])):
        take = jnp.minimum(counts[expert], remaining)
        accepted.append(take)
        remaining = jnp.maximum(remaining - take, 0)
    return jnp.stack(accepted, axis=0)


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
    dispatch_implementation: DispatchImplementation = "xla",
) -> tuple[Float[Array, "TL D"], Int[Array, ""]]:
    """Ring-style EP routed path: all-gather dispatch + psum-scatter collect."""
    # #2710 ring EP strategy: gather tokens and their selected-expert routing
    # assignments across expert shards, then psum-scatter back to local tokens.
    with jax.named_scope("gather"):
        x_global = jax.lax.all_gather(x_local, "expert", tiled=True)
        selected_experts_global = jax.lax.all_gather(selected_experts_local, "expert", tiled=True)
        combine_weights_global = jax.lax.all_gather(combine_weights_local, "expert", tiled=True)

        tokens = x_global.shape[0]
        topk = selected_experts_global.shape[1]
        assignments = tokens * topk
        expert_flat = selected_experts_global.reshape(assignments)
        weight_flat = combine_weights_global.reshape(assignments)

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
        local_expert = expert_flat - expert_start
        local_mask = jnp.logical_and(local_expert >= 0, local_expert < local_experts)

        # Keep only the assignments this shard will execute, ordered by
        # (local expert id, original flat position). This avoids the global
        # argsort + fused takes over all assignments that dominated high-EP
        # shapes, while preserving the grouped layout expected by ragged_dot.
        local_expert = jnp.where(local_mask, local_expert, 0)
        # TPU lowers this small-expert count reduction better as a dense
        # compare+sum than as `bincount`.
        expert_ids = jnp.arange(local_experts, dtype=jnp.int32)
        local_mask_i32 = local_mask.astype(jnp.int32)
        counts = jnp.sum(
            (local_expert[:, None] == expert_ids[None, :]).astype(jnp.int32) * local_mask_i32[:, None],
            axis=0,
            dtype=jnp.int32,
        )
        accepted_counts = _prefix_cap_counts(counts, capacity=local_capacity)
        accepted_total = jnp.sum(accepted_counts, dtype=jnp.int32)
        dropped_local = jnp.sum(counts, dtype=jnp.int32) - accepted_total
        valid = jnp.arange(local_capacity, dtype=jnp.int32) < accepted_total

        flat_pos = jnp.arange(assignments, dtype=jnp.int32)
        order_key = local_expert * assignments + flat_pos
        max_order_key = local_experts * assignments
        selection_key = jnp.where(local_mask, max_order_key - order_key, -1)
        _, local_idx = jax.lax.top_k(selection_key, local_capacity)

        token_local = jnp.floor_divide(local_idx, topk)
        weight_local = jnp.take(weight_flat, local_idx, axis=0).astype(x_local.dtype)

        x_take = _take_rows_impl(x_global, token_local, implementation=dispatch_implementation)
        x_dispatch = jnp.where(valid[:, None], x_take, jnp.zeros_like(x_take))
        weight_dispatch = jnp.where(valid, weight_local, jnp.zeros_like(weight_local))
    group_sizes = accepted_counts
    # `local_idx` pads by appending invalid rows at the end; keep GMM segment
    # boundaries aligned by attributing padding to the final expert segment.
    group_sizes = group_sizes.at[-1].add(local_capacity - jnp.sum(group_sizes, dtype=jnp.int32))

    with jax.named_scope("moe_up_down"):
        w13_out = ragged_dot(x_dispatch, moe_w13_local, group_sizes)
        moe_dim = moe_w2_local.shape[1]
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
        out_dispatch = ragged_dot(activation_fn(gate) * up, moe_w2_local, group_sizes)

    with jax.named_scope("scatter"):
        weighted_dispatch = out_dispatch * weight_dispatch[:, None]
        return_impl = _ep_return_implementation()
        if return_impl == "scatter_psum_scatter":
            out_global = jnp.zeros_like(x_global).at[token_local].add(weighted_dispatch, mode="drop")
            # #2710 ring EP strategy: collect only this shard's token slice after
            # reducing contributions from experts across the EP mesh.
            out_local = jax.lax.psum_scatter(out_global, "expert", scatter_dimension=0, tiled=True)
        elif return_impl == "sorted_scatter_psum_scatter":
            order = jnp.argsort(token_local, stable=True)
            sorted_token_local = jnp.take(token_local, order, axis=0)
            sorted_weighted_dispatch = jnp.take(weighted_dispatch, order, axis=0)
            out_global = jnp.zeros_like(x_global).at[sorted_token_local].add(sorted_weighted_dispatch, mode="drop")
            out_local = jax.lax.psum_scatter(out_global, "expert", scatter_dimension=0, tiled=True)
        elif return_impl == "owner_bucket_psum_scatter":
            local_tokens = x_local.shape[0]
            owner = jnp.floor_divide(token_local, local_tokens)
            owner_token = token_local - owner * local_tokens
            out_by_owner = jnp.zeros((ep_size, local_tokens, x_local.shape[1]), dtype=x_local.dtype)
            out_by_owner = out_by_owner.at[owner, owner_token].add(weighted_dispatch, mode="drop")
            out_local = jax.lax.psum_scatter(out_by_owner, "expert", scatter_dimension=0, tiled=True)
            out_local = jnp.squeeze(out_local, axis=0)
        elif return_impl == "owner_bucket_all_to_all_local_scatter":
            local_tokens = x_local.shape[0]
            owner = jnp.floor_divide(token_local, local_tokens)
            owner_token = token_local - owner * local_tokens
            bucket = max(1, math.ceil(local_capacity / ep_size * _ep_return_owner_bucket_factor()))
            bucket_pos = jnp.arange(bucket, dtype=jnp.int32)
            row_pos = jnp.arange(local_capacity, dtype=jnp.int32)
            key_base = local_capacity - row_pos

            send_tokens = []
            send_values = []
            bucket_overflow_local = jnp.array(0, dtype=jnp.int32)
            for owner_idx in range(ep_size):
                owner_mask = owner == owner_idx
                owner_total = jnp.sum(owner_mask.astype(jnp.int32), dtype=jnp.int32)
                owner_accept = jnp.minimum(owner_total, bucket)
                bucket_overflow_local = bucket_overflow_local + jnp.maximum(owner_total - bucket, 0)
                selection_key = jnp.where(owner_mask, key_base, -1)
                _, owner_rows = jax.lax.top_k(selection_key, bucket)
                owner_valid = bucket_pos < owner_accept

                owner_tokens = jnp.take(owner_token, owner_rows, axis=0)
                owner_updates = jnp.take(weighted_dispatch, owner_rows, axis=0)
                owner_tokens = jnp.where(owner_valid, owner_tokens, 0)
                owner_updates = jnp.where(owner_valid[:, None], owner_updates, jnp.zeros_like(owner_updates))
                send_tokens.append(owner_tokens)
                send_values.append(owner_updates)

            send_tokens_arr = jnp.stack(send_tokens, axis=0)
            send_values_arr = jnp.stack(send_values, axis=0)
            recv_tokens = jax.lax.all_to_all(send_tokens_arr, "expert", split_axis=0, concat_axis=0, tiled=True)
            recv_values = jax.lax.all_to_all(send_values_arr, "expert", split_axis=0, concat_axis=0, tiled=True)
            recv_tokens = recv_tokens.reshape(ep_size * bucket)
            recv_values = recv_values.reshape(ep_size * bucket, x_local.shape[1])
            out_local = jnp.zeros_like(x_local).at[recv_tokens].add(recv_values, mode="drop")
            dropped_local = dropped_local + bucket_overflow_local
        else:
            raise ValueError(
                "GRUG_MOE_EP_RETURN_IMPL must be one of "
                "{'scatter_psum_scatter', 'sorted_scatter_psum_scatter', 'owner_bucket_psum_scatter', "
                "'owner_bucket_all_to_all_local_scatter'}, "
                f"got {return_impl!r}"
            )
        dropped_total = jax.lax.psum(dropped_local, ("data", "expert"))
    return out_local, dropped_total


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
    dispatch_implementation: DispatchImplementation = "xla",
) -> Float[Array, "T D"] | tuple[Float[Array, "T D"], Int[Array, ""]]:
    """Functional routed MoE MLP core used by Grug modules and benchmarks.

    This helper handles dispatch/permute/unpermute (+EP collectives) from
    precomputed token-to-expert assignments. Routing logits/top-k selection
    stays in the caller (e.g. model MLP block).

    Set `report_capacity_overflow=True` to also return a scalar count of
    dropped expert assignments from EP capacity clipping.
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
        out, dropped = _moe_mlp_local(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation_fn=activation_fn,
            num_experts=num_experts,
            dispatch_implementation=dispatch_implementation,
        )
        if report_capacity_overflow:
            return out, dropped
        return out

    batch_spec = _batch_spec_from_x(x, mesh)
    local_expert_spec = P("expert", None, None) if has_expert_axis else P(None, None, None)

    if has_expert_axis and expert_axis_size > 1:
        if dispatch_implementation in ("sparsecore_pipeline", "sparsecore_expert_pipeline"):
            raise NotImplementedError(
                f"dispatch_implementation={dispatch_implementation!r} is only implemented for the local/non-EP moe_mlp path"
            )
        if num_experts % expert_axis_size != 0:
            raise ValueError(f"num_experts={num_experts} must be divisible by expert axis size={expert_axis_size}")

        # #2710: prefer ring EP collectives when a real expert mesh is present.
        shard_fn = shard_map(
            partial(
                _moe_mlp_ep_ring_local,
                activation_fn=activation_fn,
                num_experts=num_experts,
                capacity_factor=capacity_factor,
                dispatch_implementation=dispatch_implementation,
            ),
            mesh=mesh,
            in_specs=(
                batch_spec,
                batch_spec,
                batch_spec,
                P("expert", None, None),
                P("expert", None, None),
            ),
            out_specs=(batch_spec, P()),
            check_vma=False,
        )
        out, dropped = shard_fn(x, selected_experts, combine_weights, w_up_gate, w_down)
        if report_capacity_overflow:
            return out, dropped
        return out

    # Fallback path for no expert axis (or expert axis size 1) keeps routing
    # semantics without EP collectives.
    shard_fn = shard_map(
        partial(
            _moe_mlp_local,
            activation_fn=activation_fn,
            num_experts=num_experts,
            dispatch_implementation=dispatch_implementation,
        ),
        mesh=mesh,
        in_specs=(
            batch_spec,
            batch_spec,
            batch_spec,
            local_expert_spec,
            local_expert_spec,
        ),
        out_specs=(batch_spec, P()),
        check_vma=False,
    )
    out, dropped = shard_fn(x, selected_experts, combine_weights, w_up_gate, w_down)
    if report_capacity_overflow:
        return out, dropped
    return out


__all__ = [
    "DispatchImplementation",
    "MoeActivation",
    "moe_mlp",
]
