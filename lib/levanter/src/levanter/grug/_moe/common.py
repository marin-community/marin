# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared types, routing helpers, and layout utilities for Grug MoE."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple, TypeAlias, cast, get_args

import jax
import jax.numpy as jnp
from haliax.jax_utils import named_call
from jax.sharding import PartitionSpec as P
from shape_extensions import Int, IntTuple, IntVar

from levanter.utils.activation import ActivationFunctionEnum

_DEFAULT_EP_CAPACITY_FACTOR = 1.25
# #2710 used 1.25 as the practical EP ring default to avoid over/under-packing.


def _pack_pairs_u32[Batch: IntTuple, F: IntVar](
    a: jax.Array[[*Batch, F]], b: jax.Array[[*Batch, F]]
) -> jax.Array[[*Batch, 2 * F]]:
    """Interleave two 16-bit ``[..., F]`` arrays as ``[..., 2F]``."""
    # bitcast_convert_type's stub drops the shape (dtype-size changes are not modeled), so pin it.
    ai: jax.Array[[*Batch, F]] = jax.lax.bitcast_convert_type(a, jnp.uint16).astype(jnp.uint32)
    bi: jax.Array[[*Batch, F]] = jax.lax.bitcast_convert_type(b, jnp.uint16).astype(jnp.uint32)
    packed = ai | (bi << jnp.uint32(16))
    # A uint32 -> 16-bit bitcast appends an axis of 2, little end first, so `a` leads.
    result: jax.Array[[*Batch, 2 * F]] = jax.lax.bitcast_convert_type(packed, a.dtype).reshape(
        *a.shape[:-1], 2 * a.shape[-1]
    )
    return result


def _unpack_pairs_u32[Batch: IntTuple, F2: IntVar](
    x: jax.Array[[*Batch, F2]],
) -> tuple[jax.Array[[*Batch, F2 // 2]], jax.Array[[*Batch, F2 // 2]]]:
    """``[..., 2F]`` of a 16-bit dtype -> ``([..., F], [..., F])``, undoing ``_pack_pairs_u32``."""
    pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    packed = jax.lax.bitcast_convert_type(pairs, jnp.uint32)
    lo = (packed & jnp.uint32(0xFFFF)).astype(jnp.uint16)
    hi = (packed >> jnp.uint32(16)).astype(jnp.uint16)
    lo_out: jax.Array[[*Batch, F2 // 2]] = jax.lax.bitcast_convert_type(lo, x.dtype)
    hi_out: jax.Array[[*Batch, F2 // 2]] = jax.lax.bitcast_convert_type(hi, x.dtype)
    return lo_out, hi_out


# `bitcast_convert_type` has no AD rule, so the interleave carries its own transpose.
@jax.custom_vjp
def _interleave_halves[Batch: IntTuple, F: IntVar](
    gate: jax.Array[[*Batch, F]], up: jax.Array[[*Batch, F]]
) -> jax.Array[[*Batch, 2 * F]]:
    return _pack_pairs_u32(gate, up)


def _interleave_halves_fwd[Batch: IntTuple, F: IntVar](gate: jax.Array[[*Batch, F]], up: jax.Array[[*Batch, F]]):
    return _pack_pairs_u32(gate, up), None


def _interleave_halves_bwd[Batch: IntTuple, F2: IntVar](
    _residual: None, ct: jax.Array[[*Batch, F2]]
) -> tuple[jax.Array[[*Batch, F2 // 2]], jax.Array[[*Batch, F2 // 2]]]:
    return _unpack_pairs_u32(ct)


_interleave_halves.defvjp(_interleave_halves_fwd, _interleave_halves_bwd)


def _interleave_gate_up[Batch: IntTuple, I2: IntVar](
    moe_w13: jax.Array[[*Batch, I2]], moe_dim: int
) -> jax.Array[[*Batch, I2]]:
    """grug w13 [E,H,2I] gate=[:I], up=[I:] -> interleaved [g0,u0,g1,u1,...] (QuACK layout)."""
    # The split's width check matters here: the packed path would broadcast mismatched halves
    # against each other and return a wrong-width array instead of raising.
    gate, up = split_moe_w13_output(moe_w13, intermediate_dim=moe_dim, interleaved=False)
    if moe_w13.dtype.itemsize != 2:
        stacked: jax.Array[[*Batch, I2]] = jnp.stack([gate, up], axis=-1).reshape(moe_w13.shape)
        return stacked
    # pyrefly: ignore[bad-return]  # 2 * (I2 // 2) == I2 because moe_dim is even; not solved symbolically.
    return _interleave_halves(gate, up)


def _swiglu_gate_up_backward[Batch: IntTuple, I2: IntVar](
    gu: jax.Array[[*Batch, I2]], dh: jax.Array[[*Batch, I2 // 2]]
) -> jax.Array[[*Batch, I2]]:
    """Cotangent of the interleaved gate/up pre-activations, given the SwiGLU output's."""
    if gu.dtype.itemsize == 2:
        gate, up = _unpack_pairs_u32(gu)
    else:
        # pyrefly: ignore[bad-assignment]  # step-2 slice from 0 infers ceil(I2/2); I2 is even at runtime.
        gate: jax.Array[[*Batch, I2 // 2]] = gu[..., 0::2]
        up: jax.Array[[*Batch, I2 // 2]] = gu[..., 1::2]
    sg = jax.nn.sigmoid(gate)
    silu = gate * sg
    dgate = dh * up * (sg + silu * (1.0 - sg))
    dup = dh * silu
    if gu.dtype.itemsize == 2:
        # pyrefly: ignore[bad-return]  # 2 * (I2 // 2) == I2 because moe_dim is even; not solved symbolically.
        return _pack_pairs_u32(dgate.astype(gu.dtype), dup.astype(gu.dtype))
    packed: jax.Array[[*Batch, I2]] = jnp.stack([dgate, dup], axis=-1).reshape(gu.shape)
    return packed


PspecAxis: TypeAlias = str | tuple[str, ...] | None
MoeActivation: TypeAlias = ActivationFunctionEnum | Callable[[jax.Array], jax.Array]
MoeImplementation: TypeAlias = Literal[
    "ring",  # Expert-parallel all-gather + psum-scatter backend.
    "ragged_all_to_all",  # Expert-parallel ragged all-to-all backend.
    "fixed_all_to_all",  # Expert-parallel all-to-all with fixed sender/expert cells.
    "fixed_pooled_wave_all_to_all",  # Destination-pooled static waves with fixed receiver buffers.
    "deepep",  # Expert-parallel DeepEP intranode dispatch/combine backend.
    "scatter",  # Single-process grouped GMM with scatter-add combine.
    "sonic",  # Single-process raw Sonic Triton gather/combine backend.
    "sonic_cute",  # Single-process QuACK SM100 (Blackwell/B200) grouped-GEMM backend.
]
_VALID_MOE_IMPLEMENTATIONS = get_args(MoeImplementation)
_EP_MOE_IMPLEMENTATIONS = (
    "ring",
    "ragged_all_to_all",
    "fixed_all_to_all",
    "fixed_pooled_wave_all_to_all",
    "deepep",
)
# Local means no collectives over an expert axis. These backends can still run
# under ordinary data/model sharding through the no-EP shard_map path.
_LOCAL_MOE_IMPLEMENTATIONS = (
    "scatter",
    "sonic",
    "sonic_cute",
)

_CHECKPOINT_DISPATCH_INPUT = "grug_moe_dispatch_input"
_CHECKPOINT_EXPERT_HIDDEN = "grug_moe_expert_hidden"
_CHECKPOINT_DISPATCH_OUTPUT = "grug_moe_dispatch_output"
_CHECKPOINT_MOE_OUTPUT = "grug_moe_output"

# Checkpoint names every MoE backend tags on its dispatch tensors. A remat
# policy of jax.checkpoint_policies.save_only_these_names(*MOE_REMAT_SAVE_NAMES)
# keeps these alive for backward instead of re-running expert dispatch —
# including the EP collectives — during the recompute.
MOE_REMAT_SAVE_NAMES = (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _CHECKPOINT_MOE_OUTPUT,
)


class CapacityDrops(NamedTuple):
    """Valid assignments a backend dropped before and after transport."""

    sender_dropped: jax.Array[[]]
    receiver_dropped: jax.Array[[]]

    @property
    def dropped(self) -> jax.Array[[]]:
        return self.sender_dropped + self.receiver_dropped


class MoeDispatchCounts(NamedTuple):
    """Assignment counts omitted from expert dispatch."""

    sender_dropped: jax.Array[[]]
    receiver_dropped: jax.Array[[]]
    padding_skipped: jax.Array[[]]

    @property
    def dropped(self) -> jax.Array[[]]:
        return self.sender_dropped + self.receiver_dropped


def padding_skipped_assignments[T: IntVar](token_valid: jax.Array[[T]], *, topk: int) -> jax.Array[[]]:
    """Count the expert assignments that padded tokens would otherwise have made."""
    return jnp.sum(~token_valid, dtype=jnp.int32) * topk


@dataclass(frozen=True)
class MoEExpertMlpPspecs:
    """Logical sharding axes for local MoE expert MLP weights."""

    expert: PspecAxis = "expert"
    hidden: PspecAxis = "data"
    intermediate: PspecAxis = "model"

    @property
    def w_gate_up(self) -> P:
        return P(self.expert, self.hidden, self.intermediate)

    @property
    def w_down(self) -> P:
        return P(self.expert, self.intermediate, self.hidden)


def resolve_moe_implementation(implementation: MoeImplementation | str | None) -> MoeImplementation:
    if implementation is None:
        return "ring"
    if implementation not in _VALID_MOE_IMPLEMENTATIONS:
        valid = ", ".join(repr(choice) for choice in _VALID_MOE_IMPLEMENTATIONS)
        raise ValueError(f"implementation must be one of {valid} or None, got {implementation!r}")
    return cast(MoeImplementation, implementation)


def split_moe_w13_output[Batch: IntTuple, I2: IntVar](
    w13_out: jax.Array[[*Batch, I2]], *, intermediate_dim: int, interleaved: bool
) -> tuple[jax.Array[[*Batch, I2 // 2]], jax.Array[[*Batch, I2 // 2]]]:
    expected = 2 * intermediate_dim
    if w13_out.shape[-1] != expected:
        raise ValueError(f"w13 output last dimension must be {expected}, got shape={w13_out.shape}")
    if interleaved:
        # pyrefly: ignore[bad-assignment]  # step-2 slice from 0 infers ceil(I2/2); I2 is even at runtime.
        gate: jax.Array[[*Batch, I2 // 2]] = w13_out[..., 0::2]
        up: jax.Array[[*Batch, I2 // 2]] = w13_out[..., 1::2]
        return gate, up
    gate, up = jnp.split(w13_out, [intermediate_dim], axis=-1)
    return gate, up


def _init_weight(key: jax.Array[[]], shape: tuple[int, ...], std: float) -> jax.Array:
    # `shape` is an untyped tuple, so pyrefly cannot infer the result's rank.
    return std * jax.random.truncated_normal(key, -3, 3, shape)


@named_call
def _prepare_moe_dispatch[T: IntVar, K: IntVar, H: IntVar, E: IntVar](
    x: jax.Array[[T, H]],
    selected_experts: jax.Array[[T, K]],
    combine_weights: jax.Array[[T, K]],
    token_valid: jax.Array[[T]],
    *,
    num_experts: Int[E],
) -> tuple[
    jax.Array[[T * K, H]],
    jax.Array[[T * K]],
    jax.Array[[T * K]],
    jax.Array[[E]],
]:
    """Flatten + argsort by expert into grouped layout for GMM."""
    # #2704: keep argsort-grouped dispatch as the canonical compact routing
    # strategy, matching the behavior carried forward from 89318a910.
    tokens, topk = selected_experts.shape
    assignment_valid = _assignment_validity(token_valid, tokens=tokens, topk=topk)
    expert_ids = jnp.where(assignment_valid, selected_experts.reshape(tokens * topk), num_experts)
    dispatch_weights = jnp.where(assignment_valid, combine_weights.reshape(tokens * topk), 0)

    sort_idx = jnp.argsort(expert_ids, axis=0)
    token_ids = jnp.arange(tokens * topk, dtype=jnp.int32) // topk
    token_ids_sort: jax.Array[[T * K]] = token_ids[sort_idx]
    x_sort: jax.Array[[T * K, H]] = x[token_ids_sort]
    w_sort: jax.Array[[T * K]] = dispatch_weights[sort_idx].astype(x.dtype)
    group_sizes: jax.Array[[E]] = jnp.bincount(expert_ids, length=num_experts).astype(jnp.int32)
    return x_sort, w_sort, token_ids_sort, group_sizes


@named_call
def _prepare_moe_dispatch_indices_with_assignment_ids[T: IntVar, K: IntVar, E: IntVar](
    selected_experts: jax.Array[[T, K]],
    token_valid: jax.Array[[T]],
    *,
    num_experts: Int[E],
) -> tuple[
    jax.Array[[T * K]],
    jax.Array[[T, K]],
    jax.Array[[E]],
    jax.Array[[T * K]],
]:
    """Prepare expert-sorted token ids plus reverse positions without gathering x."""
    tokens, topk = selected_experts.shape
    assignments = tokens * topk
    assignment_valid = _assignment_validity(token_valid, tokens=tokens, topk=topk)
    expert_ids = jnp.where(assignment_valid, selected_experts.reshape(assignments), num_experts)

    sort_idx = jnp.argsort(expert_ids, axis=0)
    assignment_ids = jnp.arange(assignments, dtype=jnp.int32)
    sorted_assignment_ids: jax.Array[[T * K]] = assignment_ids[sort_idx]
    token_ids_sort: jax.Array[[T * K]] = sorted_assignment_ids // topk

    sorted_positions = jnp.arange(assignments, dtype=jnp.int32)
    dispatch_positions_flat = jnp.zeros((assignments,), dtype=jnp.int32).at[sort_idx].set(sorted_positions)
    dispatch_positions: jax.Array[[T, K]] = dispatch_positions_flat.reshape(tokens, topk)

    group_sizes: jax.Array[[E]] = jnp.bincount(expert_ids, length=num_experts).astype(jnp.int32)
    return token_ids_sort, dispatch_positions, group_sizes, sorted_assignment_ids


def _assignment_validity[T: IntVar, K: IntVar](
    token_valid: jax.Array[[T]],
    *,
    tokens: Int[T],
    topk: Int[K],
) -> jax.Array[[T * K]]:
    return jnp.broadcast_to(token_valid[:, None], (tokens, topk)).reshape(tokens * topk)


def _scaled_capacity(
    assignments: jax.Array[[]],
    *,
    capacity_factor: float,
    divisor: int = 1,
    minimum: int = 1,
    maximum: int,
) -> jax.Array[[]]:
    """Return a JIT-safe logical capacity derived from dynamic assignment demand.

    ``maximum`` must be the static physical capacity the caller sized its buffers with.
    The returned capacity is clamped to ``[minimum, maximum]``.
    """
    # Keep large assignment counts precise without changing the surrounding dtype defaults.
    with jax.enable_x64():
        scaled = jnp.ceil(assignments.astype(jnp.float64) * capacity_factor / divisor)
        return jnp.clip(scaled, minimum, maximum).astype(jnp.int32)


def _zero_dropped_assignments() -> jax.Array[[]]:
    return jnp.array(0, dtype=jnp.int32)


def _chunk_capacity_drops[E1: IntVar](
    # `caps` entries may be traced scalars derived from a dynamic assignment count, so they stay
    # `int | jax.Array[[]]` rather than a symbolic dimension.
    cu: jax.Array[[E1]],
    bounds: Sequence[int],
    caps: Sequence[int | jax.Array[[]]],
) -> jax.Array[[]]:
    """Count assignments lost to per-chunk static capacity."""
    total = jnp.zeros((), jnp.int32)
    for chunk, cap in enumerate(caps):
        count = cu[bounds[chunk + 1]] - cu[bounds[chunk]]
        total = total + jnp.maximum(count - cap, 0).astype(jnp.int32)
    return total


def _zero_inactive_grouped_rows[R: IntVar, C: IntVar](
    values: jax.Array[[R, C]], cumulative_group_sizes: jax.Array
) -> jax.Array[[R, C]]:
    """Zero the rows past the last expert group, which the grouped kernels never write."""
    active_rows = cumulative_group_sizes[-1]
    return jnp.where(jnp.arange(values.shape[0])[:, None] < active_rows, values, jnp.zeros((), values.dtype))
