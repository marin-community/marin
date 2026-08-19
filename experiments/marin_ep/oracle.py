# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dense reference implementation of the Marin EP MoE contract.

This is the oracle every Marin EP implementation (correctness simulator,
GPU kernel) is tested against. It follows `experiments/marin_ep/SPEC.md`:
the pooled per-expert capacity keep rule (Section 2) and the dense fp32
value computation with fp32 combine accumulation (Sections 3-5). It is
deliberately simple and global: no sharding, no message passing, plain
loops over experts.

The value function is written in JAX so that autodiff supplies reference
gradients (spec invariant I5).
"""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


def expert_capacity(num_assignments_global: int, num_experts: int, capacity_factor: float) -> int:
    """Per-expert row capacity `C` (SPEC Section 2).

    Clamped to `num_assignments_global`: no expert can receive more rows than
    exist, so any larger capacity factor means dropless.
    """
    raw = max(1, math.ceil(capacity_factor * num_assignments_global / num_experts))
    return min(raw, num_assignments_global)


def kept_rows_per_expert(total_counts: np.ndarray, *, capacity: int, group_size: int) -> np.ndarray:
    """Kept rows per expert under the group-pooled waterfilling rule (SPEC S2).

    Each expert keeps its floor `min(N_g, C)`; the group's slack
    (unused floor capacity) is granted to overflowing experts in ascending
    expert order. The canonical implementation shared by the oracle, the
    correctness simulator, and the perf models.
    """
    num_experts = total_counts.shape[0]
    if num_experts % group_size:
        raise ValueError(f"group_size={group_size} must divide num_experts={num_experts}")
    counts = np.asarray(total_counts, dtype=np.int64).reshape(-1, group_size)
    floor = np.minimum(counts, capacity)
    overflow = counts - floor
    slack = (capacity - counts).clip(min=0).sum(axis=1, keepdims=True)
    granted_cum = np.minimum(np.cumsum(overflow, axis=1), slack)
    grants = np.diff(granted_cum, axis=1, prepend=0)
    return (floor + grants).reshape(num_experts)


def pooled_keep_mask(
    selected_experts: np.ndarray,
    *,
    num_experts: int,
    capacity: int,
    group_size: int = 1,
) -> tuple[np.ndarray, int]:
    """Kept-assignment mask under the group-pooled capacity rule.

    `selected_experts` is the global routing `[S, T, K]` (any leading shape;
    only the flattened global order matters). Arrival order for an expert is
    the global flat assignment order, so the result is independent of how
    tokens are partitioned across devices (spec invariant I3; `group_size`
    is a static spec parameter, deliberately decoupled from the mesh).

    Returns the boolean keep mask with the input's shape and the exact
    global dropped-assignment count (spec invariant I2).
    """
    flat = np.asarray(selected_experts).reshape(-1).astype(np.int64)
    if flat.size and (flat.min() < 0 or flat.max() >= num_experts):
        raise ValueError(f"expert ids must be in [0, {num_experts}), got range [{flat.min()}, {flat.max()}]")
    counts = np.bincount(flat, minlength=num_experts)
    kept = kept_rows_per_expert(counts, capacity=capacity, group_size=group_size)
    order = np.argsort(flat, kind="stable")
    segment_start = np.cumsum(counts) - counts
    rank_sorted = np.arange(flat.size, dtype=np.int64) - segment_start[flat[order]]
    rank = np.empty_like(rank_sorted)
    rank[order] = rank_sorted
    keep = (rank < kept[flat]).reshape(np.asarray(selected_experts).shape)
    dropped = int((counts - kept).sum())
    return keep, dropped


def moe_oracle(
    x: Float[Array, "S T H"],
    selected_experts: Array,
    combine_weights: Float[Array, "S T K"],
    w13: Float[Array, "E H I2"],
    w2: Float[Array, "E I H"],
    keep: Array,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> Float[Array, "S T H"]:
    """Dense fp32 MoE forward over the global batch.

    Every token is run through every expert it selects (kept assignments
    only); contributions are weighted by `combine_weights` and accumulated
    in fp32. Differentiable in `x`, `combine_weights`, `w13`, `w2`; `keep`
    and `selected_experts` are non-differentiable constants.
    """
    num_experts, hidden_dim, i2 = w13.shape
    intermediate_dim = w2.shape[1]
    if i2 != 2 * intermediate_dim:
        raise ValueError(f"w13 last dim {i2} must be 2 * w2 intermediate dim {intermediate_dim}")

    lead_shape = x.shape[:-1]
    x_flat = x.reshape(-1, hidden_dim).astype(jnp.float32)
    experts_flat = selected_experts.reshape(x_flat.shape[0], -1)
    weights_flat = combine_weights.reshape(experts_flat.shape).astype(jnp.float32)
    keep_flat = keep.reshape(experts_flat.shape)

    y = jnp.zeros_like(x_flat)
    for g in range(num_experts):
        hidden = x_flat @ w13[g].astype(jnp.float32)
        gate, up = jnp.split(hidden, [intermediate_dim], axis=-1)
        z = (activation_fn(gate) * up) @ w2[g].astype(jnp.float32)
        gate_weight = jnp.sum(jnp.where(keep_flat & (experts_flat == g), weights_flat, 0.0), axis=-1)
        y = y + gate_weight[:, None] * z
    return y.reshape(*lead_shape, hidden_dim)
