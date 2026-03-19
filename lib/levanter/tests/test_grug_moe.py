# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest

from levanter.grug import grug_moe as grug_moe_lib


def _reference_expert_padded_batched(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    local_expert_capacity: int,
    rhs_contract_axis: int,
) -> jax.Array:
    if rhs_contract_axis == 1:
        rhs_prepared = rhs
    elif rhs_contract_axis == 2:
        rhs_prepared = jnp.swapaxes(rhs, 1, 2)
    else:
        raise ValueError(f"Unsupported rhs_contract_axis={rhs_contract_axis}")

    local_experts = rhs_prepared.shape[0]
    total_rows = lhs.shape[0]
    row_ids = jnp.arange(total_rows, dtype=jnp.int32)
    segment_ends = jnp.cumsum(group_sizes, dtype=jnp.int32)
    total_valid = jnp.sum(group_sizes, dtype=jnp.int32)
    valid = row_ids < total_valid

    segment_starts = jnp.concatenate(
        [jnp.zeros((1,), dtype=jnp.int32), segment_ends[:-1].astype(jnp.int32)],
        axis=0,
    )
    expert_ids = jnp.searchsorted(segment_ends, row_ids, side="right").astype(jnp.int32)
    expert_ids_clipped = jnp.clip(expert_ids, 0, max(0, local_experts - 1))
    within_expert = row_ids - jnp.take(segment_starts, expert_ids_clipped, axis=0)
    valid = valid & (expert_ids < local_experts) & (within_expert < local_expert_capacity)

    flat_capacity = local_experts * local_expert_capacity
    flat_indices = expert_ids * local_expert_capacity + within_expert
    capacity_positions = jnp.arange(local_expert_capacity, dtype=jnp.int32)
    gather_rows = segment_starts[:, None] + capacity_positions[None, :]
    gather_valid = capacity_positions[None, :] < group_sizes[:, None]
    lhs_with_zero = jnp.concatenate([lhs, jnp.zeros((1, lhs.shape[-1]), dtype=lhs.dtype)], axis=0)
    packed_lhs = jnp.take(
        lhs_with_zero,
        jnp.where(gather_valid, gather_rows, total_rows).reshape(-1),
        axis=0,
    ).reshape(local_experts, local_expert_capacity, lhs.shape[-1])
    packed_out = jax.lax.dot_general(
        packed_lhs,
        rhs_prepared,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
    )
    packed_out_flat = packed_out.reshape(flat_capacity, packed_out.shape[-1])
    gather_indices = jnp.where(valid, flat_indices, 0)
    out = jnp.take(packed_out_flat, gather_indices, axis=0)
    return jnp.where(valid[:, None], out, 0)


@pytest.mark.parametrize("rhs_contract_axis", [1, 2])
def test_expert_padded_batched_matches_reference_and_grads(rhs_contract_axis: int):
    lhs = (jnp.arange(15, dtype=jnp.float32).reshape(5, 3) - 7.0) / 5.0
    rhs_base = (jnp.arange(3 * 3 * 4, dtype=jnp.float32).reshape(3, 3, 4) - 9.0) / 7.0
    rhs = rhs_base if rhs_contract_axis == 1 else jnp.swapaxes(rhs_base, 1, 2)
    group_sizes = jnp.array([2, 1, 1], dtype=jnp.int32)
    local_expert_capacity = 3

    expected = _reference_expert_padded_batched(
        lhs,
        rhs,
        group_sizes,
        local_expert_capacity=local_expert_capacity,
        rhs_contract_axis=rhs_contract_axis,
    )
    actual = grug_moe_lib._ragged_dot_expert_padded_batched(
        lhs,
        rhs,
        group_sizes,
        local_expert_capacity=local_expert_capacity,
        rhs_contract_axis=rhs_contract_axis,
    )

    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)

    weights = jnp.linspace(-0.75, 0.9, actual.size, dtype=actual.dtype).reshape(actual.shape)

    reference_loss = lambda lhs_in, rhs_in: jnp.sum(
        _reference_expert_padded_batched(
            lhs_in,
            rhs_in,
            group_sizes,
            local_expert_capacity=local_expert_capacity,
            rhs_contract_axis=rhs_contract_axis,
        )
        * weights
    )
    actual_loss = lambda lhs_in, rhs_in: jnp.sum(
        grug_moe_lib._ragged_dot_expert_padded_batched(
            lhs_in,
            rhs_in,
            group_sizes,
            local_expert_capacity=local_expert_capacity,
            rhs_contract_axis=rhs_contract_axis,
        )
        * weights
    )

    grad_lhs_expected, grad_rhs_expected = jax.grad(reference_loss, argnums=(0, 1))(lhs, rhs)
    grad_lhs_actual, grad_rhs_actual = jax.grad(actual_loss, argnums=(0, 1))(lhs, rhs)

    assert jnp.allclose(grad_lhs_actual, grad_lhs_expected, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(grad_rhs_actual, grad_rhs_expected, rtol=1e-5, atol=1e-5)
