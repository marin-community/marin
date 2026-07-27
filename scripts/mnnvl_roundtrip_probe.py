# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check MNNVL peer-write transport values and gradients across one GPU node."""

import jax
import jax.numpy as jnp
import numpy as np
from iris.runtime.jax_init import initialize_jax
from levanter.grug._moe.ep_ragged_all_to_all import _mnnvl_dispatch_gather
from levanter.kernels.mnnvl.fabric_transport_ffi import (
    ensure_mnnvl_runtime,
    mnnvl_combine,
    mnnvl_dispatch,
    probe_mnnvl_peer_writes,
)

_HIDDEN_DIM = 64
_ROWS_PER_DESTINATION = 256


def _end_to_end_loss(
    values: jax.Array,
    combine_weights: jax.Array,
    expert_factor: jax.Array,
    token_sources: jax.Array,
    dispatch_positions: jax.Array,
    keep: jax.Array,
    destinations: jax.Array,
    slots: jax.Array,
    output_rows: int,
) -> jax.Array:
    received, source_ranks, source_slots = _mnnvl_dispatch_gather(
        values,
        token_sources,
        dispatch_positions,
        keep,
        destinations,
        slots,
        output_rows,
    )
    expert_outputs = (received * expert_factor).astype(jnp.bfloat16)
    returned = mnnvl_combine(
        expert_outputs,
        source_ranks,
        source_slots,
        destinations,
        slots,
        token_sources.shape[0],
    )
    token_outputs = jnp.sum(
        returned.reshape(*dispatch_positions.shape, values.shape[1]).astype(jnp.float32) * combine_weights[:, :, None],
        axis=1,
    )
    return jnp.sum(token_outputs)


def _check_end_to_end_custom_vjp(rank: int, world_size: int) -> None:
    tokens = _ROWS_PER_DESTINATION
    topk = world_size
    send_rows = tokens * topk
    output_rows = send_rows
    assignment = jnp.arange(send_rows, dtype=jnp.int32)
    token_sources = assignment // topk
    dispatch_positions = assignment.reshape(tokens, topk)
    keep = (assignment % 17 != 0).reshape(tokens, topk)
    destinations = (rank + assignment % world_size) % world_size
    slots = rank * _ROWS_PER_DESTINATION + assignment // world_size
    destinations = jnp.where(keep.reshape(-1), destinations, world_size)
    slots = jnp.where(keep.reshape(-1), slots, output_rows)

    value_row = jnp.arange(tokens, dtype=jnp.float32)[:, None]
    value_col = jnp.arange(_HIDDEN_DIM, dtype=jnp.float32)[None, :]
    values = (rank * 0.25 + value_row * 0.003 + value_col * 0.001).astype(jnp.bfloat16)
    combine_weights = (0.1 + (assignment.reshape(tokens, topk) % 7) * 0.05).astype(jnp.float32)
    expert_factor = jnp.asarray(rank + 1, dtype=jnp.bfloat16)

    loss_and_grad = jax.jit(jax.value_and_grad(_end_to_end_loss, argnums=(0, 1, 2)), static_argnums=(8,))
    loss, (value_gradient, weight_gradient, factor_gradient) = loss_and_grad(
        values,
        combine_weights,
        expert_factor,
        token_sources,
        dispatch_positions,
        keep,
        destinations,
        slots,
        output_rows,
    )
    factor_gradient.block_until_ready()

    assignment_np = np.arange(send_rows, dtype=np.int32)
    keep_np = assignment_np % 17 != 0
    destination_np = (rank + assignment_np % world_size) % world_size
    factor_np = (destination_np + 1).reshape(tokens, topk)
    keep_matrix_np = keep_np.reshape(tokens, topk)
    weights_np = np.asarray(combine_weights)
    values_np = np.asarray(values, dtype=np.float32)
    expected_coefficient = np.sum(weights_np * factor_np * keep_matrix_np, axis=1)
    expected_value_gradient = np.broadcast_to(expected_coefficient[:, None], values_np.shape)
    expected_weight_gradient = factor_np * np.sum(values_np, axis=1)[:, None] * keep_matrix_np
    expected_loss = np.sum(values_np * expected_value_gradient)

    expected_factor_gradient = 0.0
    for sender in range(world_size):
        sender_values = (
            sender * 0.25
            + np.arange(tokens, dtype=np.float32)[:, None] * 0.003
            + np.arange(_HIDDEN_DIM, dtype=np.float32)[None, :] * 0.001
        ).astype(jnp.bfloat16)
        sender_values = np.asarray(sender_values, dtype=np.float32)
        sender_destination = (sender + assignment_np % world_size) % world_size
        sender_weights = weights_np.reshape(-1)
        sender_tokens = assignment_np // topk
        accepted = np.logical_and(keep_np, sender_destination == rank)
        expected_factor_gradient += np.sum(
            sender_weights[accepted] * np.sum(sender_values[sender_tokens[accepted]], axis=1)
        )

    np.testing.assert_allclose(float(loss), expected_loss, rtol=2e-3, atol=1e-2)
    np.testing.assert_allclose(
        np.asarray(value_gradient, dtype=np.float32),
        expected_value_gradient,
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        np.asarray(weight_gradient, dtype=np.float32),
        expected_weight_gradient,
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        float(factor_gradient),
        expected_factor_gradient,
        rtol=2e-2,
        atol=2e-2,
    )


def _loss(
    values: jax.Array,
    destinations: jax.Array,
    slots: jax.Array,
    output_rows: int,
) -> jax.Array:
    received, _, _ = mnnvl_dispatch(values, destinations, slots, output_rows)
    return jnp.sum(received.astype(jnp.float32) ** 2)


def main() -> None:
    initialize_jax()
    world_size = jax.process_count()
    rank = jax.process_index()
    send_rows = world_size * _ROWS_PER_DESTINATION
    output_rows = send_rows
    ensure_mnnvl_runtime(
        buffer_rows=send_rows,
        row_bytes=_HIDDEN_DIM * jnp.dtype(jnp.bfloat16).itemsize,
    )

    probe = probe_mnnvl_peer_writes()
    expected_probe = np.arange(world_size, dtype=np.int32) * 1000 + rank
    np.testing.assert_array_equal(probe, expected_probe)

    row = jnp.arange(send_rows, dtype=jnp.int32)
    destinations = row // _ROWS_PER_DESTINATION
    local_slot = row % _ROWS_PER_DESTINATION
    slots = rank * _ROWS_PER_DESTINATION + local_slot
    values = jnp.broadcast_to(
        (rank * send_rows + row)[:, None],
        (send_rows, _HIDDEN_DIM),
    ).astype(jnp.bfloat16)

    value_and_grad = jax.jit(jax.value_and_grad(_loss), static_argnums=(3,))
    for _ in range(3):
        loss, gradient = value_and_grad(values, destinations, slots, output_rows)
        gradient.block_until_ready()
        np.testing.assert_array_equal(
            np.asarray(gradient, dtype=np.float32),
            np.asarray(2 * values, dtype=np.float32),
        )
        if not np.isfinite(float(loss)):
            raise AssertionError(f"non-finite MNNVL loss on rank {rank}: {loss}")

    _check_end_to_end_custom_vjp(rank, world_size)

    if rank == 0:
        print(f"MNNVL_ROUNDTRIP_AND_CUSTOM_VJP_PASS world_size={world_size} send_rows={send_rows}", flush=True)


if __name__ == "__main__":
    main()
