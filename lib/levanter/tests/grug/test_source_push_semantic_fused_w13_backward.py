# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from levanter.grug._moe.source_push_plan import build_source_push_semantic_plan_jax
from levanter.grug._moe.source_push_semantic_fused_w13_backward import (
    SourcePushSemanticFusedW13BackwardConfig,
    source_push_semantic_fused_w13_backward,
    source_push_semantic_fused_w13_backward_generation_accounting,
    source_push_semantic_fused_w13_backward_metadata_jax,
)


CONFIG = SourcePushSemanticFusedW13BackwardConfig()


def _plan(*, rows_per_src_dst_capacity: int = 12, rows_per_expert_capacity: int | None = None):
    selected_experts = jnp.asarray(
        [
            [[0, 2], [0, 3], [1, 2], [0, 3], [1, 2], [0, 3]],
            [[3, 1], [2, 0], [3, 0], [2, 1], [3, 1], [2, 0]],
        ],
        dtype=jnp.int32,
    )
    route_weights = jnp.asarray(
        [
            [[0.7, 0.3], [0.6, 0.4], [0.8, 0.2], [0.5, 0.5], [0.9, 0.1], [0.4, 0.6]],
            [[0.2, 0.8], [0.3, 0.7], [0.55, 0.45], [0.1, 0.9], [0.65, 0.35], [0.75, 0.25]],
        ],
        dtype=jnp.float32,
    )
    return build_source_push_semantic_plan_jax(
        selected_experts,
        route_weights,
        ep_size=2,
        experts_per_rank=2,
        rows_per_src_dst_capacity=rows_per_src_dst_capacity,
        rows_per_expert_capacity=rows_per_expert_capacity,
        capacity_factor=4.0,
    )


def _inputs(plan=None):
    plan = _plan() if plan is None else plan
    x = ((jnp.arange(2 * 6 * 256, dtype=jnp.float32).reshape(2, 6, 256) % 19 - 9) / 16).astype(jnp.bfloat16)
    w13 = ((jnp.arange(2 * 2 * 256 * 128, dtype=jnp.float32).reshape(2, 2, 256, 128) % 17 - 8) / 32).astype(
        jnp.bfloat16
    )
    dz13 = ((jnp.arange(2 * 2 * 256 * 128, dtype=jnp.float32).reshape(2, 2, 256, 128) % 23 - 11) / 64).astype(
        jnp.bfloat16
    )
    return x, dz13, w13, plan


def test_fused_w13_backward_metadata_inverts_chunk_rows_and_rotates_peers():
    x, _dz13, _w13, plan = _inputs()
    metadata = jax.jit(
        lambda x_arg, plan_arg: source_push_semantic_fused_w13_backward_metadata_jax(
            x_arg,
            plan_arg,
            send_chunks_per_dst=1,
            rows_per_expert_capacity=256,
        )
    )(x, plan)

    route_valid = np.asarray(metadata.route_valid)
    assert np.count_nonzero(route_valid) == int(np.asarray(jnp.sum(plan.xcounts)))
    token_ids = np.asarray(metadata.forward.token_ids)
    for source, token, route in np.argwhere(route_valid):
        dst_ordinal = int(metadata.route_dst_ordinal[source, token, route])
        chunk = int(metadata.route_chunk[source, token, route])
        block = int(metadata.route_block[source, token, route])
        row = int(metadata.route_row[source, token, route])
        assert token_ids[source, dst_ordinal, chunk, block, row] == token

    np.testing.assert_array_equal(
        np.asarray(metadata.recv_return_consumed_target[0, 1]),
        np.asarray(metadata.send_return_consumed_target[1, 1]),
    )
    expected_return_tiles = np.sum(np.asarray(metadata.forward.send_valid_rows), axis=-1) * 2
    np.testing.assert_array_equal(np.asarray(metadata.send_return_consumed_target), expected_return_tiles)


def test_fused_w13_backward_metadata_maps_three_peer_source_ordinals_to_send_targets():
    selected_experts = jnp.asarray(
        [
            [[0, 1], [0, 1], [0, 1]],
            [[2, 0], [2, 0], [2, 0]],
            [[2, 1], [2, 1], [2, 1]],
        ],
        dtype=jnp.int32,
    )
    plan = build_source_push_semantic_plan_jax(
        selected_experts,
        jnp.ones(selected_experts.shape, dtype=jnp.float32),
        ep_size=3,
        experts_per_rank=1,
        rows_per_src_dst_capacity=6,
        capacity_factor=4.0,
    )
    metadata = source_push_semantic_fused_w13_backward_metadata_jax(
        jnp.zeros((3, 3, 256), dtype=jnp.bfloat16),
        plan,
        send_chunks_per_dst=1,
        rows_per_expert_capacity=256,
    )

    send_targets = np.asarray(metadata.send_return_consumed_target)
    recv_targets = np.asarray(metadata.recv_return_consumed_target)
    for destination in range(3):
        for source_ordinal in range(3):
            source = (destination + source_ordinal) % 3
            destination_ordinal = (-source_ordinal) % 3
            np.testing.assert_array_equal(
                recv_targets[destination, source_ordinal],
                send_targets[source, destination_ordinal],
            )


def test_fused_w13_backward_metadata_invalidates_routes_clipped_from_forward_send():
    selected_experts = jnp.zeros((2, 40, 2), dtype=jnp.int32)
    plan = build_source_push_semantic_plan_jax(
        selected_experts,
        jnp.ones(selected_experts.shape, dtype=jnp.float32),
        ep_size=2,
        experts_per_rank=2,
        rows_per_src_dst_capacity=80,
        capacity_factor=4.0,
    )
    x = jnp.zeros((2, 40, 256), dtype=jnp.bfloat16)

    metadata = source_push_semantic_fused_w13_backward_metadata_jax(
        x,
        plan,
        send_chunks_per_dst=1,
        rows_per_expert_capacity=64,
    )

    route_valid = np.asarray(metadata.route_valid)
    send_valid_rows = np.asarray(metadata.forward.send_valid_rows)
    assert np.count_nonzero(route_valid) == int(np.sum(send_valid_rows))
    assert np.count_nonzero(route_valid) < int(np.asarray(jnp.sum(plan.xcounts)))

    token_ids = np.asarray(metadata.forward.token_ids)
    for source, token, route in np.argwhere(route_valid):
        dst_ordinal = int(metadata.route_dst_ordinal[source, token, route])
        chunk = int(metadata.route_chunk[source, token, route])
        block = int(metadata.route_block[source, token, route])
        row = int(metadata.route_row[source, token, route])
        assert row < send_valid_rows[source, dst_ordinal, chunk, block]
        assert token_ids[source, dst_ordinal, chunk, block, row] == token


def test_fused_w13_backward_generation_accounting_reuses_slots_and_counts_live_returns():
    first = source_push_semantic_fused_w13_backward_generation_accounting(
        0,
        ep_size=8,
        hidden_dim=2560,
        valid_rows=177,
    )
    reused = source_push_semantic_fused_w13_backward_generation_accounting(
        CONFIG.inbox_slots,
        ep_size=8,
        hidden_dim=2560,
        valid_rows=31,
    )

    assert (first.slot, first.generation, first.empty_generation, first.released_generation) == (0, 1, 1, 2)
    assert reused.slot == first.slot
    assert reused.generation == 2
    assert reused.send_done_generation == 2 * first.send_done_generation
    assert (first.dx_ready_generation, reused.dx_ready_generation) == (1, 2)
    assert reused.compute_done_generation == 2 * first.compute_done_generation
    assert first.returned_route_tiles == 177 * (2560 // CONFIG.block_hidden)
    assert reused.returned_route_tiles == 31 * (2560 // CONFIG.block_hidden)


def test_fused_w13_backward_interpret_matches_independent_route_reference():
    x, dz13, w13, plan = _inputs()
    result = jax.jit(
        lambda x_arg, dz_arg, w_arg, plan_arg: source_push_semantic_fused_w13_backward(
            x_arg,
            dz_arg,
            w_arg,
            plan_arg,
            send_chunks_per_dst=1,
            rows_per_expert_capacity=256,
            interpret=True,
        )
    )(x, dz13, w13, plan)

    expected_dx, expected_dw = _independent_backward_reference(x, dz13, w13, plan)
    np.testing.assert_allclose(np.asarray(result.dx), expected_dx, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(np.asarray(result.dw13), expected_dw, rtol=2e-4, atol=2e-4)
    assert int(result.queue_overflow_routes) == 0
    assert int(result.layout_overflow_rows) == 0


def test_fused_w13_backward_reference_masks_padding_and_reports_independent_overflow():
    overflow_plan = _plan(rows_per_src_dst_capacity=2, rows_per_expert_capacity=2)
    x, dz13, w13, _ = _inputs(overflow_plan)
    metadata = source_push_semantic_fused_w13_backward_metadata_jax(
        x,
        overflow_plan,
        send_chunks_per_dst=1,
        rows_per_expert_capacity=64,
    )
    clean_dz = jnp.where(metadata.forward.valid[..., None], dz13[:, :, :64], 0)
    dirty_dz = jnp.where(metadata.forward.valid[..., None], clean_dz, jnp.full_like(clean_dz, 1.0e4))

    clean = source_push_semantic_fused_w13_backward(
        x,
        clean_dz,
        w13,
        overflow_plan,
        send_chunks_per_dst=1,
        rows_per_expert_capacity=64,
        interpret=True,
    )
    dirty = source_push_semantic_fused_w13_backward(
        x,
        dirty_dz,
        w13,
        overflow_plan,
        send_chunks_per_dst=1,
        rows_per_expert_capacity=64,
        interpret=True,
    )
    clean_dx, clean_dw = clean.dx, clean.dw13
    dirty_dx, dirty_dw = dirty.dx, dirty.dw13
    np.testing.assert_array_equal(np.asarray(dirty_dx), np.asarray(clean_dx))
    np.testing.assert_array_equal(np.asarray(dirty_dw), np.asarray(clean_dw))
    assert int(overflow_plan.metadata_overflow_routes) > 0

    queue_selected = jnp.asarray(
        [
            [[0, 1], [2, 3], [4, 5], [6, 7], [0, 1], [2, 3]],
            [[8, 9], [10, 11], [12, 13], [14, 15], [8, 9], [10, 11]],
        ],
        dtype=jnp.int32,
    )
    queue_plan = build_source_push_semantic_plan_jax(
        queue_selected,
        jnp.ones(queue_selected.shape, dtype=jnp.float32),
        ep_size=2,
        experts_per_rank=8,
        rows_per_src_dst_capacity=12,
        capacity_factor=4.0,
    )
    queue_overflow_metadata = source_push_semantic_fused_w13_backward_metadata_jax(
        _inputs()[0],
        queue_plan,
        send_chunks_per_dst=1,
        rows_per_expert_capacity=256,
    )
    assert int(queue_overflow_metadata.forward.queue_overflow_routes) > 0

    layout_selected = jnp.zeros((2, 6, 2), dtype=jnp.int32)
    layout_plan = build_source_push_semantic_plan_jax(
        layout_selected,
        jnp.ones(layout_selected.shape, dtype=jnp.float32),
        ep_size=2,
        experts_per_rank=2,
        rows_per_src_dst_capacity=12,
        capacity_factor=4.0,
    )
    layout_overflow_metadata = source_push_semantic_fused_w13_backward_metadata_jax(
        _inputs()[0], layout_plan, send_chunks_per_dst=1, rows_per_expert_capacity=64
    )
    assert int(layout_overflow_metadata.forward.layout_overflow_rows) > 0


def _independent_backward_reference(x, dz13, w13, plan):
    x_host = np.asarray(x, dtype=np.float32)
    dz_host = np.asarray(dz13, dtype=np.float32)
    w_host = np.asarray(w13, dtype=np.float32)
    xcounts = np.asarray(plan.xcounts)
    pair_bases = np.asarray(plan.pair_expert_base)
    assignment_ids = np.asarray(plan.assignment_ids)
    rounded_counts = ((xcounts + CONFIG.compute_m - 1) // CONFIG.compute_m) * CONFIG.compute_m
    source_bases = np.zeros_like(xcounts)
    source_bases[1:] = np.cumsum(rounded_counts[:-1], axis=0)
    expected_dx = np.zeros_like(x_host, dtype=np.float32)
    expected_dw = np.zeros_like(w_host, dtype=np.float32)

    for source in range(xcounts.shape[0]):
        for destination in range(xcounts.shape[1]):
            for expert in range(xcounts.shape[2]):
                for local_row in range(int(xcounts[source, destination, expert])):
                    pair_row = int(pair_bases[source, destination, expert]) + local_row
                    assignment = int(assignment_ids[source, destination, pair_row])
                    token = assignment // plan.topk
                    expert_row = int(source_bases[source, destination, expert]) + local_row
                    dz_row = dz_host[destination, expert, expert_row]
                    expected_dx[source, token] += dz_row @ w_host[destination, expert].T
                    expected_dw[destination, expert] += np.outer(x_host[source, token], dz_row)
    return expected_dx, expected_dw
