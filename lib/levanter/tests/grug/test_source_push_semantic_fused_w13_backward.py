# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np

from levanter.grug._moe.source_push_plan import build_source_push_semantic_plan_jax
from levanter.grug._moe.source_push_semantic_fused_w13_backward import (
    SourcePushSemanticFusedW13BackwardConfig,
    _make_source_push_semantic_fused_w13_backward_kernel,
    source_push_semantic_fused_w13_backward,
    source_push_semantic_fused_w13_backward_metadata_jax,
    source_push_semantic_fused_w13_backward_schedule,
)


CONFIG = SourcePushSemanticFusedW13BackwardConfig()


def test_fused_w13_backward_transport_rows_and_hidden_copy_width_are_independent():
    config = SourcePushSemanticFusedW13BackwardConfig(send_m=128, send_hidden_block=256)

    config.validate()
    assert config.compute_blocks_per_send == 2


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


def _inputs(plan=None, *, output_dim: int = 128):
    plan = _plan() if plan is None else plan
    x = ((jnp.arange(2 * 6 * 512, dtype=jnp.float32).reshape(2, 6, 512) % 19 - 9) / 16).astype(jnp.bfloat16)
    w13_shape = (2, 2, 512, output_dim)
    w13 = ((jnp.arange(np.prod(w13_shape), dtype=jnp.float32).reshape(w13_shape) % 17 - 8) / 32).astype(jnp.bfloat16)
    dz13_shape = (2, 2, 256, output_dim)
    dz13 = ((jnp.arange(np.prod(dz13_shape), dtype=jnp.float32).reshape(dz13_shape) % 23 - 11) / 64).astype(
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
        destination = (source + dst_ordinal) % x.shape[0]
        source_ordinal = (-dst_ordinal) % x.shape[0]
        assert (
            metadata.forward.recv_expert[destination, source_ordinal, chunk, block]
            == metadata.forward.send_expert[source, dst_ordinal, chunk, block]
        )
        assert (
            metadata.forward.recv_row_start[destination, source_ordinal, chunk, block]
            == metadata.forward.send_row_start[source, dst_ordinal, chunk, block]
        )

    ready = np.asarray(metadata.combine_ready_generation)
    for source, token, route in np.argwhere(route_valid):
        destination_ordinal = int(metadata.route_dst_ordinal[source, token, route])
        chunk = int(metadata.route_chunk[source, token, route])
        assert ready[source, 0, destination_ordinal, chunk % CONFIG.inbox_slots] >= (chunk // CONFIG.inbox_slots + 1)

    recv_valid_rows = np.asarray(metadata.forward.recv_valid_rows)
    recv_expert = np.asarray(metadata.forward.recv_expert)
    recv_row_start = np.asarray(metadata.forward.recv_row_start)
    expected_compact_rows = np.zeros_like(np.asarray(metadata.compact_valid_rows))
    for destination, source_ordinal, chunk, block in np.argwhere(recv_valid_rows > 0):
        expert = recv_expert[destination, source_ordinal, chunk, block]
        compact_block = recv_row_start[destination, source_ordinal, chunk, block] // CONFIG.compute_m
        expected_compact_rows[destination, expert, compact_block] = recv_valid_rows[
            destination, source_ordinal, chunk, block
        ]
    np.testing.assert_array_equal(np.asarray(metadata.compact_valid_rows), expected_compact_rows)
    assert np.any((expected_compact_rows > 0) & (expected_compact_rows < CONFIG.compute_m))
    assert np.count_nonzero(expected_compact_rows) < expected_compact_rows.size


def test_fused_w13_backward_metadata_tracks_three_peer_chunk_readiness():
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
        jnp.zeros((3, 3, 512), dtype=jnp.bfloat16),
        plan,
        send_chunks_per_dst=1,
        rows_per_expert_capacity=256,
    )

    route_valid = np.asarray(metadata.route_valid)
    ready = np.asarray(metadata.combine_ready_generation)
    for source, token, route in np.argwhere(route_valid):
        destination_ordinal = int(metadata.route_dst_ordinal[source, token, route])
        chunk = int(metadata.route_chunk[source, token, route])
        assert ready[source, 0, destination_ordinal, chunk % CONFIG.inbox_slots] == (chunk // CONFIG.inbox_slots + 1)


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
    x = jnp.zeros((2, 40, 512), dtype=jnp.bfloat16)

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


def test_fused_w13_backward_schedule_keeps_semantic_roles_resident():
    schedule = source_push_semantic_fused_w13_backward_schedule(
        ep_size=8,
        experts_per_rank=8,
        rows_per_expert_capacity=4096,
        hidden_dim=2560,
        output_dim=2560,
        tokens_per_source=32768,
        send_chunks_per_dst=25,
    )

    assert schedule.hidden_tiles == 20
    assert schedule.output_tiles == 20
    assert schedule.dw_output_groups == 10
    assert schedule.compact_m_blocks == 64
    assert schedule.staging_jobs == 16000
    assert schedule.dx_jobs == 16000
    assert schedule.dw_jobs == 1600
    assert schedule.max_dw_jobs_per_program == 50
    assert schedule.token_blocks == 512
    assert schedule.rounds == 3
    assert schedule.staging_programs == 64
    assert schedule.dx_program_start == 64
    assert schedule.dx_programs == 32
    assert schedule.dw_program_start == 96
    assert schedule.dw_programs == 32
    assert schedule.combine_programs == 64
    assert schedule.active_combine_programs == 64
    assert schedule.total_programs == 128
    assert schedule.max_stage_readiness_signals == 16000
    assert schedule.compact_readiness_slots == 10240
    assert schedule.dx_readiness_signals == 200
    assert schedule.readiness_waits == 983040


def test_fused_w13_backward_dw_schedule_groups_adjacent_output_tiles_and_handles_odd_tail():
    target = source_push_semantic_fused_w13_backward_schedule(
        ep_size=8,
        experts_per_rank=32,
        rows_per_expert_capacity=4096,
        hidden_dim=2560,
        output_dim=2560,
        tokens_per_source=32768,
        send_chunks_per_dst=25,
    )
    odd_tail = source_push_semantic_fused_w13_backward_schedule(
        ep_size=2,
        experts_per_rank=2,
        rows_per_expert_capacity=256,
        hidden_dim=512,
        output_dim=384,
        tokens_per_source=6,
        send_chunks_per_dst=1,
    )

    assert target.output_tiles == 20
    assert target.dw_output_groups == 10
    assert target.dw_jobs == 32 * 20 * 10
    assert target.max_dw_jobs_per_program == 200
    assert odd_tail.output_tiles == 3
    assert odd_tail.dw_output_groups == 2
    assert odd_tail.dw_jobs == 2 * 4 * 2
    assert odd_tail.max_dw_jobs_per_program == 1


def test_fused_w13_backward_dw_source_shares_x_across_two_output_accumulators():
    source = inspect.getsource(_make_source_push_semantic_fused_w13_backward_kernel)
    dw_source = source[source.index("def _own_dw_tiles") :]

    assert "output_tile_n0 = output_group * DW_OUTPUT_TILES_PER_JOB" in dw_source
    assert "output_tile_n1 = output_tile_n0 + 1" in dw_source
    assert "x_smem.at[stage]" in dw_source
    assert "dz_n0_smem.at[stage]" in dw_source
    assert "dz_n1_smem.at[stage]" in dw_source
    assert "acc_n0_ref=mgpu.ACC" in dw_source
    assert "acc_n1_ref=mgpu.ACC" in dw_source
    assert "@pl.when(has_n1)" in dw_source


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


def test_fused_w13_backward_twenty_output_tiles_match_independent_route_reference():
    x, dz13, w13, plan = _inputs(output_dim=2560)

    result = source_push_semantic_fused_w13_backward(
        x,
        dz13,
        w13,
        plan,
        send_chunks_per_dst=1,
        rows_per_expert_capacity=256,
        interpret=True,
    )

    expected_dx, expected_dw = _independent_backward_reference(x, dz13, w13, plan)
    np.testing.assert_allclose(np.asarray(result.dx), expected_dx, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(np.asarray(result.dw13), expected_dw, rtol=2e-4, atol=2e-4)


def test_fused_w13_backward_odd_output_tile_matches_independent_route_reference():
    x, dz13, w13, plan = _inputs(output_dim=384)

    result = source_push_semantic_fused_w13_backward(
        x,
        dz13,
        w13,
        plan,
        send_chunks_per_dst=1,
        rows_per_expert_capacity=256,
        interpret=True,
    )

    expected_dx, expected_dw = _independent_backward_reference(x, dz13, w13, plan)
    np.testing.assert_allclose(np.asarray(result.dx), expected_dx, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(np.asarray(result.dw13), expected_dw, rtol=2e-4, atol=2e-4)


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
