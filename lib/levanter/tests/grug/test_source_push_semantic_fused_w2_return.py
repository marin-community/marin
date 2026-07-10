# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from levanter.grug._moe.source_push_plan import build_source_push_semantic_plan_jax
from levanter.grug._moe.source_push_semantic_fused_w2_return import (
    SourcePushSemanticFusedW2ReturnConfig,
    source_push_semantic_fused_w2_return,
    source_push_semantic_fused_w2_return_metadata_jax,
    source_push_semantic_fused_w2_return_schedule,
)


CONFIG = SourcePushSemanticFusedW2ReturnConfig()
SOURCE_COUNT = 2
TOKENS = 64
TOPK = 2
EXPERTS_PER_RANK = 2
ROWS_PER_EXPERT = 128
ENTRIES_PER_DST = 4
INTERMEDIATE = 64
HIDDEN = 128


def test_fused_w2_return_target_schedule_matches_stable_inbox_worker_topology():
    schedule = source_push_semantic_fused_w2_return_schedule(
        ep_size=8,
        hidden_dim=2560,
        tokens_per_source=32768,
        entries_per_dst=288,
    )

    assert schedule.hidden_tiles == 20
    assert schedule.hidden_tile_jobs == 10
    assert schedule.rounds == 24
    assert schedule.chunk_owner_programs == 16
    assert schedule.compute_programs == 240
    assert schedule.producer_programs == 256
    assert schedule.combine_programs == 32
    assert schedule.active_combine_programs == 32
    assert schedule.total_programs == 288
    assert schedule.readiness_signals == 2304
    assert schedule.readiness_waits == 983040


def _inputs(*, rows_per_expert: int = ROWS_PER_EXPERT):
    token = jnp.arange(TOKENS, dtype=jnp.int32)[None, :, None]
    source = jnp.arange(SOURCE_COUNT, dtype=jnp.int32)[:, None, None]
    route = jnp.arange(TOPK, dtype=jnp.int32)[None, None, :]
    selected_experts = (token * 3 + source + route * 2) % (SOURCE_COUNT * EXPERTS_PER_RANK)
    route_weights = (0.25 + 0.125 * route + 0.01 * (token % 5)).astype(jnp.float32)
    route_weights = jnp.broadcast_to(route_weights, selected_experts.shape)
    plan = build_source_push_semantic_plan_jax(
        selected_experts,
        route_weights,
        ep_size=SOURCE_COUNT,
        experts_per_rank=EXPERTS_PER_RANK,
        rows_per_src_dst_capacity=128,
        capacity_factor=4.0,
    )
    z = (
        (
            jnp.arange(
                SOURCE_COUNT * EXPERTS_PER_RANK * rows_per_expert * 2 * INTERMEDIATE, dtype=jnp.float32
            ).reshape(SOURCE_COUNT, EXPERTS_PER_RANK, rows_per_expert, 2 * INTERMEDIATE)
            % 19
        )
        - 9
    ) / 16
    w_down = (
        (
            jnp.arange(SOURCE_COUNT * EXPERTS_PER_RANK * INTERMEDIATE * HIDDEN, dtype=jnp.float32).reshape(
                SOURCE_COUNT, EXPERTS_PER_RANK, INTERMEDIATE, HIDDEN
            )
            % 17
        )
        - 8
    ) / 64
    return z.astype(jnp.bfloat16), w_down.astype(jnp.bfloat16), plan


def _host_reference(z, w_down, plan, metadata):
    z_host = np.asarray(z, dtype=np.float32)
    w_host = np.asarray(w_down, dtype=np.float32)
    assignment_ids = np.asarray(plan.assignment_ids)
    pair_valid = np.asarray(plan.valid_mask)
    pair_expert_base = np.asarray(plan.pair_expert_base)
    xcounts = np.asarray(plan.xcounts)
    route_weights = np.asarray(plan.route_weights, dtype=np.float32)
    route_valid = np.asarray(metadata.route_valid)
    queue_entry = np.asarray(metadata.queue_entry)
    queue_row = np.asarray(metadata.queue_row)
    queue_dst = np.asarray(metadata.queue_dst_ordinal)

    rounded_counts = ((xcounts + CONFIG.compute_m - 1) // CONFIG.compute_m) * CONFIG.compute_m
    source_bases = np.cumsum(rounded_counts, axis=0, dtype=np.int32) - rounded_counts
    expected_return = np.zeros(
        (SOURCE_COUNT, SOURCE_COUNT, metadata.entries_per_dst, CONFIG.compute_m, HIDDEN),
        dtype=np.float32,
    )
    expected_y = np.zeros((SOURCE_COUNT, TOKENS, HIDDEN), dtype=np.float32)

    for src in range(SOURCE_COUNT):
        for dst in range(SOURCE_COUNT):
            for pair_row in range(assignment_ids.shape[-1]):
                if not pair_valid[src, dst, pair_row]:
                    continue
                assignment = int(assignment_ids[src, dst, pair_row])
                token = assignment // TOPK
                route_slot = assignment % TOPK
                if not route_valid[src, token, route_slot]:
                    continue
                local_expert = next(
                    expert
                    for expert in range(EXPERTS_PER_RANK)
                    if pair_expert_base[src, dst, expert]
                    <= pair_row
                    < pair_expert_base[src, dst, expert] + xcounts[src, dst, expert]
                )
                local_row = pair_row - int(pair_expert_base[src, dst, local_expert])
                expert_row = int(source_bases[src, dst, local_expert]) + local_row
                gate = z_host[dst, local_expert, expert_row, :INTERMEDIATE]
                up = z_host[dst, local_expert, expert_row, INTERMEDIATE:]
                h = gate / (1.0 + np.exp(-gate)) * up
                route_y = h @ w_host[dst, local_expert]
                route_y = np.asarray(jnp.asarray(route_y, dtype=jnp.bfloat16), dtype=np.float32)
                dst_ordinal = int(queue_dst[src, token, route_slot])
                entry = int(queue_entry[src, token, route_slot])
                row = int(queue_row[src, token, route_slot])
                expected_return[src, dst_ordinal, entry, row] = route_y
                expected_y[src, token] += route_y * route_weights[src, dst, pair_row]

    expected_y = np.asarray(jnp.asarray(expected_y, dtype=jnp.bfloat16), dtype=np.float32)
    return expected_y, expected_return


def test_fused_w2_return_metadata_maps_source_queue_to_destination_rows():
    z, _w_down, plan = _inputs()
    metadata = jax.jit(
        lambda plan_arg: source_push_semantic_fused_w2_return_metadata_jax(
            plan_arg,
            rows_per_expert_capacity=z.shape[2],
            entries_per_dst=ENTRIES_PER_DST,
        )
    )(plan)

    queue_expert = np.asarray(metadata.queue_local_expert)
    queue_row_start = np.asarray(metadata.queue_local_row_start)
    queue_valid_rows = np.asarray(metadata.queue_valid_rows)
    recv_expert = np.asarray(metadata.recv_local_expert)
    recv_row_start = np.asarray(metadata.recv_expert_row_start)
    recv_valid_rows = np.asarray(metadata.recv_valid_rows)
    rounded_counts = ((np.asarray(plan.xcounts) + CONFIG.compute_m - 1) // CONFIG.compute_m) * CONFIG.compute_m
    source_bases = np.cumsum(rounded_counts, axis=0, dtype=np.int32) - rounded_counts

    for dst in range(SOURCE_COUNT):
        for source_ordinal in range(SOURCE_COUNT):
            src = (dst + source_ordinal) % SOURCE_COUNT
            dst_ordinal = (-source_ordinal) % SOURCE_COUNT
            np.testing.assert_array_equal(recv_expert[dst, source_ordinal], queue_expert[src, dst_ordinal])
            np.testing.assert_array_equal(recv_valid_rows[dst, source_ordinal], queue_valid_rows[src, dst_ordinal])
            for entry in range(ENTRIES_PER_DST):
                if recv_valid_rows[dst, source_ordinal, entry] == 0:
                    assert recv_row_start[dst, source_ordinal, entry] == 0
                    continue
                expert = recv_expert[dst, source_ordinal, entry]
                assert recv_row_start[dst, source_ordinal, entry] == (
                    source_bases[src, dst, expert] + queue_row_start[src, dst_ordinal, entry]
                )


def test_fused_w2_return_metadata_tracks_required_rolling_slot_generations():
    z, _w_down, plan = _inputs()
    metadata = source_push_semantic_fused_w2_return_metadata_jax(
        plan,
        rows_per_expert_capacity=z.shape[2],
        entries_per_dst=ENTRIES_PER_DST,
    )
    expected = np.zeros_like(np.asarray(metadata.combine_ready_generation))
    queue_dst = np.asarray(metadata.queue_dst_ordinal)
    queue_entry = np.asarray(metadata.queue_entry)
    route_valid = np.asarray(metadata.route_valid)

    for source in range(SOURCE_COUNT):
        for token in range(TOKENS):
            token_block = token // CONFIG.combine_token_block
            for route_slot in range(TOPK):
                if not route_valid[source, token, route_slot]:
                    continue
                destination = queue_dst[source, token, route_slot]
                entry = queue_entry[source, token, route_slot]
                slot = entry % 12
                expected[source, token_block, destination, slot] = max(
                    expected[source, token_block, destination, slot],
                    entry // 12 + 1,
                )

    np.testing.assert_array_equal(np.asarray(metadata.combine_ready_generation), expected)


def test_fused_w2_return_interpret_matches_independent_route_reference():
    z, w_down, plan = _inputs()
    result = jax.jit(
        lambda z_arg, w_arg, plan_arg: source_push_semantic_fused_w2_return(
            z_arg,
            w_arg,
            plan_arg,
            entries_per_dst=ENTRIES_PER_DST,
            interpret=True,
        )
    )(z, w_down, plan)
    metadata = source_push_semantic_fused_w2_return_metadata_jax(
        plan,
        rows_per_expert_capacity=z.shape[2],
        entries_per_dst=ENTRIES_PER_DST,
    )
    expected_y, expected_return = _host_reference(z, w_down, plan, metadata)

    np.testing.assert_allclose(np.asarray(result.return_y, dtype=np.float32), expected_return, rtol=2e-2, atol=0.5)
    np.testing.assert_allclose(np.asarray(result.y, dtype=np.float32), expected_y, rtol=2e-2, atol=0.5)
    assert result.return_y.dtype == jnp.bfloat16
    assert result.y.dtype == jnp.bfloat16
    assert int(result.queue_overflow_routes) == 0
    assert int(result.layout_overflow_rows) == 0


def test_fused_w2_return_masks_queue_and_layout_overflow_exactly():
    z, w_down, plan = _inputs(rows_per_expert=64)
    result = source_push_semantic_fused_w2_return(
        z,
        w_down,
        plan,
        entries_per_dst=1,
        interpret=True,
    )
    metadata = source_push_semantic_fused_w2_return_metadata_jax(
        plan,
        rows_per_expert_capacity=z.shape[2],
        entries_per_dst=1,
    )
    expected_y, expected_return = _host_reference(z, w_down, plan, metadata)

    np.testing.assert_array_equal(np.asarray(result.return_y, dtype=np.float32), expected_return)
    np.testing.assert_array_equal(np.asarray(result.y, dtype=np.float32), expected_y)
    assert int(result.queue_overflow_routes) > 0
    assert int(result.layout_overflow_rows) > 0
    valid_rows = np.asarray(metadata.queue_valid_rows)
    return_y = np.asarray(result.return_y)
    for src in range(SOURCE_COUNT):
        for dst_ordinal in range(SOURCE_COUNT):
            for entry in range(metadata.entries_per_dst):
                np.testing.assert_array_equal(
                    return_y[src, dst_ordinal, entry, valid_rows[src, dst_ordinal, entry] :], 0
                )
