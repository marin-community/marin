# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np

from levanter.grug._moe.source_push_plan import (
    build_source_push_semantic_plan_jax,
    source_push_semantic_queue_metadata_jax,
)
from levanter.grug._moe.source_push_semantic_fused_w2_backward import (
    SourcePushSemanticFusedW2BackwardConfig,
    _make_source_push_semantic_fused_w2_backward_kernel,
    source_push_semantic_fused_w2_backward,
    source_push_semantic_fused_w2_backward_generation_accounting,
    source_push_semantic_fused_w2_backward_metadata_jax,
)


CONFIG = SourcePushSemanticFusedW2BackwardConfig()
TEST_HIDDEN_DIM = 512


def test_fused_w2_backward_transport_rows_are_independent_from_compute_rows():
    config = SourcePushSemanticFusedW2BackwardConfig(send_m=128)

    config.validate()
    assert config.compute_blocks_per_send == 2


def _plan(*, rows_per_pair: int = 12, rows_per_rank: int = 6):
    selected_experts = jnp.asarray(
        [
            [[0, 2], [0, 3], [1, 2], [0, 3], [1, 2], [0, 3]],
            [[3, 1], [2, 0], [3, 0], [2, 1], [3, 1], [2, 0]],
        ],
        dtype=jnp.int32,
    )[:, :rows_per_rank]
    route_weights = (jnp.arange(selected_experts.size, dtype=jnp.float32).reshape(selected_experts.shape) % 7 + 1) / 8
    return build_source_push_semantic_plan_jax(
        selected_experts,
        route_weights,
        ep_size=2,
        experts_per_rank=2,
        rows_per_src_dst_capacity=rows_per_pair,
        capacity_factor=4.0,
    )


def _inputs(*, rows_per_expert_capacity: int = 256):
    plan = _plan()
    dy = (
        (jnp.arange(2 * 6 * TEST_HIDDEN_DIM, dtype=jnp.float32).reshape(2, 6, TEST_HIDDEN_DIM) % 17 - 8) / 16
    ).astype(jnp.bfloat16)
    return_y = (
        (
            jnp.arange(2 * 2 * 4 * 64 * TEST_HIDDEN_DIM, dtype=jnp.float32).reshape(2, 2, 4, 64, TEST_HIDDEN_DIM) % 13
            - 6
        )
        / 32
    ).astype(jnp.bfloat16)
    z13_expert = (
        (
            jnp.arange(2 * 2 * rows_per_expert_capacity * 256, dtype=jnp.float32).reshape(
                2, 2, rows_per_expert_capacity, 256
            )
            % 11
            - 5
        )
        / 32
    ).astype(jnp.bfloat16)
    w_down = (
        (jnp.arange(2 * 2 * 128 * TEST_HIDDEN_DIM, dtype=jnp.float32).reshape(2, 2, 128, TEST_HIDDEN_DIM) % 7 - 3) / 64
    ).astype(jnp.bfloat16)
    return dy, return_y, z13_expert, w_down, plan


def _independent_reference(dy, return_y, z13_expert, w_down, plan, *, capacity: int):
    dy_host = np.asarray(dy, dtype=np.float32)
    return_y_host = np.asarray(return_y, dtype=np.float32)
    z13_host = np.asarray(z13_expert, dtype=np.float32)
    gate_host, up_host = np.split(z13_host, 2, axis=-1)
    silu_host = gate_host / (1.0 + np.exp(-gate_host))
    h_host = np.asarray(jnp.asarray(silu_host * up_host).astype(jnp.bfloat16), dtype=np.float32)
    w_host = np.asarray(w_down, dtype=np.float32)
    assignment_ids = np.asarray(plan.assignment_ids)
    pair_valid = np.asarray(plan.valid_mask)
    pair_bases = np.asarray(plan.pair_expert_base)
    counts = np.asarray(plan.xcounts)
    weights = np.asarray(plan.route_weights, dtype=np.float32)
    rounded_counts = ((counts + CONFIG.compute_m - 1) // CONFIG.compute_m) * CONFIG.compute_m
    source_bases = np.zeros((2, 2, 2), dtype=np.int32)
    source_bases[:, 1, :] = rounded_counts[0]
    queue = source_push_semantic_queue_metadata_jax(plan, return_row_block=64, entries_per_dst=4)
    queue_expert = np.asarray(queue.local_expert)
    queue_row_start = np.asarray(queue.local_row_start)
    queue_valid_rows = np.asarray(queue.valid_rows)

    d_h = np.zeros((2, 2, capacity, 128), dtype=np.float32)
    d_w2 = np.zeros(w_host.shape, dtype=np.float32)
    d_route = np.zeros((2, 6, 2), dtype=np.float32)
    valid = np.zeros((2, 2, capacity), dtype=np.bool_)
    for src in range(2):
        for dst_ordinal in range(2):
            dst = (src + dst_ordinal) % 2
            for entry in range(4):
                expert = int(queue_expert[src, dst_ordinal, entry])
                for row in range(int(queue_valid_rows[src, dst_ordinal, entry])):
                    local_row = int(queue_row_start[src, dst_ordinal, entry]) + row
                    pair_row = int(pair_bases[src, dst, expert]) + local_row
                    if not pair_valid[src, dst, pair_row]:
                        continue
                    expert_row = int(source_bases[dst, src, expert]) + local_row
                    if expert_row >= capacity:
                        continue
                    assignment = int(assignment_ids[src, dst, pair_row])
                    token = assignment // plan.topk
                    slot = assignment % plan.topk
                    dy_token = dy_host[src, token]
                    dy_route = dy_token * weights[src, dst, pair_row]
                    d_h[dst, expert, expert_row] = dy_route @ w_host[dst, expert].T
                    d_w2[dst, expert] += np.outer(h_host[dst, expert, expert_row], dy_route)
                    d_route[src, token, slot] += np.dot(dy_token, return_y_host[src, dst_ordinal, entry, row])
                    valid[dst, expert, expert_row] = True
    sigmoid_gate = 1.0 / (1.0 + np.exp(-gate_host))
    d_silu_gate = sigmoid_gate * (1.0 + gate_host * (1.0 - sigmoid_gate))
    d_z13 = np.concatenate((d_h * up_host * d_silu_gate, d_h * silu_host), axis=-1)
    d_z13 = np.asarray(jnp.asarray(d_z13).astype(jnp.bfloat16), dtype=np.float32)
    d_z13[~valid] = 0.0
    return d_z13, d_w2, d_route, valid


def test_fused_w2_backward_metadata_is_jittable_and_preserves_source_routes():
    dy, _route_y, _h, _w, plan = _inputs()
    metadata = jax.jit(
        lambda dy_arg, plan_arg: source_push_semantic_fused_w2_backward_metadata_jax(
            dy_arg,
            plan_arg,
            send_chunks_per_dst=1,
            rows_per_expert_capacity=256,
        )
    )(dy, plan)

    assert metadata.token_ids.shape == (2, 2, 1, 4, 64)
    assert metadata.route_slots.shape == metadata.token_ids.shape
    assert int(metadata.live_send_blocks + metadata.masked_send_blocks) == 2 * 2 * 1 * 4
    assert int(metadata.live_send_blocks) > 0
    assert int(metadata.masked_send_blocks) > 0
    np.testing.assert_array_equal(np.asarray(metadata.route_slots)[~np.asarray(metadata.row_valid)], 0)
    np.testing.assert_array_equal(np.asarray(metadata.route_weights)[~np.asarray(metadata.row_valid)], 0)

    produced_compact_blocks = set()
    for dst in range(2):
        for src_ordinal in range(2):
            src = (dst + src_ordinal) % 2
            dst_ordinal = (-src_ordinal) % 2
            np.testing.assert_array_equal(
                np.asarray(metadata.recv_expert[dst, src_ordinal]),
                np.asarray(metadata.send_expert[src, dst_ordinal]),
            )
            np.testing.assert_array_equal(
                np.asarray(metadata.recv_row_start[dst, src_ordinal]),
                np.asarray(metadata.send_row_start[src, dst_ordinal]),
            )
            for chunk in range(metadata.send_chunks_per_dst):
                for block in range(CONFIG.compute_blocks_per_send):
                    if int(metadata.recv_valid_rows[dst, src_ordinal, chunk, block]) == 0:
                        continue
                    compact_block = (
                        dst,
                        int(metadata.recv_expert[dst, src_ordinal, chunk, block]),
                        int(metadata.recv_row_start[dst, src_ordinal, chunk, block]) // CONFIG.compute_m,
                    )
                    assert compact_block not in produced_compact_blocks
                    produced_compact_blocks.add(compact_block)

    for src in range(2):
        for dst_ordinal in range(2):
            dst = (src + dst_ordinal) % 2
            for block in range(4):
                expert = int(metadata.send_expert[src, dst_ordinal, 0, block])
                count = int(metadata.send_valid_rows[src, dst_ordinal, 0, block])
                if expert < 0:
                    assert count == 0
                    continue
                pair_start = int(plan.pair_expert_base[src, dst, expert]) + block * 0
                # Queue blocks are expert-local; token/slot identity must come from the semantic assignment id.
                for row in range(count):
                    token = int(metadata.token_ids[src, dst_ordinal, 0, block, row])
                    slot = int(metadata.route_slots[src, dst_ordinal, 0, block, row])
                    assert token * plan.topk + slot in np.asarray(plan.assignment_ids[src, dst])
                assert pair_start >= 0


def test_fused_w2_backward_generation_accounting_tracks_direct_compact_producers():
    first = source_push_semantic_fused_w2_backward_generation_accounting(
        0,
        hidden_dim=2560,
        intermediate_dim=1280,
        send_chunks_per_dst=24,
    )
    later = source_push_semantic_fused_w2_backward_generation_accounting(
        12,
        hidden_dim=2560,
        intermediate_dim=1280,
        send_chunks_per_dst=24,
    )
    next_chunk = source_push_semantic_fused_w2_backward_generation_accounting(
        1,
        hidden_dim=2560,
        intermediate_dim=1280,
        send_chunks_per_dst=24,
    )
    assert first.chunk == 0
    assert first.owner == 0
    assert next_chunk.owner == 1
    assert first.helper_tiles == CONFIG.compute_blocks_per_send * (2560 // CONFIG.send_hidden_block)
    assert first.helper_tiles == 20
    assert (first.helper_tiles + CONFIG.helper_programs_per_peer - 1) // CONFIG.helper_programs_per_peer == 4
    assert first.prepare_generation == 1
    assert first.helper_done_generation == first.helper_tiles
    assert first.expert_block_ready_arrivals == 2560 // CONFIG.send_hidden_block
    assert first.expert_block_ready_arrivals == 5
    assert later.chunk == 12
    assert later.owner == 0
    assert later.helper_tiles == first.helper_tiles
    assert later.prepare_generation == 13
    assert later.helper_done_generation == 13 * first.helper_done_generation
    assert later.expert_block_ready_arrivals == first.expert_block_ready_arrivals


def test_fused_w2_backward_config_reduces_producer_residency():
    assert CONFIG.chunk_owner_programs_per_peer == 2
    assert CONFIG.send_hidden_block == 512
    assert CONFIG.helper_programs_per_peer == 5
    assert CONFIG.consumer_programs_per_peer == 20
    producer_programs = 8 * (CONFIG.chunk_owner_programs_per_peer + CONFIG.helper_programs_per_peer)
    total_programs = producer_programs + 8 * CONFIG.consumer_programs_per_peer
    assert producer_programs == 56
    assert total_programs == 216
    assert 132 - producer_programs == 76


def test_fused_w2_backward_interpret_matches_independent_rough_route_reference():
    dy, return_y, z13_expert, w_down, plan = _inputs()
    observed = jax.jit(
        lambda dy_arg, return_y_arg, z13_arg, w_arg, plan_arg: source_push_semantic_fused_w2_backward(
            dy_arg,
            return_y_arg,
            z13_arg,
            w_arg,
            plan_arg,
            send_chunks_per_dst=1,
            rows_per_expert_capacity=256,
            interpret=True,
        )
    )(dy, return_y, z13_expert, w_down, plan)
    expected_dz13, expected_dw2, expected_droute, expected_valid = _independent_reference(
        dy, return_y, z13_expert, w_down, plan, capacity=256
    )

    assert observed.d_z13.dtype == jnp.bfloat16
    np.testing.assert_allclose(np.asarray(observed.d_z13, dtype=np.float32), expected_dz13, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(np.asarray(observed.d_w2), expected_dw2, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(np.asarray(observed.d_route_weight), expected_droute, rtol=2e-4, atol=2e-4)
    np.testing.assert_array_equal(np.asarray(observed.valid), expected_valid)
    np.testing.assert_array_equal(np.asarray(observed.d_z13)[~expected_valid], 0)
    assert int(observed.queue_overflow_routes) == 0
    assert int(observed.layout_overflow_rows) == 0


def test_fused_w2_backward_reports_queue_and_layout_overflow_and_masks_outputs():
    selected = jnp.asarray(
        [
            [[0, 8], [1, 9], [2, 10], [3, 11], [4, 12], [0, 8]],
            [[0, 8], [1, 9], [2, 10], [3, 11], [4, 12], [0, 8]],
        ],
        dtype=jnp.int32,
    )
    plan = build_source_push_semantic_plan_jax(
        selected,
        jnp.ones(selected.shape, dtype=jnp.float32),
        ep_size=2,
        experts_per_rank=8,
        rows_per_src_dst_capacity=12,
        capacity_factor=4.0,
    )
    dy = jnp.ones((2, 6, 256), dtype=jnp.bfloat16)
    metadata = source_push_semantic_fused_w2_backward_metadata_jax(
        dy,
        plan,
        send_chunks_per_dst=1,
        rows_per_expert_capacity=64,
    )

    assert int(metadata.queue_overflow_routes) > 0
    assert int(metadata.layout_overflow_rows) > 0
    assert int(metadata.masked_send_blocks) > 0
    np.testing.assert_array_equal(np.asarray(metadata.route_weights)[~np.asarray(metadata.row_valid)], 0)


def test_fused_w2_backward_kernel_contract_streams_compact_rows_to_owned_dw2_tiles():
    source = inspect.getsource(_make_source_push_semantic_fused_w2_backward_kernel)

    assert "mgpu.remote_ref" in source
    assert "mgpu.wgmma" in source
    assert "mgpu.transpose_ref" in source
    assert "lowering_semantics=mgpu.LoweringSemantics.Lane" in source
    assert "all_gather" not in source
    assert "config.inbox_slots" not in source
    assert "jax.ShapeDtypeStruct((experts_per_rank, rows_per_expert_capacity, hidden_dim), dtype)" in source
    assert "pl.dot" not in source
    assert "(chunk % config.chunk_owner_programs_per_peer) == owner" in source
    assert "prepare_sem.at[dst]" in source
    assert "value=(chunk + 1) * helper_tiles" in source
    assert "mgpu.SemaphoreType.REGULAR((experts_per_rank, compact_m_blocks))" in source
    assert "producer_programs_per_peer = consumer_start" in source
    assert "producer_programs = ep_size * producer_programs_per_peer" in source
    assert "total_programs = producer_programs + dw2_owner_programs" in source
    assert "is_producer = physical_program < producer_programs" in source
    assert "physical_program // producer_programs_per_peer" in source
    assert "(physical_program - producer_programs) // config.consumer_programs_per_peer" in source
    assert "consumer_start + (physical_program - producer_programs) % config.consumer_programs_per_peer" in source
    assert "grid=(total_programs,)" in source
    assert 'grid_names=("physical_program",)' in source
    assert "pl.program_id(1)" not in source
    assert "(worker >= helper_start) & (worker < consumer_start)" in source
    assert "tile = helper + helper_iteration * config.helper_programs_per_peer" in source
    assert "lower_smem" in source
    assert "upper_smem" in source
    assert source.count("mgpu.copy_smem_to_gmem(") == 2
    assert source.count("mgpu.wait_smem_to_gmem(0, wait_read_only=False)") == 1
    assert "pl.ds(hidden_start, config.send_hidden_block // 2)" in source
    assert "hidden_start + config.send_hidden_block // 2" in source
    assert "for route_hidden_start in range(0, hidden_dim, config.send_hidden_block)" in source
    assert "@pl.when(hidden_tile == 0)" in source
    assert "remote_dy_expert = mgpu.remote_ref" in source
    assert "pl.ds(row_start, config.compute_m)" in source
    assert "_signal_remote_compact_block(compact_ready_sem, peer, expert, compact_block)" in source
    assert "pl.semaphore_signal(helper_done_sem.at[peer])" in source
    assert "@pl.when(worker >= consumer_start)" in source
    assert source.count("compact_ready_sem.at[expert, compact_block]") >= 3
    assert source.count("value=send_hidden_tiles") == 2
    assert "global_owner = peer_ordinal * config.consumer_programs_per_peer + consumer" in source
    assert "tile = global_owner + tile_iteration * dw2_owner_programs" in source
    assert "@pl.loop(0, compact_m_blocks)" in source
    assert "valid_ref[expert, row_start]" in source
    assert "] = acc_ref[...]" in source
    assert "mgpu.atomic_add" not in source
    assert "jax.nn.silu(gate_smem[:, :].astype(jnp.float32))" in source
    assert "d_z13_ref" in source
    assert "d_h_ref" not in source
