# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import re

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh

from levanter.grug._moe import source_push_semantic_fused_w13 as fused_w13
from levanter.grug._moe.source_push_plan import build_source_push_semantic_plan_jax
from levanter.grug._moe.source_push_semantic_fused_w13 import (
    SourcePushSemanticFusedW13Config,
    SourcePushSemanticFusedW13Metadata,
    RAW_GATHER_HIDDEN_LAYOUT,
    RAW_GATHER_ROW_LAYOUT,
    RAW_GATHER_ROWS,
    TOKEN_TRANSFER_ROWS,
    _make_source_push_semantic_fused_w13_kernel,
    source_push_semantic_fused_w13,
    source_push_semantic_fused_w13_generation_accounting,
    source_push_semantic_fused_w13_metadata_jax,
    source_push_semantic_fused_w13_reference_jax,
)


CONFIG = SourcePushSemanticFusedW13Config()


def _plan():
    selected_experts = jnp.asarray(
        [
            [[0, 2], [0, 3], [1, 2], [0, 3], [1, 2], [0, 3]],
            [[3, 1], [2, 0], [3, 0], [2, 1], [3, 1], [2, 0]],
        ],
        dtype=jnp.int32,
    )
    return build_source_push_semantic_plan_jax(
        selected_experts,
        jnp.ones(selected_experts.shape, dtype=jnp.float32),
        ep_size=2,
        experts_per_rank=2,
        rows_per_src_dst_capacity=12,
        capacity_factor=4.0,
    )


def _inputs():
    plan = _plan()
    x = (jnp.arange(2 * 6 * 256, dtype=jnp.float32).reshape(2, 6, 256) % 17 - 8).astype(jnp.bfloat16)
    weights = ((jnp.arange(2 * 2 * 256 * 256, dtype=jnp.float32).reshape(2, 2, 256, 256) % 13 - 6) / 64).astype(
        jnp.bfloat16
    )
    return x, weights, plan


def test_fused_w13_metadata_aggregates_four_semantic_blocks_and_rotates_peers():
    x, _weights, plan = _inputs()
    metadata = jax.jit(
        lambda x_arg, plan_arg: source_push_semantic_fused_w13_metadata_jax(
            x_arg,
            plan_arg,
            send_chunks_per_dst=1,
            rows_per_expert_capacity=256,
        )
    )(x, plan)

    assert metadata.token_ids.shape == (2, 2, 1, 4, 64)
    np.testing.assert_array_equal(np.asarray(metadata.recv_expert[0, 1]), np.asarray(metadata.send_expert[1, 1]))
    np.testing.assert_array_equal(np.asarray(metadata.recv_expert[1, 1]), np.asarray(metadata.send_expert[0, 1]))
    np.testing.assert_array_equal(
        np.asarray(metadata.recv_valid_rows[0, 0]), np.asarray(metadata.send_valid_rows[0, 0])
    )
    assert np.all(np.asarray(metadata.send_valid_rows) <= CONFIG.compute_m)

    token_ids = np.asarray(metadata.token_ids)
    experts = np.asarray(metadata.send_expert)
    valid_rows = np.asarray(metadata.send_valid_rows)
    pair_bases = np.asarray(plan.pair_expert_base)
    semantic_token_ids = np.asarray(plan.token_ids)
    for src in range(2):
        for dst_ordinal in range(2):
            dst = (src + dst_ordinal) % 2
            for block in range(4):
                expert = experts[src, dst_ordinal, 0, block]
                if expert < 0:
                    continue
                pair_start = pair_bases[src, dst, expert]
                count = valid_rows[src, dst_ordinal, 0, block]
                np.testing.assert_array_equal(
                    token_ids[src, dst_ordinal, 0, block, :count],
                    semantic_token_ids[src, dst, pair_start : pair_start + count],
                )


def test_fused_w13_generation_accounting_reuses_slots_with_cumulative_targets():
    first = source_push_semantic_fused_w13_generation_accounting(0, hidden_dim=2560, intermediate_dim=1280)
    next_chunk = source_push_semantic_fused_w13_generation_accounting(1, hidden_dim=2560, intermediate_dim=1280)
    reused = source_push_semantic_fused_w13_generation_accounting(
        CONFIG.inbox_slots, hidden_dim=2560, intermediate_dim=1280
    )

    assert (first.slot, first.generation, first.empty_generation, first.released_generation) == (0, 1, 1, 2)
    assert (first.owner, next_chunk.owner) == (0, 1)
    assert first.producer_side_programs == 12
    assert first.consumer_programs == 20
    assert CONFIG.worker_programs_per_peer == 32
    assert first.helper_tiles == CONFIG.compute_blocks_per_send * (2560 // CONFIG.send_k)
    assert first.helper_tiles_per_compute_block == 2560 // CONFIG.send_k
    assert first.prepare_generation == first.generation
    assert first.helper_done_generation == first.helper_tiles_per_compute_block
    assert reused.slot == first.slot
    assert reused.generation == 2
    assert reused.owner == first.owner
    assert reused.helper_tiles == first.helper_tiles
    assert reused.helper_done_generation == 2 * first.helper_done_generation
    assert reused.consumer_done_generation == 2 * first.consumer_done_generation
    assert reused.empty_generation == first.released_generation


def test_fused_w13_kernel_uses_block_readiness_and_slot_wide_release():
    source = inspect.getsource(_make_source_push_semantic_fused_w13_kernel)

    assert "(ep_size, config.inbox_slots, blocks_per_send)" in source
    assert "helper_done_sem.at[peer, slot, block]" in source
    assert "value=generation * (hidden_dim // config.send_k)" in source
    assert "helper_done_sem.at[dst, slot]" not in source
    assert "consumer_done_sem.at[peer, slot]" in source
    assert "value=generation * consumer_jobs" in source
    assert "mgpu.wgmma(" in source


def test_fused_w13_transport_rows_are_independent_from_compute_rows():
    config = SourcePushSemanticFusedW13Config(send_m=128)

    config.validate()
    assert config.compute_blocks_per_send == 2


def test_fused_w13_raw_gather_groups_eight_rows_without_serial_row_loop():
    source = inspect.getsource(_make_source_push_semantic_fused_w13_kernel)

    assert RAW_GATHER_ROWS == 8
    assert CONFIG.compute_m // RAW_GATHER_ROWS == 8
    assert "range(config.compute_m // RAW_GATHER_ROWS)" in source
    assert "pl.ds(row_start, RAW_GATHER_ROWS)" in source
    assert "x_ref[safe_tokens[:, None], hidden_offsets[None, :]]" in source
    assert "RAW_GATHER_HIDDEN_LAYOUT" in source
    assert "RAW_GATHER_ROW_LAYOUT" in source
    assert "mgpu.Layout.TILED" not in source
    assert re.search(r"pl\.loop\(0,\s*config\.compute_m\)", source) is None
    assert TOKEN_TRANSFER_ROWS == 128
    assert "token_smem=mgpu.SMEM((TOKEN_TRANSFER_ROWS,), dtype=jnp.int32)" in source
    assert "tuple(token_smem[row_start + row] for row in range(RAW_GATHER_ROWS))" in source

    assert RAW_GATHER_HIDDEN_LAYOUT.to_mgpu().registers_shape((CONFIG.send_k,)) == (1,)
    assert RAW_GATHER_ROW_LAYOUT.to_mgpu().registers_shape((RAW_GATHER_ROWS,)) == (1, 1)


def test_fused_w13_reference_zeros_invalid_and_source_padding_rows():
    x, weights, plan = _inputs()
    metadata = source_push_semantic_fused_w13_metadata_jax(
        x,
        plan,
        send_chunks_per_dst=1,
        rows_per_expert_capacity=256,
    )
    z = source_push_semantic_fused_w13_reference_jax(x, weights, metadata)

    assert np.count_nonzero(np.asarray(metadata.valid)) == int(np.asarray(jnp.sum(plan.xcounts)))
    np.testing.assert_array_equal(np.asarray(z)[~np.asarray(metadata.valid)], 0)
    assert np.any(np.asarray(z)[np.asarray(metadata.valid)] != 0)


def test_fused_w13_interpret_matches_independent_semantic_scatter_reference():
    x, weights, plan = _inputs()
    result = jax.jit(
        lambda x_arg, weights_arg, plan_arg: source_push_semantic_fused_w13(
            x_arg,
            weights_arg,
            plan_arg,
            send_chunks_per_dst=1,
            rows_per_expert_capacity=256,
            interpret=True,
        )
    )(x, weights, plan)

    expected = np.zeros((2, 2, 256, 256), dtype=np.float32)
    x_host = np.asarray(x, dtype=np.float32)
    weights_host = np.asarray(weights, dtype=np.float32)
    assignment_ids = np.asarray(plan.assignment_ids)
    valid = np.asarray(plan.valid_mask)
    xcounts = np.asarray(plan.xcounts)
    pair_bases = np.asarray(plan.pair_expert_base)
    rounded_counts = ((xcounts + CONFIG.compute_m - 1) // CONFIG.compute_m) * CONFIG.compute_m
    padded_src_bases = np.zeros((2, 2, 2), dtype=np.int32)
    padded_src_bases[:, 1, :] = rounded_counts[0]
    expected_valid = np.zeros((2, 2, 256), dtype=np.bool_)
    for src in range(2):
        for dst in range(2):
            for expert in range(2):
                for local_row in range(int(xcounts[src, dst, expert])):
                    pair_row = int(pair_bases[src, dst, expert]) + local_row
                    assert valid[src, dst, pair_row]
                    token = assignment_ids[src, dst, pair_row] // plan.topk
                    expert_row = int(padded_src_bases[dst, src, expert]) + local_row
                    expected[dst, expert, expert_row] = x_host[src, token] @ weights_host[dst, expert]
                    expected_valid[dst, expert, expert_row] = True

    np.testing.assert_allclose(np.asarray(result.z, dtype=np.float32), expected, rtol=2e-2, atol=0.25)
    np.testing.assert_array_equal(np.asarray(result.valid), expected_valid)
    assert int(result.queue_overflow_routes) == 0
    assert int(result.layout_overflow_rows) == 0


def test_fused_w13_sharded_wrapper_specs_match_input_ranks(monkeypatch):
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("expert",), axis_types=(AxisType.Explicit,))
    x = jnp.zeros((1, 1, 256), dtype=jnp.bfloat16)
    weights = jnp.zeros((1, 1, 256, 256), dtype=jnp.bfloat16)
    metadata = SourcePushSemanticFusedW13Metadata(
        token_ids=jnp.zeros((1, 1, 1, 4, 64), dtype=jnp.int32),
        send_expert=jnp.zeros((1, 1, 1, 4), dtype=jnp.int32),
        send_row_start=jnp.zeros((1, 1, 1, 4), dtype=jnp.int32),
        send_valid_rows=jnp.zeros((1, 1, 1, 4), dtype=jnp.int32),
        recv_expert=jnp.zeros((1, 1, 1, 4), dtype=jnp.int32),
        recv_row_start=jnp.zeros((1, 1, 1, 4), dtype=jnp.int32),
        recv_valid_rows=jnp.zeros((1, 1, 1, 4), dtype=jnp.int32),
        valid=jnp.zeros((1, 1, 256), dtype=jnp.bool_),
        queue_overflow_routes=jnp.asarray(0, dtype=jnp.int32),
        layout_overflow_rows=jnp.asarray(0, dtype=jnp.int32),
        rows_per_expert_capacity=256,
        send_chunks_per_dst=1,
    )
    captured_specs = None

    def fake_shard_map(_fn, *, in_specs, **_kwargs):
        nonlocal captured_specs
        captured_specs = in_specs

        def run(*args):
            for spec, arg in zip(in_specs, args, strict=True):
                assert len(spec) <= arg.ndim
            return jnp.zeros((1, 1, 256, 256), dtype=jnp.bfloat16)

        return run

    monkeypatch.setattr(fused_w13, "shard_map", fake_shard_map)
    monkeypatch.setattr(fused_w13, "_make_source_push_semantic_fused_w13_kernel", lambda **_kwargs: None)

    result = fused_w13._source_push_semantic_fused_w13_sharded(
        x,
        weights,
        metadata,
        config=CONFIG,
        mesh=mesh,
    )

    assert result.shape == (1, 1, 256, 256)
    assert captured_specs is not None
