# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.grug._moe.source_push_inbox import PushInboxConfig
from levanter.grug._moe.source_push_plan import (
    build_source_push_semantic_plan_jax,
    source_push_semantic_pair_expert_ids_jax,
    source_push_semantic_queue_metadata_jax,
    source_push_semantic_w13_reference_jax,
)
from levanter.grug._moe.source_push_semantic_inbox_pallas import (
    SourcePushSemanticInboxLayout,
    SourcePushSemanticPermuteW13Result,
    source_push_semantic_inbox_kernel_inputs_jax,
    source_push_semantic_inbox_layout_jax,
    source_push_semantic_inbox_metadata_jax,
    source_push_semantic_inbox_pack_pallas_mgpu,
    source_push_semantic_inbox_w13_pallas_mgpu,
    source_push_semantic_permute_w13_pallas_mgpu,
)


def _semantic_plan():
    selected_experts = jnp.asarray(
        [
            [[0, 2], [1, 3], [0, 3], [1, 2], [0, 2]],
            [[3, 1], [2, 0], [3, 0], [2, 1], [3, 1]],
        ],
        dtype=jnp.int32,
    )
    route_weights = jnp.ones(selected_experts.shape, dtype=jnp.float32)
    return build_source_push_semantic_plan_jax(
        selected_experts,
        route_weights,
        ep_size=2,
        experts_per_rank=2,
        rows_per_src_dst_capacity=10,
        capacity_factor=4.0,
    )


def _inbox_config(*, row_block: int, entries_per_dst: int) -> PushInboxConfig:
    return PushInboxConfig(
        ep_size=2,
        entries_per_rank=entries_per_dst,
        inbox_slots=2,
        hidden_dim=64,
        intermediate_dim=64,
        block_m=row_block,
        block_n=64,
        block_k=64,
        n_group=1,
        experts_per_rank=2,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=4,
        send_pipeline_depth=1,
        n_groups_per_job=1,
        tokens_per_rank=5,
        topk=2,
        capacity_factor=4.0,
    )


def _source_padded_w13_reference_jax(x, w_gate_up, plan, layout: SourcePushSemanticInboxLayout):
    z_pair, _ = source_push_semantic_w13_reference_jax(x, w_gate_up, plan)
    expert = source_push_semantic_pair_expert_ids_jax(plan)
    source_index = jnp.arange(plan.xcounts.shape[0], dtype=jnp.int32)[:, None, None]
    dst_index = jnp.arange(plan.xcounts.shape[1], dtype=jnp.int32)[None, :, None]
    pair_row = jnp.arange(plan.assignment_ids.shape[-1], dtype=jnp.int32)[None, None, :]
    local_row = pair_row - plan.pair_expert_base.at[source_index, dst_index, expert].get()
    padded_row = layout.src_base_by_expert.at[dst_index, source_index, expert].get() + local_row
    valid = plan.valid_mask & (padded_row < layout.rows_per_expert_capacity)
    scatter_row = jnp.where(valid, padded_row, layout.rows_per_expert_capacity)
    expected_z = jnp.zeros(
        (
            plan.xcounts.shape[1],
            plan.xcounts.shape[2],
            layout.rows_per_expert_capacity,
            w_gate_up.shape[-1],
        ),
        dtype=jnp.bfloat16,
    )
    expected_z = expected_z.at[dst_index, expert, scatter_row].set(
        jnp.where(valid[..., None], z_pair.astype(jnp.bfloat16), 0),
        mode="drop",
    )
    intermediate_dim = expected_z.shape[-1] // 2
    gate = expected_z[..., :intermediate_dim].astype(jnp.float32)
    up = expected_z[..., intermediate_dim:].astype(jnp.float32)
    expected_h = jax.nn.silu(gate) * up
    return expected_z, expected_h, layout.valid


def test_source_push_semantic_inbox_kernel_inputs_are_block_aligned_and_jittable():
    plan = _semantic_plan()
    row_block = 2
    entries_per_dst = 4
    queue = source_push_semantic_queue_metadata_jax(
        plan,
        return_row_block=row_block,
        entries_per_dst=entries_per_dst,
    )
    x = jnp.arange(2 * 5 * 4, dtype=jnp.float32).reshape(2, 5, 4)
    rows_per_expert_capacity = 8

    inputs = jax.jit(
        lambda x_arg, semantic_plan, semantic_queue: source_push_semantic_inbox_kernel_inputs_jax(
            x_arg,
            semantic_plan,
            semantic_queue,
            rows_per_expert_capacity=rows_per_expert_capacity,
        )
    )(x, plan, queue)

    x_host = np.asarray(x)
    counts = np.asarray(plan.xcounts)
    pair_bases = np.asarray(plan.pair_expert_base)
    token_ids = np.asarray(plan.token_ids)
    rounded_counts = ((counts + row_block - 1) // row_block) * row_block
    padded_rows_per_expert = np.zeros((2, 2), dtype=np.int32)
    for dst in range(2):
        for expert in range(2):
            padded_rows_per_expert[dst, expert] = sum(rounded_counts[:, dst, expert])
    padded_expert_base = np.zeros_like(padded_rows_per_expert)
    padded_expert_base[:, 1] = rows_per_expert_capacity
    padded_src_base = np.zeros((2, 2, 2), dtype=np.int32)
    padded_src_base[:, 1, :] = rounded_counts[0]

    expected_x = np.zeros((2, 2, entries_per_dst, row_block, 4), dtype=np.float32)
    expected_send_meta = np.zeros((2, 2, entries_per_dst, 4), dtype=np.int32)
    expected_send_meta[..., 1] = -1
    for src in range(2):
        for dst_ordinal in range(2):
            dst = (src + dst_ordinal) % 2
            entry = 0
            for expert in range(2):
                for local_row_start in range(0, int(counts[src, dst, expert]), row_block):
                    valid_rows = min(row_block, int(counts[src, dst, expert]) - local_row_start)
                    padded_row_start = (
                        padded_expert_base[dst, expert] + padded_src_base[dst, src, expert] + local_row_start
                    )
                    expected_send_meta[src, dst_ordinal, entry] = (
                        src,
                        expert,
                        padded_row_start,
                        valid_rows,
                    )
                    pair_row_start = pair_bases[src, dst, expert] + local_row_start
                    for row in range(valid_rows):
                        token = token_ids[src, dst, pair_row_start + row]
                        expected_x[src, dst_ordinal, entry, row] = x_host[src, token]
                    entry += 1

    expected_recv_meta = np.zeros_like(expected_send_meta)
    for dst in range(2):
        for source_ordinal in range(2):
            src = (dst + source_ordinal) % 2
            dst_ordinal = (dst - src) % 2
            expected_recv_meta[dst, source_ordinal] = expected_send_meta[src, dst_ordinal]

    expected_valid = np.zeros((2, 2, rows_per_expert_capacity), dtype=np.bool_)
    for dst in range(2):
        for expert in range(2):
            for src in range(2):
                start = padded_src_base[dst, src, expert]
                count = counts[src, dst, expert]
                expected_valid[dst, expert, start : start + count] = True

    np.testing.assert_array_equal(np.asarray(inputs.packed_x), expected_x)
    np.testing.assert_array_equal(np.asarray(inputs.send_meta), expected_send_meta)
    np.testing.assert_array_equal(np.asarray(inputs.recv_meta), expected_recv_meta)
    np.testing.assert_array_equal(np.asarray(inputs.layout.expert_base), padded_expert_base)
    np.testing.assert_array_equal(np.asarray(inputs.layout.src_base_by_expert), padded_src_base)
    np.testing.assert_array_equal(np.asarray(inputs.layout.rounded_rows_per_expert), padded_rows_per_expert)
    np.testing.assert_array_equal(np.asarray(inputs.layout.transport_rows_by_src_dst_expert), counts)
    np.testing.assert_array_equal(np.asarray(inputs.layout.valid), expected_valid)
    assert inputs.layout.rows_per_expert_capacity == rows_per_expert_capacity
    assert int(inputs.layout.overflow_rows) == 0


def test_source_push_semantic_inbox_layout_only_reports_source_bases_validity_and_overflow():
    plan = _semantic_plan()
    queue = source_push_semantic_queue_metadata_jax(
        plan,
        return_row_block=2,
        entries_per_dst=4,
    )
    rows_per_expert_capacity = 2

    layout = jax.jit(
        lambda semantic_plan, semantic_queue: source_push_semantic_inbox_layout_jax(
            semantic_plan,
            semantic_queue,
            rows_per_expert_capacity=rows_per_expert_capacity,
        )
    )(plan, queue)

    rounded_counts = ((np.asarray(plan.xcounts) + 1) // 2) * 2
    expected_source_bases = np.zeros((2, 2, 2), dtype=np.int32)
    expected_source_bases[:, 1, :] = rounded_counts[0]
    expected_rounded_rows = np.sum(rounded_counts, axis=0)
    expected_overflow = np.sum(np.maximum(expected_rounded_rows - rows_per_expert_capacity, 0))
    expected_valid = np.zeros((2, 2, rows_per_expert_capacity), dtype=np.bool_)
    for destination in range(2):
        for expert in range(2):
            for source in range(2):
                start = expected_source_bases[destination, source, expert]
                count = min(
                    int(plan.xcounts[source, destination, expert]),
                    max(rows_per_expert_capacity - start, 0),
                )
                expected_valid[destination, expert, start : start + count] = True

    np.testing.assert_array_equal(np.asarray(layout.src_base_by_expert), expected_source_bases)
    np.testing.assert_array_equal(np.asarray(layout.rounded_rows_per_expert), expected_rounded_rows)
    np.testing.assert_array_equal(np.asarray(layout.valid), expected_valid)
    assert int(layout.overflow_rows) == expected_overflow


def test_source_push_semantic_inbox_pallas_pack_matches_reference_and_is_source_sharded():
    selected_experts = jnp.asarray([[[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]], dtype=jnp.int32)
    plan = build_source_push_semantic_plan_jax(
        selected_experts,
        jnp.ones(selected_experts.shape, dtype=jnp.float32),
        ep_size=1,
        experts_per_rank=2,
        rows_per_src_dst_capacity=10,
        capacity_factor=4.0,
    )
    queue = source_push_semantic_queue_metadata_jax(
        plan,
        return_row_block=2,
        entries_per_dst=6,
    )
    x = jnp.arange(5 * 8, dtype=jnp.float32).reshape(1, 5, 8)
    metadata = jax.jit(
        lambda x_arg, semantic_plan, semantic_queue: source_push_semantic_inbox_metadata_jax(
            x_arg,
            semantic_plan,
            semantic_queue,
            rows_per_expert_capacity=8,
        )
    )(x, plan, queue)
    reference = source_push_semantic_inbox_kernel_inputs_jax(
        x,
        plan,
        queue,
        rows_per_expert_capacity=8,
    ).packed_x
    devices = np.asarray(jax.devices()[:1])
    mesh = Mesh(devices, axis_names=("expert",), axis_types=(AxisType.Explicit,))

    packed = source_push_semantic_inbox_pack_pallas_mgpu(
        x,
        metadata.token_ids,
        metadata.valid_mask,
        mesh=mesh,
        interpret=True,
    )

    np.testing.assert_array_equal(np.asarray(packed), np.asarray(reference))
    assert isinstance(packed.sharding, NamedSharding)
    assert packed.sharding.spec == P("expert", None, None, None, None)
    assert {shard.data.shape[0] for shard in packed.addressable_shards} == {x.shape[0] // devices.size}


def test_source_push_semantic_inbox_pallas_pack_matches_rough_b64_source_padded_reference():
    selected_experts = jnp.asarray(
        [
            [[0, 1], [0, 2], [0, 3], [1, 2], [0, 2], [1, 3], [0, 1], [2, 3], [0, 3]],
            [[3, 2], [3, 1], [2, 0], [3, 0], [2, 1], [3, 1], [2, 0], [1, 0], [3, 2]],
        ],
        dtype=jnp.int32,
    )
    plan = build_source_push_semantic_plan_jax(
        selected_experts,
        jnp.ones(selected_experts.shape, dtype=jnp.float32),
        ep_size=2,
        experts_per_rank=2,
        rows_per_src_dst_capacity=18,
        capacity_factor=4.0,
    )
    queue = jax.jit(
        lambda semantic_plan: source_push_semantic_queue_metadata_jax(
            semantic_plan,
            return_row_block=64,
            entries_per_dst=4,
        )
    )(plan)
    x = jnp.arange(2 * 9 * 10, dtype=jnp.float32).reshape(2, 9, 10)
    rows_per_expert_capacity = 128

    metadata = jax.jit(
        lambda x_arg, semantic_plan, semantic_queue: source_push_semantic_inbox_metadata_jax(
            x_arg,
            semantic_plan,
            semantic_queue,
            rows_per_expert_capacity=rows_per_expert_capacity,
        )
    )(x, plan, queue)
    reference = jax.jit(
        lambda x_arg, semantic_plan, semantic_queue: source_push_semantic_inbox_kernel_inputs_jax(
            x_arg,
            semantic_plan,
            semantic_queue,
            rows_per_expert_capacity=rows_per_expert_capacity,
        ).packed_x
    )(x, plan, queue)
    packed = source_push_semantic_inbox_pack_pallas_mgpu(
        x,
        metadata.token_ids,
        metadata.valid_mask,
        interpret=True,
    )

    np.testing.assert_array_equal(np.asarray(packed), np.asarray(reference))
    np.testing.assert_array_equal(np.asarray(packed)[~np.asarray(metadata.valid_mask)], 0)
    np.testing.assert_array_equal(np.asarray(metadata.layout.src_base_by_expert)[:, 1, :], 64)
    assert np.unique(np.asarray(plan.xcounts)[np.asarray(plan.xcounts) > 0]).size > 1
    assert np.any(~np.asarray(metadata.valid_mask))
    assert np.any(~np.asarray(metadata.layout.valid))
    assert int(queue.overflow_entries) == 0
    assert int(queue.overflow_routes) == 0
    assert int(metadata.layout.overflow_rows) == 0


def test_source_push_semantic_inbox_w13_interpret_matches_independent_expert_major_reference():
    plan = _semantic_plan()
    row_block = 8
    entries_per_dst = 2
    queue = jax.jit(
        lambda semantic_plan: source_push_semantic_queue_metadata_jax(
            semantic_plan,
            return_row_block=row_block,
            entries_per_dst=entries_per_dst,
        )
    )(plan)
    config = _inbox_config(row_block=row_block, entries_per_dst=entries_per_dst)

    x = jnp.zeros((2, 5, 64), dtype=jnp.bfloat16)
    source_token_value = jnp.arange(1, 11, dtype=jnp.float32).reshape(2, 5).astype(jnp.bfloat16)
    x = x.at[:, :, 0].set(source_token_value)
    w_gate_up = jnp.zeros((2, 2, 64, 128), dtype=jnp.bfloat16)
    expert_scale = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.bfloat16)
    w_gate_up = w_gate_up.at[:, :, 0, :64].set(expert_scale[:, :, None])
    w_gate_up = w_gate_up.at[:, :, 0, 64:].set(jnp.asarray(0.5, dtype=jnp.bfloat16))

    observed = jax.jit(
        lambda x_arg, w_arg, semantic_plan, semantic_queue: source_push_semantic_inbox_w13_pallas_mgpu(
            x_arg,
            w_arg,
            semantic_plan,
            semantic_queue,
            config=config,
            interpret=True,
        )
    )(x, w_gate_up, plan, queue)

    expected_z, expected_h, expected_valid = _source_padded_w13_reference_jax(
        x,
        w_gate_up,
        plan,
        observed.layout,
    )
    expected_inputs = source_push_semantic_inbox_kernel_inputs_jax(
        x,
        plan,
        queue,
        rows_per_expert_capacity=observed.layout.rows_per_expert_capacity,
    )

    np.testing.assert_array_equal(np.asarray(observed.valid), np.asarray(expected_valid))
    np.testing.assert_allclose(np.asarray(observed.z, dtype=np.float32), np.asarray(expected_z), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(observed.h), np.asarray(expected_h), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(observed.packed_x), np.asarray(expected_inputs.packed_x))
    for src in range(2):
        for dst_ordinal in range(2):
            dst = (src + dst_ordinal) % 2
            np.testing.assert_array_equal(
                np.asarray(observed.recv_x[dst, src]),
                np.asarray(expected_inputs.packed_x[src, dst_ordinal]),
            )
    assert int(queue.overflow_entries) == 0
    assert int(queue.overflow_routes) == 0
    assert int(observed.layout.overflow_rows) == 0
    assert np.any(np.asarray(queue.valid_rows) < row_block)
    assert np.any(~np.asarray(observed.valid))
    assert not np.any(np.asarray(observed.z)[~np.asarray(observed.valid)])
    assert not np.any(np.asarray(observed.h)[~np.asarray(observed.valid)])


def test_source_push_semantic_permute_w13_jit_matches_rough_reference_with_queue_slot_reuse():
    selected_experts = jnp.asarray(
        [
            [[0, 1], [0, 2], [0, 3], [1, 2], [0, 2], [1, 3], [0, 1], [2, 3], [0, 3]],
            [[3, 2], [3, 1], [2, 0], [3, 0], [2, 1], [3, 1], [2, 0], [1, 0], [3, 2]],
        ],
        dtype=jnp.int32,
    )
    plan = build_source_push_semantic_plan_jax(
        selected_experts,
        jnp.ones(selected_experts.shape, dtype=jnp.float32),
        ep_size=2,
        experts_per_rank=2,
        rows_per_src_dst_capacity=18,
        capacity_factor=4.0,
    )
    config = _inbox_config(row_block=2, entries_per_dst=6)
    x = jnp.arange(2 * 9 * 64, dtype=jnp.float32).reshape(2, 9, 64).astype(jnp.bfloat16) / 128
    w_gate_up = jnp.arange(2 * 2 * 64 * 128, dtype=jnp.float32).reshape(2, 2, 64, 128).astype(jnp.bfloat16) / 4096

    observed = jax.jit(
        lambda x_arg, w_arg, semantic_plan: source_push_semantic_permute_w13_pallas_mgpu(
            x_arg,
            w_arg,
            semantic_plan,
            config=config,
            interpret=True,
        )
    )(x, w_gate_up, plan)
    expected_z, expected_h, expected_valid = _source_padded_w13_reference_jax(
        x,
        w_gate_up,
        plan,
        observed.layout,
    )

    required_entries = np.sum((np.asarray(plan.xcounts) + config.block_m - 1) // config.block_m, axis=2)
    assert np.max(required_entries) > config.inbox_slots
    assert np.unique(np.asarray(plan.xcounts)[np.asarray(plan.xcounts) > 0]).size > 1
    assert SourcePushSemanticPermuteW13Result._fields == (
        "z",
        "h",
        "valid",
        "layout",
        "queue_overflow_routes",
        "layout_overflow_rows",
    )
    np.testing.assert_array_equal(np.asarray(observed.valid), np.asarray(expected_valid))
    np.testing.assert_allclose(np.asarray(observed.z, dtype=np.float32), np.asarray(expected_z), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(observed.h), np.asarray(expected_h), rtol=1e-5, atol=1e-5)
    assert int(observed.queue_overflow_routes) == 0
    assert int(observed.layout_overflow_rows) == 0
    assert not np.any(np.asarray(observed.z)[~np.asarray(observed.valid)])
    assert not np.any(np.asarray(observed.h)[~np.asarray(observed.valid)])


def test_source_push_semantic_permute_w13_reports_queue_and_layout_overflow():
    selected_experts = jnp.zeros((2, 10, 1), dtype=jnp.int32)
    plan = build_source_push_semantic_plan_jax(
        selected_experts,
        jnp.ones(selected_experts.shape, dtype=jnp.float32),
        ep_size=2,
        experts_per_rank=2,
        rows_per_src_dst_capacity=10,
        capacity_factor=4.0,
    )
    config = _inbox_config(row_block=4, entries_per_dst=1)
    x = jnp.zeros((2, 10, 64), dtype=jnp.bfloat16)
    x = x.at[:, :, 0].set(jnp.arange(1, 21, dtype=jnp.float32).reshape(2, 10).astype(jnp.bfloat16))
    w_gate_up = jnp.zeros((2, 2, 64, 128), dtype=jnp.bfloat16)
    w_gate_up = w_gate_up.at[0, 0, 0, :].set(jnp.asarray(0.5, dtype=jnp.bfloat16))

    observed = jax.jit(
        lambda x_arg, w_arg, semantic_plan: source_push_semantic_permute_w13_pallas_mgpu(
            x_arg,
            w_arg,
            semantic_plan,
            config=config,
            interpret=True,
        )
    )(x, w_gate_up, plan)
    expected_z, expected_h, expected_valid = _source_padded_w13_reference_jax(
        x,
        w_gate_up,
        plan,
        observed.layout,
    )

    assert int(observed.queue_overflow_routes) == 12
    assert int(observed.layout_overflow_rows) == 20
    np.testing.assert_array_equal(np.asarray(observed.valid), np.asarray(expected_valid))
    np.testing.assert_allclose(np.asarray(observed.z, dtype=np.float32), np.asarray(expected_z), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(observed.h), np.asarray(expected_h), rtol=1e-5, atol=1e-5)
    assert np.count_nonzero(np.asarray(observed.valid)) == 4
    assert not np.any(np.asarray(observed.z)[~np.asarray(observed.valid)])
    assert not np.any(np.asarray(observed.h)[~np.asarray(observed.valid)])
