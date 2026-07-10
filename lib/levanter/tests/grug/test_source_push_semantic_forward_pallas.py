# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug._moe.source_push_plan import (
    build_source_push_semantic_plan_jax,
    source_push_semantic_expert_major_to_pair_jax,
    source_push_semantic_gather_x_jax,
    source_push_semantic_pair_to_expert_major_jax,
    source_push_semantic_w13_reference_jax,
    source_push_semantic_x_to_expert_major_jax,
)
from levanter.grug._moe.source_push_semantic_forward_pallas import (
    SourcePushSemanticGatherXPallasBlockSizes,
    SourcePushSemanticW13ExpertMajorPallasBlockSizes,
    SourcePushSemanticW13PallasBlockSizes,
    source_push_semantic_expert_major_source_token_lookup_jax,
    source_push_semantic_gather_x_pallas_mgpu,
    source_push_semantic_w13_expert_major_pallas_mgpu,
    source_push_semantic_w13_from_x_expert_pallas_mgpu,
    source_push_semantic_w13_from_x_pair_pallas_mgpu,
    source_push_semantic_w13_pallas_scaffold_mgpu,
    source_push_semantic_x_to_expert_major_direct_pallas_mgpu,
    source_push_semantic_x_to_expert_major_lookup_pallas_mgpu,
    source_push_semantic_x_to_expert_major_pallas_scaffold_mgpu,
)


def _semantic_plan():
    selected_experts = jnp.asarray(
        [
            [[0, 2], [1, 3], [0, 1]],
            [[2, 0], [3, 1], [2, 3]],
        ],
        dtype=jnp.int32,
    )
    combine_weights = jnp.asarray(
        [
            [[0.7, 0.3], [0.6, 0.4], [0.9, 0.1]],
            [[0.5, 0.5], [0.2, 0.8], [0.1, 0.9]],
        ],
        dtype=jnp.float32,
    )
    return build_source_push_semantic_plan_jax(
        selected_experts,
        combine_weights,
        ep_size=2,
        experts_per_rank=2,
        rows_per_src_dst_capacity=6,
        capacity_factor=2.0,
    )


def _skewed_invalid_tail_plan():
    selected_experts = jnp.asarray([[[0]], [[0]], [[0]]], dtype=jnp.int32)
    combine_weights = jnp.ones((3, 1, 1), dtype=jnp.float32)
    return build_source_push_semantic_plan_jax(
        selected_experts,
        combine_weights,
        ep_size=3,
        experts_per_rank=1,
        rows_per_src_dst_capacity=3,
        capacity_factor=10.0,
    )


def _padded_source_row_bases():
    return jnp.asarray(
        [
            [[0, 1], [4, 5]],
            [[2, 0], [6, 4]],
        ],
        dtype=jnp.int32,
    )


def _scatter_pair_rows_at_source_bases(pair_values, plan, source_row_bases, rows_per_expert_capacity):
    pair_values = np.asarray(pair_values)
    counts = np.asarray(plan.xcounts)
    pair_bases = np.asarray(plan.pair_expert_base)
    source_row_bases = np.asarray(source_row_bases)
    expected = np.zeros(
        (counts.shape[1], counts.shape[2], rows_per_expert_capacity, pair_values.shape[-1]),
        dtype=pair_values.dtype,
    )
    valid = np.zeros(expected.shape[:3], dtype=np.bool_)
    for src in range(counts.shape[0]):
        for dst in range(counts.shape[1]):
            for expert in range(counts.shape[2]):
                count = int(counts[src, dst, expert])
                pair_base = int(pair_bases[src, dst, expert])
                source_base = int(source_row_bases[dst, src, expert])
                expected[dst, expert, source_base : source_base + count] = pair_values[
                    src, dst, pair_base : pair_base + count
                ]
                valid[dst, expert, source_base : source_base + count] = True
    return expected, valid


def test_source_push_semantic_gather_x_pallas_interpret_matches_jax_reference():
    plan = _semantic_plan()
    x = jnp.arange(2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8).astype(jnp.bfloat16)

    observed = source_push_semantic_gather_x_pallas_mgpu(
        x,
        plan,
        block_sizes=SourcePushSemanticGatherXPallasBlockSizes(row_block=2, hidden_block=4),
        interpret=True,
    )
    expected = source_push_semantic_gather_x_jax(x, plan).astype(jnp.bfloat16)

    np.testing.assert_array_equal(np.asarray(observed), np.asarray(expected))


def test_source_push_semantic_x_to_expert_major_jax_places_source_rows_by_expert():
    plan = _semantic_plan()
    x = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4)
    rows_per_expert_capacity = 4

    x_expert, valid = jax.jit(
        lambda x_arg: source_push_semantic_x_to_expert_major_jax(
            x_arg,
            plan,
            rows_per_expert_capacity=rows_per_expert_capacity,
        )
    )(x)

    expected = np.zeros((2, 2, rows_per_expert_capacity, 4), dtype=np.float32)
    expected_valid = np.zeros((2, 2, rows_per_expert_capacity), dtype=np.bool_)
    for src in range(plan.assignment_ids.shape[0]):
        for dst in range(plan.assignment_ids.shape[1]):
            for expert in range(plan.xcounts.shape[-1]):
                count = int(np.asarray(plan.xcounts)[src, dst, expert])
                pair_base = int(np.asarray(plan.pair_expert_base)[src, dst, expert])
                expert_base = int(np.asarray(plan.src_base_by_expert)[dst, src, expert])
                for local_row in range(count):
                    pair_row = pair_base + local_row
                    expert_row = expert_base + local_row
                    token = int(np.asarray(plan.token_ids)[src, dst, pair_row])
                    expected[dst, expert, expert_row] = np.asarray(x)[src, token]
                    expected_valid[dst, expert, expert_row] = True

    np.testing.assert_array_equal(np.asarray(valid), expected_valid)
    np.testing.assert_array_equal(np.asarray(x_expert), expected)


def test_source_push_semantic_x_to_expert_major_ignores_invalid_tail_rows_between_sources():
    plan = _skewed_invalid_tail_plan()
    x = jnp.arange(3 * 1 * 4, dtype=jnp.float32).reshape(3, 1, 4)

    x_expert, valid = source_push_semantic_x_to_expert_major_jax(
        x,
        plan,
        rows_per_expert_capacity=3,
    )

    np.testing.assert_array_equal(np.asarray(valid[0, 0]), np.asarray([True, True, True]))
    np.testing.assert_array_equal(np.asarray(x_expert[0, 0]), np.asarray(x[:, 0, :]))


def test_source_push_semantic_pair_to_expert_major_ignores_invalid_tail_rows_between_sources():
    plan = _skewed_invalid_tail_plan()
    pair_values = jnp.arange(3 * 3 * 3 * 2, dtype=jnp.float32).reshape(3, 3, 3, 2)

    expert_values, valid = source_push_semantic_pair_to_expert_major_jax(
        pair_values,
        plan,
        rows_per_expert_capacity=3,
    )

    np.testing.assert_array_equal(np.asarray(valid[0, 0]), np.asarray([True, True, True]))
    expected = np.asarray(pair_values)[:, 0, 0, :]
    np.testing.assert_array_equal(np.asarray(expert_values[0, 0]), expected)


def test_source_push_semantic_x_to_expert_major_scaffold_reconstructs_pair_rows():
    plan = _semantic_plan()
    x = jnp.arange(2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8).astype(jnp.bfloat16)
    rows_per_expert_capacity = 4

    x_expert, valid = source_push_semantic_x_to_expert_major_pallas_scaffold_mgpu(
        x,
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
        block_sizes=SourcePushSemanticGatherXPallasBlockSizes(row_block=2, hidden_block=4),
        interpret=True,
    )
    reconstructed = source_push_semantic_expert_major_to_pair_jax(x_expert, plan)
    expected_pair = source_push_semantic_gather_x_jax(x, plan)
    expected_valid = (
        jnp.arange(rows_per_expert_capacity, dtype=jnp.int32)[None, None, :] < plan.rows_per_local_expert[:, :, None]
    )

    np.testing.assert_array_equal(np.asarray(valid), np.asarray(expected_valid))
    np.testing.assert_array_equal(np.asarray(reconstructed), np.asarray(expected_pair))


def test_source_push_semantic_x_to_expert_major_direct_interpret_reconstructs_pair_rows():
    plan = _semantic_plan()
    x = jnp.arange(2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8).astype(jnp.bfloat16)
    rows_per_expert_capacity = 4

    x_expert, valid = source_push_semantic_x_to_expert_major_direct_pallas_mgpu(
        x,
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
        block_sizes=SourcePushSemanticGatherXPallasBlockSizes(row_block=2, hidden_block=4),
        interpret=True,
    )
    reconstructed = source_push_semantic_expert_major_to_pair_jax(x_expert, plan)
    expected_pair = source_push_semantic_gather_x_jax(x, plan)
    expected_valid = (
        jnp.arange(rows_per_expert_capacity, dtype=jnp.int32)[None, None, :] < plan.rows_per_local_expert[:, :, None]
    )

    np.testing.assert_array_equal(np.asarray(valid), np.asarray(expected_valid))
    np.testing.assert_array_equal(np.asarray(reconstructed), np.asarray(expected_pair))


def test_source_push_semantic_x_to_expert_major_direct_padded_bases_place_rows_and_leave_gaps_invalid():
    plan = _semantic_plan()
    source_row_bases = _padded_source_row_bases()
    rows_per_expert_capacity = 9
    x = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4)
    expected_x, expected_valid = _scatter_pair_rows_at_source_bases(
        source_push_semantic_gather_x_jax(x, plan),
        plan,
        source_row_bases,
        rows_per_expert_capacity,
    )

    def pack(x_arg):
        return source_push_semantic_x_to_expert_major_direct_pallas_mgpu(
            x_arg,
            plan,
            rows_per_expert_capacity=rows_per_expert_capacity,
            source_row_base_by_expert=source_row_bases,
            block_sizes=SourcePushSemanticGatherXPallasBlockSizes(row_block=1, hidden_block=4),
            interpret=True,
        )

    for observed_x, observed_valid in (pack(x), jax.jit(pack)(x)):
        np.testing.assert_array_equal(np.asarray(observed_valid), expected_valid)
        np.testing.assert_array_equal(np.asarray(observed_x), expected_x)


def test_source_push_semantic_x_to_expert_major_direct_explicit_compact_bases_match_default():
    plan = _semantic_plan()
    x = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4)
    rows_per_expert_capacity = 4

    compact = source_push_semantic_x_to_expert_major_direct_pallas_mgpu(
        x,
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
        interpret=True,
    )
    explicit = source_push_semantic_x_to_expert_major_direct_pallas_mgpu(
        x,
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
        source_row_base_by_expert=plan.src_base_by_expert,
        interpret=True,
    )

    np.testing.assert_array_equal(np.asarray(explicit[0]), np.asarray(compact[0]))
    np.testing.assert_array_equal(np.asarray(explicit[1]), np.asarray(compact[1]))


def test_source_push_semantic_x_to_expert_major_direct_rejects_invalid_padded_bases():
    plan = _semantic_plan()
    x = jnp.ones((2, 3, 4), dtype=jnp.float32)
    invalid_requests = (
        (jnp.zeros((2, 2, 3), dtype=jnp.int32), 9, ValueError),
        (jnp.zeros((2, 2, 2), dtype=jnp.float32), 9, ValueError),
        (_padded_source_row_bases(), 6, (ValueError, RuntimeError)),
    )

    for source_row_bases, rows_per_expert_capacity, error_type in invalid_requests:
        with pytest.raises(error_type, match="source_row_base_by_expert|shape|dtype"):
            result = source_push_semantic_x_to_expert_major_direct_pallas_mgpu(
                x,
                plan,
                rows_per_expert_capacity=rows_per_expert_capacity,
                source_row_base_by_expert=source_row_bases,
                interpret=True,
            )
            jax.block_until_ready(result)


def test_source_push_semantic_expert_major_source_token_lookup_reconstructs_rows():
    plan = _semantic_plan()
    x = jnp.arange(2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8).astype(jnp.bfloat16)
    rows_per_expert_capacity = 4

    source_lookup, token_lookup, valid = source_push_semantic_expert_major_source_token_lookup_jax(
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )
    reconstructed = x.at[source_lookup, token_lookup].get()
    reconstructed = jnp.where(valid[..., None], reconstructed, jnp.zeros((), dtype=reconstructed.dtype))
    expected, expected_valid = source_push_semantic_x_to_expert_major_jax(
        x,
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )

    np.testing.assert_array_equal(np.asarray(valid), np.asarray(expected_valid))
    np.testing.assert_array_equal(np.asarray(reconstructed), np.asarray(expected))


def test_source_push_semantic_expert_major_source_token_lookup_ignores_invalid_tail_rows_between_sources():
    plan = _skewed_invalid_tail_plan()

    source_lookup, token_lookup, valid = source_push_semantic_expert_major_source_token_lookup_jax(
        plan,
        rows_per_expert_capacity=3,
    )

    np.testing.assert_array_equal(np.asarray(valid[0, 0]), np.asarray([True, True, True]))
    np.testing.assert_array_equal(np.asarray(source_lookup[0, 0]), np.asarray([0, 1, 2]))
    np.testing.assert_array_equal(np.asarray(token_lookup[0, 0]), np.asarray([0, 0, 0]))


def test_source_push_semantic_x_to_expert_major_lookup_interpret_reconstructs_pair_rows():
    plan = _semantic_plan()
    x = jnp.arange(2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8).astype(jnp.bfloat16)
    rows_per_expert_capacity = 4

    x_expert, valid = source_push_semantic_x_to_expert_major_lookup_pallas_mgpu(
        x,
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
        block_sizes=SourcePushSemanticGatherXPallasBlockSizes(row_block=2, hidden_block=4),
        interpret=True,
    )
    reconstructed = source_push_semantic_expert_major_to_pair_jax(x_expert, plan)
    expected_pair = source_push_semantic_gather_x_jax(x, plan)
    expected_valid = (
        jnp.arange(rows_per_expert_capacity, dtype=jnp.int32)[None, None, :] < plan.rows_per_local_expert[:, :, None]
    )

    np.testing.assert_array_equal(np.asarray(valid), np.asarray(expected_valid))
    np.testing.assert_array_equal(np.asarray(reconstructed), np.asarray(expected_pair))


def test_source_push_semantic_w13_pallas_scaffold_interpret_matches_reference():
    plan = _semantic_plan()
    x = (jnp.arange(2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8) / 16).astype(jnp.bfloat16)
    w_gate_up = (jnp.arange(2 * 2 * 8 * 16, dtype=jnp.float32).reshape(2, 2, 8, 16) / 128).astype(jnp.bfloat16)

    observed_z, observed_h = source_push_semantic_w13_pallas_scaffold_mgpu(
        x,
        w_gate_up,
        plan,
        block_sizes=SourcePushSemanticGatherXPallasBlockSizes(row_block=2, hidden_block=4),
        w13_block_sizes=SourcePushSemanticW13PallasBlockSizes(row_block=1, hidden_block=4, intermediate_block=4),
        interpret=True,
    )
    expected_z, expected_h = source_push_semantic_w13_reference_jax(x, w_gate_up, plan)

    np.testing.assert_allclose(np.asarray(observed_z), np.asarray(expected_z), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(observed_h), np.asarray(expected_h), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_w13_from_x_pair_pallas_interpret_matches_reference():
    plan = _semantic_plan()
    x = (jnp.arange(2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8) / 32).astype(jnp.bfloat16)
    w_gate_up = (jnp.arange(2 * 2 * 8 * 16, dtype=jnp.float32).reshape(2, 2, 8, 16) / 256).astype(jnp.bfloat16)
    x_pair = source_push_semantic_gather_x_jax(x, plan)

    observed_z, observed_h = source_push_semantic_w13_from_x_pair_pallas_mgpu(
        x_pair,
        w_gate_up,
        plan,
        block_sizes=SourcePushSemanticW13PallasBlockSizes(row_block=1, hidden_block=4, intermediate_block=4),
        interpret=True,
    )
    expected_z, expected_h = source_push_semantic_w13_reference_jax(x, w_gate_up, plan)

    np.testing.assert_allclose(np.asarray(observed_z), np.asarray(expected_z), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(observed_h), np.asarray(expected_h), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_w13_from_x_expert_pallas_interpret_matches_reference():
    plan = _semantic_plan()
    x = (jnp.arange(2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8) / 32).astype(jnp.bfloat16)
    w_gate_up = (jnp.arange(2 * 2 * 8 * 16, dtype=jnp.float32).reshape(2, 2, 8, 16) / 256).astype(jnp.bfloat16)
    rows_per_expert_capacity = int(np.max(np.asarray(plan.rows_per_local_expert)))
    x_expert, valid = source_push_semantic_x_to_expert_major_jax(
        x,
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )

    observed_z, observed_h = source_push_semantic_w13_from_x_expert_pallas_mgpu(
        x_expert,
        w_gate_up,
        valid,
        block_sizes=SourcePushSemanticW13ExpertMajorPallasBlockSizes(
            row_block=1,
            hidden_block=4,
            intermediate_block=4,
        ),
        interpret=True,
    )
    observed_z_pair = source_push_semantic_expert_major_to_pair_jax(observed_z, plan)
    observed_h_pair = source_push_semantic_expert_major_to_pair_jax(observed_h, plan)
    expected_z_pair, expected_h_pair = source_push_semantic_w13_reference_jax(x, w_gate_up, plan)

    np.testing.assert_allclose(np.asarray(observed_z_pair), np.asarray(expected_z_pair), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(observed_h_pair), np.asarray(expected_h_pair), atol=1e-4, rtol=1e-4)
    np.testing.assert_array_equal(
        np.asarray(observed_z[~valid]), np.zeros((int((~valid).sum()), 16), dtype=np.float32)
    )
    np.testing.assert_array_equal(np.asarray(observed_h[~valid]), np.zeros((int((~valid).sum()), 8), dtype=np.float32))


def test_source_push_semantic_w13_expert_major_pallas_interpret_matches_reference():
    plan = _semantic_plan()
    x = (jnp.arange(2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8) / 32).astype(jnp.bfloat16)
    w_gate_up = (jnp.arange(2 * 2 * 8 * 16, dtype=jnp.float32).reshape(2, 2, 8, 16) / 256).astype(jnp.bfloat16)
    rows_per_expert_capacity = int(np.max(np.asarray(plan.rows_per_local_expert)))

    observed_z, observed_h, observed_valid = source_push_semantic_w13_expert_major_pallas_mgpu(
        x,
        w_gate_up,
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
        block_sizes=SourcePushSemanticW13ExpertMajorPallasBlockSizes(
            row_block=1,
            hidden_block=4,
            intermediate_block=4,
        ),
        interpret=True,
    )
    observed_z_pair = source_push_semantic_expert_major_to_pair_jax(observed_z, plan)
    observed_h_pair = source_push_semantic_expert_major_to_pair_jax(observed_h, plan)
    expected_z_pair, expected_h_pair = source_push_semantic_w13_reference_jax(x, w_gate_up, plan)
    expected_valid = (
        jnp.arange(rows_per_expert_capacity, dtype=jnp.int32)[None, None, :] < plan.rows_per_local_expert[:, :, None]
    )

    np.testing.assert_array_equal(np.asarray(observed_valid), np.asarray(expected_valid))
    np.testing.assert_allclose(np.asarray(observed_z_pair), np.asarray(expected_z_pair), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(observed_h_pair), np.asarray(expected_h_pair), atol=1e-4, rtol=1e-4)


def test_source_push_semantic_w13_expert_major_padded_bases_jit_matches_pair_reference():
    plan = _semantic_plan()
    source_row_bases = _padded_source_row_bases()
    rows_per_expert_capacity = 9
    x = (jnp.arange(2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8) / 32).astype(jnp.bfloat16)
    w_gate_up = (jnp.arange(2 * 2 * 8 * 16, dtype=jnp.float32).reshape(2, 2, 8, 16) / 256).astype(jnp.bfloat16)
    expected_z_pair, expected_h_pair = source_push_semantic_w13_reference_jax(x, w_gate_up, plan)
    expected_z, expected_valid = _scatter_pair_rows_at_source_bases(
        expected_z_pair,
        plan,
        source_row_bases,
        rows_per_expert_capacity,
    )
    expected_h, _ = _scatter_pair_rows_at_source_bases(
        expected_h_pair,
        plan,
        source_row_bases,
        rows_per_expert_capacity,
    )

    observed_z, observed_h, observed_valid = jax.jit(
        lambda x_arg, w_arg: source_push_semantic_w13_expert_major_pallas_mgpu(
            x_arg,
            w_arg,
            plan,
            rows_per_expert_capacity=rows_per_expert_capacity,
            source_row_base_by_expert=source_row_bases,
            block_sizes=SourcePushSemanticW13ExpertMajorPallasBlockSizes(
                row_block=1,
                hidden_block=4,
                intermediate_block=4,
            ),
            pack_block_sizes=SourcePushSemanticGatherXPallasBlockSizes(row_block=1, hidden_block=4),
            interpret=True,
        )
    )(x, w_gate_up)

    np.testing.assert_array_equal(np.asarray(observed_valid), expected_valid)
    np.testing.assert_allclose(np.asarray(observed_z), expected_z, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(observed_h), expected_h, atol=1e-4, rtol=1e-4)


def test_source_push_semantic_w13_separate_pack_interpret_matches_reference_gradients():
    plan = _semantic_plan()
    x = jnp.arange(2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8) / 32
    w_gate_up = jnp.arange(2 * 2 * 8 * 16, dtype=jnp.float32).reshape(2, 2, 8, 16) / 256
    rows_per_expert_capacity = int(np.max(np.asarray(plan.rows_per_local_expert)))

    def packed_loss(x_arg, w_arg):
        x_expert, valid = source_push_semantic_x_to_expert_major_direct_pallas_mgpu(
            x_arg,
            plan,
            rows_per_expert_capacity=rows_per_expert_capacity,
            block_sizes=SourcePushSemanticGatherXPallasBlockSizes(row_block=2, hidden_block=4),
            interpret=True,
        )
        z_expert, h_expert = source_push_semantic_w13_from_x_expert_pallas_mgpu(
            x_expert,
            w_arg,
            valid,
            block_sizes=SourcePushSemanticW13ExpertMajorPallasBlockSizes(
                row_block=1,
                hidden_block=4,
                intermediate_block=4,
            ),
            interpret=True,
        )
        return jnp.sum(z_expert) + jnp.sum(h_expert)

    def reference_loss(x_arg, w_arg):
        z_pair, h_pair = source_push_semantic_w13_reference_jax(x_arg, w_arg, plan)
        return jnp.sum(z_pair) + jnp.sum(h_pair)

    observed_dx, observed_dw = jax.grad(packed_loss, argnums=(0, 1))(x, w_gate_up)
    expected_dx, expected_dw = jax.grad(reference_loss, argnums=(0, 1))(x, w_gate_up)

    np.testing.assert_allclose(np.asarray(observed_dx), np.asarray(expected_dx), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(observed_dw), np.asarray(expected_dw), atol=1e-4, rtol=1e-4)
