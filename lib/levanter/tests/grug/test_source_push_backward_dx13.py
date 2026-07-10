# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug._moe import source_push_backward_dx13 as dx13
from levanter.grug._moe.source_push_backward_w2 import _source_push_w2_backward_expert_blocks_reference
from levanter.grug._moe.source_push_mlp import SourcePushMlpRouteTable, source_push_mlp_route_table_from_plan
from levanter.grug._moe.source_push_plan import SourcePushPlan, build_source_push_plan


EP_SIZE = 2
EXPERTS_PER_RANK = 2
TOKENS_PER_RANK = 4
TOPK = 2
BLOCK_M = 2
HIDDEN_DIM = 3
INTERMEDIATE_DIM = 2


def _small_route_table() -> tuple[SourcePushPlan, SourcePushMlpRouteTable]:
    selected_experts = jnp.array(
        [
            [[0, 0], [1, 2], [3, 0], [2, 3]],
            [[2, 2], [3, 1], [0, 2], [1, 1]],
        ],
        dtype=jnp.int32,
    )
    route_weights = jnp.ones_like(selected_experts, dtype=jnp.float32)
    plan = build_source_push_plan(
        selected_experts,
        route_weights,
        ep_size=EP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        block_m=BLOCK_M,
        capacity_factor=3.0,
    )
    assert int(plan.dropped_routes) == 0
    return plan, source_push_mlp_route_table_from_plan(plan)


def _dx13_inputs(route_table: SourcePushMlpRouteTable):
    capacity = route_table.expert_capacity
    d_activation = jnp.linspace(
        -0.35,
        0.45,
        EP_SIZE * EXPERTS_PER_RANK * capacity * INTERMEDIATE_DIM,
        dtype=jnp.float32,
    ).reshape(EP_SIZE, EXPERTS_PER_RANK, capacity, INTERMEDIATE_DIM)
    z = jnp.linspace(
        -0.5,
        0.65,
        EP_SIZE * EXPERTS_PER_RANK * capacity * 2 * INTERMEDIATE_DIM,
        dtype=jnp.float32,
    ).reshape(EP_SIZE, EXPERTS_PER_RANK, capacity, 2 * INTERMEDIATE_DIM)
    w13 = jnp.linspace(
        -0.25,
        0.55,
        EP_SIZE * EXPERTS_PER_RANK * HIDDEN_DIM * 2 * INTERMEDIATE_DIM,
        dtype=jnp.float32,
    ).reshape(EP_SIZE, EXPERTS_PER_RANK, HIDDEN_DIM, 2 * INTERMEDIATE_DIM)
    return d_activation, z, w13


def _materialized_dx13_reference(d_activation, z, w13, valid_by_expert):
    valid_f = valid_by_expert.astype(jnp.float32)
    gate, up = jnp.split(z.astype(jnp.float32) * valid_f[..., None], 2, axis=-1)
    d_activation = d_activation.astype(jnp.float32) * valid_f[..., None]
    silu_gate = jax.nn.silu(gate)
    sigmoid_gate = jax.nn.sigmoid(gate)
    d_silu_gate = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
    d_z = jnp.concatenate([d_activation * up * d_silu_gate, d_activation * silu_gate], axis=-1)
    return jnp.einsum("deco,deho->dech", d_z, w13.astype(jnp.float32)) * valid_f[..., None]


def test_source_push_dx13_expert_major_matches_materialized_dz_w13_transpose():
    _plan, route_table = _small_route_table()
    d_activation, z, w13 = _dx13_inputs(route_table)

    observed = dx13.source_push_dx13_expert_major_reference(
        d_activation,
        z,
        w13,
        route_table.valid_by_expert,
    )
    expected = _materialized_dx13_reference(d_activation, z, w13, route_table.valid_by_expert)
    output = dx13.source_push_dx13_push_compact_reference(
        d_activation,
        z,
        w13,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.route_slot_by_expert,
        route_table.valid_by_expert,
    )

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(output.dx_expert_major), np.asarray(expected), atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(
        np.asarray(output.source_rank_by_expert), np.asarray(route_table.source_rank_by_expert)
    )
    np.testing.assert_array_equal(np.asarray(output.token_id_by_expert), np.asarray(route_table.token_id_by_expert))
    np.testing.assert_array_equal(
        np.asarray(output.route_slot_by_expert), np.asarray(route_table.route_slot_by_expert)
    )
    np.testing.assert_array_equal(np.asarray(output.valid_by_expert), np.asarray(route_table.valid_by_expert))


def test_source_push_dx13_xla_expert_major_matches_materialized_dz_w13_transpose():
    _plan, route_table = _small_route_table()
    d_activation, z, w13 = _dx13_inputs(route_table)

    observed = dx13.source_push_dx13_expert_major_xla(
        d_activation,
        z,
        w13,
        route_table.valid_by_expert,
    )
    expected = _materialized_dx13_reference(d_activation, z, w13, route_table.valid_by_expert)

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-6, rtol=1e-6)


def test_source_push_dx13_xla_push_compact_matches_reference_contract():
    _plan, route_table = _small_route_table()
    d_activation, z, w13 = _dx13_inputs(route_table)

    expected = dx13.source_push_dx13_push_compact_reference(
        d_activation,
        z,
        w13,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.route_slot_by_expert,
        route_table.valid_by_expert,
    )
    observed = dx13.source_push_dx13_push_compact_xla(
        d_activation,
        z,
        w13,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.route_slot_by_expert,
        route_table.valid_by_expert,
    )

    np.testing.assert_allclose(
        np.asarray(observed.dx_expert_major),
        np.asarray(expected.dx_expert_major),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(observed.source_rank_by_expert),
        np.asarray(route_table.source_rank_by_expert),
    )
    np.testing.assert_array_equal(np.asarray(observed.token_id_by_expert), np.asarray(route_table.token_id_by_expert))
    np.testing.assert_array_equal(
        np.asarray(observed.route_slot_by_expert),
        np.asarray(route_table.route_slot_by_expert),
    )
    np.testing.assert_array_equal(np.asarray(observed.valid_by_expert), np.asarray(route_table.valid_by_expert))


def test_source_push_dx13_pallas_mgpu_interpret_matches_reference_compact_contract():
    _plan, route_table = _small_route_table()
    d_activation, z, w13 = _dx13_inputs(route_table)

    expected = dx13.source_push_dx13_push_compact_reference(
        d_activation,
        z,
        w13,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.route_slot_by_expert,
        route_table.valid_by_expert,
    )
    observed = dx13.source_push_dx13_push_compact(
        d_activation,
        z,
        w13,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.route_slot_by_expert,
        route_table.valid_by_expert,
        implementation=dx13.SOURCE_PUSH_DX13_IMPLEMENTATION_PALLAS_MGPU,
        interpret=True,
    )

    np.testing.assert_allclose(
        np.asarray(observed.dx_expert_major),
        np.asarray(expected.dx_expert_major),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(observed.source_rank_by_expert),
        np.asarray(route_table.source_rank_by_expert),
    )
    np.testing.assert_array_equal(np.asarray(observed.token_id_by_expert), np.asarray(route_table.token_id_by_expert))
    np.testing.assert_array_equal(
        np.asarray(observed.route_slot_by_expert),
        np.asarray(route_table.route_slot_by_expert),
    )
    np.testing.assert_array_equal(np.asarray(observed.valid_by_expert), np.asarray(route_table.valid_by_expert))


def test_source_push_dx13_store_zero_interpret_returns_expert_major_zero_buffer():
    _plan, route_table = _small_route_table()
    d_activation, z, w13 = _dx13_inputs(route_table)

    observed = dx13.source_push_dx13_expert_major_store_zero_pallas_mgpu(
        d_activation,
        z,
        w13,
        route_table.valid_by_expert,
        interpret=True,
    )

    assert observed.shape == (EP_SIZE, EXPERTS_PER_RANK, route_table.expert_capacity, HIDDEN_DIM)
    assert observed.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(observed), np.zeros(observed.shape, dtype=np.float32))


def test_source_push_dx13_resolved_block_sizes_match_gpu_transfer_floor():
    defaults = dx13.SourcePushDx13PallasBlockSizes.get_default()

    observed = dx13.source_push_dx13_pallas_resolved_block_sizes(defaults)
    interpret_observed = dx13.source_push_dx13_pallas_resolved_block_sizes(defaults, interpret=True)

    assert defaults.row_block < dx13.MIN_MOSAIC_INT32_TRANSFER_ELEMENTS
    assert observed == dx13.SourcePushDx13PallasBlockSizes(
        row_block=dx13.MIN_MOSAIC_INT32_TRANSFER_ELEMENTS,
        hidden_block=defaults.hidden_block,
        output_block=defaults.output_block,
    )
    assert interpret_observed == defaults


def test_source_push_dx13_masks_invalid_dirty_padding():
    _plan, route_table = _small_route_table()
    assert np.any(~np.asarray(route_table.valid_by_expert))
    d_activation, z, w13 = _dx13_inputs(route_table)
    valid = route_table.valid_by_expert
    clean_d_activation = jnp.where(valid[..., None], d_activation, jnp.zeros_like(d_activation))
    clean_z = jnp.where(valid[..., None], z, jnp.zeros_like(z))
    dirty_d_activation = jnp.where(valid[..., None], clean_d_activation, jnp.full_like(clean_d_activation, 1.0e5))
    dirty_z = jnp.where(valid[..., None], clean_z, jnp.full_like(clean_z, -1.0e5))

    expected = dx13.source_push_dx13_expert_major_reference(clean_d_activation, clean_z, w13, valid)
    observed = dx13.source_push_dx13_expert_major_reference(dirty_d_activation, dirty_z, w13, valid)

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(
        np.asarray(observed)[~np.asarray(valid)],
        np.zeros_like(np.asarray(observed)[~np.asarray(valid)]),
    )


def test_source_push_dx13_consumes_w2_d_activation_boundary():
    _plan, route_table = _small_route_table()
    capacity = route_table.expert_capacity
    _unused_d_activation, h, w13 = _dx13_inputs(route_table)
    route_weight = jnp.linspace(
        0.05,
        0.95,
        EP_SIZE * EXPERTS_PER_RANK * capacity,
        dtype=jnp.float32,
    ).reshape(EP_SIZE, EXPERTS_PER_RANK, capacity)
    dy = jnp.linspace(
        -0.45,
        0.55,
        EP_SIZE * EXPERTS_PER_RANK * capacity * HIDDEN_DIM,
        dtype=jnp.float32,
    ).reshape(EP_SIZE, EXPERTS_PER_RANK, capacity, HIDDEN_DIM)
    w2 = jnp.linspace(
        -0.3,
        0.4,
        EP_SIZE * EXPERTS_PER_RANK * INTERMEDIATE_DIM * HIDDEN_DIM,
        dtype=jnp.float32,
    ).reshape(EP_SIZE, EXPERTS_PER_RANK, INTERMEDIATE_DIM, HIDDEN_DIM)
    w2_output = _source_push_w2_backward_expert_blocks_reference(
        h,
        route_weight,
        dy,
        w2,
        route_table.valid_by_expert,
    )

    assert w2_output.d_activation is not None
    observed = dx13.source_push_dx13_expert_major_reference(
        w2_output.d_activation,
        h,
        w13,
        route_table.valid_by_expert,
    )
    expected = jnp.einsum("deco,deho->dech", w2_output.d_h, w13.astype(jnp.float32))
    expected = expected * route_table.valid_by_expert.astype(jnp.float32)[..., None]

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-6, rtol=1e-6)


def test_source_push_dx13_route_buffer_and_combine_use_source_return_metadata():
    _plan, route_table = _small_route_table()
    capacity = route_table.expert_capacity
    dx_expert_major = (
        jnp.arange(EP_SIZE * EXPERTS_PER_RANK * capacity * HIDDEN_DIM, dtype=jnp.float32).reshape(
            EP_SIZE,
            EXPERTS_PER_RANK,
            capacity,
            HIDDEN_DIM,
        )
        * 0.25
        + 1.0
    )
    dx_expert_major = jnp.where(route_table.valid_by_expert[..., None], dx_expert_major, 9999.0)

    compact_output = dx13.SourcePushDx13CompactOutput(
        dx_expert_major,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.route_slot_by_expert,
        route_table.valid_by_expert,
    )
    epilogue = dx13.source_push_dx13_route_buffer_epilogue_reference(
        compact_output,
        tokens_per_source=route_table.tokens_per_source,
        topk=route_table.topk,
    )
    fields_epilogue = dx13.source_push_dx13_route_buffer_epilogue_from_fields_reference(
        dx_expert_major,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.route_slot_by_expert,
        route_table.valid_by_expert,
        tokens_per_source=route_table.tokens_per_source,
        topk=route_table.topk,
    )
    direct_combined_dx = dx13.source_push_dx13_combine_source_tokens_reference(
        dx_expert_major,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.valid_by_expert,
        tokens_per_source=route_table.tokens_per_source,
    )

    expected_routes = np.zeros((EP_SIZE, TOKENS_PER_RANK, TOPK, HIDDEN_DIM), dtype=np.float32)
    dx_host = np.asarray(dx_expert_major)
    for dst, expert, row in np.argwhere(np.asarray(route_table.valid_by_expert)):
        src = int(route_table.source_rank_by_expert[dst, expert, row])
        token = int(route_table.token_id_by_expert[dst, expert, row])
        slot = int(route_table.route_slot_by_expert[dst, expert, row])
        expected_routes[src, token, slot] += dx_host[dst, expert, row]

    np.testing.assert_allclose(np.asarray(epilogue.dx_routes), expected_routes, atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(fields_epilogue.dx_routes), expected_routes, atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(epilogue.dx), expected_routes.sum(axis=2), atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(fields_epilogue.dx), expected_routes.sum(axis=2), atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(direct_combined_dx), expected_routes.sum(axis=2), atol=0, rtol=0)
    np.testing.assert_allclose(
        np.asarray(epilogue.dx[0, 0]),
        np.asarray(epilogue.dx_routes[0, 0, 0] + epilogue.dx_routes[0, 0, 1]),
        atol=0,
        rtol=0,
    )

    duplicate_route_rows = [
        tuple(row)
        for row in np.argwhere(
            np.asarray(route_table.valid_by_expert)
            & (np.asarray(route_table.source_rank_by_expert) == 0)
            & (np.asarray(route_table.token_id_by_expert) == 0)
        )
    ]
    assert len(duplicate_route_rows) == TOPK
    for dst, expert, row in duplicate_route_rows:
        slot = int(route_table.route_slot_by_expert[dst, expert, row])
        np.testing.assert_allclose(
            np.asarray(epilogue.dx_routes[0, 0, slot]),
            dx_host[dst, expert, row],
            atol=0,
            rtol=0,
        )


def test_source_push_dx13_push_route_buffer_reference_matches_compact_epilogue():
    _plan, route_table = _small_route_table()
    d_activation, z, w13 = _dx13_inputs(route_table)

    compact_output = dx13.source_push_dx13_push_compact_reference(
        d_activation,
        z,
        w13,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.route_slot_by_expert,
        route_table.valid_by_expert,
    )
    expected = dx13.source_push_dx13_route_buffer_epilogue_reference(
        compact_output,
        tokens_per_source=route_table.tokens_per_source,
        topk=route_table.topk,
    )
    observed = dx13.source_push_dx13_push_route_buffer_reference(
        d_activation,
        z,
        w13,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.route_slot_by_expert,
        route_table.valid_by_expert,
        tokens_per_source=route_table.tokens_per_source,
        topk=route_table.topk,
    )

    np.testing.assert_allclose(np.asarray(observed.dx_routes), np.asarray(expected.dx_routes), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(observed.dx), np.asarray(expected.dx), atol=1e-6, rtol=1e-6)

    dx_routes_host = np.asarray(observed.dx_routes)
    dx_expert_major_host = np.asarray(compact_output.dx_expert_major)
    for dst, expert, row in np.argwhere(np.asarray(route_table.valid_by_expert)):
        src = int(route_table.source_rank_by_expert[dst, expert, row])
        token = int(route_table.token_id_by_expert[dst, expert, row])
        slot = int(route_table.route_slot_by_expert[dst, expert, row])
        np.testing.assert_allclose(
            dx_routes_host[src, token, slot],
            dx_expert_major_host[dst, expert, row],
            atol=1e-6,
            rtol=1e-6,
        )


def test_source_push_dx13_compact_assignment_slots_map_expert_rows_to_source_queue_slots():
    plan, route_table = _small_route_table()
    slots = dx13.source_push_dx13_compact_assignment_slots_from_plan(
        plan,
        expert_capacity=route_table.expert_capacity,
    )

    source_rank = np.asarray(slots.source_rank_by_expert)
    dst_ordinal = np.asarray(slots.dst_ordinal_by_expert)
    entry_by_expert = np.asarray(slots.entry_by_expert)
    row_in_entry_by_expert = np.asarray(slots.row_in_entry_by_expert)
    valid_by_expert = np.asarray(slots.valid_by_expert)
    token_ids = np.asarray(plan.token_ids)
    route_slots = np.asarray(plan.route_slots)

    np.testing.assert_array_equal(valid_by_expert, np.asarray(route_table.valid_by_expert))
    for dst, expert, expert_row in np.argwhere(np.asarray(route_table.valid_by_expert)):
        src = int(source_rank[dst, expert, expert_row])
        dst_ord = int(dst_ordinal[dst, expert, expert_row])
        entry = int(entry_by_expert[dst, expert, expert_row])
        row = int(row_in_entry_by_expert[dst, expert, expert_row])

        assert (src + dst_ord) % EP_SIZE == dst
        assert token_ids[src, dst_ord, entry, row] == int(route_table.token_id_by_expert[dst, expert, expert_row])
        assert route_slots[src, dst_ord, entry, row] == int(route_table.route_slot_by_expert[dst, expert, expert_row])


def test_source_push_dx13_compact_assignment_slots_from_route_table_fields_match_plan():
    plan, route_table = _small_route_table()
    expected = dx13.source_push_dx13_compact_assignment_slots_from_plan(
        plan,
        expert_capacity=route_table.expert_capacity,
    )
    observed = dx13.source_push_dx13_compact_assignment_slots_from_fields(
        route_table.source_rank_by_expert,
        route_table.dst_ordinal_by_expert,
        route_table.entry_by_expert,
        route_table.row_in_entry_by_expert,
        route_table.valid_by_expert,
    )

    valid = np.asarray(expected.valid_by_expert)
    np.testing.assert_array_equal(
        np.asarray(observed.source_rank_by_expert)[valid],
        np.asarray(expected.source_rank_by_expert)[valid],
    )
    np.testing.assert_array_equal(
        np.asarray(observed.dst_ordinal_by_expert)[valid],
        np.asarray(expected.dst_ordinal_by_expert)[valid],
    )
    np.testing.assert_array_equal(
        np.asarray(observed.entry_by_expert)[valid], np.asarray(expected.entry_by_expert)[valid]
    )
    np.testing.assert_array_equal(
        np.asarray(observed.row_in_entry_by_expert)[valid],
        np.asarray(expected.row_in_entry_by_expert)[valid],
    )
    np.testing.assert_array_equal(np.asarray(observed.valid_by_expert), np.asarray(expected.valid_by_expert))


def test_source_push_dx13_push_contrib_returns_source_compact_assignment_buffer():
    plan, route_table = _small_route_table()
    d_activation, z, w13 = _dx13_inputs(route_table)
    slots = dx13.source_push_dx13_compact_assignment_slots_from_plan(
        plan,
        expert_capacity=route_table.expert_capacity,
    )

    observed = dx13.source_push_dx13_push_contrib_reference(
        d_activation,
        z,
        w13,
        slots,
        queue_shape=plan.valid_mask.shape,
    )
    combined = dx13.source_push_dx13_source_compact_to_route_buffer_reference(observed, plan)
    compact_output = dx13.source_push_dx13_push_compact_reference(
        d_activation,
        z,
        w13,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.route_slot_by_expert,
        route_table.valid_by_expert,
    )
    expected = dx13.source_push_dx13_route_buffer_epilogue_reference(
        compact_output,
        tokens_per_source=route_table.tokens_per_source,
        topk=route_table.topk,
    )

    dx_contrib = np.asarray(observed.dx_contrib)
    dx_expert_major = np.asarray(compact_output.dx_expert_major)
    source_rank = np.asarray(slots.source_rank_by_expert)
    dst_ordinal = np.asarray(slots.dst_ordinal_by_expert)
    entry_by_expert = np.asarray(slots.entry_by_expert)
    row_in_entry_by_expert = np.asarray(slots.row_in_entry_by_expert)

    assert observed.dx_contrib.shape == (*plan.valid_mask.shape, HIDDEN_DIM)
    np.testing.assert_allclose(np.asarray(combined.dx_routes), np.asarray(expected.dx_routes), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(combined.dx), np.asarray(expected.dx), atol=1e-6, rtol=1e-6)
    for dst, expert, expert_row in np.argwhere(np.asarray(route_table.valid_by_expert)):
        src = int(source_rank[dst, expert, expert_row])
        dst_ord = int(dst_ordinal[dst, expert, expert_row])
        entry = int(entry_by_expert[dst, expert, expert_row])
        row = int(row_in_entry_by_expert[dst, expert, expert_row])
        np.testing.assert_allclose(
            dx_contrib[src, dst_ord, entry, row],
            dx_expert_major[dst, expert, expert_row],
            atol=1e-6,
            rtol=1e-6,
        )


def test_source_push_dx13_push_compact_contrib_boundary_requires_separate_source_combine():
    plan, route_table = _small_route_table()
    d_activation, z, w13 = _dx13_inputs(route_table)
    slots = dx13.source_push_dx13_compact_assignment_slots_from_plan(
        plan,
        expert_capacity=route_table.expert_capacity,
    )

    observed = dx13.source_push_dx13_push_compact_contrib(
        d_activation,
        z,
        w13,
        slots,
        queue_shape=plan.valid_mask.shape,
    )
    combined = dx13.source_push_dx13_source_compact_combine_reference(observed, plan)
    compact_rows = dx13.source_push_dx13_push_compact_reference(
        d_activation,
        z,
        w13,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.route_slot_by_expert,
        route_table.valid_by_expert,
    )
    expected = dx13.source_push_dx13_route_buffer_epilogue_reference(
        compact_rows,
        tokens_per_source=route_table.tokens_per_source,
        topk=route_table.topk,
    )

    assert isinstance(observed, dx13.SourcePushDx13SourceCompactOutput)
    assert observed.dx_contrib.shape == (*plan.valid_mask.shape, HIDDEN_DIM)
    assert not hasattr(observed, "dx")
    assert not hasattr(observed, "dx_routes")
    np.testing.assert_allclose(np.asarray(combined), np.asarray(expected.dx), atol=1e-6, rtol=1e-6)


def test_source_push_dx13_source_compact_combine_skips_route_buffer_with_dirty_padding():
    plan, route_table = _small_route_table()
    d_activation, z, w13 = _dx13_inputs(route_table)
    slots = dx13.source_push_dx13_compact_assignment_slots_from_plan(
        plan,
        expert_capacity=route_table.expert_capacity,
    )
    compact_output = dx13.source_push_dx13_push_contrib_reference(
        d_activation,
        z,
        w13,
        slots,
        queue_shape=plan.valid_mask.shape,
    )
    dirty_contrib = jnp.where(
        plan.valid_mask[..., None],
        compact_output.dx_contrib,
        jnp.full_like(compact_output.dx_contrib, 1.0e5),
    )
    dirty_output = dx13.SourcePushDx13SourceCompactOutput(dx_contrib=dirty_contrib)

    observed = dx13.source_push_dx13_source_compact_combine_reference(dirty_output, plan)
    route_buffer_output = dx13.source_push_dx13_source_compact_to_route_buffer_reference(dirty_output, plan)
    compact_rows = dx13.source_push_dx13_push_compact_reference(
        d_activation,
        z,
        w13,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.route_slot_by_expert,
        route_table.valid_by_expert,
    )
    expected = dx13.source_push_dx13_route_buffer_epilogue_reference(
        compact_rows,
        tokens_per_source=route_table.tokens_per_source,
        topk=route_table.topk,
    )

    assert observed.shape == (EP_SIZE, TOKENS_PER_RANK, HIDDEN_DIM)
    assert np.any(~np.asarray(plan.valid_mask))
    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected.dx), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(route_buffer_output.dx), np.asarray(expected.dx), atol=1e-6, rtol=1e-6)


def test_source_push_dx13_push_contrib_pallas_interpret_matches_reference():
    plan, route_table = _small_route_table()
    d_activation, z, w13 = _dx13_inputs(route_table)
    slots = dx13.source_push_dx13_compact_assignment_slots_from_fields(
        route_table.source_rank_by_expert,
        route_table.dst_ordinal_by_expert,
        route_table.entry_by_expert,
        route_table.row_in_entry_by_expert,
        route_table.valid_by_expert,
    )

    expected = dx13.source_push_dx13_push_contrib_reference(
        d_activation,
        z,
        w13,
        slots,
        queue_shape=plan.valid_mask.shape,
    )
    observed = dx13.source_push_dx13_push_contrib(
        d_activation,
        z,
        w13,
        slots,
        queue_shape=plan.valid_mask.shape,
        implementation=dx13.SOURCE_PUSH_DX13_IMPLEMENTATION_PALLAS_MGPU,
        interpret=True,
    )

    np.testing.assert_allclose(np.asarray(observed.dx_contrib), np.asarray(expected.dx_contrib), atol=1e-6, rtol=1e-6)


def test_source_push_dx13_source_compact_block_contiguous_predicate_accepts_full_queue_entries():
    slots = dx13.source_push_dx13_compact_assignment_slots_from_fields(
        source_rank_by_expert=jnp.array([[[0, 0, 1, 1]], [[0, 0, 1, 1]]], dtype=jnp.int32),
        dst_ordinal_by_expert=jnp.array([[[0, 0, 1, 1]], [[0, 0, 1, 1]]], dtype=jnp.int32),
        entry_by_expert=jnp.array([[[0, 0, 0, 0]], [[1, 1, 1, 1]]], dtype=jnp.int32),
        row_in_entry_by_expert=jnp.array([[[0, 1, 0, 1]], [[0, 1, 0, 1]]], dtype=jnp.int32),
        valid_by_expert=jnp.ones((2, 1, 4), dtype=jnp.bool_),
    )

    assert dx13.source_push_dx13_source_compact_slots_are_block_contiguous(
        slots,
        queue_shape=(2, 2, 2, 2),
        row_block=2,
    )


def test_source_push_dx13_source_compact_block_contiguous_predicate_rejects_row_shuffle():
    slots = dx13.source_push_dx13_compact_assignment_slots_from_fields(
        source_rank_by_expert=jnp.array([[[0, 0]], [[1, 1]]], dtype=jnp.int32),
        dst_ordinal_by_expert=jnp.array([[[0, 0]], [[0, 0]]], dtype=jnp.int32),
        entry_by_expert=jnp.array([[[0, 0]], [[0, 0]]], dtype=jnp.int32),
        row_in_entry_by_expert=jnp.array([[[1, 0]], [[0, 1]]], dtype=jnp.int32),
        valid_by_expert=jnp.ones((2, 1, 2), dtype=jnp.bool_),
    )

    assert not dx13.source_push_dx13_source_compact_slots_are_block_contiguous(
        slots,
        queue_shape=(2, 1, 1, 2),
        row_block=2,
    )
    with pytest.raises(ValueError, match="row_in_entry"):
        dx13.source_push_dx13_push_contrib_block_contiguous_pallas_mgpu(
            jnp.zeros((2, 1, 2, INTERMEDIATE_DIM), dtype=jnp.float32),
            jnp.zeros((2, 1, 2, 2 * INTERMEDIATE_DIM), dtype=jnp.float32),
            jnp.zeros((2, 1, HIDDEN_DIM, 2 * INTERMEDIATE_DIM), dtype=jnp.float32),
            slots,
            queue_shape=(2, 1, 1, 2),
            block_sizes=dx13.SourcePushDx13PallasBlockSizes(row_block=2, hidden_block=HIDDEN_DIM, output_block=1),
            interpret=True,
        )


def test_source_push_dx13_push_contrib_block_contiguous_interpret_matches_reference():
    slots = dx13.source_push_dx13_compact_assignment_slots_from_fields(
        source_rank_by_expert=jnp.array([[[0, 0, 1, 1]], [[0, 0, 1, 1]]], dtype=jnp.int32),
        dst_ordinal_by_expert=jnp.array([[[0, 0, 1, 1]], [[0, 0, 1, 1]]], dtype=jnp.int32),
        entry_by_expert=jnp.array([[[0, 0, 0, 0]], [[1, 1, 1, 1]]], dtype=jnp.int32),
        row_in_entry_by_expert=jnp.array([[[0, 1, 0, 1]], [[0, 1, 0, 1]]], dtype=jnp.int32),
        valid_by_expert=jnp.ones((2, 1, 4), dtype=jnp.bool_),
    )
    d_activation = jnp.linspace(-0.3, 0.4, 2 * 1 * 4 * INTERMEDIATE_DIM, dtype=jnp.float32).reshape(
        2, 1, 4, INTERMEDIATE_DIM
    )
    z = jnp.linspace(-0.2, 0.5, 2 * 1 * 4 * 2 * INTERMEDIATE_DIM, dtype=jnp.float32).reshape(
        2, 1, 4, 2 * INTERMEDIATE_DIM
    )
    w13 = jnp.linspace(-0.1, 0.6, 2 * 1 * HIDDEN_DIM * 2 * INTERMEDIATE_DIM, dtype=jnp.float32).reshape(
        2, 1, HIDDEN_DIM, 2 * INTERMEDIATE_DIM
    )

    expected = dx13.source_push_dx13_push_contrib_reference(
        d_activation,
        z,
        w13,
        slots,
        queue_shape=(2, 2, 2, 2),
    )
    observed = dx13.source_push_dx13_push_contrib_block_contiguous_pallas_mgpu(
        d_activation,
        z,
        w13,
        slots,
        queue_shape=(2, 2, 2, 2),
        block_sizes=dx13.SourcePushDx13PallasBlockSizes(row_block=2, hidden_block=HIDDEN_DIM, output_block=1),
        interpret=True,
    )

    np.testing.assert_allclose(np.asarray(observed.dx_contrib), np.asarray(expected.dx_contrib), atol=1e-6, rtol=1e-6)


def test_source_push_dx13_push_contrib_rejects_mismatched_source_compact_slot_shape():
    plan, route_table = _small_route_table()
    d_activation, z, w13 = _dx13_inputs(route_table)
    bad_slots = dx13.SourcePushDx13SourceCompactSlots(
        source_rank_by_expert=route_table.source_rank_by_expert[:, :, :-1],
        dst_ordinal_by_expert=route_table.dst_ordinal_by_expert,
        entry_by_expert=route_table.entry_by_expert,
        row_in_entry_by_expert=route_table.row_in_entry_by_expert,
        valid_by_expert=route_table.valid_by_expert,
    )

    with pytest.raises(ValueError, match="source_rank_by_expert shape"):
        dx13.source_push_dx13_push_contrib_reference(
            d_activation,
            z,
            w13,
            bad_slots,
            queue_shape=plan.valid_mask.shape,
        )


def test_source_push_dx13_push_contrib_pallas_requires_gpu_backend():
    plan, route_table = _small_route_table()
    d_activation, z, w13 = _dx13_inputs(route_table)
    slots = dx13.source_push_dx13_compact_assignment_slots_from_fields(
        route_table.source_rank_by_expert,
        route_table.dst_ordinal_by_expert,
        route_table.entry_by_expert,
        route_table.row_in_entry_by_expert,
        route_table.valid_by_expert,
    )

    if jax.default_backend() == "gpu":
        pytest.skip("CPU-only guard is not meaningful on GPU")
    with pytest.raises(NotImplementedError, match="source-compact push requires a GPU backend"):
        dx13.source_push_dx13_push_contrib(
            d_activation,
            z,
            w13,
            slots,
            queue_shape=plan.valid_mask.shape,
            implementation=dx13.SOURCE_PUSH_DX13_IMPLEMENTATION_PALLAS_MGPU,
        )


def test_source_push_dx13_source_grouped_reconstructs_route_buffer_without_token_scatter_at_producer():
    plan, route_table = _small_route_table()
    d_activation, z, w13 = _dx13_inputs(route_table)

    compact_output = dx13.source_push_dx13_push_compact_reference(
        d_activation,
        z,
        w13,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.route_slot_by_expert,
        route_table.valid_by_expert,
    )
    grouped = dx13.source_push_dx13_source_grouped_from_fields_reference(
        compact_output.dx_expert_major,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.route_slot_by_expert,
        route_table.valid_by_expert,
        plan.src_base_by_expert,
    )
    observed = dx13.source_push_dx13_source_grouped_to_route_buffer_reference(
        grouped,
        tokens_per_source=route_table.tokens_per_source,
        topk=route_table.topk,
    )
    expected = dx13.source_push_dx13_route_buffer_epilogue_reference(
        compact_output,
        tokens_per_source=route_table.tokens_per_source,
        topk=route_table.topk,
    )

    np.testing.assert_allclose(np.asarray(observed.dx_routes), np.asarray(expected.dx_routes), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(observed.dx), np.asarray(expected.dx), atol=1e-6, rtol=1e-6)

    grouped_dx = np.asarray(grouped.dx_by_source)
    grouped_tokens = np.asarray(grouped.token_id_by_source)
    grouped_slots = np.asarray(grouped.route_slot_by_source)
    grouped_valid = np.asarray(grouped.valid_by_source)
    dx_host = np.asarray(compact_output.dx_expert_major)
    src_base = np.asarray(plan.src_base_by_expert)
    for dst, expert, row in np.argwhere(np.asarray(route_table.valid_by_expert)):
        src = int(route_table.source_rank_by_expert[dst, expert, row])
        source_row = row - int(src_base[dst, src, expert])
        np.testing.assert_allclose(
            grouped_dx[dst, src, expert, source_row],
            dx_host[dst, expert, row],
            atol=1e-6,
            rtol=1e-6,
        )
        assert grouped_tokens[dst, src, expert, source_row] == int(route_table.token_id_by_expert[dst, expert, row])
        assert grouped_slots[dst, src, expert, source_row] == int(route_table.route_slot_by_expert[dst, expert, row])
        assert grouped_valid[dst, src, expert, source_row]


def test_source_push_dx13_source_grouped_public_api_matches_reference():
    plan, route_table = _small_route_table()
    d_activation, z, w13 = _dx13_inputs(route_table)
    compact_output = dx13.source_push_dx13_push_compact_reference(
        d_activation,
        z,
        w13,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.route_slot_by_expert,
        route_table.valid_by_expert,
    )

    expected = dx13.source_push_dx13_source_grouped_from_fields_reference(
        compact_output.dx_expert_major,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.route_slot_by_expert,
        route_table.valid_by_expert,
        plan.src_base_by_expert,
    )
    observed = dx13.source_push_dx13_source_grouped_from_fields(
        compact_output.dx_expert_major,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.route_slot_by_expert,
        route_table.valid_by_expert,
        plan.src_base_by_expert,
    )

    np.testing.assert_allclose(
        np.asarray(observed.dx_by_source), np.asarray(expected.dx_by_source), atol=1e-6, rtol=1e-6
    )
    np.testing.assert_array_equal(np.asarray(observed.token_id_by_source), np.asarray(expected.token_id_by_source))
    np.testing.assert_array_equal(np.asarray(observed.route_slot_by_source), np.asarray(expected.route_slot_by_source))
    np.testing.assert_array_equal(np.asarray(observed.valid_by_source), np.asarray(expected.valid_by_source))


def test_source_push_dx13_source_grouped_pallas_interpret_matches_reference():
    plan, route_table = _small_route_table()
    d_activation, z, w13 = _dx13_inputs(route_table)
    compact_output = dx13.source_push_dx13_push_compact_reference(
        d_activation,
        z,
        w13,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.route_slot_by_expert,
        route_table.valid_by_expert,
    )

    expected = dx13.source_push_dx13_source_grouped_from_fields_reference(
        compact_output.dx_expert_major,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.route_slot_by_expert,
        route_table.valid_by_expert,
        plan.src_base_by_expert,
    )
    observed = dx13.source_push_dx13_source_grouped_from_fields(
        compact_output.dx_expert_major,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.route_slot_by_expert,
        route_table.valid_by_expert,
        plan.src_base_by_expert,
        implementation=dx13.SOURCE_PUSH_DX13_IMPLEMENTATION_PALLAS_MGPU,
        interpret=True,
    )

    np.testing.assert_allclose(
        np.asarray(observed.dx_by_source), np.asarray(expected.dx_by_source), atol=1e-6, rtol=1e-6
    )
    np.testing.assert_array_equal(np.asarray(observed.token_id_by_source), np.asarray(expected.token_id_by_source))
    np.testing.assert_array_equal(np.asarray(observed.route_slot_by_source), np.asarray(expected.route_slot_by_source))
    np.testing.assert_array_equal(np.asarray(observed.valid_by_source), np.asarray(expected.valid_by_source))


def test_source_push_dx13_push_source_grouped_reference_matches_compact_then_grouped():
    plan, route_table = _small_route_table()
    d_activation, z, w13 = _dx13_inputs(route_table)

    compact_output = dx13.source_push_dx13_push_compact_reference(
        d_activation,
        z,
        w13,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.route_slot_by_expert,
        route_table.valid_by_expert,
    )
    expected = dx13.source_push_dx13_source_grouped_from_fields_reference(
        compact_output.dx_expert_major,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.route_slot_by_expert,
        route_table.valid_by_expert,
        plan.src_base_by_expert,
    )
    observed = dx13.source_push_dx13_push_source_grouped_reference(
        d_activation,
        z,
        w13,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.route_slot_by_expert,
        route_table.valid_by_expert,
        plan.src_base_by_expert,
    )

    np.testing.assert_allclose(
        np.asarray(observed.dx_by_source), np.asarray(expected.dx_by_source), atol=1e-6, rtol=1e-6
    )
    np.testing.assert_array_equal(np.asarray(observed.token_id_by_source), np.asarray(expected.token_id_by_source))
    np.testing.assert_array_equal(np.asarray(observed.route_slot_by_source), np.asarray(expected.route_slot_by_source))
    np.testing.assert_array_equal(np.asarray(observed.valid_by_source), np.asarray(expected.valid_by_source))


def test_source_push_dx13_push_route_buffer_public_api_matches_reference():
    _plan, route_table = _small_route_table()
    d_activation, z, w13 = _dx13_inputs(route_table)

    expected = dx13.source_push_dx13_push_route_buffer_reference(
        d_activation,
        z,
        w13,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.route_slot_by_expert,
        route_table.valid_by_expert,
        tokens_per_source=route_table.tokens_per_source,
        topk=route_table.topk,
    )
    observed = dx13.source_push_dx13_push_route_buffer(
        d_activation,
        z,
        w13,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.route_slot_by_expert,
        route_table.valid_by_expert,
        tokens_per_source=route_table.tokens_per_source,
        topk=route_table.topk,
    )

    np.testing.assert_allclose(np.asarray(observed.dx_routes), np.asarray(expected.dx_routes), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(observed.dx), np.asarray(expected.dx), atol=1e-6, rtol=1e-6)


def test_source_push_dx13_push_route_buffer_pallas_interpret_matches_reference():
    _plan, route_table = _small_route_table()
    d_activation, z, w13 = _dx13_inputs(route_table)

    expected = dx13.source_push_dx13_push_route_buffer_reference(
        d_activation,
        z,
        w13,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.route_slot_by_expert,
        route_table.valid_by_expert,
        tokens_per_source=route_table.tokens_per_source,
        topk=route_table.topk,
    )
    observed = dx13.source_push_dx13_push_route_buffer(
        d_activation,
        z,
        w13,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.route_slot_by_expert,
        route_table.valid_by_expert,
        tokens_per_source=route_table.tokens_per_source,
        topk=route_table.topk,
        implementation=dx13.SOURCE_PUSH_DX13_IMPLEMENTATION_PALLAS_MGPU,
        interpret=True,
    )

    np.testing.assert_allclose(np.asarray(observed.dx_routes), np.asarray(expected.dx_routes), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(observed.dx), np.asarray(expected.dx), atol=1e-6, rtol=1e-6)


def test_source_push_dx13_push_route_buffer_pallas_requires_remote_store_kernel():
    _plan, route_table = _small_route_table()
    d_activation, z, w13 = _dx13_inputs(route_table)

    with pytest.raises(NotImplementedError):
        dx13.source_push_dx13_push_route_buffer(
            d_activation,
            z,
            w13,
            route_table.source_rank_by_expert,
            route_table.token_id_by_expert,
            route_table.route_slot_by_expert,
            route_table.valid_by_expert,
            tokens_per_source=route_table.tokens_per_source,
            topk=route_table.topk,
            implementation=dx13.SOURCE_PUSH_DX13_IMPLEMENTATION_PALLAS_MGPU,
        )
