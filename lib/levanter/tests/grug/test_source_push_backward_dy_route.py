# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import mosaic_gpu as mgpu

import levanter.grug._moe.source_push_mlp as source_push_mlp
from levanter.grug._moe import source_push_forward
from levanter.grug._moe import source_push_backward_dy_route as dy_route
from levanter.grug._moe.source_push_backward_dy_route import (
    SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_PALLAS_MGPU,
    SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_REFERENCE,
    _source_push_backward_dy_to_h_rows,
    _source_push_backward_dy_to_expert_major,
    _source_push_backward_dy_to_expert_major_pallas_call,
    _source_push_h_flat_indices,
)
from levanter.grug._moe.source_push_inbox import PushInboxConfig
from levanter.grug._moe.source_push_mlp import source_push_mlp_route_table_from_plan


EP_SIZE = 2
EXPERTS_PER_RANK = 2
BLOCK_M = 2
HIDDEN_DIM = 5
INTERMEDIATE_DIM = 2
TOPK = 2


def test_source_push_backward_dy_route_default_row_block_satisfies_mosaic_copy_floor():
    block_sizes = dy_route.SourcePushDyRoutePallasBlockSizes.get_default()

    assert block_sizes.row_block == dy_route.MIN_SOURCE_PUSH_DY_ROUTE_GPU_ROW_BLOCK
    assert block_sizes.row_block * np.dtype(np.int32).itemsize % 128 == 0


def test_source_push_backward_dy_route_compact_metadata_stays_in_gmem():
    in_specs, _out_spec = dy_route._source_push_backward_dy_route_compact_block_specs(
        row_block=dy_route.DEFAULT_SOURCE_PUSH_DY_ROUTE_ROW_BLOCK,
        hidden_block=dy_route.DEFAULT_SOURCE_PUSH_DY_ROUTE_HIDDEN_BLOCK,
    )

    _dy_spec, source_rank_spec, token_id_spec, valid_spec = in_specs
    assert source_rank_spec.memory_space == mgpu.GMEM
    assert token_id_spec.memory_space == mgpu.GMEM
    assert valid_spec.memory_space == mgpu.GMEM


def test_source_push_backward_dy_route_source_padded_matches_compact_reference_with_padding():
    route_assignments = jnp.array(
        [
            [[0, 0], [2, 3], [1, 2]],
            [[2, 2], [0, 1], [3, 0]],
        ],
        dtype=jnp.int32,
    )
    config, host_inputs, route_table = _forward_inputs_for_routes(
        route_assignments,
        entries_per_rank=4,
        use_exact_expert_major=False,
    )
    dy = _dy_values(config)

    observed = _assert_matches_compact_dy_reference(config, host_inputs, route_table, dy)

    assert observed.dtype == jnp.float32
    valid_rows = _flat_valid_mask(config, host_inputs)
    assert int(np.sum(valid_rows)) == int(route_assignments.size)
    np.testing.assert_array_equal(
        np.asarray(observed)[~valid_rows],
        np.zeros_like(np.asarray(observed)[~valid_rows]),
    )

    duplicate_token_rows = np.all(np.asarray(observed) == np.asarray(dy[0, 0]), axis=-1)
    assert int(np.sum(duplicate_token_rows)) == 2


def test_source_push_backward_dy_route_exact_layout_matches_compact_reference():
    route_assignments = jnp.array(
        [
            [[0, 2], [1, 3], [0, 2], [1, 3]],
            [[2, 0], [3, 1], [2, 0], [3, 1]],
        ],
        dtype=jnp.int32,
    )
    config, host_inputs, route_table = _forward_inputs_for_routes(
        route_assignments,
        entries_per_rank=2,
        use_exact_expert_major=True,
    )
    dy = _dy_values(config)

    observed = _assert_matches_compact_dy_reference(config, host_inputs, route_table, dy)

    assert host_inputs.use_exact_expert_major
    valid_rows = _flat_valid_mask(config, host_inputs)
    assert int(np.sum(valid_rows)) == int(route_assignments.size)
    np.testing.assert_array_equal(
        np.asarray(observed)[~valid_rows],
        np.zeros_like(np.asarray(observed)[~valid_rows]),
    )


def test_source_push_backward_dy_route_compact_reference_matches_compact_mlp_reference():
    route_assignments = jnp.array(
        [
            [[0, 0], [2, 3], [1, 2]],
            [[2, 2], [0, 1], [3, 0]],
        ],
        dtype=jnp.int32,
    )
    _config, _host_inputs, route_table = _forward_inputs_for_routes(
        route_assignments,
        entries_per_rank=4,
        use_exact_expert_major=False,
    )
    dy = _dy_values_for_route_table(route_table)

    observed = _assert_compact_matches_compact_dy_reference(
        route_table,
        dy,
        implementation=SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_REFERENCE,
    )

    assert observed.shape == (
        route_table.ep_size,
        route_table.experts_per_rank,
        route_table.expert_capacity,
        dy.shape[-1],
    )
    np.testing.assert_array_equal(
        np.asarray(observed)[~np.asarray(route_table.valid_by_expert)],
        np.zeros_like(np.asarray(observed)[~np.asarray(route_table.valid_by_expert)]),
    )


def test_source_push_backward_dy_route_compact_pallas_interpret_matches_compact_mlp_reference():
    route_assignments = jnp.array(
        [
            [[0, 2], [1, 3], [0, 2], [1, 3]],
            [[2, 0], [3, 1], [2, 0], [3, 1]],
        ],
        dtype=jnp.int32,
    )
    _config, _host_inputs, route_table = _forward_inputs_for_routes(
        route_assignments,
        entries_per_rank=2,
        use_exact_expert_major=True,
    )
    dy = _dy_values_for_route_table(route_table)

    observed = _assert_compact_matches_compact_dy_reference(
        route_table,
        dy,
        implementation=SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_PALLAS_MGPU,
        interpret=True,
    )

    np.testing.assert_array_equal(
        np.asarray(observed)[~np.asarray(route_table.valid_by_expert)],
        np.zeros_like(np.asarray(observed)[~np.asarray(route_table.valid_by_expert)]),
    )


def test_source_push_backward_dy_route_compact_pallas_call_interpret_matches_reference():
    route_assignments = jnp.array(
        [
            [[0, 2], [1, 3], [0, 2], [1, 3]],
            [[2, 0], [3, 1], [2, 0], [3, 1]],
        ],
        dtype=jnp.int32,
    )
    _config, _host_inputs, route_table = _forward_inputs_for_routes(
        route_assignments,
        entries_per_rank=2,
        use_exact_expert_major=True,
    )
    dy = _dy_values_for_route_table(route_table)

    observed = _source_push_backward_dy_to_expert_major_pallas_call(
        dy,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.valid_by_expert,
        row_block=2,
        hidden_block=1,
        interpret=True,
    )
    expected = _source_push_backward_dy_to_expert_major(
        dy,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.valid_by_expert,
        implementation=SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_REFERENCE,
    )

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=0, rtol=0)


def test_source_push_backward_dy_route_pallas_interpret_matches_source_padded_reference_and_zeros_invalid_rows():
    route_assignments = jnp.array(
        [
            [[0, 0], [2, 3], [1, 2]],
            [[2, 2], [0, 1], [3, 0]],
        ],
        dtype=jnp.int32,
    )
    config, host_inputs, _route_table = _forward_inputs_for_routes(
        route_assignments,
        entries_per_rank=4,
        use_exact_expert_major=False,
    )
    dy = _dy_values(config)

    _assert_pallas_interpret_matches_reference(config, host_inputs, dy)


def test_source_push_backward_dy_route_pallas_interpret_matches_exact_reference_and_zeros_invalid_rows():
    route_assignments = jnp.array(
        [
            [[0, 2], [1, 3], [0, 2], [1, 3]],
            [[2, 0], [3, 1], [2, 0], [3, 1]],
        ],
        dtype=jnp.int32,
    )
    config, host_inputs, _route_table = _forward_inputs_for_routes(
        route_assignments,
        entries_per_rank=2,
        use_exact_expert_major=True,
    )
    dy = _dy_values(config)

    _assert_pallas_interpret_matches_reference(config, host_inputs, dy)


def _forward_inputs_for_routes(
    route_assignments: jnp.ndarray,
    *,
    entries_per_rank: int,
    use_exact_expert_major: bool,
):
    tokens_per_rank = route_assignments.shape[1]
    config = PushInboxConfig(
        ep_size=EP_SIZE,
        entries_per_rank=entries_per_rank,
        inbox_slots=2,
        hidden_dim=HIDDEN_DIM,
        intermediate_dim=INTERMEDIATE_DIM,
        block_m=BLOCK_M,
        block_k=1,
        block_n=1,
        experts_per_rank=EXPERTS_PER_RANK,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=4,
        send_pipeline_depth=1,
        n_groups_per_job=1,
        routing="balanced",
        tokens_per_rank=tokens_per_rank,
        topk=TOPK,
        capacity_factor=10.0,
    )
    x = jnp.zeros((EP_SIZE, tokens_per_rank, HIDDEN_DIM), dtype=jnp.float32)
    route_weights = jnp.ones(route_assignments.shape, dtype=jnp.float32)
    w13 = jnp.zeros((EP_SIZE, EXPERTS_PER_RANK, HIDDEN_DIM, 2 * INTERMEDIATE_DIM), dtype=jnp.float32)
    w2 = jnp.zeros((EP_SIZE, EXPERTS_PER_RANK, INTERMEDIATE_DIM, HIDDEN_DIM), dtype=jnp.float32)
    host_inputs = source_push_forward.make_source_push_forward_inputs(
        config,
        x,
        route_assignments,
        route_weights,
        w13,
        w2,
        input_mode="exact_source_push_plan" if use_exact_expert_major else "real_arrays",
        use_exact_expert_major=use_exact_expert_major,
    )
    route_table = source_push_mlp_route_table_from_plan(
        host_inputs.plan,
        src_base_by_expert=host_inputs.src_base_by_expert,
    )
    assert int(host_inputs.plan.dropped_routes) == 0
    return config, host_inputs, route_table


def _dy_values(config: PushInboxConfig):
    return jnp.arange(
        1,
        1 + config.ep_size * config.tokens_per_rank * config.hidden_dim,
        dtype=jnp.float32,
    ).reshape(config.ep_size, config.tokens_per_rank, config.hidden_dim)


def _dy_values_for_route_table(route_table):
    return jnp.arange(
        1,
        1 + route_table.ep_size * route_table.tokens_per_source * HIDDEN_DIM,
        dtype=jnp.float32,
    ).reshape(route_table.ep_size, route_table.tokens_per_source, HIDDEN_DIM)


def _assert_matches_compact_dy_reference(config, host_inputs, route_table, dy):
    observed = _source_push_backward_dy_to_h_rows(config, host_inputs, dy)
    expert_base = jnp.asarray(host_inputs.expert_base, dtype=jnp.int32)

    for expert in range(config.experts_per_rank):
        route_indices = source_push_mlp._source_push_mlp_expert_route_indices(route_table, expert)
        expected = source_push_mlp._source_push_mlp_dy_to_expert_major(
            dy,
            route_indices.safe_src,
            route_indices.safe_token,
            route_indices.valid_f,
        )
        observed_for_expert = source_push_mlp._source_push_mlp_h_flat_for_expert(
            route_table,
            expert_base,
            observed,
            expert,
        )
        np.testing.assert_allclose(np.asarray(observed_for_expert), np.asarray(expected), atol=0, rtol=0)

    return observed


def _assert_compact_matches_compact_dy_reference(
    route_table,
    dy,
    *,
    implementation,
    interpret=False,
):
    observed = _source_push_backward_dy_to_expert_major(
        dy,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.valid_by_expert,
        implementation=implementation,
        block_sizes=dy_route.SourcePushDyRoutePallasBlockSizes(row_block=2, hidden_block=1),
        interpret=interpret,
    )

    for expert in range(route_table.experts_per_rank):
        route_indices = source_push_mlp._source_push_mlp_expert_route_indices(route_table, expert)
        expected = source_push_mlp._source_push_mlp_dy_to_expert_major(
            dy,
            route_indices.safe_src,
            route_indices.safe_token,
            route_indices.valid_f,
        )
        np.testing.assert_allclose(np.asarray(observed[:, expert]), np.asarray(expected), atol=0, rtol=0)

    return observed


def _assert_pallas_interpret_matches_reference(config, host_inputs, dy):
    inverse_indices = dy_route._source_push_dy_route_inverse_indices(
        host_inputs.plan,
        host_inputs.send_meta,
        host_inputs.expert_base,
        host_inputs.src_base_by_expert,
        hidden_rows_per_rank=config.hidden_rows_per_rank,
        use_exact_expert_major=host_inputs.use_exact_expert_major,
    )
    expected = _source_push_backward_dy_to_h_rows(config, host_inputs, dy)
    observed = dy_route._source_push_backward_dy_to_h_rows_pallas_mgpu(
        dy,
        inverse_indices,
        block_sizes=dy_route.SourcePushDyRoutePallasBlockSizes(row_block=2, hidden_block=1),
        interpret=True,
    )

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=0, rtol=0)
    valid_rows = _flat_valid_mask(config, host_inputs)
    np.testing.assert_array_equal(
        np.asarray(observed)[~valid_rows],
        np.zeros_like(np.asarray(observed)[~valid_rows]),
    )


def _flat_valid_mask(config, host_inputs):
    flat_dst, flat_row, valid_mask = _source_push_h_flat_indices(
        host_inputs.plan,
        host_inputs.send_meta,
        host_inputs.expert_base,
        host_inputs.src_base_by_expert,
        use_exact_expert_major=host_inputs.use_exact_expert_major,
    )
    flat_dst = np.asarray(flat_dst)
    flat_row = np.asarray(flat_row)
    valid_mask = np.asarray(valid_mask)
    row_mask = np.zeros((config.ep_size, config.hidden_rows_per_rank), dtype=np.bool_)
    for dst, row, valid in zip(flat_dst.reshape(-1), flat_row.reshape(-1), valid_mask.reshape(-1), strict=True):
        if not valid:
            continue
        assert not row_mask[dst, row]
        row_mask[dst, row] = True
    return row_mask
