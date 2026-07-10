# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import mosaic_gpu as mgpu
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from levanter.grug._moe import source_push_forward
from levanter.grug._moe import source_push_mlp
from levanter.grug._moe.source_push_backward_w13 import (
    SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_TILED,
    SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_PALLAS_MGPU,
    SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_REFERENCE,
    SOURCE_PUSH_X_TO_W13_ROWS_IMPLEMENTATION_PALLAS_MGPU,
    SourcePushW13BackwardTiledBlockSizes,
    SourcePushXToW13RowsPallasBlockSizes,
    _pad_w13_compact_dh_for_row_block,
    _source_push_w13_backward_expert_blocks_compact_dx_source_gather_dw13,
    _source_push_w13_backward_expert_blocks_dw13_only_exact_flat_pallas_mgpu,
    _source_push_w13_backward_expert_blocks_dw13_only_pallas_mgpu,
    _source_push_w13_backward_expert_blocks_dx_only_pallas_mgpu,
    _source_push_w13_backward_expert_blocks_local_linear_dw13_only_pallas_mgpu,
    _source_push_w13_backward_expert_blocks_local_swiglu_gate_dw13_only_pallas_mgpu,
    _source_push_w13_backward_expert_blocks_local_swiglu_dw13_only_pallas_mgpu,
    _source_push_w13_backward_expert_blocks_local_swiglu_persistent_dw13_only_pallas_mgpu,
    _source_push_w13_backward_expert_blocks_local_swiglu_split_dw13_only_pallas_mgpu,
    _source_push_w13_backward_expert_blocks_local_swiglu_up_dw13_only_pallas_mgpu,
    _source_push_w13_backward_expert_blocks_pallas_mgpu,
    _source_push_w13_backward_expert_blocks_prefilled_x_dw13_only_pallas_mgpu,
    _source_push_w13_backward_pallas_mgpu,
    _source_push_x_to_w13_rows_block_specs,
    _source_push_w13_dw13_source_padded_partials_pallas_mgpu,
    _source_push_w13_dw13_expert_blocks_source_gather_pallas_mgpu,
    estimate_source_push_w13_backward_cost,
    source_push_w13_backward,
    source_push_w13_backward_expert_blocks_dw13_only_xla,
    source_push_w13_backward_expert_blocks_local_swiglu_dw13_only_xla,
    source_push_w13_backward_expert_blocks_source_gather_dw13_only,
    source_push_w13_backward_expert_blocks_source_padded_dw13_only_xla,
    source_push_w13_backward_expert_blocks_reference,
    source_push_w13_backward_expert_blocks_tiled_reference,
    source_push_w13_backward_reference,
    source_push_w13_dw13_local_linear_reference,
    source_push_w13_dw13_local_swiglu_branch_reference,
    source_push_w13_dw13_local_swiglu_reference,
    source_push_w13_dw13_expert_blocks_source_gather_tiled_reference,
    source_push_w13_dx_expert_blocks_reference,
    source_push_x_to_w13_rows,
    source_push_x_to_w13_rows_reference,
)
from levanter.grug._moe.source_push_inbox import PushInboxConfig
from levanter.grug._moe.source_push_mlp import source_push_mlp_route_table_from_plan


EP_SIZE = 2
EXPERTS_PER_RANK = 2
TOKENS_PER_RANK = 3
TOPK = 2
BLOCK_M = 4
HIDDEN_DIM = 4
INTERMEDIATE_DIM = 3


def _small_source_push_w13_case(*, block_m: int = BLOCK_M, use_exact_expert_major: bool = False):
    route_assignments = jnp.array(
        [
            [[0, 2], [1, 2], [3, 0]],
            [[2, 0], [3, 1], [3, 2]],
        ],
        dtype=jnp.int32,
    )
    route_weights = jnp.array(
        [
            [[0.50, 0.25], [0.75, 0.125], [0.375, 0.625]],
            [[0.20, 0.80], [0.60, 0.40], [0.30, 0.70]],
        ],
        dtype=jnp.float32,
    )
    x = jnp.linspace(
        -0.2,
        0.4,
        EP_SIZE * TOKENS_PER_RANK * HIDDEN_DIM,
        dtype=jnp.float32,
    ).reshape(EP_SIZE, TOKENS_PER_RANK, HIDDEN_DIM)
    w13 = jnp.linspace(
        -0.3,
        0.5,
        EP_SIZE * EXPERTS_PER_RANK * HIDDEN_DIM * 2 * INTERMEDIATE_DIM,
        dtype=jnp.float32,
    ).reshape(EP_SIZE, EXPERTS_PER_RANK, HIDDEN_DIM, 2 * INTERMEDIATE_DIM)
    config = PushInboxConfig(
        ep_size=EP_SIZE,
        entries_per_rank=4,
        inbox_slots=2,
        hidden_dim=HIDDEN_DIM,
        intermediate_dim=INTERMEDIATE_DIM,
        block_m=block_m,
        block_k=1,
        block_n=1,
        experts_per_rank=EXPERTS_PER_RANK,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=4,
        send_pipeline_depth=1,
        n_groups_per_job=1,
        routing="balanced",
        tokens_per_rank=TOKENS_PER_RANK,
        topk=TOPK,
        capacity_factor=4.0,
    )
    w2 = jnp.zeros((EP_SIZE, EXPERTS_PER_RANK, INTERMEDIATE_DIM, HIDDEN_DIM), dtype=jnp.float32)
    host_inputs = source_push_forward.make_source_push_forward_inputs(
        config,
        x,
        route_assignments,
        route_weights,
        w13,
        w2,
        use_exact_expert_major=use_exact_expert_major,
    )
    route_table = source_push_mlp_route_table_from_plan(
        host_inputs.plan,
        src_base_by_expert=host_inputs.src_base_by_expert,
    )
    assert int(host_inputs.plan.dropped_routes) == 0
    return config, host_inputs, route_table, x, w13


def _flat_live_row_mask(config, host_inputs):
    send_meta = np.asarray(host_inputs.send_meta, dtype=np.int32)
    mask = np.zeros((config.ep_size, config.hidden_rows_per_rank), dtype=np.bool_)
    for src in range(config.ep_size):
        for dst_ordinal in range(config.ep_size):
            dst = (src + dst_ordinal) % config.ep_size
            for entry in range(config.entries_per_rank):
                row_start = int(send_meta[src, dst_ordinal, entry, 2])
                if host_inputs.use_exact_expert_major:
                    expert = int(send_meta[src, dst_ordinal, entry, 1])
                    row_start += int(host_inputs.expert_base[dst, expert])
                    row_start += int(host_inputs.src_base_by_expert[dst, src, expert])
                valid_rows = int(send_meta[src, dst_ordinal, entry, 3])
                mask[dst, row_start : row_start + valid_rows] = True
    return mask


def _dirty_dh(config, host_inputs):
    d_h = jnp.linspace(
        -0.3,
        0.5,
        config.ep_size * config.hidden_rows_per_rank * 2 * config.intermediate_dim,
        dtype=jnp.float32,
    ).reshape(config.ep_size, config.hidden_rows_per_rank, 2 * config.intermediate_dim)
    invalid = jnp.asarray(~_flat_live_row_mask(config, host_inputs), dtype=jnp.float32)
    return d_h + invalid[..., None] * jnp.asarray(100.0, dtype=jnp.float32)


def _compact_x_from_route_table(x, route_table):
    safe_src = jnp.where(route_table.valid_by_expert, route_table.source_rank_by_expert, 0)
    safe_token = jnp.where(route_table.valid_by_expert, route_table.token_id_by_expert, 0)
    x_expert_major = x.at[safe_src, safe_token].get()
    return x_expert_major * route_table.valid_by_expert[..., None].astype(jnp.float32)


def _compact_swiglu_inputs(config, route_table):
    d_activation = jnp.linspace(
        -0.25,
        0.35,
        config.ep_size * config.experts_per_rank * route_table.valid_by_expert.shape[-1] * config.intermediate_dim,
        dtype=jnp.float32,
    ).reshape(config.ep_size, config.experts_per_rank, route_table.valid_by_expert.shape[-1], config.intermediate_dim)
    z = jnp.linspace(
        -0.45,
        0.55,
        config.ep_size * config.experts_per_rank * route_table.valid_by_expert.shape[-1] * 2 * config.intermediate_dim,
        dtype=jnp.float32,
    ).reshape(
        config.ep_size,
        config.experts_per_rank,
        route_table.valid_by_expert.shape[-1],
        2 * config.intermediate_dim,
    )
    invalid = (~route_table.valid_by_expert)[..., None].astype(jnp.float32)
    return d_activation + invalid * 100.0, z + invalid * 100.0


def test_source_push_w13_backward_reference_matches_existing_per_expert_helpers():
    config, host_inputs, route_table, x, w13 = _small_source_push_w13_case()
    expert_base = jnp.asarray(host_inputs.expert_base, dtype=jnp.int32)
    d_h = _dirty_dh(config, host_inputs)

    observed = source_push_w13_backward_reference(
        x,
        d_h,
        w13,
        host_inputs.plan,
        jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        expert_base,
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
    )

    for expert in range(EXPERTS_PER_RANK):
        route_indices = source_push_mlp._source_push_mlp_expert_route_indices(route_table, expert)
        d_h_block = source_push_mlp._source_push_mlp_h_flat_for_expert(route_table, expert_base, d_h, expert)
        d_h_block = d_h_block * route_indices.valid_f[..., None]
        expected = source_push_mlp._source_push_mlp_x_w13_backward_for_expert(
            x,
            route_indices,
            d_h_block,
            w13[:, expert].astype(jnp.float32),
        )
        observed_x_block = source_push_mlp._source_push_mlp_h_flat_for_expert(
            route_table,
            expert_base,
            observed.x_expert_major,
            expert,
        )
        observed_dx_block = source_push_mlp._source_push_mlp_h_flat_for_expert(
            route_table,
            expert_base,
            observed.dx_expert_major,
            expert,
        )
        expected_x_block = source_push_mlp._source_push_mlp_x_to_expert_major(
            x,
            route_indices.safe_src,
            route_indices.safe_token,
            route_indices.valid_f,
        )

        np.testing.assert_allclose(
            np.asarray(observed_x_block * route_indices.valid_f[..., None]),
            np.asarray(expected_x_block),
            atol=0,
            rtol=0,
        )
        np.testing.assert_allclose(
            np.asarray(observed_dx_block * route_indices.valid_f[..., None]),
            np.asarray(expected.dx_block),
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(observed.dw13[:, expert]),
            np.asarray(expected.dw13_block),
            atol=1e-6,
            rtol=1e-6,
        )


@pytest.mark.parametrize(
    "block_m,use_exact_expert_major",
    [
        (BLOCK_M, False),
        (1, True),
    ],
)
def test_source_push_w13_backward_compact_reference_matches_flat_reference(block_m, use_exact_expert_major):
    config, host_inputs, route_table, x, w13 = _small_source_push_w13_case(
        block_m=block_m,
        use_exact_expert_major=use_exact_expert_major,
    )
    expert_base = jnp.asarray(host_inputs.expert_base, dtype=jnp.int32)
    d_h_flat = _dirty_dh(config, host_inputs)
    d_h_blocks = jnp.zeros(
        (EP_SIZE, EXPERTS_PER_RANK, route_table.expert_capacity, 2 * INTERMEDIATE_DIM),
        dtype=d_h_flat.dtype,
    )
    for expert in range(EXPERTS_PER_RANK):
        d_h_blocks = d_h_blocks.at[:, expert].set(
            source_push_mlp._source_push_mlp_h_flat_for_expert(route_table, expert_base, d_h_flat, expert)
        )

    expected = source_push_w13_backward_reference(
        x,
        d_h_flat,
        w13,
        host_inputs.plan,
        jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        expert_base,
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
        use_exact_expert_major=use_exact_expert_major,
    )
    observed = source_push_w13_backward_expert_blocks_reference(
        x,
        d_h_blocks,
        w13,
        host_inputs.plan,
        jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        expert_base,
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
        use_exact_expert_major=use_exact_expert_major,
    )

    for expert in range(EXPERTS_PER_RANK):
        route_indices = source_push_mlp._source_push_mlp_expert_route_indices(route_table, expert)
        expected_x_block = source_push_mlp._source_push_mlp_h_flat_for_expert(
            route_table,
            expert_base,
            expected.x_expert_major,
            expert,
        )
        expected_dx_block = source_push_mlp._source_push_mlp_h_flat_for_expert(
            route_table,
            expert_base,
            expected.dx_expert_major,
            expert,
        )
        np.testing.assert_allclose(
            np.asarray(observed.x_expert_major[:, expert] * route_indices.valid_f[..., None]),
            np.asarray(expected_x_block * route_indices.valid_f[..., None]),
            atol=0,
            rtol=0,
        )
        np.testing.assert_allclose(
            np.asarray(observed.dx_expert_major[:, expert] * route_indices.valid_f[..., None]),
            np.asarray(expected_dx_block * route_indices.valid_f[..., None]),
            atol=1e-6,
            rtol=1e-6,
        )
    np.testing.assert_allclose(np.asarray(observed.dw13), np.asarray(expected.dw13), atol=1e-6, rtol=1e-6)


def test_source_push_w13_backward_compact_tiled_matches_compact_reference_without_x_materialization():
    config, host_inputs, route_table, x, w13 = _small_source_push_w13_case(block_m=1, use_exact_expert_major=True)
    expert_base = jnp.asarray(host_inputs.expert_base, dtype=jnp.int32)
    d_h_flat = _dirty_dh(config, host_inputs)
    d_h_blocks = jnp.zeros(
        (EP_SIZE, EXPERTS_PER_RANK, route_table.expert_capacity, 2 * INTERMEDIATE_DIM),
        dtype=d_h_flat.dtype,
    )
    for expert in range(EXPERTS_PER_RANK):
        d_h_blocks = d_h_blocks.at[:, expert].set(
            source_push_mlp._source_push_mlp_h_flat_for_expert(route_table, expert_base, d_h_flat, expert)
        )
    block_sizes = SourcePushW13BackwardTiledBlockSizes(row_block=2, hidden_block=2, output_block=3)

    expected = source_push_w13_backward_expert_blocks_reference(
        x,
        d_h_blocks,
        w13,
        host_inputs.plan,
        jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        expert_base,
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
        use_exact_expert_major=True,
    )
    observed = source_push_w13_backward_expert_blocks_tiled_reference(
        x,
        d_h_blocks,
        w13,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.valid_by_expert,
        block_sizes=block_sizes,
    )

    assert observed.x_expert_major.shape == (0,)
    np.testing.assert_allclose(
        np.asarray(observed.dx_expert_major),
        np.asarray(expected.dx_expert_major),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(observed.dw13), np.asarray(expected.dw13), atol=1e-6, rtol=1e-6)


def test_source_push_w13_backward_compact_tiled_preserves_destination_sharding():
    config, host_inputs, route_table, x, w13 = _small_source_push_w13_case(block_m=1, use_exact_expert_major=True)
    expert_base = jnp.asarray(host_inputs.expert_base, dtype=jnp.int32)
    d_h_flat = _dirty_dh(config, host_inputs)
    d_h_blocks = jnp.zeros(
        (EP_SIZE, EXPERTS_PER_RANK, route_table.expert_capacity, 2 * INTERMEDIATE_DIM),
        dtype=d_h_flat.dtype,
    )
    for expert in range(EXPERTS_PER_RANK):
        d_h_blocks = d_h_blocks.at[:, expert].set(
            source_push_mlp._source_push_mlp_h_flat_for_expert(route_table, expert_base, d_h_flat, expert)
        )
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("expert",))
    d_h_blocks = jax.device_put(d_h_blocks, NamedSharding(mesh, P("expert", None, None, None)))

    observed = source_push_w13_backward_expert_blocks_tiled_reference(
        x,
        d_h_blocks,
        w13,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.valid_by_expert,
        block_sizes=SourcePushW13BackwardTiledBlockSizes(row_block=2, hidden_block=2, output_block=3),
    )

    assert isinstance(observed.dx_expert_major.sharding, NamedSharding)
    assert observed.dx_expert_major.sharding.spec == P("expert", None, None, None)


def test_source_push_w13_backward_decomposition_helpers_match_compact_reference():
    config, host_inputs, route_table, x, w13 = _small_source_push_w13_case(block_m=1, use_exact_expert_major=True)
    expert_base = jnp.asarray(host_inputs.expert_base, dtype=jnp.int32)
    d_h_flat = _dirty_dh(config, host_inputs)
    d_h_blocks = jnp.zeros(
        (EP_SIZE, EXPERTS_PER_RANK, route_table.expert_capacity, 2 * INTERMEDIATE_DIM),
        dtype=d_h_flat.dtype,
    )
    for expert in range(EXPERTS_PER_RANK):
        d_h_blocks = d_h_blocks.at[:, expert].set(
            source_push_mlp._source_push_mlp_h_flat_for_expert(route_table, expert_base, d_h_flat, expert)
        )
    block_sizes = SourcePushW13BackwardTiledBlockSizes(row_block=2, hidden_block=2, output_block=3)

    expected = source_push_w13_backward_expert_blocks_reference(
        x,
        d_h_blocks,
        w13,
        host_inputs.plan,
        jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        expert_base,
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
        use_exact_expert_major=True,
    )
    observed_dx = source_push_w13_dx_expert_blocks_reference(
        d_h_blocks,
        w13,
        route_table.valid_by_expert,
    )
    observed_dw13 = source_push_w13_dw13_expert_blocks_source_gather_tiled_reference(
        x,
        d_h_blocks,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.valid_by_expert,
        block_sizes=block_sizes,
    )

    np.testing.assert_allclose(
        np.asarray(observed_dx),
        np.asarray(expected.dx_expert_major),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(observed_dw13), np.asarray(expected.dw13), atol=1e-6, rtol=1e-6)


def test_source_push_w13_backward_source_gather_dw13_only_avoids_x_output():
    config, host_inputs, route_table, x, w13 = _small_source_push_w13_case(block_m=1, use_exact_expert_major=True)
    expert_base = jnp.asarray(host_inputs.expert_base, dtype=jnp.int32)
    d_h_flat = _dirty_dh(config, host_inputs)
    d_h_blocks = jnp.zeros(
        (EP_SIZE, EXPERTS_PER_RANK, route_table.expert_capacity, 2 * INTERMEDIATE_DIM),
        dtype=d_h_flat.dtype,
    )
    for expert in range(EXPERTS_PER_RANK):
        d_h_blocks = d_h_blocks.at[:, expert].set(
            source_push_mlp._source_push_mlp_h_flat_for_expert(route_table, expert_base, d_h_flat, expert)
        )
    block_sizes = SourcePushW13BackwardTiledBlockSizes(row_block=2, hidden_block=2, output_block=3)

    expected = source_push_w13_backward_expert_blocks_reference(
        x,
        d_h_blocks,
        w13,
        host_inputs.plan,
        jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        expert_base,
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
        use_exact_expert_major=True,
    )
    observed = source_push_w13_backward_expert_blocks_source_gather_dw13_only(
        x,
        d_h_blocks,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.valid_by_expert,
        block_sizes=block_sizes,
    )

    assert observed.x_expert_major.shape == (0,)
    np.testing.assert_allclose(np.asarray(observed.dw13), np.asarray(expected.dw13), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(observed.dx_expert_major),
        np.zeros_like(np.asarray(expected.dx_expert_major)),
        atol=0,
        rtol=0,
    )


def test_source_push_w13_backward_source_gather_dw13_pallas_interpreter_matches_reference():
    config, host_inputs, route_table, x, w13 = _small_source_push_w13_case(block_m=1, use_exact_expert_major=True)
    expert_base = jnp.asarray(host_inputs.expert_base, dtype=jnp.int32)
    d_h_flat = _dirty_dh(config, host_inputs)
    d_h_blocks = jnp.zeros(
        (EP_SIZE, EXPERTS_PER_RANK, route_table.expert_capacity, 2 * INTERMEDIATE_DIM),
        dtype=d_h_flat.dtype,
    )
    for expert in range(EXPERTS_PER_RANK):
        d_h_blocks = d_h_blocks.at[:, expert].set(
            source_push_mlp._source_push_mlp_h_flat_for_expert(route_table, expert_base, d_h_flat, expert)
        )
    block_sizes = SourcePushW13BackwardTiledBlockSizes(row_block=2, hidden_block=2, output_block=3)

    expected = source_push_w13_backward_expert_blocks_reference(
        x,
        d_h_blocks,
        w13,
        host_inputs.plan,
        jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        expert_base,
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
        use_exact_expert_major=True,
    )
    observed_dw13 = _source_push_w13_dw13_expert_blocks_source_gather_pallas_mgpu(
        x,
        d_h_blocks,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.valid_by_expert,
        block_sizes=block_sizes,
        interpret=True,
    )

    np.testing.assert_allclose(np.asarray(observed_dw13), np.asarray(expected.dw13), atol=1e-6, rtol=1e-6)


def test_source_push_w13_backward_source_gather_dw13_pallas_requires_gpu_lowering_on_cpu():
    _config, _host_inputs, route_table, x, _w13 = _small_source_push_w13_case(
        block_m=1,
        use_exact_expert_major=True,
    )
    d_h_blocks = jnp.zeros(
        (EP_SIZE, EXPERTS_PER_RANK, route_table.expert_capacity, 2 * INTERMEDIATE_DIM),
        dtype=jnp.float32,
    )

    with pytest.raises(NotImplementedError, match="requires a GPU backend"):
        _source_push_w13_dw13_expert_blocks_source_gather_pallas_mgpu(
            x,
            d_h_blocks,
            route_table.source_rank_by_expert,
            route_table.token_id_by_expert,
            route_table.valid_by_expert,
            block_sizes=SourcePushW13BackwardTiledBlockSizes(row_block=2, hidden_block=2, output_block=3),
        )


def test_source_push_w13_backward_dx_only_pallas_interpreter_matches_compact_reference():
    config, host_inputs, route_table, _x, w13 = _small_source_push_w13_case(block_m=1, use_exact_expert_major=True)
    expert_base = jnp.asarray(host_inputs.expert_base, dtype=jnp.int32)
    d_h_flat = _dirty_dh(config, host_inputs)
    d_h_blocks = jnp.zeros(
        (EP_SIZE, EXPERTS_PER_RANK, route_table.expert_capacity, 2 * INTERMEDIATE_DIM),
        dtype=d_h_flat.dtype,
    )
    for expert in range(EXPERTS_PER_RANK):
        d_h_blocks = d_h_blocks.at[:, expert].set(
            source_push_mlp._source_push_mlp_h_flat_for_expert(route_table, expert_base, d_h_flat, expert)
        )
    expected_dx = source_push_w13_dx_expert_blocks_reference(
        d_h_blocks,
        w13,
        route_table.valid_by_expert,
    )

    observed = _source_push_w13_backward_expert_blocks_dx_only_pallas_mgpu(
        d_h_blocks,
        w13,
        route_table.valid_by_expert,
        block_sizes=SourcePushW13BackwardTiledBlockSizes(row_block=2, hidden_block=2, output_block=3),
        interpret=True,
    )

    assert observed.x_expert_major.shape == (0,)
    np.testing.assert_allclose(np.asarray(observed.dx_expert_major), np.asarray(expected_dx), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(observed.dw13), np.zeros_like(np.asarray(w13)), atol=0, rtol=0)


def test_source_push_w13_backward_dw13_only_pallas_interpreter_matches_compact_reference():
    config, host_inputs, route_table, x, w13 = _small_source_push_w13_case(block_m=1, use_exact_expert_major=True)
    expert_base = jnp.asarray(host_inputs.expert_base, dtype=jnp.int32)
    d_h_flat = _dirty_dh(config, host_inputs)
    d_h_blocks = jnp.zeros(
        (EP_SIZE, EXPERTS_PER_RANK, route_table.expert_capacity, 2 * INTERMEDIATE_DIM),
        dtype=d_h_flat.dtype,
    )
    for expert in range(EXPERTS_PER_RANK):
        d_h_blocks = d_h_blocks.at[:, expert].set(
            source_push_mlp._source_push_mlp_h_flat_for_expert(route_table, expert_base, d_h_flat, expert)
        )
    expected = source_push_w13_backward_expert_blocks_reference(
        x,
        d_h_blocks,
        w13,
        host_inputs.plan,
        jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        expert_base,
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
        use_exact_expert_major=True,
    )

    observed = _source_push_w13_backward_expert_blocks_dw13_only_pallas_mgpu(
        x,
        d_h_blocks,
        w13,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.valid_by_expert,
        block_sizes=SourcePushW13BackwardTiledBlockSizes(row_block=2, hidden_block=2, output_block=3),
        interpret=True,
    )

    assert observed.x_expert_major.shape == (0,)
    assert observed.dx_expert_major.shape == (0,)
    np.testing.assert_allclose(np.asarray(observed.dw13), np.asarray(expected.dw13), atol=1e-6, rtol=1e-6)


def test_source_push_w13_backward_prefilled_x_dw13_only_pallas_interpreter_matches_compact_reference():
    config, host_inputs, route_table, x, w13 = _small_source_push_w13_case(block_m=1, use_exact_expert_major=True)
    expert_base = jnp.asarray(host_inputs.expert_base, dtype=jnp.int32)
    d_h_flat = _dirty_dh(config, host_inputs)
    d_h_blocks = jnp.zeros(
        (EP_SIZE, EXPERTS_PER_RANK, route_table.expert_capacity, 2 * INTERMEDIATE_DIM),
        dtype=d_h_flat.dtype,
    )
    for expert in range(EXPERTS_PER_RANK):
        d_h_blocks = d_h_blocks.at[:, expert].set(
            source_push_mlp._source_push_mlp_h_flat_for_expert(route_table, expert_base, d_h_flat, expert)
        )
    expected = source_push_w13_backward_expert_blocks_reference(
        x,
        d_h_blocks,
        w13,
        host_inputs.plan,
        jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        expert_base,
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
        use_exact_expert_major=True,
    )
    x_expert_major = _compact_x_from_route_table(x, route_table)

    observed = _source_push_w13_backward_expert_blocks_prefilled_x_dw13_only_pallas_mgpu(
        x_expert_major,
        d_h_blocks,
        w13,
        route_table.valid_by_expert,
        block_sizes=SourcePushW13BackwardTiledBlockSizes(row_block=2, hidden_block=2, output_block=3),
        interpret=True,
    )

    assert observed.x_expert_major.shape == (0,)
    assert observed.dx_expert_major.shape == (0,)
    np.testing.assert_allclose(np.asarray(observed.dw13), np.asarray(expected.dw13), atol=1e-6, rtol=1e-6)


def test_source_push_w13_backward_dw13_only_exact_flat_pallas_interpreter_matches_compact_reference():
    config, host_inputs, route_table, x, w13 = _small_source_push_w13_case(block_m=1, use_exact_expert_major=True)
    expert_base = jnp.asarray(host_inputs.expert_base, dtype=jnp.int32)
    d_h_flat = _dirty_dh(config, host_inputs)
    d_h_blocks = jnp.zeros(
        (EP_SIZE, EXPERTS_PER_RANK, route_table.expert_capacity, 2 * INTERMEDIATE_DIM),
        dtype=d_h_flat.dtype,
    )
    for expert in range(EXPERTS_PER_RANK):
        d_h_blocks = d_h_blocks.at[:, expert].set(
            source_push_mlp._source_push_mlp_h_flat_for_expert(route_table, expert_base, d_h_flat, expert)
        )
    expected = source_push_w13_backward_expert_blocks_reference(
        x,
        d_h_blocks,
        w13,
        host_inputs.plan,
        jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        expert_base,
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
        use_exact_expert_major=True,
    )

    observed = _source_push_w13_backward_expert_blocks_dw13_only_exact_flat_pallas_mgpu(
        x,
        d_h_blocks,
        w13,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.valid_by_expert,
        block_sizes=SourcePushW13BackwardTiledBlockSizes(row_block=2, hidden_block=2, output_block=3),
        interpret=True,
    )

    assert observed.x_expert_major.shape == (0,)
    assert observed.dx_expert_major.shape == (0,)
    np.testing.assert_allclose(np.asarray(observed.dw13), np.asarray(expected.dw13), atol=1e-6, rtol=1e-6)


def test_source_push_w13_backward_dw13_only_xla_matches_compact_reference():
    config, host_inputs, route_table, x, w13 = _small_source_push_w13_case(block_m=1, use_exact_expert_major=True)
    expert_base = jnp.asarray(host_inputs.expert_base, dtype=jnp.int32)
    d_h_flat = _dirty_dh(config, host_inputs)
    d_h_blocks = jnp.zeros(
        (EP_SIZE, EXPERTS_PER_RANK, route_table.expert_capacity, 2 * INTERMEDIATE_DIM),
        dtype=d_h_flat.dtype,
    )
    for expert in range(EXPERTS_PER_RANK):
        d_h_blocks = d_h_blocks.at[:, expert].set(
            source_push_mlp._source_push_mlp_h_flat_for_expert(route_table, expert_base, d_h_flat, expert)
        )
    expected = source_push_w13_backward_expert_blocks_reference(
        x,
        d_h_blocks,
        w13,
        host_inputs.plan,
        jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        expert_base,
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
        use_exact_expert_major=True,
    )

    observed = source_push_w13_backward_expert_blocks_dw13_only_xla(
        x,
        d_h_blocks,
        w13,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.valid_by_expert,
        block_sizes=SourcePushW13BackwardTiledBlockSizes(row_block=2, hidden_block=2, output_block=3),
    )

    assert observed.x_expert_major.shape == (0,)
    assert observed.dx_expert_major.shape == (0,)
    np.testing.assert_allclose(np.asarray(observed.dw13), np.asarray(expected.dw13), atol=1e-6, rtol=1e-6)


def test_source_push_w13_local_swiglu_dw13_reference_matches_materialized_dz():
    config, _host_inputs, route_table, x, _w13 = _small_source_push_w13_case(block_m=1, use_exact_expert_major=True)
    x_expert_major = _compact_x_from_route_table(x, route_table)
    d_activation, z = _compact_swiglu_inputs(config, route_table)
    valid_f = route_table.valid_by_expert.astype(jnp.float32)

    gate, up = jnp.split(z.astype(jnp.float32) * valid_f[..., None], 2, axis=-1)
    d_activation_clean = d_activation.astype(jnp.float32) * valid_f[..., None]
    silu_gate = jax.nn.silu(gate)
    sigmoid_gate = jax.nn.sigmoid(gate)
    d_silu_gate = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
    d_z = jnp.concatenate([d_activation_clean * up * d_silu_gate, d_activation_clean * silu_gate], axis=-1)
    expected = jnp.einsum("dech,deco->deho", x_expert_major.astype(jnp.float32), d_z)

    observed = source_push_w13_dw13_local_swiglu_reference(
        x_expert_major + (~route_table.valid_by_expert)[..., None].astype(jnp.float32) * 50.0,
        d_activation,
        z,
        route_table.valid_by_expert,
    )

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-6, rtol=1e-6)


def test_source_push_w13_local_linear_dw13_reference_matches_materialized_diagnostic_dz():
    config, _host_inputs, route_table, x, _w13 = _small_source_push_w13_case(block_m=1, use_exact_expert_major=True)
    x_expert_major = _compact_x_from_route_table(x, route_table)
    d_activation, z = _compact_swiglu_inputs(config, route_table)
    valid_f = route_table.valid_by_expert.astype(jnp.float32)

    gate, up = jnp.split(z.astype(jnp.float32) * valid_f[..., None], 2, axis=-1)
    d_activation_clean = d_activation.astype(jnp.float32) * valid_f[..., None]
    d_z = jnp.concatenate([d_activation_clean * up, d_activation_clean * gate], axis=-1)
    expected = jnp.einsum("dech,deco->deho", x_expert_major.astype(jnp.float32), d_z)

    observed = source_push_w13_dw13_local_linear_reference(
        x_expert_major + (~route_table.valid_by_expert)[..., None].astype(jnp.float32) * 50.0,
        d_activation,
        z,
        route_table.valid_by_expert,
    )

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=1e-6, rtol=1e-6)


def test_source_push_w13_local_swiglu_dw13_masks_invalid_rows_before_derivative():
    config, _host_inputs, route_table, x, _w13 = _small_source_push_w13_case(block_m=1, use_exact_expert_major=True)
    x_expert_major = _compact_x_from_route_table(x, route_table)
    d_activation, z = _compact_swiglu_inputs(config, route_table)
    valid = route_table.valid_by_expert
    invalid = ~valid
    expected = source_push_w13_dw13_local_swiglu_reference(x_expert_major, d_activation, z, valid)
    dirty_x = jnp.where(invalid[..., None], jnp.full_like(x_expert_major, jnp.nan), x_expert_major)
    dirty_d_activation = jnp.where(invalid[..., None], jnp.full_like(d_activation, jnp.nan), d_activation)
    dirty_z = jnp.where(invalid[..., None], jnp.full_like(z, jnp.nan), z)

    observed = _source_push_w13_backward_expert_blocks_local_swiglu_dw13_only_pallas_mgpu(
        dirty_x,
        dirty_d_activation,
        dirty_z,
        valid,
        block_sizes=SourcePushW13BackwardTiledBlockSizes(row_block=2, hidden_block=2, output_block=3),
        interpret=True,
    )

    assert observed.x_expert_major.shape == (0,)
    assert observed.dx_expert_major.shape == (0,)
    assert not np.isnan(np.asarray(observed.dw13)).any()
    np.testing.assert_allclose(np.asarray(observed.dw13), np.asarray(expected), atol=1e-6, rtol=1e-6)


def test_source_push_w13_local_linear_dw13_pallas_interpreter_matches_reference():
    config, _host_inputs, route_table, x, _w13 = _small_source_push_w13_case(block_m=1, use_exact_expert_major=True)
    x_expert_major = _compact_x_from_route_table(x, route_table)
    d_activation, z = _compact_swiglu_inputs(config, route_table)
    expected = source_push_w13_dw13_local_linear_reference(
        x_expert_major,
        d_activation,
        z,
        route_table.valid_by_expert,
    )

    observed = _source_push_w13_backward_expert_blocks_local_linear_dw13_only_pallas_mgpu(
        x_expert_major,
        d_activation,
        z,
        route_table.valid_by_expert,
        block_sizes=SourcePushW13BackwardTiledBlockSizes(row_block=2, hidden_block=2, output_block=3),
        interpret=True,
    )

    assert observed.x_expert_major.shape == (0,)
    assert observed.dx_expert_major.shape == (0,)
    np.testing.assert_allclose(np.asarray(observed.dw13), np.asarray(expected), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("branch", "wrapper"),
    [
        ("gate", _source_push_w13_backward_expert_blocks_local_swiglu_gate_dw13_only_pallas_mgpu),
        ("up", _source_push_w13_backward_expert_blocks_local_swiglu_up_dw13_only_pallas_mgpu),
    ],
)
def test_source_push_w13_local_swiglu_branch_dw13_pallas_interpreter_matches_reference(branch, wrapper):
    config, _host_inputs, route_table, x, _w13 = _small_source_push_w13_case(block_m=1, use_exact_expert_major=True)
    x_expert_major = _compact_x_from_route_table(x, route_table)
    d_activation, z = _compact_swiglu_inputs(config, route_table)
    expected = source_push_w13_dw13_local_swiglu_branch_reference(
        x_expert_major,
        d_activation,
        z,
        route_table.valid_by_expert,
        branch=branch,
    )
    full_expected = source_push_w13_dw13_local_swiglu_reference(
        x_expert_major,
        d_activation,
        z,
        route_table.valid_by_expert,
    )
    if branch == "gate":
        expected_slice = full_expected[..., :INTERMEDIATE_DIM]
    else:
        expected_slice = full_expected[..., INTERMEDIATE_DIM:]

    observed = wrapper(
        x_expert_major,
        d_activation,
        z,
        route_table.valid_by_expert,
        block_sizes=SourcePushW13BackwardTiledBlockSizes(row_block=2, hidden_block=2, output_block=3),
        interpret=True,
    )

    assert observed.x_expert_major.shape == (0,)
    assert observed.dx_expert_major.shape == (0,)
    assert observed.dw13.shape[-1] == INTERMEDIATE_DIM
    np.testing.assert_allclose(np.asarray(expected), np.asarray(expected_slice), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(observed.dw13), np.asarray(expected), atol=1e-6, rtol=1e-6)


def test_source_push_w13_local_swiglu_dw13_api_rejects_full_dz_input():
    config, _host_inputs, route_table, x, _w13 = _small_source_push_w13_case(block_m=1, use_exact_expert_major=True)
    x_expert_major = _compact_x_from_route_table(x, route_table)
    _d_activation, z = _compact_swiglu_inputs(config, route_table)
    full_dz = jnp.zeros_like(z)

    with pytest.raises(ValueError, match=r"z output dim .* must be 2 \* d_activation dim"):
        source_push_w13_dw13_local_swiglu_reference(
            x_expert_major,
            full_dz,
            z,
            route_table.valid_by_expert,
        )


def test_source_push_w13_local_swiglu_dw13_pallas_interpreter_matches_reference():
    config, _host_inputs, route_table, x, _w13 = _small_source_push_w13_case(block_m=1, use_exact_expert_major=True)
    x_expert_major = _compact_x_from_route_table(x, route_table)
    d_activation, z = _compact_swiglu_inputs(config, route_table)
    expected = source_push_w13_dw13_local_swiglu_reference(
        x_expert_major,
        d_activation,
        z,
        route_table.valid_by_expert,
    )

    observed = _source_push_w13_backward_expert_blocks_local_swiglu_dw13_only_pallas_mgpu(
        x_expert_major,
        d_activation,
        z,
        route_table.valid_by_expert,
        block_sizes=SourcePushW13BackwardTiledBlockSizes(row_block=2, hidden_block=2, output_block=3),
        interpret=True,
    )

    assert observed.x_expert_major.shape == (0,)
    assert observed.dx_expert_major.shape == (0,)
    np.testing.assert_allclose(np.asarray(observed.dw13), np.asarray(expected), atol=1e-6, rtol=1e-6)


def test_source_push_w13_local_swiglu_split_dw13_pallas_interpreter_matches_reference():
    config, _host_inputs, route_table, x, _w13 = _small_source_push_w13_case(block_m=1, use_exact_expert_major=True)
    x_expert_major = _compact_x_from_route_table(x, route_table)
    d_activation, z = _compact_swiglu_inputs(config, route_table)
    expected = source_push_w13_dw13_local_swiglu_reference(
        x_expert_major,
        d_activation,
        z,
        route_table.valid_by_expert,
    )

    observed = _source_push_w13_backward_expert_blocks_local_swiglu_split_dw13_only_pallas_mgpu(
        x_expert_major,
        d_activation,
        z,
        route_table.valid_by_expert,
        block_sizes=SourcePushW13BackwardTiledBlockSizes(row_block=2, hidden_block=2, output_block=3),
        interpret=True,
    )

    assert observed.x_expert_major.shape == (0,)
    assert observed.dx_expert_major.shape == (0,)
    np.testing.assert_allclose(np.asarray(observed.dw13), np.asarray(expected), atol=1e-6, rtol=1e-6)


def test_source_push_w13_local_swiglu_dw13_xla_matches_reference():
    config, _host_inputs, route_table, x, _w13 = _small_source_push_w13_case(block_m=1, use_exact_expert_major=True)
    x_expert_major = _compact_x_from_route_table(x, route_table)
    d_activation, z = _compact_swiglu_inputs(config, route_table)
    expected = source_push_w13_dw13_local_swiglu_reference(
        x_expert_major,
        d_activation,
        z,
        route_table.valid_by_expert,
    )

    observed = source_push_w13_backward_expert_blocks_local_swiglu_dw13_only_xla(
        x_expert_major,
        d_activation,
        z,
        route_table.valid_by_expert,
    )

    assert observed.x_expert_major.shape == (0,)
    assert observed.dx_expert_major.shape == (0,)
    np.testing.assert_allclose(np.asarray(observed.dw13), np.asarray(expected), atol=1e-6, rtol=1e-6)


def test_source_push_w13_local_swiglu_persistent_dw13_interpreter_matches_reference_with_finite_invalid_rows():
    config, _host_inputs, route_table, x, _w13 = _small_source_push_w13_case(block_m=1, use_exact_expert_major=True)
    x_expert_major = _compact_x_from_route_table(x, route_table)
    d_activation, z = _compact_swiglu_inputs(config, route_table)
    valid = route_table.valid_by_expert
    invalid = ~valid
    expected = source_push_w13_dw13_local_swiglu_reference(x_expert_major, d_activation, z, valid)
    dirty_x = jnp.where(invalid[..., None], jnp.zeros_like(x_expert_major), x_expert_major)
    dirty_d_activation = jnp.where(invalid[..., None], d_activation + 50.0, d_activation)
    dirty_z = jnp.where(invalid[..., None], z + 50.0, z)

    observed = _source_push_w13_backward_expert_blocks_local_swiglu_persistent_dw13_only_pallas_mgpu(
        dirty_x,
        dirty_d_activation,
        dirty_z,
        valid,
        block_sizes=SourcePushW13BackwardTiledBlockSizes(row_block=2, hidden_block=2, output_block=3),
        interpret=True,
    )

    assert observed.x_expert_major.shape == (0,)
    assert observed.dx_expert_major.shape == (0,)
    assert not np.isnan(np.asarray(observed.dw13)).any()
    np.testing.assert_allclose(np.asarray(observed.dw13), np.asarray(expected), atol=1e-6, rtol=1e-6)


def test_source_push_w13_backward_source_padded_dw13_only_xla_matches_compact_reference():
    config, host_inputs, route_table, x, w13 = _small_source_push_w13_case(block_m=2, use_exact_expert_major=False)
    expert_base = jnp.asarray(host_inputs.expert_base, dtype=jnp.int32)
    d_h_flat = _dirty_dh(config, host_inputs)
    d_h_blocks = jnp.zeros(
        (EP_SIZE, EXPERTS_PER_RANK, route_table.expert_capacity, 2 * INTERMEDIATE_DIM),
        dtype=d_h_flat.dtype,
    )
    for expert in range(EXPERTS_PER_RANK):
        d_h_blocks = d_h_blocks.at[:, expert].set(
            source_push_mlp._source_push_mlp_h_flat_for_expert(route_table, expert_base, d_h_flat, expert)
        )
    expected = source_push_w13_backward_expert_blocks_reference(
        x,
        d_h_blocks,
        w13,
        host_inputs.plan,
        jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        expert_base,
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
    )

    observed = source_push_w13_backward_expert_blocks_source_padded_dw13_only_xla(
        x,
        d_h_blocks,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.valid_by_expert,
        host_inputs.src_base_by_expert,
        block_sizes=SourcePushW13BackwardTiledBlockSizes(row_block=2, hidden_block=2, output_block=3),
    )

    assert observed.x_expert_major.shape == (0,)
    assert observed.dx_expert_major.shape == (0,)
    np.testing.assert_allclose(np.asarray(observed.dw13), np.asarray(expected.dw13), atol=1e-6, rtol=1e-6)


def test_source_push_w13_backward_source_padded_dw13_pallas_interpreter_matches_xla_reference():
    config, host_inputs, route_table, x, _w13 = _small_source_push_w13_case(block_m=2, use_exact_expert_major=False)
    expert_base = jnp.asarray(host_inputs.expert_base, dtype=jnp.int32)
    d_h_flat = _dirty_dh(config, host_inputs)
    d_h_blocks = jnp.zeros(
        (EP_SIZE, EXPERTS_PER_RANK, route_table.expert_capacity, 2 * INTERMEDIATE_DIM),
        dtype=d_h_flat.dtype,
    )
    for expert in range(EXPERTS_PER_RANK):
        d_h_blocks = d_h_blocks.at[:, expert].set(
            source_push_mlp._source_push_mlp_h_flat_for_expert(route_table, expert_base, d_h_flat, expert)
        )
    block_sizes = SourcePushW13BackwardTiledBlockSizes(row_block=2, hidden_block=2, output_block=3)

    expected = source_push_w13_backward_expert_blocks_source_padded_dw13_only_xla(
        x,
        d_h_blocks,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.valid_by_expert,
        host_inputs.src_base_by_expert,
        block_sizes=block_sizes,
    )
    partials = _source_push_w13_dw13_source_padded_partials_pallas_mgpu(
        x,
        d_h_blocks,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.valid_by_expert,
        host_inputs.src_base_by_expert,
        block_sizes=block_sizes,
        interpret=True,
    )
    observed_dw13 = jnp.sum(partials, axis=0)

    assert partials.shape == (EP_SIZE, EP_SIZE, EXPERTS_PER_RANK, HIDDEN_DIM, 2 * INTERMEDIATE_DIM)
    np.testing.assert_allclose(np.asarray(observed_dw13), np.asarray(expected.dw13), atol=1e-6, rtol=1e-6)


def test_source_push_w13_backward_dx_only_pads_unaligned_capacity():
    d_h = jnp.ones((2, 3, 5, 4), dtype=jnp.bfloat16)
    valid = jnp.ones((2, 3, 5), dtype=jnp.bool_)

    padded_d_h, padded_valid = _pad_w13_compact_dh_for_row_block(d_h, valid, row_block=4)

    assert padded_d_h.shape == (2, 3, 8, 4)
    assert padded_valid.shape == (2, 3, 8)
    np.testing.assert_array_equal(np.asarray(padded_d_h[:, :, :5, :]), np.asarray(d_h))
    np.testing.assert_array_equal(np.asarray(padded_valid[:, :, :5]), np.asarray(valid))
    np.testing.assert_array_equal(np.asarray(padded_valid[:, :, 5:]), np.zeros((2, 3, 3), dtype=bool))


def test_source_push_w13_backward_compact_pallas_interpreter_matches_compact_reference():
    config, host_inputs, route_table, x, w13 = _small_source_push_w13_case(block_m=1, use_exact_expert_major=True)
    expert_base = jnp.asarray(host_inputs.expert_base, dtype=jnp.int32)
    d_h_flat = _dirty_dh(config, host_inputs)
    d_h_blocks = jnp.zeros(
        (EP_SIZE, EXPERTS_PER_RANK, route_table.expert_capacity, 2 * INTERMEDIATE_DIM),
        dtype=d_h_flat.dtype,
    )
    for expert in range(EXPERTS_PER_RANK):
        d_h_blocks = d_h_blocks.at[:, expert].set(
            source_push_mlp._source_push_mlp_h_flat_for_expert(route_table, expert_base, d_h_flat, expert)
        )
    block_sizes = SourcePushW13BackwardTiledBlockSizes(row_block=2, hidden_block=2, output_block=3)

    expected = source_push_w13_backward_expert_blocks_reference(
        x,
        d_h_blocks,
        w13,
        host_inputs.plan,
        jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        expert_base,
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
        use_exact_expert_major=True,
    )
    observed = _source_push_w13_backward_expert_blocks_pallas_mgpu(
        x,
        d_h_blocks,
        w13,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.valid_by_expert,
        block_sizes=block_sizes,
        interpret=True,
        return_x_expert_major=True,
    )

    np.testing.assert_allclose(
        np.asarray(observed.x_expert_major),
        np.asarray(expected.x_expert_major),
        atol=0,
        rtol=0,
    )
    np.testing.assert_allclose(
        np.asarray(observed.dx_expert_major),
        np.asarray(expected.dx_expert_major),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(observed.dw13), np.asarray(expected.dw13), atol=1e-6, rtol=1e-6)


def test_source_push_w13_backward_compact_pallas_default_avoids_x_output():
    config, host_inputs, route_table, x, w13 = _small_source_push_w13_case(block_m=1, use_exact_expert_major=True)
    expert_base = jnp.asarray(host_inputs.expert_base, dtype=jnp.int32)
    d_h_flat = _dirty_dh(config, host_inputs)
    d_h_blocks = jnp.zeros(
        (EP_SIZE, EXPERTS_PER_RANK, route_table.expert_capacity, 2 * INTERMEDIATE_DIM),
        dtype=d_h_flat.dtype,
    )
    for expert in range(EXPERTS_PER_RANK):
        d_h_blocks = d_h_blocks.at[:, expert].set(
            source_push_mlp._source_push_mlp_h_flat_for_expert(route_table, expert_base, d_h_flat, expert)
        )
    block_sizes = SourcePushW13BackwardTiledBlockSizes(row_block=2, hidden_block=2, output_block=3)

    expected = source_push_w13_backward_expert_blocks_reference(
        x,
        d_h_blocks,
        w13,
        host_inputs.plan,
        jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        expert_base,
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
        use_exact_expert_major=True,
    )
    observed = _source_push_w13_backward_expert_blocks_pallas_mgpu(
        x,
        d_h_blocks,
        w13,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.valid_by_expert,
        block_sizes=block_sizes,
        interpret=True,
    )

    assert observed.x_expert_major.shape == (0,)
    np.testing.assert_allclose(
        np.asarray(observed.dx_expert_major),
        np.asarray(expected.dx_expert_major),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(observed.dw13), np.asarray(expected.dw13), atol=1e-6, rtol=1e-6)


def test_source_push_w13_backward_compact_dx_source_gather_dw13_matches_reference_without_x_materialization():
    config, host_inputs, route_table, x, w13 = _small_source_push_w13_case(block_m=1, use_exact_expert_major=True)
    expert_base = jnp.asarray(host_inputs.expert_base, dtype=jnp.int32)
    d_h_flat = _dirty_dh(config, host_inputs)
    d_h_blocks = jnp.zeros(
        (EP_SIZE, EXPERTS_PER_RANK, route_table.expert_capacity, 2 * INTERMEDIATE_DIM),
        dtype=d_h_flat.dtype,
    )
    for expert in range(EXPERTS_PER_RANK):
        d_h_blocks = d_h_blocks.at[:, expert].set(
            source_push_mlp._source_push_mlp_h_flat_for_expert(route_table, expert_base, d_h_flat, expert)
        )
    block_sizes = SourcePushW13BackwardTiledBlockSizes(row_block=2, hidden_block=2, output_block=3)

    expected = source_push_w13_backward_expert_blocks_reference(
        x,
        d_h_blocks,
        w13,
        host_inputs.plan,
        jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        expert_base,
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
        use_exact_expert_major=True,
    )
    observed = _source_push_w13_backward_expert_blocks_compact_dx_source_gather_dw13(
        x,
        d_h_blocks,
        w13,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        route_table.valid_by_expert,
        block_sizes=block_sizes,
        interpret=True,
    )

    assert observed.x_expert_major.shape == (0,)
    np.testing.assert_allclose(
        np.asarray(observed.dx_expert_major),
        np.asarray(expected.dx_expert_major),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(observed.dw13), np.asarray(expected.dw13), atol=1e-6, rtol=1e-6)


def test_source_push_w13_backward_compact_pallas_requires_gpu_lowering_on_cpu():
    config, host_inputs, route_table, x, w13 = _small_source_push_w13_case(block_m=1, use_exact_expert_major=True)
    d_h_blocks = jnp.zeros(
        (EP_SIZE, EXPERTS_PER_RANK, route_table.expert_capacity, 2 * INTERMEDIATE_DIM),
        dtype=jnp.float32,
    )

    with pytest.raises(NotImplementedError, match="requires a GPU backend"):
        _source_push_w13_backward_expert_blocks_pallas_mgpu(
            x,
            d_h_blocks,
            w13,
            route_table.source_rank_by_expert,
            route_table.token_id_by_expert,
            route_table.valid_by_expert,
            block_sizes=SourcePushW13BackwardTiledBlockSizes(row_block=2, hidden_block=2, output_block=3),
        )


def test_source_push_w13_backward_masks_invalid_flat_rows():
    config, host_inputs, _route_table, x, w13 = _small_source_push_w13_case()
    d_h_dirty = _dirty_dh(config, host_inputs)
    live_mask = _flat_live_row_mask(config, host_inputs)
    d_h_clean = jnp.where(jnp.asarray(live_mask)[..., None], d_h_dirty, jnp.zeros((), dtype=d_h_dirty.dtype))

    dirty = source_push_w13_backward(
        x,
        d_h_dirty,
        w13,
        host_inputs.plan,
        jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        jnp.asarray(host_inputs.expert_base, dtype=jnp.int32),
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
    )
    clean = source_push_w13_backward_reference(
        x,
        d_h_clean,
        w13,
        host_inputs.plan,
        jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        jnp.asarray(host_inputs.expert_base, dtype=jnp.int32),
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
    )

    np.testing.assert_allclose(np.asarray(dirty.dx_expert_major), np.asarray(clean.dx_expert_major), atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(dirty.dw13), np.asarray(clean.dw13), atol=0, rtol=0)
    np.testing.assert_array_equal(
        np.asarray(dirty.x_expert_major)[~live_mask],
        np.zeros((np.size(live_mask) - int(np.sum(live_mask)), HIDDEN_DIM), dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("block_m", "use_exact_expert_major"),
    [
        (BLOCK_M, False),
        (1, True),
    ],
)
def test_source_push_w13_backward_tiled_matches_reference_without_x_materialization(
    block_m: int,
    use_exact_expert_major: bool,
):
    config, host_inputs, _route_table, x, w13 = _small_source_push_w13_case(
        block_m=block_m,
        use_exact_expert_major=use_exact_expert_major,
    )
    d_h = _dirty_dh(config, host_inputs)
    block_sizes = SourcePushW13BackwardTiledBlockSizes(row_block=1, hidden_block=2, output_block=3)

    expected = source_push_w13_backward_reference(
        x,
        d_h,
        w13,
        host_inputs.plan,
        jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        jnp.asarray(host_inputs.expert_base, dtype=jnp.int32),
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
        use_exact_expert_major=use_exact_expert_major,
    )
    observed = source_push_w13_backward(
        x,
        d_h,
        w13,
        host_inputs.plan,
        jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        jnp.asarray(host_inputs.expert_base, dtype=jnp.int32),
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
        use_exact_expert_major=use_exact_expert_major,
        implementation=SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_TILED,
        tiled_block_sizes=block_sizes,
    )

    assert observed.x_expert_major.shape == (0,)
    np.testing.assert_allclose(
        np.asarray(observed.dx_expert_major),
        np.asarray(expected.dx_expert_major),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(observed.dw13), np.asarray(expected.dw13), atol=1e-6, rtol=1e-6)


def test_source_push_w13_backward_tiled_can_return_x_for_debug_parity():
    config, host_inputs, _route_table, x, w13 = _small_source_push_w13_case()
    d_h = _dirty_dh(config, host_inputs)
    block_sizes = SourcePushW13BackwardTiledBlockSizes(row_block=1, hidden_block=2, output_block=3)

    expected = source_push_w13_backward_reference(
        x,
        d_h,
        w13,
        host_inputs.plan,
        jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        jnp.asarray(host_inputs.expert_base, dtype=jnp.int32),
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
    )
    observed = source_push_w13_backward(
        x,
        d_h,
        w13,
        host_inputs.plan,
        jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        jnp.asarray(host_inputs.expert_base, dtype=jnp.int32),
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
        implementation=SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_TILED,
        tiled_block_sizes=block_sizes,
        return_x_expert_major=True,
    )

    np.testing.assert_allclose(
        np.asarray(observed.x_expert_major), np.asarray(expected.x_expert_major), atol=0, rtol=0
    )


def test_source_push_w13_backward_bf16_inputs_match_float32_reference_with_tolerance():
    config, host_inputs, _route_table, x, w13 = _small_source_push_w13_case()
    live_mask = _flat_live_row_mask(config, host_inputs)
    d_h = jnp.where(
        jnp.asarray(live_mask)[..., None],
        _dirty_dh(config, host_inputs),
        jnp.zeros((config.ep_size, config.hidden_rows_per_rank, 2 * config.intermediate_dim), dtype=jnp.float32),
    )

    float32_output = source_push_w13_backward_reference(
        x,
        d_h,
        w13,
        host_inputs.plan,
        jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        jnp.asarray(host_inputs.expert_base, dtype=jnp.int32),
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
    )
    bf16_output = source_push_w13_backward_reference(
        x.astype(jnp.bfloat16),
        d_h.astype(jnp.bfloat16),
        w13.astype(jnp.bfloat16),
        host_inputs.plan,
        jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        jnp.asarray(host_inputs.expert_base, dtype=jnp.int32),
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
    )

    np.testing.assert_allclose(
        np.asarray(bf16_output.dx_expert_major),
        np.asarray(float32_output.dx_expert_major),
        atol=2e-2,
        rtol=2e-2,
    )
    np.testing.assert_allclose(
        np.asarray(bf16_output.dw13),
        np.asarray(float32_output.dw13),
        atol=2e-2,
        rtol=2e-2,
    )


@pytest.mark.parametrize(
    ("block_m", "use_exact_expert_major"),
    [
        (BLOCK_M, False),
        (1, True),
    ],
)
def test_source_push_w13_backward_pallas_interpreter_matches_reference(
    block_m: int,
    use_exact_expert_major: bool,
):
    config, host_inputs, _route_table, x, w13 = _small_source_push_w13_case(
        block_m=block_m,
        use_exact_expert_major=use_exact_expert_major,
    )
    d_h = _dirty_dh(config, host_inputs)

    expected = source_push_w13_backward_reference(
        x,
        d_h,
        w13,
        host_inputs.plan,
        jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        jnp.asarray(host_inputs.expert_base, dtype=jnp.int32),
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
        use_exact_expert_major=use_exact_expert_major,
    )
    observed = _source_push_w13_backward_pallas_mgpu(
        x,
        d_h,
        w13,
        host_inputs.plan,
        jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        jnp.asarray(host_inputs.expert_base, dtype=jnp.int32),
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
        use_exact_expert_major=use_exact_expert_major,
        block_sizes=SourcePushXToW13RowsPallasBlockSizes(row_block=1, hidden_block=1),
        interpret=True,
    )

    np.testing.assert_allclose(
        np.asarray(observed.x_expert_major), np.asarray(expected.x_expert_major), atol=0, rtol=0
    )
    np.testing.assert_allclose(
        np.asarray(observed.dx_expert_major),
        np.asarray(expected.dx_expert_major),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(observed.dw13), np.asarray(expected.dw13), atol=1e-6, rtol=1e-6)


def test_source_push_w13_backward_pallas_selector_requires_gpu_lowering_on_cpu():
    config, host_inputs, _route_table, x, w13 = _small_source_push_w13_case()

    with pytest.raises(NotImplementedError, match="requires a GPU backend"):
        source_push_w13_backward(
            x,
            _dirty_dh(config, host_inputs),
            w13,
            host_inputs.plan,
            jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
            jnp.asarray(host_inputs.expert_base, dtype=jnp.int32),
            jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
            implementation=SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_PALLAS_MGPU,
        )


def test_source_push_mlp_w13_backward_selector_keeps_gpu_default_on_reference(monkeypatch):
    assert (
        source_push_mlp._source_push_mlp_backward_w13_implementation(
            source_push_mlp.SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU
        )
        == SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_REFERENCE
    )

    monkeypatch.setattr(source_push_mlp.jax, "default_backend", lambda: "gpu")

    assert (
        source_push_mlp._source_push_mlp_backward_w13_implementation(
            source_push_mlp.SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU
        )
        == SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_REFERENCE
    )


def test_source_push_x_to_w13_rows_reference_uses_forward_flat_row_layout():
    config, host_inputs, route_table, x, _w13 = _small_source_push_w13_case()
    observed = source_push_x_to_w13_rows_reference(
        x,
        host_inputs.plan,
        jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        jnp.asarray(host_inputs.expert_base, dtype=jnp.int32),
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
        hidden_rows_per_rank=config.hidden_rows_per_rank,
    )

    for route in range(route_table.source_rank.shape[0]):
        src = int(route_table.source_rank[route])
        token = int(route_table.token_id[route])
        dst = int(route_table.destination_rank[route])
        expert = int(route_table.local_expert[route])
        expert_row = int(route_table.expert_row[route])
        flat_row = int(host_inputs.expert_base[dst, expert]) + expert_row
        np.testing.assert_allclose(
            np.asarray(observed[dst, flat_row]),
            np.asarray(x[src, token], dtype=np.float32),
            atol=0,
            rtol=0,
        )


def test_source_push_x_to_w13_rows_pallas_inputs_are_gmem_refs():
    in_specs, out_spec = _source_push_x_to_w13_rows_block_specs(row_block=128, hidden_block=8)

    assert all(spec.memory_space == mgpu.GMEM for spec in in_specs)
    assert out_spec.memory_space is None
    assert out_spec.block_shape == (None, 128, 8)


@pytest.mark.parametrize(
    ("block_m", "use_exact_expert_major"),
    [
        (BLOCK_M, False),
        (1, True),
    ],
)
def test_source_push_x_to_w13_rows_pallas_interpreter_matches_reference(
    block_m: int,
    use_exact_expert_major: bool,
):
    config, host_inputs, _route_table, x, _w13 = _small_source_push_w13_case(
        block_m=block_m,
        use_exact_expert_major=use_exact_expert_major,
    )
    expected = source_push_x_to_w13_rows_reference(
        x,
        host_inputs.plan,
        jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        jnp.asarray(host_inputs.expert_base, dtype=jnp.int32),
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
        hidden_rows_per_rank=config.hidden_rows_per_rank,
        use_exact_expert_major=use_exact_expert_major,
    )

    observed = source_push_x_to_w13_rows(
        x,
        host_inputs.plan,
        jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        jnp.asarray(host_inputs.expert_base, dtype=jnp.int32),
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
        hidden_rows_per_rank=config.hidden_rows_per_rank,
        use_exact_expert_major=use_exact_expert_major,
        implementation=SOURCE_PUSH_X_TO_W13_ROWS_IMPLEMENTATION_PALLAS_MGPU,
        block_sizes=SourcePushXToW13RowsPallasBlockSizes(row_block=1, hidden_block=1),
        interpret=True,
    )
    live_mask = _flat_live_row_mask(config, host_inputs)

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=0, rtol=0)
    np.testing.assert_array_equal(
        np.asarray(observed)[~live_mask],
        np.zeros((np.size(live_mask) - int(np.sum(live_mask)), HIDDEN_DIM), dtype=np.float32),
    )


def test_source_push_w13_backward_target_cost_estimate_is_per_rank_stage_math():
    estimate = estimate_source_push_w13_backward_cost(
        useful_rows_per_rank=32768 * 4,
        padded_rows_per_rank=8 * 288 * 64,
        hidden_dim=2560,
        intermediate_dim=1280,
    )

    assert estimate.w13_backward_flops_per_rank == 3_435_973_836_800
    assert estimate.x_remat_bytes_per_rank == 671_088_640
    assert estimate.x_remat_padded_bytes_per_rank == 754_974_720
    assert estimate.math_seconds_at_reference_tflops_per_rank == pytest.approx(0.0137438953472)
