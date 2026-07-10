# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np

from levanter.grug._moe import source_push_backward_return
import levanter.grug._moe.source_push_mlp as source_push_mlp
from levanter.grug._moe.source_push_plan import (
    SourcePushPlan,
    build_source_push_plan,
    source_push_source_padded_row_bases,
)


EP_SIZE = 2
EXPERTS_PER_RANK = 2
BLOCK_M = 2
TOKENS_PER_RANK = 4
TOPK = 2
HIDDEN_DIM = 3


def _duplicate_route_plan() -> SourcePushPlan:
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
    return plan


def _expert_major_rows(plan: SourcePushPlan, *, src_base_by_expert=None, capacity: int | None = None):
    route_table = source_push_mlp.source_push_mlp_route_table_from_plan(
        plan,
        src_base_by_expert=src_base_by_expert,
    )
    if capacity is None:
        capacity = route_table.expert_capacity
    dx = (
        jnp.arange(EP_SIZE * EXPERTS_PER_RANK * capacity * HIDDEN_DIM, dtype=jnp.float32).reshape(
            EP_SIZE,
            EXPERTS_PER_RANK,
            capacity,
            HIDDEN_DIM,
        )
        * 0.25
        + 1.0
    )
    d_route = (
        jnp.arange(EP_SIZE * EXPERTS_PER_RANK * capacity, dtype=jnp.float32).reshape(
            EP_SIZE,
            EXPERTS_PER_RANK,
            capacity,
        )
        * 0.125
        - 0.5
    )
    return route_table, dx, d_route


def _existing_per_expert_return(plan: SourcePushPlan, dx_expert_major, d_route_block, *, src_base_by_expert=None):
    route_table = source_push_mlp.source_push_mlp_route_table_from_plan(
        plan,
        src_base_by_expert=src_base_by_expert,
    )
    dx = jnp.zeros((EP_SIZE, plan.tokens_per_source, plan.topk, HIDDEN_DIM), dtype=dx_expert_major.dtype)
    d_route_weights = jnp.zeros((EP_SIZE, plan.tokens_per_source, plan.topk), dtype=d_route_block.dtype)
    source_dx = jnp.zeros((EP_SIZE, plan.tokens_per_source, HIDDEN_DIM), dtype=dx_expert_major.dtype)
    for expert in range(route_table.experts_per_rank):
        route_indices = source_push_mlp._source_push_mlp_expert_route_indices(route_table, expert)
        masked_dx_block = dx_expert_major[:, expert] * route_indices.valid[..., None]
        masked_d_route_block = d_route_block[:, expert] * route_indices.valid
        source_grads = source_push_mlp._source_push_mlp_return_expert_backward_to_sources(
            source_dx,
            d_route_weights,
            route_indices,
            masked_dx_block,
            masked_d_route_block,
        )
        source_dx = source_grads.dx
        d_route_weights = source_grads.d_route_weights

        route_buffer = dx.at[route_indices.safe_src, route_indices.safe_token, route_indices.safe_slot].add(
            masked_dx_block,
        )
        dx = route_buffer
    return source_dx, d_route_weights, dx


def test_source_push_backward_return_matches_existing_per_expert_accumulation_with_duplicates():
    plan = _duplicate_route_plan()
    _route_table, dx_expert_major, d_route_block = _expert_major_rows(plan)

    observed = source_push_backward_return.source_push_backward_return_reference(
        dx_expert_major,
        d_route_block,
        plan,
    )
    expected_dx, expected_d_route_weights, _expected_dx_routes = _existing_per_expert_return(
        plan,
        dx_expert_major,
        d_route_block,
    )
    jitted = jax.jit(
        lambda dx_arg, d_route_arg: source_push_backward_return.source_push_backward_return(
            dx_arg,
            d_route_arg,
            plan,
        )
    )(dx_expert_major, d_route_block)

    np.testing.assert_allclose(np.asarray(observed.dx), np.asarray(expected_dx), atol=0, rtol=0)
    np.testing.assert_allclose(
        np.asarray(observed.d_route_weights),
        np.asarray(expected_d_route_weights),
        atol=0,
        rtol=0,
    )
    np.testing.assert_allclose(np.asarray(jitted.dx), np.asarray(observed.dx), atol=0, rtol=0)
    np.testing.assert_allclose(
        np.asarray(jitted.d_route_weights),
        np.asarray(observed.d_route_weights),
        atol=0,
        rtol=0,
    )

    route_buffer = source_push_backward_return.source_push_backward_return_route_buffer_jax(
        dx_expert_major,
        d_route_block,
        plan,
    )
    np.testing.assert_allclose(
        np.asarray(observed.dx[0, 0]),
        np.asarray(route_buffer.dx_routes[0, 0, 0] + route_buffer.dx_routes[0, 0, 1]),
        atol=0,
        rtol=0,
    )
    assert np.count_nonzero(np.asarray(route_buffer.dx_routes[0, 0])) == TOPK * HIDDEN_DIM


def test_source_push_backward_return_masks_source_padded_invalid_rows():
    plan = _duplicate_route_plan()
    rounded_counts, _expert_base, src_base_by_expert = source_push_source_padded_row_bases(plan, BLOCK_M)
    padded_capacity = int(np.max(np.sum(rounded_counts, axis=0)))
    _route_table, dx_expert_major, d_route_block = _expert_major_rows(
        plan,
        src_base_by_expert=src_base_by_expert,
        capacity=padded_capacity,
    )
    live_mask = _live_expert_row_mask(plan, src_base_by_expert, padded_capacity)
    assert np.any(~live_mask)

    clean_dx = jnp.where(jnp.asarray(live_mask[..., None]), dx_expert_major, jnp.zeros_like(dx_expert_major))
    clean_d_route = jnp.where(jnp.asarray(live_mask), d_route_block, jnp.zeros_like(d_route_block))
    poisoned_dx = jnp.where(jnp.asarray(live_mask[..., None]), clean_dx, jnp.full_like(clean_dx, 1.0e6))
    poisoned_d_route = jnp.where(jnp.asarray(live_mask), clean_d_route, jnp.full_like(clean_d_route, -1.0e6))

    expected = source_push_backward_return.source_push_backward_return_reference(
        clean_dx,
        clean_d_route,
        plan,
        src_base_by_expert=jnp.asarray(src_base_by_expert, dtype=jnp.int32),
    )
    observed = source_push_backward_return.source_push_backward_return_reference(
        poisoned_dx,
        poisoned_d_route,
        plan,
        src_base_by_expert=jnp.asarray(src_base_by_expert, dtype=jnp.int32),
    )

    np.testing.assert_allclose(np.asarray(observed.dx), np.asarray(expected.dx), atol=0, rtol=0)
    np.testing.assert_allclose(
        np.asarray(observed.d_route_weights),
        np.asarray(expected.d_route_weights),
        atol=0,
        rtol=0,
    )


def test_source_push_backward_return_route_indices_are_token_slot_inverse():
    plan = _duplicate_route_plan()
    rounded_counts, _expert_base, src_base_by_expert = source_push_source_padded_row_bases(plan, BLOCK_M)
    route_indices = source_push_backward_return.source_push_backward_return_route_indices_jax(
        plan,
        src_base_by_expert=jnp.asarray(src_base_by_expert, dtype=jnp.int32),
    )
    route_dst = np.asarray(route_indices.dst)
    route_expert = np.asarray(route_indices.expert)
    route_row = np.asarray(route_indices.row)
    route_valid = np.asarray(route_indices.valid)
    valid_mask = np.asarray(plan.valid_mask)
    token_ids = np.asarray(plan.token_ids)
    route_slots = np.asarray(plan.route_slots)
    local_experts = np.asarray(plan.local_experts)
    local_row_starts = np.asarray(plan.local_row_starts)

    assert np.all(route_valid)
    for src, dst_ord, entry, row in np.argwhere(valid_mask):
        dst = (src + dst_ord) % EP_SIZE
        token = token_ids[src, dst_ord, entry, row]
        slot = route_slots[src, dst_ord, entry, row]
        expert = local_experts[src, dst_ord, entry]
        compact_row = src_base_by_expert[dst, src, expert] + local_row_starts[src, dst_ord, entry] + row
        assert route_dst[src, token, slot] == dst
        assert route_expert[src, token, slot] == expert
        assert route_row[src, token, slot] == compact_row
    assert int(np.max(route_row)) < int(np.max(np.sum(rounded_counts, axis=0)))


def test_source_push_backward_return_accepts_precomputed_route_indices():
    plan = _duplicate_route_plan()
    _route_table, dx_expert_major, d_route_block = _expert_major_rows(plan)
    route_indices = source_push_backward_return.source_push_backward_return_route_indices_jax(plan)

    expected = source_push_backward_return.source_push_backward_return_reference(
        dx_expert_major,
        d_route_block,
        plan,
    )
    observed = source_push_backward_return.source_push_backward_return(
        dx_expert_major,
        d_route_block,
        plan,
        route_indices=route_indices,
    )

    np.testing.assert_allclose(np.asarray(observed.dx), np.asarray(expected.dx), atol=0, rtol=0)
    np.testing.assert_allclose(
        np.asarray(observed.d_route_weights),
        np.asarray(expected.d_route_weights),
        atol=0,
        rtol=0,
    )


def test_source_push_backward_return_pallas_uses_compact_route_indices(monkeypatch):
    plan = _duplicate_route_plan()
    _route_table, dx_expert_major, d_route_block = _expert_major_rows(plan)
    route_indices = source_push_backward_return.source_push_backward_return_route_indices_jax(plan)
    expected = source_push_backward_return.source_push_backward_return(
        dx_expert_major,
        d_route_block,
        plan,
        route_indices=route_indices,
    )
    calls = []

    def fail_rebuild(*args, **kwargs):
        raise AssertionError("route indices should have been supplied by the caller")

    def fake_pallas(dx, d_route, passed_route_indices, *, block_sizes=None):
        calls.append((dx.shape, d_route.shape, passed_route_indices, block_sizes))
        return source_push_backward_return._source_push_backward_return_compact_from_indices_jax(
            dx,
            d_route,
            passed_route_indices,
        )

    monkeypatch.setattr(
        source_push_backward_return,
        "source_push_backward_return_route_indices_jax",
        fail_rebuild,
    )
    monkeypatch.setattr(
        source_push_backward_return,
        "_source_push_backward_return_compact_pallas_mgpu",
        fake_pallas,
    )

    observed = source_push_backward_return.source_push_backward_return(
        dx_expert_major,
        d_route_block,
        plan,
        route_indices=route_indices,
        implementation=source_push_backward_return.SOURCE_PUSH_BACKWARD_RETURN_IMPLEMENTATION_PALLAS_MGPU,
    )

    assert len(calls) == 1
    dx_shape, d_route_shape, passed_route_indices, block_sizes = calls[0]
    assert dx_shape == dx_expert_major.shape
    assert d_route_shape == d_route_block.shape
    assert passed_route_indices is route_indices
    assert block_sizes is None
    np.testing.assert_allclose(np.asarray(observed.dx), np.asarray(expected.dx), atol=0, rtol=0)
    np.testing.assert_allclose(
        np.asarray(observed.d_route_weights),
        np.asarray(expected.d_route_weights),
        atol=0,
        rtol=0,
    )


def test_source_push_backward_return_pallas_mesh_uses_compact_direct_gather(monkeypatch):
    plan = _duplicate_route_plan()
    _route_table, dx_expert_major, d_route_block = _expert_major_rows(plan)
    expected = source_push_backward_return.source_push_backward_return_reference(dx_expert_major, d_route_block, plan)
    expected_indices = source_push_backward_return.source_push_backward_return_route_indices_jax(plan)
    calls = []
    fake_mesh = object()

    def fake_rebuild(*args, **kwargs):
        calls.append(("rebuild", args, kwargs))
        return expected_indices

    def fake_direct_gather(mesh, dx, d_route, route_indices, *, block_sizes):
        calls.append(("direct_gather", mesh, dx.shape, d_route.shape, route_indices, block_sizes))
        return source_push_backward_return._source_push_backward_return_compact_from_indices_jax(
            dx,
            d_route,
            route_indices,
        )

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        source_push_backward_return,
        "source_push_backward_return_route_indices_jax",
        fake_rebuild,
    )
    monkeypatch.setattr(
        source_push_backward_return,
        "_source_push_backward_return_compact_direct_gather_mgpu",
        fake_direct_gather,
    )

    observed = source_push_backward_return.source_push_backward_return(
        dx_expert_major,
        d_route_block,
        plan,
        implementation=source_push_backward_return.SOURCE_PUSH_BACKWARD_RETURN_IMPLEMENTATION_PALLAS_MGPU,
        block_sizes=source_push_backward_return.SourcePushBackwardReturnPallasBlockSizes(hidden_block=1),
        mesh=fake_mesh,
    )

    assert len(calls) == 2
    rebuild_call, direct_call = calls
    assert rebuild_call[0] == "rebuild"
    assert direct_call[0] == "direct_gather"
    _, mesh, dx_shape, d_route_shape, route_indices, block_sizes = direct_call
    assert mesh is fake_mesh
    assert dx_shape == dx_expert_major.shape
    assert d_route_shape == d_route_block.shape
    assert route_indices is expected_indices
    assert block_sizes.hidden_block == 1
    np.testing.assert_allclose(np.asarray(observed.dx), np.asarray(expected.dx), atol=0, rtol=0)
    np.testing.assert_allclose(
        np.asarray(observed.d_route_weights),
        np.asarray(expected.d_route_weights),
        atol=0,
        rtol=0,
    )


def test_source_push_backward_return_compact_pallas_interpret_matches_reference():
    plan = _duplicate_route_plan()
    _route_table, dx_expert_major, d_route_block = _expert_major_rows(plan)
    route_indices = source_push_backward_return.source_push_backward_return_route_indices_jax(plan)

    expected = source_push_backward_return.source_push_backward_return_reference(
        dx_expert_major,
        d_route_block,
        plan,
    )
    observed = source_push_backward_return._source_push_backward_return_compact_pallas_mgpu(
        dx_expert_major,
        d_route_block,
        route_indices,
        block_sizes=source_push_backward_return.SourcePushBackwardReturnPallasBlockSizes(hidden_block=1),
        interpret=True,
    )

    np.testing.assert_allclose(np.asarray(observed.dx), np.asarray(expected.dx), atol=0, rtol=0)
    np.testing.assert_allclose(
        np.asarray(observed.d_route_weights),
        np.asarray(expected.d_route_weights),
        atol=0,
        rtol=0,
    )


def test_source_push_backward_return_compact_pallas_uses_int32_validity(monkeypatch):
    plan = _duplicate_route_plan()
    _route_table, dx_expert_major, d_route_block = _expert_major_rows(plan)
    route_indices = source_push_backward_return.source_push_backward_return_route_indices_jax(plan)
    valid_dtypes = []

    def fake_pallas_call(
        dx, d_route, route_dst, route_expert, route_row, route_valid, *, hidden_block, interpret, mesh
    ):
        del hidden_block, interpret, mesh
        valid_dtypes.append(route_valid.dtype)
        return source_push_backward_return._source_push_backward_return_compact_from_indices_reference(
            dx,
            d_route,
            route_dst,
            route_expert,
            route_row,
            route_valid,
        )

    monkeypatch.setattr(
        source_push_backward_return,
        "_source_push_backward_return_compact_pallas_call",
        fake_pallas_call,
    )

    observed = source_push_backward_return._source_push_backward_return_compact_pallas_mgpu(
        dx_expert_major,
        d_route_block,
        route_indices,
        block_sizes=source_push_backward_return.SourcePushBackwardReturnPallasBlockSizes(hidden_block=1),
        interpret=True,
    )
    expected = source_push_backward_return.source_push_backward_return_reference(
        dx_expert_major,
        d_route_block,
        plan,
    )

    assert valid_dtypes == [jnp.dtype(jnp.int32)]
    np.testing.assert_allclose(np.asarray(observed.dx), np.asarray(expected.dx), atol=0, rtol=0)
    np.testing.assert_allclose(
        np.asarray(observed.d_route_weights),
        np.asarray(expected.d_route_weights),
        atol=0,
        rtol=0,
    )


def test_source_push_backward_return_flat_matches_source_padded_compact_layout():
    plan = _duplicate_route_plan()
    compact_dx, compact_d_route, flat_dx, flat_d_route, expert_base, src_base_by_expert = _source_padded_flat_rows(
        plan,
    )

    compact = source_push_backward_return.source_push_backward_return_reference(
        compact_dx,
        compact_d_route,
        plan,
        src_base_by_expert=jnp.asarray(src_base_by_expert, dtype=jnp.int32),
    )
    flat = source_push_backward_return.source_push_backward_return_flat_reference(
        jnp.asarray(flat_dx),
        jnp.asarray(flat_d_route),
        plan,
        expert_base=jnp.asarray(expert_base, dtype=jnp.int32),
        src_base_by_expert=jnp.asarray(src_base_by_expert, dtype=jnp.int32),
    )

    np.testing.assert_allclose(np.asarray(flat.dx), np.asarray(compact.dx), atol=0, rtol=0)
    np.testing.assert_allclose(
        np.asarray(flat.d_route_weights),
        np.asarray(compact.d_route_weights),
        atol=0,
        rtol=0,
    )


def test_source_push_backward_return_flat_route_indices_are_token_slot_inverse():
    plan = _duplicate_route_plan()
    _compact_dx, _compact_d_route, _flat_dx, _flat_d_route, expert_base, src_base_by_expert = _source_padded_flat_rows(
        plan,
    )
    route_indices = source_push_backward_return.source_push_backward_return_flat_route_indices_jax(
        plan,
        expert_base=jnp.asarray(expert_base, dtype=jnp.int32),
        src_base_by_expert=jnp.asarray(src_base_by_expert, dtype=jnp.int32),
    )
    route_dst = np.asarray(route_indices.dst)
    route_row = np.asarray(route_indices.row)
    route_valid = np.asarray(route_indices.valid)
    valid_mask = np.asarray(plan.valid_mask)
    token_ids = np.asarray(plan.token_ids)
    route_slots = np.asarray(plan.route_slots)
    local_experts = np.asarray(plan.local_experts)
    local_row_starts = np.asarray(plan.local_row_starts)

    assert np.all(route_valid)
    for src, dst_ord, entry, row in np.argwhere(valid_mask):
        dst = (src + dst_ord) % EP_SIZE
        token = token_ids[src, dst_ord, entry, row]
        slot = route_slots[src, dst_ord, entry, row]
        expert = local_experts[src, dst_ord, entry]
        flat_row = (
            expert_base[dst, expert]
            + src_base_by_expert[dst, src, expert]
            + local_row_starts[src, dst_ord, entry]
            + row
        )
        assert route_dst[src, token, slot] == dst
        assert route_row[src, token, slot] == flat_row


def test_source_push_backward_return_flat_accepts_precomputed_route_indices():
    plan = _duplicate_route_plan()
    _compact_dx, _compact_d_route, flat_dx, flat_d_route, expert_base, src_base_by_expert = _source_padded_flat_rows(
        plan,
    )
    route_indices = source_push_backward_return.source_push_backward_return_flat_route_indices_jax(
        plan,
        expert_base=jnp.asarray(expert_base, dtype=jnp.int32),
        src_base_by_expert=jnp.asarray(src_base_by_expert, dtype=jnp.int32),
    )

    expected = source_push_backward_return.source_push_backward_return_flat_reference(
        jnp.asarray(flat_dx),
        jnp.asarray(flat_d_route),
        plan,
        expert_base=jnp.asarray(expert_base, dtype=jnp.int32),
        src_base_by_expert=jnp.asarray(src_base_by_expert, dtype=jnp.int32),
    )
    observed = source_push_backward_return.source_push_backward_return_flat(
        jnp.asarray(flat_dx),
        jnp.asarray(flat_d_route),
        plan,
        expert_base=jnp.asarray(expert_base, dtype=jnp.int32),
        src_base_by_expert=jnp.asarray(src_base_by_expert, dtype=jnp.int32),
        route_indices=route_indices,
    )

    np.testing.assert_allclose(np.asarray(observed.dx), np.asarray(expected.dx), atol=0, rtol=0)
    np.testing.assert_allclose(
        np.asarray(observed.d_route_weights),
        np.asarray(expected.d_route_weights),
        atol=0,
        rtol=0,
    )


def test_source_push_backward_return_flat_pallas_uses_precomputed_route_indices(monkeypatch):
    plan = _duplicate_route_plan()
    _compact_dx, _compact_d_route, flat_dx, flat_d_route, expert_base, src_base_by_expert = _source_padded_flat_rows(
        plan,
    )
    route_indices = source_push_backward_return.source_push_backward_return_flat_route_indices_jax(
        plan,
        expert_base=jnp.asarray(expert_base, dtype=jnp.int32),
        src_base_by_expert=jnp.asarray(src_base_by_expert, dtype=jnp.int32),
    )
    expected = source_push_backward_return.source_push_backward_return_flat_reference(
        jnp.asarray(flat_dx),
        jnp.asarray(flat_d_route),
        plan,
        expert_base=jnp.asarray(expert_base, dtype=jnp.int32),
        src_base_by_expert=jnp.asarray(src_base_by_expert, dtype=jnp.int32),
        route_indices=route_indices,
    )
    calls = []

    def fail_rebuild(*args, **kwargs):
        raise AssertionError("route indices should have been supplied by the caller")

    def fake_pallas(dx, d_route, passed_route_indices, *, block_sizes=None):
        calls.append((passed_route_indices, block_sizes))
        return source_push_backward_return._source_push_backward_return_flat_from_indices_jax(
            dx,
            d_route,
            passed_route_indices,
        )

    monkeypatch.setattr(
        source_push_backward_return,
        "source_push_backward_return_flat_route_indices_jax",
        fail_rebuild,
    )
    monkeypatch.setattr(
        source_push_backward_return,
        "_source_push_backward_return_flat_pallas_mgpu",
        fake_pallas,
    )

    observed = source_push_backward_return.source_push_backward_return_flat(
        jnp.asarray(flat_dx),
        jnp.asarray(flat_d_route),
        plan,
        expert_base=jnp.asarray(expert_base, dtype=jnp.int32),
        src_base_by_expert=jnp.asarray(src_base_by_expert, dtype=jnp.int32),
        route_indices=route_indices,
        implementation=source_push_backward_return.SOURCE_PUSH_BACKWARD_RETURN_IMPLEMENTATION_PALLAS_MGPU,
    )

    assert len(calls) == 1
    assert calls[0][0] is route_indices
    assert calls[0][1] is None
    np.testing.assert_allclose(np.asarray(observed.dx), np.asarray(expected.dx), atol=0, rtol=0)
    np.testing.assert_allclose(
        np.asarray(observed.d_route_weights),
        np.asarray(expected.d_route_weights),
        atol=0,
        rtol=0,
    )


def test_source_push_backward_return_flat_pallas_mesh_uses_remote_route_buffer(monkeypatch):
    plan = _duplicate_route_plan()
    _compact_dx, _compact_d_route, flat_dx, flat_d_route, expert_base, src_base_by_expert = _source_padded_flat_rows(
        plan,
    )
    route_indices = source_push_backward_return.source_push_backward_return_flat_route_indices_jax(
        plan,
        expert_base=jnp.asarray(expert_base, dtype=jnp.int32),
        src_base_by_expert=jnp.asarray(src_base_by_expert, dtype=jnp.int32),
    )
    expected_buffer = source_push_backward_return.source_push_backward_return_flat_route_buffer_jax(
        jnp.asarray(flat_dx),
        jnp.asarray(flat_d_route),
        plan,
        expert_base=jnp.asarray(expert_base, dtype=jnp.int32),
        src_base_by_expert=jnp.asarray(src_base_by_expert, dtype=jnp.int32),
    )
    calls = []
    fake_mesh = object()

    def fake_remote_route_buffer(mesh, dx, d_route, route_rows, *, route_shape, block_sizes):
        calls.append((mesh, route_rows, route_shape, block_sizes, dx.shape, d_route.shape))
        return expected_buffer

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        source_push_backward_return,
        "_source_push_backward_return_flat_remote_write_route_buffer_mgpu",
        fake_remote_route_buffer,
    )

    observed = source_push_backward_return.source_push_backward_return_flat(
        jnp.asarray(flat_dx),
        jnp.asarray(flat_d_route),
        plan,
        expert_base=jnp.asarray(expert_base, dtype=jnp.int32),
        src_base_by_expert=jnp.asarray(src_base_by_expert, dtype=jnp.int32),
        route_indices=route_indices,
        implementation=source_push_backward_return.SOURCE_PUSH_BACKWARD_RETURN_IMPLEMENTATION_PALLAS_MGPU,
        block_sizes=source_push_backward_return.SourcePushBackwardReturnPallasBlockSizes(hidden_block=1),
        mesh=fake_mesh,
    )

    assert len(calls) == 1
    mesh, route_rows, route_shape, block_sizes, dx_shape, d_route_shape = calls[0]
    assert mesh is fake_mesh
    assert route_rows.row.shape == plan.valid_mask.shape
    assert route_shape == route_indices.valid.shape
    assert block_sizes.hidden_block == 1
    assert dx_shape == flat_dx.shape
    assert d_route_shape == flat_d_route.shape
    np.testing.assert_allclose(np.asarray(observed.dx), np.asarray(jnp.sum(expected_buffer.dx_routes, axis=2)))
    np.testing.assert_allclose(np.asarray(observed.d_route_weights), np.asarray(expected_buffer.d_route_weights))


def test_source_push_backward_return_flat_pallas_mesh_skips_route_index_rebuild(monkeypatch):
    plan = _duplicate_route_plan()
    _compact_dx, _compact_d_route, flat_dx, flat_d_route, expert_base, src_base_by_expert = _source_padded_flat_rows(
        plan,
    )
    expected_buffer = source_push_backward_return.source_push_backward_return_flat_route_buffer_jax(
        jnp.asarray(flat_dx),
        jnp.asarray(flat_d_route),
        plan,
        expert_base=jnp.asarray(expert_base, dtype=jnp.int32),
        src_base_by_expert=jnp.asarray(src_base_by_expert, dtype=jnp.int32),
    )
    calls = []
    fake_mesh = object()

    def fail_rebuild(*args, **kwargs):
        raise AssertionError("mesh return should use queue rows directly")

    def fake_remote_route_buffer(mesh, dx, d_route, route_rows, *, route_shape, block_sizes):
        calls.append((mesh, route_rows, route_shape, block_sizes, dx.shape, d_route.shape))
        return expected_buffer

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        source_push_backward_return,
        "source_push_backward_return_flat_route_indices_jax",
        fail_rebuild,
    )
    monkeypatch.setattr(
        source_push_backward_return,
        "_source_push_backward_return_flat_remote_write_route_buffer_mgpu",
        fake_remote_route_buffer,
    )

    observed = source_push_backward_return.source_push_backward_return_flat(
        jnp.asarray(flat_dx),
        jnp.asarray(flat_d_route),
        plan,
        expert_base=jnp.asarray(expert_base, dtype=jnp.int32),
        src_base_by_expert=jnp.asarray(src_base_by_expert, dtype=jnp.int32),
        implementation=source_push_backward_return.SOURCE_PUSH_BACKWARD_RETURN_IMPLEMENTATION_PALLAS_MGPU,
        block_sizes=source_push_backward_return.SourcePushBackwardReturnPallasBlockSizes(hidden_block=1),
        mesh=fake_mesh,
    )

    assert len(calls) == 1
    mesh, route_rows, route_shape, block_sizes, dx_shape, d_route_shape = calls[0]
    assert mesh is fake_mesh
    assert route_rows.row.shape == plan.valid_mask.shape
    assert route_shape == (EP_SIZE, TOKENS_PER_RANK, TOPK)
    assert block_sizes.hidden_block == 1
    assert dx_shape == flat_dx.shape
    assert d_route_shape == flat_d_route.shape
    np.testing.assert_allclose(np.asarray(observed.dx), np.asarray(jnp.sum(expected_buffer.dx_routes, axis=2)))
    np.testing.assert_allclose(np.asarray(observed.d_route_weights), np.asarray(expected_buffer.d_route_weights))


def test_source_push_backward_return_flat_pallas_interpret_matches_reference():
    plan = _duplicate_route_plan()
    _compact_dx, _compact_d_route, flat_dx, flat_d_route, expert_base, src_base_by_expert = _source_padded_flat_rows(
        plan,
    )
    route_indices = source_push_backward_return.source_push_backward_return_flat_route_indices_jax(
        plan,
        expert_base=jnp.asarray(expert_base, dtype=jnp.int32),
        src_base_by_expert=jnp.asarray(src_base_by_expert, dtype=jnp.int32),
    )

    expected = source_push_backward_return.source_push_backward_return_flat_reference(
        jnp.asarray(flat_dx),
        jnp.asarray(flat_d_route),
        plan,
        expert_base=jnp.asarray(expert_base, dtype=jnp.int32),
        src_base_by_expert=jnp.asarray(src_base_by_expert, dtype=jnp.int32),
    )
    observed = source_push_backward_return._source_push_backward_return_flat_pallas_mgpu(
        jnp.asarray(flat_dx),
        jnp.asarray(flat_d_route),
        route_indices,
        block_sizes=source_push_backward_return.SourcePushBackwardReturnPallasBlockSizes(hidden_block=1),
        interpret=True,
    )

    np.testing.assert_allclose(np.asarray(observed.dx), np.asarray(expected.dx), atol=0, rtol=0)
    np.testing.assert_allclose(
        np.asarray(observed.d_route_weights),
        np.asarray(expected.d_route_weights),
        atol=0,
        rtol=0,
    )


def test_source_push_backward_return_flat_pallas_uses_int32_validity(monkeypatch):
    plan = _duplicate_route_plan()
    _compact_dx, _compact_d_route, flat_dx, flat_d_route, expert_base, src_base_by_expert = _source_padded_flat_rows(
        plan,
    )
    route_indices = source_push_backward_return.source_push_backward_return_flat_route_indices_jax(
        plan,
        expert_base=jnp.asarray(expert_base, dtype=jnp.int32),
        src_base_by_expert=jnp.asarray(src_base_by_expert, dtype=jnp.int32),
    )
    valid_dtypes = []

    def fake_pallas_call(dx, d_route, route_dst, route_row, route_valid, *, hidden_block, interpret, mesh):
        del hidden_block, interpret, mesh
        valid_dtypes.append(route_valid.dtype)
        return source_push_backward_return._source_push_backward_return_flat_from_indices_reference(
            dx,
            d_route,
            route_dst,
            route_row,
            route_valid,
        )

    monkeypatch.setattr(
        source_push_backward_return,
        "_source_push_backward_return_flat_pallas_call",
        fake_pallas_call,
    )

    observed = source_push_backward_return._source_push_backward_return_flat_pallas_mgpu(
        jnp.asarray(flat_dx),
        jnp.asarray(flat_d_route),
        route_indices,
        block_sizes=source_push_backward_return.SourcePushBackwardReturnPallasBlockSizes(hidden_block=1),
        interpret=True,
    )
    expected = source_push_backward_return.source_push_backward_return_flat_reference(
        jnp.asarray(flat_dx),
        jnp.asarray(flat_d_route),
        plan,
        expert_base=jnp.asarray(expert_base, dtype=jnp.int32),
        src_base_by_expert=jnp.asarray(src_base_by_expert, dtype=jnp.int32),
    )

    assert valid_dtypes == [jnp.dtype(jnp.int32)]
    np.testing.assert_allclose(np.asarray(observed.dx), np.asarray(expected.dx), atol=0, rtol=0)
    np.testing.assert_allclose(
        np.asarray(observed.d_route_weights),
        np.asarray(expected.d_route_weights),
        atol=0,
        rtol=0,
    )


def test_source_push_backward_return_dropped_routes_remain_zero():
    selected_experts = jnp.zeros((1, 5, 2), dtype=jnp.int32)
    route_weights = jnp.ones_like(selected_experts, dtype=jnp.float32)
    plan = build_source_push_plan(
        selected_experts,
        route_weights,
        ep_size=1,
        experts_per_rank=1,
        block_m=2,
        capacity_factor=0.2,
    )
    assert int(plan.dropped_routes) > 0
    route_table = source_push_mlp.source_push_mlp_route_table_from_plan(plan)
    dx_expert_major = jnp.arange(route_table.expert_capacity * HIDDEN_DIM, dtype=jnp.float32).reshape(
        1,
        1,
        route_table.expert_capacity,
        HIDDEN_DIM,
    )
    d_route_block = jnp.arange(route_table.expert_capacity, dtype=jnp.float32).reshape(
        1,
        1,
        route_table.expert_capacity,
    )

    observed = source_push_backward_return.source_push_backward_return_reference(
        dx_expert_major,
        d_route_block,
        plan,
    )
    accepted = np.zeros(selected_experts.shape, dtype=np.bool_)
    for route in range(route_table.source_rank.shape[0]):
        accepted[
            int(route_table.source_rank[route]),
            int(route_table.token_id[route]),
            int(route_table.route_slot[route]),
        ] = True

    d_route_weights = np.asarray(observed.d_route_weights)
    np.testing.assert_array_equal(d_route_weights[~accepted], np.zeros_like(d_route_weights[~accepted]))
    assert np.any(d_route_weights[accepted] != 0)


def _source_padded_flat_rows(plan: SourcePushPlan):
    rounded_counts, expert_base, src_base_by_expert = source_push_source_padded_row_bases(plan, BLOCK_M)
    rows_per_expert = np.sum(rounded_counts, axis=0)
    compact_capacity = int(np.max(rows_per_expert))
    hidden_rows = int(np.max(expert_base + rows_per_expert))
    _route_table, compact_dx, compact_d_route = _expert_major_rows(
        plan,
        src_base_by_expert=src_base_by_expert,
        capacity=compact_capacity,
    )
    flat_dx = np.zeros((EP_SIZE, hidden_rows, HIDDEN_DIM), dtype=np.float32)
    flat_d_route = np.zeros((EP_SIZE, hidden_rows), dtype=np.float32)
    compact_dx_host = np.asarray(compact_dx)
    compact_d_route_host = np.asarray(compact_d_route)
    for dst in range(EP_SIZE):
        for expert in range(EXPERTS_PER_RANK):
            row_count = rows_per_expert[dst, expert]
            start = expert_base[dst, expert]
            flat_dx[dst, start : start + row_count] = compact_dx_host[dst, expert, :row_count]
            flat_d_route[dst, start : start + row_count] = compact_d_route_host[dst, expert, :row_count]
    return compact_dx, compact_d_route, flat_dx, flat_d_route, expert_base, src_base_by_expert


def _live_expert_row_mask(plan: SourcePushPlan, src_base_by_expert: np.ndarray, capacity: int) -> np.ndarray:
    live = np.zeros((EP_SIZE, EXPERTS_PER_RANK, capacity), dtype=np.bool_)
    valid_mask = np.asarray(plan.valid_mask)
    local_experts = np.asarray(plan.local_experts)
    local_row_starts = np.asarray(plan.local_row_starts)
    for src, dst_ord, entry, row in np.argwhere(valid_mask):
        dst = (src + dst_ord) % EP_SIZE
        expert = local_experts[src, dst_ord, entry]
        expert_row = src_base_by_expert[dst, src, expert] + local_row_starts[src, dst_ord, entry] + row
        live[dst, expert, expert_row] = True
    return live
