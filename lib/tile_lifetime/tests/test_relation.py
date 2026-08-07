# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from tile_lifetime import RelationPlanError, build_relation_plan


def _relation_plan():
    destination_indices = np.array(
        [
            [0, 1],
            [3, 2],
            [1, 3],
            [2, 0],
        ],
        dtype=np.int32,
    )
    weights = np.array(
        [
            [0.25, 0.75],
            [0.60, 0.40],
            [0.30, 0.70],
            [0.55, 0.45],
        ],
        dtype=np.float32,
    )
    plan = build_relation_plan(
        destination_indices,
        weights,
        destination_rank_by_item=np.array([0, 0, 1, 1], dtype=np.int32),
        destination_local_item_by_item=np.array([0, 1, 0, 1], dtype=np.int32),
        padding_quantum=2,
    )
    return plan, destination_indices, weights


def test_relation_plan_builds_stable_padded_destination_groups() -> None:
    plan, destination_indices, weights = _relation_plan()

    assert plan.group_destination_rank.tolist() == [0, 0, 1, 1]
    assert plan.group_destination_local_item.tolist() == [0, 1, 0, 1]
    assert plan.group_count.tolist() == [2, 2, 2, 2]
    assert plan.group_padded_count.tolist() == [2, 2, 2, 2]
    assert plan.group_offset.tolist() == [0, 2, 4, 6]
    assert not np.any(plan.row_padding)
    for source_item in range(destination_indices.shape[0]):
        for route_slot in range(destination_indices.shape[1]):
            flat_route = source_item * destination_indices.shape[1] + route_slot
            destination_row = plan.route_to_destination_row[flat_route]
            assert plan.row_source_item[destination_row] == source_item
            assert plan.row_route_slot[destination_row] == route_slot
            assert plan.row_destination_item[destination_row] == destination_indices[source_item, route_slot]
            assert plan.row_weight[destination_row] == weights[source_item, route_slot]


def test_relation_plan_dispatch_inverse_and_merge_match_direct_source_order_reference() -> None:
    plan, destination_indices, weights = _relation_plan()
    payload = np.arange(12, dtype=np.float32).reshape(4, 3)
    dispatched = plan.dispatch(payload)
    destination_scale = np.where(plan.row_valid, plan.row_destination_item + 1, 0).astype(np.float32)
    destination_output = dispatched * destination_scale[:, None]

    restored = plan.inverse_dispatch(destination_output)
    expected_restored = np.empty((4, 2, 3), dtype=np.float32)
    expected_merge = np.zeros((4, 3), dtype=np.float32)
    for source_item in range(4):
        for route_slot in range(2):
            value = payload[source_item] * (destination_indices[source_item, route_slot] + 1)
            expected_restored[source_item, route_slot] = value
            expected_merge[source_item] += value * weights[source_item, route_slot]

    np.testing.assert_array_equal(restored, expected_restored)
    merged = plan.weighted_merge(destination_output)
    np.testing.assert_allclose(merged, expected_merge, rtol=1e-7, atol=0)
    np.testing.assert_array_equal(merged, plan.weighted_merge(destination_output))
    assert plan.merge_order == "source_item ascending, then route_slot ascending, FP32 accumulation"


def test_relation_plan_coalesces_payload_by_source_item_and_destination_rank() -> None:
    plan, _, _ = _relation_plan()
    payload = np.arange(12, dtype=np.float32).reshape(4, 3)

    coalesced = plan.dispatch_coalesced(payload)
    assert coalesced.shape == (6, 3)
    assert plan.exchange_source_item.tolist() == [0, 2, 3, 1, 2, 3]
    assert plan.exchange_destination_rank.tolist() == [0, 0, 0, 1, 1, 1]
    np.testing.assert_array_equal(plan.expand_coalesced(coalesced), plan.dispatch(payload))


def test_relation_plan_marks_padding_and_keeps_it_out_of_inverse_dispatch() -> None:
    plan = build_relation_plan(
        np.array([[0, 1], [0, 2], [0, 2]], dtype=np.int32),
        np.ones((3, 2), dtype=np.float32),
        destination_rank_by_item=np.array([0, 0, 1], dtype=np.int32),
        destination_local_item_by_item=np.array([0, 1, 0], dtype=np.int32),
        padding_quantum=4,
    )
    payload = np.arange(6, dtype=np.float32).reshape(3, 2)

    assert plan.group_count.tolist() == [3, 1, 2]
    assert plan.group_padded_count.tolist() == [4, 4, 4]
    assert np.count_nonzero(plan.row_padding) == 6
    dispatched = plan.dispatch(payload)
    np.testing.assert_array_equal(dispatched[plan.row_padding], 0)
    np.testing.assert_array_equal(plan.inverse_dispatch(dispatched), np.repeat(payload[:, None, :], 2, axis=1))


def test_relation_plan_rejects_capacity_overflow_before_payload_dispatch() -> None:
    with pytest.raises(RelationPlanError) as exc_info:
        build_relation_plan(
            np.zeros((4, 2), dtype=np.int32),
            np.ones((4, 2), dtype=np.float32),
            destination_rank_by_item=np.array([0, 1], dtype=np.int32),
            destination_local_item_by_item=np.array([0, 0], dtype=np.int32),
            padding_quantum=4,
            max_routes_per_rank=6,
            max_padded_rows_per_rank=8,
        )

    assert exc_info.value.reasons == ("destination rank 0 has 8 routes, exceeding capacity 6",)


def test_relation_plan_dump_is_compact_and_inspectable() -> None:
    plan, _, _ = _relation_plan()

    assert plan.dump() == "\n".join(
        (
            "RelationPlan sources=4 slots=2 routes=8 destination_rows=8 exchange_rows=6 "
            "merge_order=source_item ascending, then route_slot ascending, FP32 accumulation",
            "  rank=0 item=0 count=2 padded=2 offset=0",
            "  rank=0 item=1 count=2 padded=2 offset=2",
            "  rank=1 item=0 count=2 padded=2 offset=4",
            "  rank=1 item=1 count=2 padded=2 offset=6",
        )
    )
