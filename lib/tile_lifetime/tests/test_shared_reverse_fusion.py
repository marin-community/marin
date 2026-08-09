# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from tile_lifetime import (
    OwnerComputeTraversal,
    SharedReverseFusionDisposition,
    plan_shared_producer_reverse_fusion,
)
from tile_lifetime.event_dataflow import TaskAxis, TaskFamily, TaskRelation


def _relation(pairs):
    source = TaskFamily("source Fold owners", (TaskAxis("source", 2),))
    target = TaskFamily("target Fold owners", (TaskAxis("target", 2),))
    return TaskRelation.from_pairs(source, target, tuple((((left,), (right,))) for left, right in pairs))


def test_shared_reverse_fusion_coalesces_small_relation_components() -> None:
    plan = plan_shared_producer_reverse_fusion(
        _relation(((0, 0), (1, 1))),
        source_accumulator_elements=4,
        target_accumulator_elements=2,
        transient_edge_elements=3,
        accumulator_bytes_per_element=4,
        local_capacity_bytes=36,
        baseline_contracts_per_edge=7,
        fused_contracts_per_edge=5,
    )

    assert plan.disposition is SharedReverseFusionDisposition.FUSED_LOCAL
    assert len(plan.components) == 2
    assert all(component.selected_traversal is OwnerComputeTraversal.SOURCE_MAJOR for component in plan.components)
    assert plan.required_local_bytes == 36
    assert plan.baseline_contract_invocations == 14
    assert plan.fused_contract_invocations == 10
    assert plan.physical_contract_reduction == 1.4
    assert plan.reasons == ()


def test_shared_reverse_fusion_rejects_split_owner_component() -> None:
    plan = plan_shared_producer_reverse_fusion(
        _relation(((0, 0), (0, 1), (1, 0), (1, 1))),
        source_accumulator_elements=4,
        target_accumulator_elements=2,
        transient_edge_elements=3,
        accumulator_bytes_per_element=4,
        local_capacity_bytes=40,
        baseline_contracts_per_edge=7,
        fused_contracts_per_edge=5,
    )

    assert plan.disposition is SharedReverseFusionDisposition.REJECTED_LOCAL_CAPACITY
    assert len(plan.components) == 1
    assert plan.components[0].selected_traversal is OwnerComputeTraversal.SOURCE_MAJOR
    assert plan.required_local_bytes == 44
    assert "external partial Fold" in plan.reasons[-1]
