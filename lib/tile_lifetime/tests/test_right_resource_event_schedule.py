# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for generic relation-grouped Event Tensor schedules."""

from __future__ import annotations

import ast
import inspect

import numpy as np

import tile_lifetime.right_resource_event_schedule as right_resource_schedule
from tile_lifetime.event_buffering import EventRealizationKind
from tile_lifetime.event_dataflow import EventSchedulingMode, execute_event_dataflow
from tile_lifetime.relation import RelationPlan, build_relation_plan
from tile_lifetime.right_resource_event_schedule import (
    RightResourceFoldEventSchedule,
    RightResourcePipelineDescriptor,
    derive_right_resource_fold_event_schedule,
)


def _relation(destination: np.ndarray, valid: np.ndarray) -> RelationPlan:
    return build_relation_plan(
        destination,
        np.ones(destination.shape, dtype=np.float32),
        edge_valid=valid,
        destination_rank_by_item=np.zeros(5, dtype=np.int32),
        destination_local_item_by_item=np.arange(5, dtype=np.int32),
        padding_quantum=1,
    )


def _descriptor(*, buffer_depth: int = 2) -> RightResourcePipelineDescriptor:
    return RightResourcePipelineDescriptor(
        grouped_body_name="grouped_body",
        fold_finalize_name="fold_finalize",
        edge_partition_by_slot=(0, 1, 0, 1),
        edge_partition_count=2,
        edge_capacity_per_task=2,
        resource_buffer_depth=buffer_depth,
        resource_payload_bytes=4096,
    )


def _fixture() -> RelationPlan:
    destination = np.asarray(
        [
            [3, 0, 2, 1],
            [0, 2, 3, 1],
            [2, 0, 1, 3],
            [1, 3, 0, 2],
        ],
        dtype=np.int32,
    )
    valid = np.asarray(
        [
            [True, True, True, False],
            [True, True, False, True],
            [True, False, True, True],
            [True, True, True, True],
        ],
        dtype=np.bool_,
    )
    return _relation(destination, valid)


def _execute(schedule: RightResourceFoldEventSchedule, relation: RelationPlan) -> np.ndarray:
    contributions = np.arange(relation.source_item.size, dtype=np.float32) + np.float32(1.0)

    def stage(coordinate: tuple[int, ...], state: dict[str, object]) -> None:
        staged = state["staged"]
        assert isinstance(staged, set)
        staged.add(coordinate[0])

    def body(coordinate: tuple[int, ...], state: dict[str, object]) -> None:
        task = coordinate[0]
        staged = state["staged"]
        partials = state["partials"]
        assert isinstance(staged, set)
        assert isinstance(partials, dict)
        assert task in staged
        begin = schedule.grouping.resource_edge_offsets[task]
        end = schedule.grouping.resource_edge_offsets[task + 1]
        for edge in schedule.grouping.resource_edges[begin:end]:
            source = int(relation.source_item[edge])
            slot = int(relation.route_slot[edge])
            partition = schedule.descriptor.edge_partition_by_slot[slot]
            resource = schedule.grouping.resource_item[task]
            partials.setdefault((source, partition), []).append(contributions[edge] * np.float32(resource + 1))

    def finalize(coordinate: tuple[int, ...], state: dict[str, object]) -> None:
        partials = state["partials"]
        output = state["output"]
        assert isinstance(partials, dict)
        assert isinstance(output, np.ndarray)
        output[coordinate] = np.sum(partials.get(coordinate, ()), dtype=np.float32)

    result = execute_event_dataflow(
        schedule.program,
        actions={
            schedule.resource_stage.name: stage,
            schedule.grouped_body.name: body,
            schedule.fold_finalize.name: finalize,
        },
        state={
            "staged": set(),
            "partials": {},
            "output": np.zeros(
                (relation.source_item_count, schedule.descriptor.edge_partition_count),
                dtype=np.float32,
            ),
        },
        scheduling_mode=EventSchedulingMode.DYNAMIC,
        random_seed=17,
    )
    output = result.state["output"]
    assert isinstance(output, np.ndarray)
    return output


def test_relation_grouping_drives_resource_and_fold_readiness() -> None:
    relation = _fixture()
    schedule = derive_right_resource_fold_event_schedule(relation, _descriptor())

    assert schedule.grouping.edge_count == relation.route_count
    assert 4 not in schedule.grouping.resource_item  # Empty right-side resource.
    assert tuple(schedule.fold_runtime_inputs.event_initial_counts) == (2, 1, 1, 2, 2, 1, 2, 2)
    assert schedule.resource_buffer.capacity == 2
    assert schedule.resource_buffer.slots == tuple(task % 2 for task in range(schedule.grouping.task_count))
    assert schedule.resource_buffer.generations == tuple(task // 2 for task in range(schedule.grouping.task_count))
    assert tuple(entry.kind for entry in schedule.realization.entries) == (
        EventRealizationKind.PHYSICAL,
        EventRealizationKind.ERASED_STREAM_ORDER,
        EventRealizationKind.PHYSICAL,
    )

    output = _execute(schedule, relation)
    expected = np.zeros_like(output)
    for edge in np.flatnonzero(relation.edge_valid.reshape(-1)):
        source = int(relation.source_item[edge])
        partition = _descriptor().edge_partition_by_slot[int(relation.route_slot[edge])]
        expected[source, partition] += np.float32(edge + 1) * np.float32(relation.destination_item[edge] + 1)
    np.testing.assert_array_equal(output, expected)


def test_relation_mutation_changes_only_runtime_fingerprint_when_shape_is_stable() -> None:
    baseline_relation = _fixture()
    baseline = derive_right_resource_fold_event_schedule(baseline_relation, _descriptor())
    destination = baseline_relation.destination_item.reshape(
        baseline_relation.source_item_count,
        baseline_relation.route_slots,
    ).copy()
    valid = baseline_relation.edge_valid.copy()
    destination[0, 0], destination[1, 0] = destination[1, 0], destination[0, 0]
    mutated_relation = _relation(destination, valid)
    mutated = derive_right_resource_fold_event_schedule(mutated_relation, _descriptor())

    assert baseline.grouping.task_count == mutated.grouping.task_count
    assert baseline.program_fingerprint == mutated.program_fingerprint
    assert baseline.runtime_fingerprint != mutated.runtime_fingerprint
    assert not np.array_equal(_execute(baseline, baseline_relation), _execute(mutated, mutated_relation))


def test_buffer_depth_is_a_schedule_choice() -> None:
    relation = _fixture()
    shallow = derive_right_resource_fold_event_schedule(relation, _descriptor(buffer_depth=1))
    deeper = derive_right_resource_fold_event_schedule(relation, _descriptor(buffer_depth=3))

    assert shallow.runtime_fingerprint == deeper.runtime_fingerprint
    assert shallow.program_fingerprint != deeper.program_fingerprint
    assert shallow.resource_buffer.slots != deeper.resource_buffer.slots
    np.testing.assert_array_equal(_execute(shallow, relation), _execute(deeper, relation))


def test_schedule_module_has_no_workload_role_vocabulary() -> None:
    source = inspect.getsource(right_resource_schedule).lower()
    tree = ast.parse(source)
    local_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    for forbidden in ("attention", "moe", "expert"):
        assert forbidden not in source
    assert not {"query", "key", "value", "q", "k", "v"} & local_names
