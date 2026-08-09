# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import MutableMapping
from dataclasses import replace

import numpy as np
import pytest

from tile_lifetime.event_dataflow import (
    EventCount,
    EventCountExpression,
    EventDataflowError,
    EventDataflowProgram,
    EventDomain,
    EventMemoryScope,
    EventSchedulingMode,
    ImperativeEventOpKind,
    IndexPair,
    MemoryVisibility,
    TaskAxis,
    TaskDependence,
    TaskFamily,
    TaskRelation,
    TraceKind,
    coarsen_event_tensor_plan,
    derive_event_tensor_plan,
    execute_event_dataflow,
    lower_event_tensor_plan,
    verify_event_dataflow_program,
    verify_event_tensor_plan,
)
from tile_lifetime.event_dataflow_examples import (
    relation_segment_dependence,
    single_dependence_event_program,
    split_fold_dependence,
    tiled_collective_dependence,
)
from tile_lifetime.relation import build_relation_plan


def _split_fold_actions(values: np.ndarray):
    def partial(coordinate: tuple[int, ...], state: dict[str, object]) -> None:
        row, partition = coordinate
        partials = state["partials"]
        assert isinstance(partials, dict)
        partials[coordinate] = float(values[row, partition])

    def finalize(coordinate: tuple[int, ...], state: dict[str, object]) -> None:
        (row,) = coordinate
        partials = state["partials"]
        output = state["output"]
        assert isinstance(partials, dict)
        assert isinstance(output, np.ndarray)
        output[row] = sum(partials[(row, partition)] for partition in range(values.shape[1]))

    return {"partial_fold": partial, "fold_finalize": finalize}


@pytest.mark.parametrize("partition_count", [1, 3, 7])
@pytest.mark.parametrize("mode", list(EventSchedulingMode))
def test_split_fold_events_execute_direct_fold_for_arbitrary_partition_count(
    partition_count: int,
    mode: EventSchedulingMode,
) -> None:
    values = np.arange(4 * partition_count, dtype=np.float32).reshape(4, partition_count) + 0.25
    dependence = split_fold_dependence(row_count=4, partition_count=partition_count)
    program = single_dependence_event_program(dependence, name="split_fold_ready", scheduling_mode=mode)

    result = execute_event_dataflow(
        program,
        actions=_split_fold_actions(values),
        state={"partials": {}, "output": np.zeros(4, dtype=np.float32)},
        scheduling_mode=mode,
    )

    np.testing.assert_array_equal(result.state["output"], np.sum(values, axis=1))
    assert program.event_plans[0].initial_count.as_mapping() == {(row,): partition_count for row in range(4)}
    execution_index = {task: index for index, task in enumerate(result.executed_tasks)}
    for row in range(4):
        finalize = next(
            task for task in result.executed_tasks if task.family == "fold_finalize" and task.coordinate == (row,)
        )
        producers = [
            task for task in result.executed_tasks if task.family == "partial_fold" and task.coordinate[0] == row
        ]
        assert all(execution_index[producer] < execution_index[finalize] for producer in producers)


def test_event_coarsening_adds_only_false_dependencies_and_preserves_fold_output() -> None:
    dependence = split_fold_dependence(row_count=2, partition_count=2)
    fine = derive_event_tensor_plan(dependence, name="fine")
    coarse = coarsen_event_tensor_plan(
        fine,
        domain=EventDomain("coarse.events", (TaskAxis("all_rows", 1),)),
        project=lambda _coordinate: (0,),
        name="coarse",
    )
    required = set(dependence.relation.pairs)
    assert set(fine.scheduled_dependence.pairs) == required
    assert required < set(coarse.scheduled_dependence.pairs)

    values = np.asarray([[1.0, 2.0], [4.0, 8.0]], dtype=np.float32)
    source = dependence.relation.source
    target = dependence.relation.target
    assert isinstance(source, TaskFamily)
    assert isinstance(target, TaskFamily)
    coarse_program = EventDataflowProgram((source, target), (dependence,), (coarse,))
    result = execute_event_dataflow(
        coarse_program,
        actions=_split_fold_actions(values),
        state={"partials": {}, "output": np.zeros(2, dtype=np.float32)},
        scheduling_mode=EventSchedulingMode.DYNAMIC,
        random_seed=7,
    )
    np.testing.assert_array_equal(result.state["output"], np.asarray([3.0, 12.0], dtype=np.float32))


def _relation_fixture(destination_indices: np.ndarray, edge_valid: np.ndarray):
    return build_relation_plan(
        destination_indices,
        np.ones(destination_indices.shape, dtype=np.float32),
        edge_valid=edge_valid,
        destination_rank_by_item=np.asarray([0, 0, 1, 1, 1], dtype=np.int32),
        destination_local_item_by_item=np.asarray([0, 1, 0, 1, 2], dtype=np.int32),
        padding_quantum=2,
    )


def _segment_actions(dependence: TaskDependence):
    sources_by_target = {
        coordinate: dependence.relation.sources_for(coordinate) for coordinate in dependence.relation.target.coordinates
    }

    def edge_ready(coordinate: tuple[int, ...], state: dict[str, object]) -> None:
        completed = state["completed"]
        assert isinstance(completed, set)
        completed.add(coordinate)

    def segment(coordinate: tuple[int, ...], state: dict[str, object]) -> None:
        completed = state["completed"]
        counts = state["counts"]
        assert isinstance(completed, set)
        assert isinstance(counts, np.ndarray)
        assert all(source in completed for source in sources_by_target[coordinate])
        counts[coordinate[0]] = len(sources_by_target[coordinate])

    return {"relation_edge_ready": edge_ready, "segment_consumer": segment}


def test_runtime_relation_mutation_changes_derived_counts_without_event_specific_logic() -> None:
    destinations = np.asarray([[0, 1, 1], [3, 1, 4], [4, 0, 3], [1, 3, 4]], dtype=np.int32)
    valid = np.asarray(
        [[True, True, False], [True, True, True], [True, False, False], [False, True, True]],
        dtype=np.bool_,
    )
    relation = _relation_fixture(destinations, valid)
    dependence = relation_segment_dependence(relation)
    program = single_dependence_event_program(
        dependence,
        name="runtime_segments",
        scheduling_mode=EventSchedulingMode.DYNAMIC,
    )
    expected = np.bincount(destinations[valid], minlength=5)
    assert tuple(program.event_plans[0].initial_count.as_mapping()[(index,)] for index in range(5)) == tuple(expected)
    assert expected[2] == 0

    first = execute_event_dataflow(
        program,
        actions=_segment_actions(dependence),
        state={"completed": set(), "counts": np.full(5, -1, dtype=np.int32)},
        scheduling_mode=EventSchedulingMode.DYNAMIC,
    )
    second = execute_event_dataflow(
        program,
        actions=_segment_actions(dependence),
        state={"completed": set(), "counts": np.full(5, -1, dtype=np.int32)},
        scheduling_mode=EventSchedulingMode.DYNAMIC,
    )
    np.testing.assert_array_equal(first.state["counts"], expected)
    assert first.trace == second.trace

    mutated_destinations = destinations.copy()
    mutated_destinations[0, 0] = 2
    mutated_valid = valid.copy()
    mutated_valid[2, 1] = True
    mutated = _relation_fixture(mutated_destinations, mutated_valid)
    mutated_dependence = relation_segment_dependence(mutated)
    mutated_program = single_dependence_event_program(
        mutated_dependence,
        name="runtime_segments",
        scheduling_mode=EventSchedulingMode.DYNAMIC,
    )
    mutated_expected = np.bincount(mutated_destinations[mutated_valid], minlength=5)
    assert mutated_program.event_plans[0].initial_count.as_mapping() != program.event_plans[0].initial_count.as_mapping()
    mutated_result = execute_event_dataflow(
        mutated_program,
        actions=_segment_actions(mutated_dependence),
        state={"completed": set(), "counts": np.full(5, -1, dtype=np.int32)},
        scheduling_mode=EventSchedulingMode.DYNAMIC,
    )
    np.testing.assert_array_equal(mutated_result.state["counts"], mutated_expected)


@pytest.mark.parametrize("mode", list(EventSchedulingMode))
def test_tiled_collective_graph_uses_same_event_derivation(mode: EventSchedulingMode) -> None:
    dependence = tiled_collective_dependence(
        output_tile_count=3,
        destination_count=2,
        partials_per_destination=4,
    )
    program = single_dependence_event_program(dependence, name="tile_transport_ready", scheduling_mode=mode)

    def produce(coordinate: tuple[int, ...], state: MutableMapping[str, object]) -> None:
        produced = state["produced"]
        assert isinstance(produced, dict)
        output_tile, destination, partial = coordinate
        produced[coordinate] = output_tile * 100 + destination * 10 + partial

    def consume(coordinate: tuple[int, ...], state: MutableMapping[str, object]) -> None:
        produced = state["produced"]
        output = state["output"]
        assert isinstance(produced, dict)
        assert isinstance(output, np.ndarray)
        output_tile, destination = coordinate
        output[coordinate] = sum(produced[(output_tile, destination, partial)] for partial in range(4))

    result = execute_event_dataflow(
        program,
        actions={"contract_output_tile": produce, "placement_change_tile": consume},
        state={"produced": {}, "output": np.zeros((3, 2), dtype=np.int32)},
        scheduling_mode=mode,
    )
    expected = np.asarray(
        [
            [sum(output_tile * 100 + destination * 10 + partial for partial in range(4)) for destination in range(2)]
            for output_tile in range(3)
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(result.state["output"], expected)
    assert program.event_plans[0].memory_scope is EventMemoryScope.SYSTEM


def test_random_legal_task_orders_preserve_exact_integer_fold() -> None:
    values = np.arange(30, dtype=np.float32).reshape(5, 6)
    dependence = split_fold_dependence(row_count=5, partition_count=6)
    program = single_dependence_event_program(
        dependence,
        name="randomized_fold",
        scheduling_mode=EventSchedulingMode.DYNAMIC,
    )
    traces = []
    for seed in range(5):
        result = execute_event_dataflow(
            program,
            actions=_split_fold_actions(values),
            state={"partials": {}, "output": np.zeros(5, dtype=np.float32)},
            scheduling_mode=EventSchedulingMode.DYNAMIC,
            random_seed=seed,
        )
        np.testing.assert_array_equal(result.state["output"], np.sum(values, axis=1))
        traces.append(result.executed_tasks)
    assert len(set(traces)) > 1


def test_event_plan_verifier_rejects_wrong_count_and_weak_visibility() -> None:
    dependence = tiled_collective_dependence(
        output_tile_count=1,
        destination_count=1,
        partials_per_destination=2,
    )
    plan = derive_event_tensor_plan(dependence, name="collective")
    wrong_count = replace(
        plan,
        initial_count=EventCountExpression((EventCount((0, 0), 1),), "incorrect test count"),
    )
    with pytest.raises(EventDataflowError, match="indegrees"):
        verify_event_tensor_plan(wrong_count)
    with pytest.raises(EventDataflowError, match="weaker"):
        derive_event_tensor_plan(dependence, name="weak", memory_scope=EventMemoryScope.CTA)


def test_task_relation_rejects_duplicate_logical_notifications() -> None:
    producer = TaskFamily("producer", (TaskAxis("i", 1),))
    consumer = TaskFamily("consumer", (TaskAxis("i", 1),))
    pair = IndexPair((0,), (0,))
    with pytest.raises(EventDataflowError, match="duplicate logical edges"):
        TaskRelation(producer, consumer, (pair, pair))


def test_program_verifier_rejects_dependency_cycle_before_interpretation() -> None:
    first = TaskFamily("first", (TaskAxis("i", 1),))
    second = TaskFamily("second", (TaskAxis("i", 1),))
    visibility = MemoryVisibility(EventMemoryScope.DEVICE)
    forward = TaskDependence(TaskRelation.from_pairs(first, second, (((0,), (0,)),)), visibility)
    backward = TaskDependence(TaskRelation.from_pairs(second, first, (((0,), (0,)),)), visibility)
    program = EventDataflowProgram(
        (first, second),
        (forward, backward),
        (
            derive_event_tensor_plan(forward, name="forward"),
            derive_event_tensor_plan(backward, name="backward"),
        ),
    )
    with pytest.raises(EventDataflowError, match="impossible cycle"):
        verify_event_dataflow_program(program)


def test_program_verifier_rejects_ambiguous_event_identity_and_unknown_families() -> None:
    first = TaskFamily("first", (TaskAxis("i", 1),))
    second = TaskFamily("second", (TaskAxis("i", 1),))
    third = TaskFamily("third", (TaskAxis("i", 1),))
    visibility = MemoryVisibility(EventMemoryScope.DEVICE)
    first_to_second = TaskDependence(
        TaskRelation.from_pairs(first, second, (((0,), (0,)),)),
        visibility,
    )
    second_to_third = TaskDependence(
        TaskRelation.from_pairs(second, third, (((0,), (0,)),)),
        visibility,
    )
    duplicate_names = EventDataflowProgram(
        (first, second, third),
        (first_to_second, second_to_third),
        (
            derive_event_tensor_plan(first_to_second, name="ready"),
            derive_event_tensor_plan(second_to_third, name="ready"),
        ),
    )
    with pytest.raises(EventDataflowError, match="plan names must be unique"):
        verify_event_dataflow_program(duplicate_names)

    missing_third = replace(duplicate_names, task_families=(first, second))
    with pytest.raises(EventDataflowError, match="endpoint must be present"):
        verify_event_dataflow_program(missing_third)


def test_generation_is_part_of_reference_event_identity() -> None:
    dependence = split_fold_dependence(row_count=1, partition_count=1)
    program = single_dependence_event_program(
        dependence,
        name="generation_test",
        scheduling_mode=EventSchedulingMode.STATIC,
    )
    values = np.ones((1, 1), dtype=np.float32)
    result = execute_event_dataflow(
        program,
        actions=_split_fold_actions(values),
        state={"partials": {}, "output": np.zeros(1, dtype=np.float32)},
        scheduling_mode=EventSchedulingMode.STATIC,
        generation=9,
    )
    assert {entry.generation for entry in result.trace} == {9}
    assert any(entry.kind is TraceKind.NOTIFY for entry in result.trace)


def test_same_event_plan_lowers_to_static_waits_or_dynamic_queue_triggers() -> None:
    dependence = split_fold_dependence(row_count=2, partition_count=3)
    plan = derive_event_tensor_plan(dependence, name="lowering")
    static = lower_event_tensor_plan(plan, scheduling_mode=EventSchedulingMode.STATIC)
    dynamic = lower_event_tensor_plan(plan, scheduling_mode=EventSchedulingMode.DYNAMIC)

    assert {operation.kind for operation in static} == {
        ImperativeEventOpKind.INITIALIZE,
        ImperativeEventOpKind.WAIT,
        ImperativeEventOpKind.NOTIFY,
    }
    assert {operation.kind for operation in dynamic} == {
        ImperativeEventOpKind.INITIALIZE,
        ImperativeEventOpKind.NOTIFY,
        ImperativeEventOpKind.TRIGGER_ENQUEUE,
    }
    assert sum(operation.kind is ImperativeEventOpKind.NOTIFY for operation in static) == 6
    assert sum(operation.kind is ImperativeEventOpKind.NOTIFY for operation in dynamic) == 6
