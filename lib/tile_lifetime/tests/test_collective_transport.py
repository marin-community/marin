# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
from dataclasses import replace
from pathlib import Path

import pytest

from tile_lifetime.collective_transport import (
    CollectiveReduction,
    ValueCompleteness,
    recover_collective_completion_plans,
)
from tile_lifetime.event_dataflow import (
    EventMemoryScope,
    EventSchedulingMode,
    ImperativeEventOpKind,
    lower_event_tensor_plan,
    verify_event_dataflow_program,
)
from tile_lifetime.event_dataflow_adapters import (
    CollectiveCompletionSchedule,
    collective_completion_task_dataflow,
)
from tile_lifetime.ir import DType
from tile_lifetime.plan import NumericalPolicy

_HLO = (
    Path(__file__).parents[1]
    / "benchmarks/artifacts/xla_grug_routed_combined_gpu_gb200_v0/transformed-gpu-pre-scheduler-hlo.txt.gz"
)


def _hlo() -> str:
    return gzip.decompress(_HLO.read_bytes()).decode()


def test_weight_contract_collectives_recover_as_partial_value_completion() -> None:
    plans = recover_collective_completion_plans(_hlo(), producer_values=("dot.6", "dot.7"))

    assert tuple(plan.transport.source_value for plan in plans) == ("dot.6", "dot.7")
    assert tuple(plan.transport.destination_value for plan in plans) == ("psum.58", "psum.59")
    assert tuple(plan.shape for plan in plans) == ("bf16[4,32,64]{2,1,0}", "bf16[4,32,32]{2,1,0}")
    assert all(plan.fold.reduction is CollectiveReduction.SUM for plan in plans)
    assert all(plan.fold.dtype is DType.BF16 for plan in plans)
    assert all(plan.fold.input_completeness is ValueCompleteness.PARTIAL for plan in plans)
    assert all(plan.fold.output_completeness is ValueCompleteness.COMPLETE for plan in plans)
    assert all(plan.fold.numerical_policy is NumericalPolicy.ALLOW_ROUNDING_REORDER for plan in plans)
    assert all(plan.transport.replica_domain.groups == ((0,),) for plan in plans)
    assert all(plan.transport.replica_domain.use_global_device_ids for plan in plans)
    assert all(plan.transport.channel_id == 1 for plan in plans)


def test_replica_group_mutation_changes_transport_without_changing_fold() -> None:
    baseline = recover_collective_completion_plans(_hlo(), producer_values=("dot.6", "dot.7"))
    mutated_hlo = _hlo().replace("replica_groups={{0}}", "replica_groups={{0,1}}")
    mutated = recover_collective_completion_plans(mutated_hlo, producer_values=("dot.6", "dot.7"))

    assert tuple(plan.fold for plan in mutated) == tuple(plan.fold for plan in baseline)
    assert all(plan.transport.replica_domain.groups == ((0, 1),) for plan in mutated)
    assert tuple(
        replace(plan.transport, replica_domain=baseline[index].transport.replica_domain)
        for index, plan in enumerate(mutated)
    ) == tuple(plan.transport for plan in baseline)


def test_reducer_mutation_changes_fold_without_changing_transport() -> None:
    hlo = _hlo()
    original = "ROOT %add.209 = bf16[] add(%psum.42, %psum.43)"
    mutated_text = "ROOT %add.209 = bf16[] maximum(%psum.42, %psum.43)"
    assert hlo.count(original) == 1
    baseline = recover_collective_completion_plans(hlo, producer_values=("dot.6", "dot.7"))
    mutated = recover_collective_completion_plans(
        hlo.replace(original, mutated_text),
        producer_values=("dot.6", "dot.7"),
    )

    assert mutated[0].fold.reduction is CollectiveReduction.MAXIMUM
    assert mutated[1].fold == baseline[1].fold
    assert tuple(plan.transport for plan in mutated) == tuple(plan.transport for plan in baseline)


def test_collective_rejects_reducer_with_different_dtype() -> None:
    hlo = _hlo()
    original = "ROOT %add.209 = bf16[] add(%psum.42, %psum.43)"
    assert hlo.count(original) == 1
    with pytest.raises(ValueError, match="reducer dtype"):
        recover_collective_completion_plans(
            hlo.replace(original, "ROOT %add.209 = f32[] add(%psum.42, %psum.43)"),
            producer_values=("dot.6",),
        )


@pytest.mark.parametrize("scheduling_mode", list(EventSchedulingMode))
def test_collective_completion_derives_system_visible_tiled_readiness(
    scheduling_mode: EventSchedulingMode,
) -> None:
    mutated_hlo = _hlo().replace("replica_groups={{0}}", "replica_groups={{0,1},{2}}")
    completion = recover_collective_completion_plans(mutated_hlo, producer_values=("dot.6",))[0]

    dataflow = collective_completion_task_dataflow(
        completion,
        schedule=CollectiveCompletionSchedule(tile_count=3, scheduling_mode=scheduling_mode),
    )

    verify_event_dataflow_program(dataflow.program)
    assert dataflow.contribution_devices == (0, 1, 2)
    assert dataflow.contribution_groups == (0, 0, 1)
    partial_plan, event_plan = dataflow.program.event_plans
    assert partial_plan.initial_count.as_mapping() == {
        (contribution, tile): 1 for contribution in range(3) for tile in range(3)
    }
    assert partial_plan.memory_scope is EventMemoryScope.DEVICE
    assert event_plan.initial_count.as_mapping() == {
        (0, 0): 2,
        (0, 1): 2,
        (0, 2): 2,
        (1, 0): 1,
        (1, 1): 1,
        (1, 2): 1,
    }
    assert event_plan.memory_scope is EventMemoryScope.SYSTEM
    assert event_plan.visibility.release_on_notify
    assert event_plan.visibility.acquire_before_consumer
    operations = lower_event_tensor_plan(event_plan, scheduling_mode=scheduling_mode)
    expected_trigger = (
        ImperativeEventOpKind.WAIT
        if scheduling_mode is EventSchedulingMode.STATIC
        else ImperativeEventOpKind.TRIGGER_ENQUEUE
    )
    assert sum(operation.kind is expected_trigger for operation in operations) == 6


def test_collective_replica_group_mutation_changes_readiness_not_fold_body() -> None:
    baseline_completion = recover_collective_completion_plans(_hlo(), producer_values=("dot.6",))[0]
    mutated_hlo = _hlo().replace("replica_groups={{0}}", "replica_groups={{0,1},{2}}")
    mutated_completion = recover_collective_completion_plans(mutated_hlo, producer_values=("dot.6",))[0]
    schedule = CollectiveCompletionSchedule(tile_count=2, scheduling_mode=EventSchedulingMode.DYNAMIC)

    baseline = collective_completion_task_dataflow(baseline_completion, schedule=schedule)
    mutated = collective_completion_task_dataflow(mutated_completion, schedule=schedule)

    assert mutated.completion.fold == baseline.completion.fold
    assert baseline.program.event_plans[1].initial_count.as_mapping() == {(0, 0): 1, (0, 1): 1}
    assert mutated.program.event_plans[1].initial_count.as_mapping() == {
        (0, 0): 2,
        (0, 1): 2,
        (1, 0): 1,
        (1, 1): 1,
    }


def test_collective_reducer_mutation_reuses_event_construction() -> None:
    hlo = _hlo()
    original = "ROOT %add.209 = bf16[] add(%psum.42, %psum.43)"
    baseline_completion = recover_collective_completion_plans(hlo, producer_values=("dot.6",))[0]
    maximum_completion = recover_collective_completion_plans(
        hlo.replace(original, "ROOT %add.209 = bf16[] maximum(%psum.42, %psum.43)"),
        producer_values=("dot.6",),
    )[0]
    schedule = CollectiveCompletionSchedule(tile_count=4, scheduling_mode=EventSchedulingMode.STATIC)

    baseline = collective_completion_task_dataflow(baseline_completion, schedule=schedule)
    maximum = collective_completion_task_dataflow(maximum_completion, schedule=schedule)

    assert baseline.completion.fold.reduction is CollectiveReduction.SUM
    assert maximum.completion.fold.reduction is CollectiveReduction.MAXIMUM
    assert maximum.program == baseline.program
