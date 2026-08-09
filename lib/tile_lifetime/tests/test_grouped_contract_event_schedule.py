# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from tile_lifetime.event_buffering import EventRealizationKind
from tile_lifetime.grouped_contract_event_schedule import (
    GroupedContractReleasePoint,
    GroupedContractSynchronizationDescriptor,
    GroupedContractWorkerAssignment,
    derive_grouped_contract_physical_event_schedule,
    verify_grouped_contract_backend_parameters,
)
from tile_lifetime.sm100_grouped_contract_event_codegen import (
    render_sm100_grouped_contract_event_include,
    sm100_bf16_grouped_contract_event_schedule,
    verify_sm100_grouped_contract_event_include,
)


def _descriptor(*, cluster_ctas: int = 2, load_pipeline_stages: int = 3):
    return GroupedContractSynchronizationDescriptor(
        workers=GroupedContractWorkerAssignment(
            cluster_ctas=cluster_ctas,
            transfer_warpgroups_per_cta=1,
            matrix_warpgroups=1,
            epilogue_warpgroups_per_cta=1,
            matrix_owner_cta=0,
        ),
        load_pipeline_stages=load_pipeline_stages,
        operand_bytes_per_transfer_owner=32_768,
        operand_release_point=GroupedContractReleasePoint.MATRIX_OPERAND_CONSUMED,
        output_release_point=GroupedContractReleasePoint.ACCUMULATOR_READ_COMPLETE,
    )


def test_grouped_contract_schedule_derives_owner_cardinalities_and_releases() -> None:
    schedule = derive_grouped_contract_physical_event_schedule(
        _descriptor(),
        reduction_partition_count=5,
    )

    assert schedule.operand_ready_count == 2
    assert schedule.operand_release_count == 1
    assert schedule.output_ready_count == 1
    assert schedule.output_release_count == 2
    assert schedule.descriptor.operand_release_point is GroupedContractReleasePoint.MATRIX_OPERAND_CONSUMED
    assert schedule.descriptor.output_release_point is GroupedContractReleasePoint.ACCUMULATOR_READ_COMPLETE
    assert schedule.dataflow.operand_buffer.capacity == 6

    last_consumers = dict(schedule.dataflow.operand_buffer.last_consumers)
    assert {consumer.family for consumer in last_consumers.values()} == {schedule.dataflow.matrix_issue.name}
    reuse = tuple(
        plan
        for plan in schedule.dataflow.program.event_plans
        if plan.notify_relation.source == schedule.dataflow.matrix_issue
        and plan.trigger_relation.target == schedule.dataflow.operand_stage
    )
    assert len(reuse) == 1
    assert reuse[0].generation_policy.value == "phased"
    assert {count.value for count in reuse[0].initial_count.counts} == {0, 1}

    physical = {entry.mechanism for entry in schedule.audit.physical}
    assert physical == {
        "cluster transaction-completion barrier",
        "phased operand-stage release barrier",
        "cluster accumulator-ready barrier",
        "phased cluster accumulator-release barrier",
    }
    assert all(entry.kind is EventRealizationKind.ERASED_PROGRAM_ORDER for entry in schedule.audit.erased)


def test_cluster_width_mutation_changes_logical_event_domains_and_counts() -> None:
    two_ctas = derive_grouped_contract_physical_event_schedule(
        _descriptor(cluster_ctas=2),
        reduction_partition_count=5,
    )
    four_ctas = derive_grouped_contract_physical_event_schedule(
        _descriptor(cluster_ctas=4),
        reduction_partition_count=5,
    )

    assert two_ctas.dataflow.operand_stage.axes[-1].extent == 2
    assert four_ctas.dataflow.operand_stage.axes[-1].extent == 4
    assert two_ctas.dataflow.epilogue.axes[-1].extent == 2
    assert four_ctas.dataflow.epilogue.axes[-1].extent == 4
    assert two_ctas.operand_ready_count == two_ctas.output_release_count == 2
    assert four_ctas.operand_ready_count == four_ctas.output_release_count == 4
    assert two_ctas.fingerprint != four_ctas.fingerprint


def test_pipeline_depth_mutation_regenerates_bounded_stage_reuse() -> None:
    depth_two = derive_grouped_contract_physical_event_schedule(
        _descriptor(load_pipeline_stages=2),
        reduction_partition_count=5,
    )
    depth_three = derive_grouped_contract_physical_event_schedule(
        _descriptor(load_pipeline_stages=3),
        reduction_partition_count=5,
    )

    assert depth_two.dataflow.operand_buffer.capacity == 4
    assert depth_three.dataflow.operand_buffer.capacity == 6
    assert depth_two.dataflow.operand_buffer.slots != depth_three.dataflow.operand_buffer.slots
    assert depth_two.dataflow.operand_buffer.generations != depth_three.dataflow.operand_buffer.generations
    assert depth_two.fingerprint != depth_three.fingerprint


def test_transaction_bytes_are_not_logical_producer_indegree() -> None:
    schedule = derive_grouped_contract_physical_event_schedule(
        _descriptor(cluster_ctas=4),
        reduction_partition_count=5,
    )

    assert schedule.operand_ready_count == 4
    assert schedule.operand_transaction_bytes == 4 * 32_768
    assert schedule.transaction_completion_enabled is True
    assert int(schedule.transaction_completion_enabled) != schedule.operand_ready_count


def test_grouped_contract_backend_audit_rejects_stale_owner_counts() -> None:
    schedule = derive_grouped_contract_physical_event_schedule(
        _descriptor(),
        reduction_partition_count=5,
    )

    verify_grouped_contract_backend_parameters(
        schedule,
        cluster_ctas=2,
        load_pipeline_stages=3,
        operand_release_count=1,
        output_release_count=2,
    )
    with pytest.raises(ValueError, match="output_release_count: expected 2, found 1"):
        verify_grouped_contract_backend_parameters(
            schedule,
            cluster_ctas=2,
            load_pipeline_stages=3,
            operand_release_count=1,
            output_release_count=1,
        )


def test_sm100_include_is_a_verified_view_of_generic_event_schedule(tmp_path: Path) -> None:
    schedule = sm100_bf16_grouped_contract_event_schedule()
    include = tmp_path / "generated_event_schedule.inc"
    include.write_text(render_sm100_grouped_contract_event_include(schedule))

    verify_sm100_grouped_contract_event_include(include, schedule)
    observed = include.read_text()
    assert "kOperandReadyLogicalCount = 2" in observed
    assert "kOutputReleaseLogicalCount = 2" in observed
    assert "kOperandTransactionBytes = 65536" in observed

    include.write_text(observed.replace("kOutputReleaseLogicalCount = 2", "kOutputReleaseLogicalCount = 1"))
    with pytest.raises(ValueError, match="does not match schedule"):
        verify_sm100_grouped_contract_event_include(include, schedule)
