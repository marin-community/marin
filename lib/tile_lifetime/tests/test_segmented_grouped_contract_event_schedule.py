# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from tile_lifetime.event_buffering import EventRealizationKind
from tile_lifetime.grouped_contract_event_schedule import (
    GroupedContractReleasePoint,
    GroupedContractSynchronizationDescriptor,
    GroupedContractWorkerAssignment,
)
from tile_lifetime.relation import build_relation_plan
from tile_lifetime.segmented_grouped_contract_event_schedule import (
    derive_same_stream_segmented_grouped_contract_schedule,
)


def _relation(destinations: np.ndarray):
    destination_count = max(4, int(np.max(destinations)) + 1)
    return build_relation_plan(
        destinations,
        np.ones_like(destinations, dtype=np.float32),
        destination_rank_by_item=np.zeros(destination_count, dtype=np.int32),
        destination_local_item_by_item=np.arange(destination_count, dtype=np.int32),
        padding_quantum=1,
    )


def _descriptor(*, cluster_ctas: int = 2):
    return GroupedContractSynchronizationDescriptor(
        workers=GroupedContractWorkerAssignment(
            cluster_ctas=cluster_ctas,
            transfer_warpgroups_per_cta=1,
            matrix_warpgroups=1,
            epilogue_warpgroups_per_cta=1,
            matrix_owner_cta=0,
        ),
        load_pipeline_stages=3,
        operand_bytes_per_transfer_owner=32_768,
        operand_release_point=GroupedContractReleasePoint.MATRIX_OPERAND_CONSUMED,
        output_release_point=GroupedContractReleasePoint.ACCUMULATOR_READ_COMPLETE,
    )


def test_runtime_relation_and_internal_contract_events_remain_distinct() -> None:
    relation = _relation(np.asarray([[0, 2], [1, 2], [2, 0]], dtype=np.int32))
    schedule = derive_same_stream_segmented_grouped_contract_schedule(
        relation,
        output_tile_count=2,
        descriptor=_descriptor(),
        reduction_partition_count=5,
    )

    expected_outer_counts = tuple(int(count) for count in relation.group_count for _ in range(2))
    assert schedule.segment_runtime_inputs.event_initial_counts == expected_outer_counts
    assert schedule.segment_runtime_inputs.initially_ready_events == (6, 7)
    assert schedule.segment_realization.entries[0].kind is EventRealizationKind.ERASED_STREAM_ORDER
    assert schedule.contract_pipeline.operand_ready_count == 2
    assert schedule.contract_pipeline.operand_transaction_bytes == 65_536


def test_relation_and_worker_mutations_change_only_their_schedule_level() -> None:
    primary_relation = _relation(np.asarray([[0, 2], [1, 2], [2, 0]], dtype=np.int32))
    mutated_relation = _relation(np.asarray([[3, 2], [1, 3], [2, 3]], dtype=np.int32))

    primary = derive_same_stream_segmented_grouped_contract_schedule(
        primary_relation,
        output_tile_count=1,
        descriptor=_descriptor(),
        reduction_partition_count=5,
    )
    relation_mutation = derive_same_stream_segmented_grouped_contract_schedule(
        mutated_relation,
        output_tile_count=1,
        descriptor=_descriptor(),
        reduction_partition_count=5,
    )
    worker_mutation = derive_same_stream_segmented_grouped_contract_schedule(
        primary_relation,
        output_tile_count=1,
        descriptor=_descriptor(cluster_ctas=4),
        reduction_partition_count=5,
    )

    assert primary.segment_runtime_inputs != relation_mutation.segment_runtime_inputs
    assert primary.contract_pipeline.fingerprint == relation_mutation.contract_pipeline.fingerprint
    assert primary.segment_runtime_inputs == worker_mutation.segment_runtime_inputs
    assert primary.contract_pipeline.fingerprint != worker_mutation.contract_pipeline.fingerprint
    assert primary.program_fingerprint == relation_mutation.program_fingerprint
    assert primary.runtime_fingerprint != relation_mutation.runtime_fingerprint
    assert primary.program_fingerprint != worker_mutation.program_fingerprint
    assert primary.runtime_fingerprint == worker_mutation.runtime_fingerprint
