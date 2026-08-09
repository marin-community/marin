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
