# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
from pathlib import Path

from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.xla_demand_sliced_contract import (
    DemandSlicedContractPhysicalMode,
    audit_demand_sliced_contract_replacement,
    plan_demand_sliced_contract_ffi,
    replace_demand_sliced_contracts_with_custom_calls,
)
from tile_lifetime.xla_low_rank_gated_product_ffi import (
    plan_generated_low_rank_contract_map_training,
    replace_generated_low_rank_contract_map_training,
)

_TARGET_PREFIX = "shuttle.generic.demand_sliced_contract.test"
_SYNTHETIC = """\
HloModule synthetic_partition_map

ENTRY %main (activation: bf16[2,3], left: bf16[2,3], middle: bf16[1,3], right: bf16[3,3]) \
-> (bf16[2,2], bf16[2,3]) {
  %activation = bf16[2,3]{1,0} parameter(0)
  %left = bf16[2,3]{0,1} parameter(1)
  %middle = bf16[1,3]{0,1} parameter(2)
  %right = bf16[3,3]{0,1} parameter(3)
  %weights = bf16[6,3]{0,1} concatenate(%left, %middle, %right), dimensions={0}
  %projection = bf16[2,6]{1,0} dot(%activation, %weights), lhs_contracting_dims={1}, rhs_contracting_dims={1}
  %left_result = bf16[2,2]{1,0} slice(%projection), slice={[0:2], [0:2]}
  %middle_result = bf16[2,1]{1,0} slice(%projection), slice={[0:2], [2:3]}
  %right_result = bf16[2,3]{1,0} slice(%projection), slice={[0:2], [3:6]}
  %middle_broadcast = bf16[2,2]{1,0} broadcast(%middle_result), dimensions={0}
  %pair_map = bf16[2,2]{1,0} multiply(%left_result, %middle_broadcast)
  ROOT %result = (bf16[2,2]{1,0}, bf16[2,3]{1,0}) tuple(%pair_map, %right_result)
}
"""


def _post_gated_product_grug_hlo() -> str:
    artifact = (
        Path(__file__).parents[1]
        / "benchmarks/artifacts/xla_grug_shared_map_h100_fused_reverses_unaccepted_e3411679_v0/"
        "transformed-gpu-pre-scheduler-hlo.txt.gz"
    )
    hlo = gzip.decompress(artifact.read_bytes()).decode()
    gated = plan_generated_low_rank_contract_map_training(
        hlo,
        forward_target_prefix="shuttle.generic.low_rank_contract_map.generated.forward.test",
        reverse_target_prefix="shuttle.generic.low_rank_contract_map.generated.reverse.test",
    )
    return replace_generated_low_rank_contract_map_training(hlo, gated)


def test_grug_demand_sliced_contract_plan_groups_six_calls_into_four_generic_abis() -> None:
    plan = plan_demand_sliced_contract_ffi(_post_gated_product_grug_hlo(), target_prefix=_TARGET_PREFIX)

    assert len(plan.calls) == 6
    assert len(plan.families) == 4
    assert plan.target_occurrences == (
        (f"{_TARGET_PREFIX}.0", 2),
        (f"{_TARGET_PREFIX}.1", 2),
        (f"{_TARGET_PREFIX}.2", 1),
        (f"{_TARGET_PREFIX}.3", 1),
    )
    for family in plan.families:
        assert family.template == "generic_multi_output_contract"
        assert family.preferred_mode is DemandSlicedContractPhysicalMode.CONCATENATED_MAINLOOP_PARTITIONED_OUTPUT
        assert family.fallback_modes == (DemandSlicedContractPhysicalMode.SPLIT_MAINLOOPS_BY_DEMAND,)
        assert family.numerical.input_dtype == "bf16"
        assert family.numerical.accumulation_dtype == "f32"
        assert family.numerical.output_dtype == "bf16"
        assert family.numerical.policy is NumericalPolicy.ALLOW_ROUNDING_REORDER
        assert not family.numerical.partitioning_changes_reduction_order
    assert tuple(len(call.outputs) for call in plan.calls) == (4, 3, 4, 3, 4, 3)
    assert tuple(len(call.inputs) for call in plan.calls) == (5, 4, 5, 4, 5, 4)


def test_grug_partition_users_expose_real_map_and_fold_attachment_value() -> None:
    plan = plan_demand_sliced_contract_ffi(_post_gated_product_grug_hlo(), target_prefix=_TARGET_PREFIX)

    query_key_projection = plan.calls[0]
    gate_up_router_projection = plan.calls[1]
    weight_adjoint = plan.calls[4]
    assert {(fold.source_partitions) for fold in query_key_projection.folds} >= {(0,), (2,)}
    assert any(opportunity.source_partitions == (0, 1) for opportunity in gate_up_router_projection.cross_partition_maps)
    assert any(
        frontier.opcode == "dot" and frontier.source_partitions == (0, 1)
        for frontier in gate_up_router_projection.frontier
    )
    assert not weight_adjoint.folds
    assert {opportunity.opcode for opportunity in weight_adjoint.maps} >= {"convert", "multiply", "add"}
    assert weight_adjoint.frontier[-1].opcode == "tuple"


def test_grug_demand_sliced_contract_replacement_preserves_layouts_users_and_collectives() -> None:
    hlo = _post_gated_product_grug_hlo()
    plan = plan_demand_sliced_contract_ffi(hlo, target_prefix=_TARGET_PREFIX)
    transformed = replace_demand_sliced_contracts_with_custom_calls(hlo, plan)
    audit = audit_demand_sliced_contract_replacement(hlo, transformed, plan)

    assert len(audit.calls) == 6
    assert audit.removed_contract_count == 6
    assert audit.removed_contract_flops == 205_824
    assert audit.target_occurrences == plan.target_occurrences
    assert len(audit.collective_instructions) == 10
    assert audit.copy_count == (0, 0)
    assert audit.transpose_count == (43, 43)
    assert transformed.count("custom_call_target=") == 29
    for call, expected in zip(audit.calls, plan.calls, strict=True):
        assert call.inputs == tuple(value.instruction for value in expected.inputs)
        assert call.outputs == tuple(value.instruction for value in expected.outputs)
        assert call.output_users == tuple(
            (output.instruction, partition.external_users)
            for output, partition in zip(expected.outputs, expected.recovered.partitions, strict=True)
        )


def test_partition_mutation_regenerates_same_generic_interface_and_exact_replacement() -> None:
    mutated = (
        _SYNTHETIC.replace("left: bf16[2,3]", "left: bf16[1,3]")
        .replace("middle: bf16[1,3]", "middle: bf16[2,3]")
        .replace("%left = bf16[2,3]{0,1}", "%left = bf16[1,3]{0,1}")
        .replace("%middle = bf16[1,3]{0,1}", "%middle = bf16[2,3]{0,1}")
        .replace(
            "%left_result = bf16[2,2]{1,0} slice(%projection), slice={[0:2], [0:2]}",
            "%left_result = bf16[2,1]{1,0} slice(%projection), slice={[0:2], [0:1]}",
        )
        .replace(
            "%middle_result = bf16[2,1]{1,0} slice(%projection), slice={[0:2], [2:3]}",
            "%middle_result = bf16[2,2]{1,0} slice(%projection), slice={[0:2], [1:3]}",
        )
        .replace(
            "%middle_broadcast = bf16[2,2]{1,0} broadcast(%middle_result), dimensions={0}",
            "%left_broadcast = bf16[2,2]{1,0} broadcast(%left_result), dimensions={0}",
        )
        .replace("multiply(%left_result, %middle_broadcast)", "multiply(%left_broadcast, %middle_result)")
    )
    original = plan_demand_sliced_contract_ffi(_SYNTHETIC, target_prefix=_TARGET_PREFIX)
    changed = plan_demand_sliced_contract_ffi(mutated, target_prefix=_TARGET_PREFIX)

    assert original.families[0].template == changed.families[0].template
    assert original.families[0].preferred_mode == changed.families[0].preferred_mode
    assert original.families[0].semantic_digest != changed.families[0].semantic_digest
    assert original.families[0].input_shapes != changed.families[0].input_shapes
    assert original.families[0].output_shapes != changed.families[0].output_shapes
    assert original.calls[0].cross_partition_maps[-1].source_partitions == (0, 1)
    assert changed.calls[0].cross_partition_maps[-1].source_partitions == (0, 1)
    transformed = replace_demand_sliced_contracts_with_custom_calls(mutated, changed)
    audit = audit_demand_sliced_contract_replacement(mutated, transformed, changed)
    assert audit.removed_contract_count == 1
    assert audit.removed_contract_flops == 72
