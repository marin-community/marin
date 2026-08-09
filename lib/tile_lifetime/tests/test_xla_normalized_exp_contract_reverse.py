# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import re
from pathlib import Path

from tile_lifetime.xla_normalized_exp_contract_reverse import (
    audit_normalized_exp_contract_reverse_hlo_replacement,
    plan_normalized_exp_contract_reverse_hlo_replacement,
    recover_normalized_exp_contract_reverse_hlo_regions,
    replace_normalized_exp_contract_reverse_hlo_region_with_custom_call,
)

_TARGET = "shuttle.generic.normalized_exp_contract_reverse.test"

_HLO = (
    Path(__file__).parents[1] / "benchmarks/artifacts/xla_grug_shared_map_h100_narrowed_unaccepted_da49b94c_v0/"
    "original-gpu-pre-scheduler-hlo.txt.gz"
)


def _hlo() -> str:
    return gzip.decompress(_HLO.read_bytes()).decode()


def test_natural_grug_hlo_recovers_normalized_exp_contract_reverse_without_names() -> None:
    hlo = _hlo()
    report = recover_normalized_exp_contract_reverse_hlo_regions(hlo)
    renamed = re.sub(r'op_name="[^"]*"', 'op_name="unrelated_diagnostic_label"', hlo)
    renamed_report = recover_normalized_exp_contract_reverse_hlo_regions(renamed)

    assert len(report.regions) == 1
    region = report.regions[0]
    assert region.score_contract.output_shape == "bf16[8,128]{1,0}"
    assert region.score_contract.dimensions.lhs_contracting == (1,)
    assert region.score_contract.dimensions.rhs_contracting == (0,)
    assert region.input_reverse_contract.output_shape == "bf16[8,32]{1,0}"
    assert region.input_reverse_contract.dimensions.lhs_contracting == (1,)
    assert region.input_reverse_contract.dimensions.rhs_contracting == (1,)
    assert region.operand_reverse_contract.output_shape == "bf16[32,128]{1,0}"
    assert region.operand_reverse_contract.dimensions.lhs_contracting == (0,)
    assert region.operand_reverse_contract.dimensions.rhs_contracting == (0,)
    assert region.saved_state.shape == "f32[8]{0}"
    assert region.fold_validity.shape == "pred[128]{0}"
    assert region.row_cotangent.shape == "f32[8]{0}"
    assert region.selected_mask.shape == "bf16[8,128]{1,0}"
    assert region.selected_indices.shape == "s32[8]{0}"
    assert region.row_validity.shape == "pred[8]{0}"
    assert region.score_cotangent.shape == "bf16[8,128]{1,0}"
    assert region.semantic_digest == renamed_report.regions[0].semantic_digest


def test_normalized_exp_contract_reverse_shape_mutation_changes_semantics() -> None:
    hlo = _hlo()
    baseline = recover_normalized_exp_contract_reverse_hlo_regions(hlo).regions[0]
    mutated_hlo = hlo.replace("[8,128]", "[8,96]").replace("[32,128]", "[32,96]")
    mutated = recover_normalized_exp_contract_reverse_hlo_regions(mutated_hlo).regions[0]

    assert mutated.score_contract.output_shape == "bf16[8,96]{1,0}"
    assert mutated.operand_reverse_contract.output_shape == "bf16[32,96]{1,0}"
    assert mutated.semantic_digest != baseline.semantic_digest


def test_normalized_exp_contract_reverse_rejects_incomplete_dataflow() -> None:
    hlo = _hlo().replace("%exp.48 =", "%unrelated.48 =", 1).replace("%exp.48)", "%unrelated.48)", 1)
    report = recover_normalized_exp_contract_reverse_hlo_regions(hlo)

    assert not report.regions


def test_natural_grug_normalized_exp_reverse_region_roundtrips_through_typed_ffi() -> None:
    hlo = _hlo()
    plan = plan_normalized_exp_contract_reverse_hlo_replacement(hlo)
    transformed = replace_normalized_exp_contract_reverse_hlo_region_with_custom_call(
        hlo,
        plan,
        target=_TARGET,
    )
    audit = audit_normalized_exp_contract_reverse_hlo_replacement(
        hlo,
        transformed,
        plan,
        target=_TARGET,
    )

    assert tuple(value.shape for value in plan.inputs) == (
        "bf16[8,32]{1,0}",
        "bf16[32,128]{1,0}",
        "f32[8]{0}",
        "pred[128]{0}",
        "f32[8]{0}",
        "s32[8]{0}",
        "pred[8]{0}",
    )
    assert tuple(value.shape for value in plan.outputs) == (
        "bf16[8,32]{1,0}",
        "bf16[32,128]{1,0}",
    )
    assert audit.call_instruction == "shuttle.generated.normalized_exp_contract_reverse"
    assert audit.inputs == tuple(value.instruction for value in plan.inputs)
    assert audit.outputs == (
        "shuttle.generated.normalized_exp_contract_reverse.output.0",
        "shuttle.generated.normalized_exp_contract_reverse.output.1",
    )
    assert audit.rewired_external_users == plan.external_users
    assert audit.dead_instructions == plan.region.internal_instructions
    assert audit.placement_paths == (
        ("shuttle.generated.normalized_exp_contract_reverse.output.0", ("reshape.198",), "psum.48"),
        ("shuttle.generated.normalized_exp_contract_reverse.output.1", ("slice.87",), "psum.49"),
    )
    assert (
        "%reshape.198 = bf16[2,4,32]{2,1,0} reshape(%shuttle.generated.normalized_exp_contract_reverse.output.0)"
        in transformed
    )
    assert (
        "%slice.87 = bf16[32,64]{1,0} slice(%shuttle.generated.normalized_exp_contract_reverse.output.1)" in transformed
    )
