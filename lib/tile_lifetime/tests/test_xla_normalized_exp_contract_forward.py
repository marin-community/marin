# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import re
from pathlib import Path

from tile_lifetime.xla_normalized_exp_contract_forward import (
    audit_normalized_exp_contract_forward_hlo_replacement,
    plan_normalized_exp_contract_forward_hlo_replacement,
    recover_normalized_exp_contract_forward_hlo_region,
    replace_normalized_exp_contract_forward_hlo_region_with_custom_call,
)
from tile_lifetime.xla_normalized_exp_contract_reverse import (
    audit_normalized_exp_contract_reverse_hlo_replacement,
    plan_normalized_exp_contract_reverse_hlo_replacement,
    replace_normalized_exp_contract_reverse_hlo_region_with_custom_call,
)

_HLO = (
    Path(__file__).parents[1] / "benchmarks/artifacts/xla_grug_shared_map_h100_narrowed_unaccepted_da49b94c_v0/"
    "original-gpu-pre-scheduler-hlo.txt.gz"
)
_TARGET = "shuttle.generic.normalized_exp_contract_forward.test"


def _hlo() -> str:
    return gzip.decompress(_HLO.read_bytes()).decode()


def test_natural_grug_recovers_compact_normalized_exp_forward_without_names() -> None:
    hlo = _hlo()
    region = recover_normalized_exp_contract_forward_hlo_region(hlo)
    renamed = re.sub(r'op_name="[^"]*"', 'op_name="unrelated_diagnostic_label"', hlo)
    renamed_region = recover_normalized_exp_contract_forward_hlo_region(renamed)

    assert region.physical_score_contract.output_shape == "bf16[128,128]{1,0}"
    assert region.compact_score_contract.lhs.shape == "bf16[8,32]{1,0}"
    assert region.compact_score_contract.rhs.shape == "bf16[32,128]{1,0}"
    assert region.fold_validity.shape == "pred[128]{0}"
    assert region.selected_indices.shape == "s32[8]{0}"
    assert region.output.shape == "f32[8]{0}"
    assert region.saved_state.shape == "f32[8]{0}"
    assert region.semantic_digest == renamed_region.semantic_digest


def test_compact_normalized_exp_forward_roundtrips_through_typed_ffi() -> None:
    hlo = _hlo()
    plan = plan_normalized_exp_contract_forward_hlo_replacement(hlo)
    transformed = replace_normalized_exp_contract_forward_hlo_region_with_custom_call(
        hlo,
        plan,
        target=_TARGET,
    )
    audit = audit_normalized_exp_contract_forward_hlo_replacement(
        hlo,
        transformed,
        plan,
        target=_TARGET,
    )

    assert tuple(value.shape for value in plan.inputs) == (
        "bf16[8,32]{1,0}",
        "bf16[32,128]{1,0}",
        "pred[128]{0}",
        "s32[8]{0}",
    )
    assert tuple(value.shape for value in plan.outputs) == ("f32[8]{0}", "f32[8]{0}")
    assert audit.call_instruction == "shuttle.generated.normalized_exp_contract_forward"
    assert audit.inputs == tuple(value.instruction for value in plan.inputs)
    assert audit.outputs == (
        "shuttle.generated.normalized_exp_contract_forward.output.0",
        "shuttle.generated.normalized_exp_contract_forward.output.1",
    )
    assert audit.rewired_external_users == plan.external_users
    assert audit.output_users == tuple(
        (output, users) for output, (_, users) in zip(audit.outputs, plan.external_users, strict=True)
    )
    assert audit.dead_instructions
    assert set(audit.dead_instructions).isdisjoint(audit.retained_boundary_instructions)


def test_compact_forward_and_reverse_compose_as_two_generic_calls() -> None:
    hlo = _hlo()
    forward_plan = plan_normalized_exp_contract_forward_hlo_replacement(hlo)
    forward = replace_normalized_exp_contract_forward_hlo_region_with_custom_call(
        hlo,
        forward_plan,
        target=_TARGET,
    )
    reverse_plan = plan_normalized_exp_contract_reverse_hlo_replacement(forward)
    transformed = replace_normalized_exp_contract_reverse_hlo_region_with_custom_call(
        forward,
        reverse_plan,
        target="shuttle.generic.normalized_exp_contract_reverse.composed_test",
    )
    reverse_audit = audit_normalized_exp_contract_reverse_hlo_replacement(
        forward,
        transformed,
        reverse_plan,
        target="shuttle.generic.normalized_exp_contract_reverse.composed_test",
    )

    assert reverse_plan.region.saved_state.instruction == "shuttle.generated.normalized_exp_contract_forward.output.1"
    assert reverse_audit.call_instruction == "shuttle.generated.normalized_exp_contract_reverse"
