# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import re
from dataclasses import replace
from pathlib import Path

from tile_lifetime.cast_scalar_program import (
    CastScalarExpression,
    CastScalarKind,
    CastScalarProgram,
    generate_cuda_scalar_body,
)
from tile_lifetime.xla_low_rank_gated_product import recover_low_rank_gated_product_training
from tile_lifetime.xla_normalized_exp_contract_forward import (
    plan_normalized_exp_contract_forward_hlo_replacement,
    replace_normalized_exp_contract_forward_hlo_region_with_custom_call,
)
from tile_lifetime.xla_normalized_exp_contract_reverse import (
    plan_normalized_exp_contract_reverse_hlo_replacement,
    replace_normalized_exp_contract_reverse_hlo_region_with_custom_call,
)

_ARTIFACT = (
    Path(__file__).parents[1] / "benchmarks/artifacts/xla_grug_shared_map_h100_narrowed_unaccepted_da49b94c_v0/"
    "transformed-gpu-pre-scheduler-hlo.txt.gz"
)


def _current_training_hlo() -> str:
    with gzip.open(_ARTIFACT, "rt") as handle:
        hlo = handle.read()
    forward = plan_normalized_exp_contract_forward_hlo_replacement(hlo)
    hlo = replace_normalized_exp_contract_forward_hlo_region_with_custom_call(
        hlo,
        forward,
        target="shuttle.test.normalized_exp.forward",
    )
    reverse = plan_normalized_exp_contract_reverse_hlo_replacement(hlo)
    return replace_normalized_exp_contract_reverse_hlo_region_with_custom_call(
        hlo,
        reverse,
        target="shuttle.test.normalized_exp.reverse",
    )


def test_natural_training_hlo_recovers_repeated_low_rank_gated_product_families() -> None:
    report = recover_low_rank_gated_product_training(_current_training_hlo())

    group_counts = sorted(
        len(tuple(plan for plan in report.forward_realizations if plan.parameter_origins == origins))
        for origins in {plan.parameter_origins for plan in report.forward_realizations}
    )
    assert len(report.forward_realizations) == 6
    assert len(report.reverse_families) == 4
    assert group_counts == [1, 1, 2, 2]
    assert len({plan.hidden_map.digest for plan in report.forward_realizations}) == 1
    assert len({plan.output_map.digest for plan in report.forward_realizations}) == 1
    assert len({plan.up_input_map.digest for plan in report.reverse_families}) == 1
    assert len({plan.hidden_vjp_map.digest for plan in report.reverse_families}) == 1
    assert len({plan.residual_vjp_map.digest for plan in report.reverse_families}) == 1
    assert report.live_contract_count == 52
    assert report.live_contract_flops == 2_232_320
    assert report.owned_contract_count == 28
    assert report.owned_contract_flops == 1_835_008


def test_low_rank_gated_product_recovery_ignores_frontend_metadata() -> None:
    hlo = _current_training_hlo()
    without_metadata = re.sub(r", metadata=\{.*$", "", hlo, flags=re.MULTILINE)

    assert recover_low_rank_gated_product_training(without_metadata) == recover_low_rank_gated_product_training(hlo)


def test_low_rank_gated_product_plan_preserves_bf16_and_collective_boundaries() -> None:
    report = recover_low_rank_gated_product_training(_current_training_hlo())

    for forward in report.forward_realizations:
        assert forward.input.shape.startswith("bf16[")
        assert forward.down_contract.output.shape.startswith("bf16[")
        assert forward.hidden.shape.startswith("bf16[")
        assert forward.up_contract.output.shape.startswith("bf16[")
        assert forward.output.shape.startswith("bf16[")
        assert forward.hidden_map.numerical_policy.value == "source_ordered"
        assert forward.output_map.numerical_policy.value == "source_ordered"
    for reverse in report.reverse_families:
        assert reverse.up_input_adjoint.output.shape.startswith("bf16[")
        assert reverse.down_input_adjoint.output.shape.startswith("bf16[")
        assert reverse.input_adjoint.shape.startswith("bf16[")
        assert reverse.down_weight_adjoint.output.shape.startswith("bf16[")
        assert reverse.up_weight_adjoint.output.shape.startswith("bf16[")
        assert reverse.upstream_collectives


def test_low_rank_gated_product_hidden_map_mutates_through_same_scalar_generator() -> None:
    plan = recover_low_rank_gated_product_training(_current_training_hlo()).forward_realizations[0]
    hidden_input = plan.hidden_map.inputs[0]
    mutated_map = CastScalarProgram(
        CastScalarExpression(
            kind=CastScalarKind.TANH,
            dtype=hidden_input.dtype,
            operands=(hidden_input,),
        )
    )
    mutated_plan = replace(plan, hidden_map=mutated_map)

    original_source = generate_cuda_scalar_body(plan.hidden_map, symbol="generated_hidden_map")
    mutated_source = generate_cuda_scalar_body(mutated_plan.hidden_map, symbol="generated_hidden_map")
    assert mutated_plan.down_contract == plan.down_contract
    assert mutated_plan.up_contract == plan.up_contract
    assert mutated_source.semantic_digest != original_source.semantic_digest
    assert "tanhf" in mutated_source.source
