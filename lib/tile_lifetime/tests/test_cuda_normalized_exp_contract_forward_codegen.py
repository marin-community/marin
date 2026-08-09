# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
from dataclasses import replace
from pathlib import Path

import pytest

from tile_lifetime.cuda_normalized_exp_contract_forward_codegen import (
    generate_cuda_normalized_exp_contract_forward_ffi,
)
from tile_lifetime.ffi_command_buffer import audit_ffi_command_buffer_eligibility
from tile_lifetime.tensor_program import ScalarExpressionKind, scalar_binary, scalar_constant, scalar_input, scalar_unary
from tile_lifetime.xla_normalized_exp_contract_forward import plan_normalized_exp_contract_forward_hlo_replacement

_HLO = (
    Path(__file__).parents[1] / "benchmarks/artifacts/xla_grug_shared_map_h100_narrowed_unaccepted_da49b94c_v0/"
    "original-gpu-pre-scheduler-hlo.txt.gz"
)
_TARGET = "shuttle.generic.normalized_exp_contract_forward.test"


def _plan():
    return plan_normalized_exp_contract_forward_hlo_replacement(gzip.decompress(_HLO.read_bytes()).decode())


def test_generated_normalized_exp_forward_owns_compact_contract_fold_selection() -> None:
    generated = generate_cuda_normalized_exp_contract_forward_ffi(_plan(), target=_TARGET)

    assert generated.rows == 8
    assert generated.reduction == 32
    assert generated.fold_extent == 128
    assert generated.shared_bytes == 2048
    assert "generated_score_map" in generated.source
    assert "fold_validity[fold]" in generated.source
    assert "selected_indices[row]" in generated.source
    assert "__float2bfloat16_rn(accumulator)" in generated.source
    assert "cublas" not in generated.source.lower()
    assert "softmax" not in generated.source.lower()
    assert "cross_entropy" not in generated.source.lower()
    assert not generated.command_buffer_compatible
    assert "cudaPeekAtLastError" in generated.source
    assert "kCmdBufferCompatible" not in generated.source


def test_generated_normalized_exp_forward_has_capture_safe_candidate() -> None:
    generated = generate_cuda_normalized_exp_contract_forward_ffi(
        _plan(),
        target=_TARGET,
        command_buffer_compatible=True,
    )

    assert generated.command_buffer_compatible
    assert audit_ffi_command_buffer_eligibility(generated.source).eligible
    assert "cudaPeekAtLastError" not in generated.source
    assert "{ffi::Traits::kCmdBufferCompatible}" in generated.source


def test_forward_score_map_mutation_regenerates_same_physical_family() -> None:
    raw_score = scalar_input("raw_score")
    cap = scalar_constant(6.0)
    soft_cap = scalar_binary(
        ScalarExpressionKind.MULTIPLY,
        cap,
        scalar_unary(
            ScalarExpressionKind.TANH,
            scalar_binary(ScalarExpressionKind.DIVIDE, raw_score, cap),
        ),
    )
    baseline = generate_cuda_normalized_exp_contract_forward_ffi(_plan(), target=_TARGET)
    mutated = generate_cuda_normalized_exp_contract_forward_ffi(
        _plan(),
        target=_TARGET,
        score_expression=soft_cap,
    )

    assert baseline.handler_symbol == mutated.handler_symbol
    assert baseline.shared_bytes == mutated.shared_bytes
    assert baseline.semantic_digest != mutated.semantic_digest
    assert baseline.source_digest != mutated.source_digest
    assert "tanhf" in mutated.source


def test_generated_normalized_exp_forward_rejects_non_contracting_trailing_axis() -> None:
    plan = _plan()
    invalid_contract = replace(
        plan.region.compact_score_contract,
        dimensions=replace(plan.region.compact_score_contract.dimensions, lhs_contracting=(0,)),
    )
    invalid_region = replace(plan.region, compact_score_contract=invalid_contract)

    with pytest.raises(ValueError, match="trailing lhs reduction axis"):
        generate_cuda_normalized_exp_contract_forward_ffi(
            replace(plan, region=invalid_region),
            target=_TARGET,
        )
