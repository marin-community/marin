# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
from pathlib import Path

import pytest

from tile_lifetime.cuda_normalized_exp_contract_reverse_codegen import (
    generate_cuda_normalized_exp_contract_reverse_ffi,
)
from tile_lifetime.tensor_program import ScalarExpressionKind, scalar_binary, scalar_constant, scalar_input, scalar_unary
from tile_lifetime.xla_normalized_exp_contract_reverse import plan_normalized_exp_contract_reverse_hlo_replacement

_HLO = (
    Path(__file__).parents[1] / "benchmarks/artifacts/xla_grug_shared_map_h100_narrowed_unaccepted_da49b94c_v0/"
    "original-gpu-pre-scheduler-hlo.txt.gz"
)
_TARGET = "shuttle.generic.normalized_exp_contract_reverse.test"


def _plan():
    return plan_normalized_exp_contract_reverse_hlo_replacement(gzip.decompress(_HLO.read_bytes()).decode())


def test_generated_normalized_exp_reverse_owns_generic_contract_map_fold_body() -> None:
    generated = generate_cuda_normalized_exp_contract_reverse_ffi(_plan(), target=_TARGET)

    assert generated.rows == 8
    assert generated.reduction == 32
    assert generated.fold_extent == 128
    assert generated.shared_bytes == 2048
    assert generated.handler_symbol == "shuttle_generic_normalized_exp_contract_reverse_test"
    assert "generated_score_map(raw_score)" in generated.source
    assert "generated_score_derivative()" in generated.source
    assert "fold_validity[fold]" in generated.source
    assert "selected_indices[row] == fold" in generated.source
    assert "__float2bfloat16_rn(score_accumulator)" in generated.source
    assert "__float2bfloat16_rn(mapped_cotangent)" in generated.source
    assert "cublas" not in generated.source.lower()
    assert "softmax" not in generated.source.lower()
    assert "cross_entropy" not in generated.source.lower()


def test_score_map_mutation_regenerates_same_physical_family() -> None:
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
    baseline = generate_cuda_normalized_exp_contract_reverse_ffi(_plan(), target=_TARGET)
    mutated = generate_cuda_normalized_exp_contract_reverse_ffi(
        _plan(),
        target=_TARGET,
        score_expression=soft_cap,
    )

    assert baseline.rows == mutated.rows
    assert baseline.reduction == mutated.reduction
    assert baseline.fold_extent == mutated.fold_extent
    assert baseline.handler_symbol == mutated.handler_symbol
    assert baseline.semantic_digest != mutated.semantic_digest
    assert baseline.source_digest != mutated.source_digest
    assert "tanhf" in mutated.source
    assert "generated_score_derivative(raw_score)" in mutated.source


def test_generated_normalized_exp_reverse_rejects_non_scalar_score_map() -> None:
    with pytest.raises(ValueError, match="exactly one raw_score"):
        generate_cuda_normalized_exp_contract_reverse_ffi(
            _plan(),
            target=_TARGET,
            score_expression=scalar_binary(
                ScalarExpressionKind.ADD,
                scalar_input("raw_score"),
                scalar_input("other"),
            ),
        )
