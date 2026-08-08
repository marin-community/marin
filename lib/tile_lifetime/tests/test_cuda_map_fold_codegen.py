# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from pathlib import Path

import pytest

from tile_lifetime.cuda_map_fold_codegen import (
    CudaMapFoldProgram,
    evaluate_scalar_expression,
    render_cuda_map_fold_include,
    shuttle_map_fold_program,
    verify_cuda_map_fold_include,
)
from tile_lifetime.tensor_program import ScalarExpressionKind, scalar_binary, scalar_input


def _replace_function(program: CudaMapFoldProgram, symbol: str, expression) -> CudaMapFoldProgram:
    functions = tuple(
        replace(function, expression=expression) if function.symbol == symbol else function
        for function in program.functions
    )
    return replace(program, functions=functions)


def _function(program: CudaMapFoldProgram, symbol: str):
    return next(function for function in program.functions if function.symbol == symbol)


def test_checked_in_cuda_include_matches_selected_scalar_ir() -> None:
    path = Path(__file__).resolve().parents[1] / "backends" / "sm100" / "mok_gmm_probe" / "generated_map_fold.inc"

    verify_cuda_map_fold_include(path, shuttle_map_fold_program())


def test_pair_map_mutation_changes_generated_cuda_without_a_new_skeleton() -> None:
    program = shuttle_map_fold_program()
    product = scalar_binary(
        ScalarExpressionKind.MULTIPLY,
        scalar_input("left"),
        scalar_input("right"),
    )

    mutated = _replace_function(program, "generated_pair_map", product)

    assert mutated.fingerprint != program.fingerprint
    assert render_cuda_map_fold_include(mutated) != render_cuda_map_fold_include(program)
    assert evaluate_scalar_expression(
        _function(mutated, "generated_pair_map").expression,
        {"left": 2.0, "right": 3.0},
    ) == pytest.approx(6.0)
    assert "return (left * right);" in render_cuda_map_fold_include(mutated)


def test_fold_mutation_changes_generated_cuda_and_reference_semantics() -> None:
    program = shuttle_map_fold_program()
    subtract = scalar_binary(
        ScalarExpressionKind.SUBTRACT,
        scalar_input("state"),
        scalar_input("contribution"),
    )

    mutated = _replace_function(program, "generated_fold_update", subtract)

    assert mutated.fingerprint != program.fingerprint
    source = render_cuda_map_fold_include(mutated)
    assert source != render_cuda_map_fold_include(program)
    assert "return __fsub_rn(state, contribution);" in source
    assert evaluate_scalar_expression(
        _function(mutated, "generated_fold_update").expression,
        {"state": 5.0, "contribution": 2.0},
    ) == pytest.approx(3.0)


def test_include_verification_rejects_drift(tmp_path: Path) -> None:
    path = tmp_path / "generated.inc"
    path.write_text("handwritten body\n")

    with pytest.raises(ValueError, match="does not match program"):
        verify_cuda_map_fold_include(path, shuttle_map_fold_program())
