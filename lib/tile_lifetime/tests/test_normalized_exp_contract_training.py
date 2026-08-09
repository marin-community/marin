# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from tile_lifetime.normalized_exp_contract_training import (
    build_normalized_exp_contract_training_program,
    execute_normalized_exp_contract_training,
    tanh_soft_cap_score_expression,
)
from tile_lifetime.tensor_program import ContractPrimitive, FoldPrimitive, MapPrimitive


def _inputs():
    generator = np.random.default_rng(17)
    return {
        "left": generator.normal(size=(3, 4)).astype(np.float32),
        "right": generator.normal(size=(4, 5)).astype(np.float32),
        "selected_indices": np.asarray([1, 3, 0], dtype=np.int32),
        "row_validity": np.asarray([True, True, False]),
        "fold_validity": np.asarray([True, True, True, True, False]),
        "output_cotangent": generator.normal(size=(3,)).astype(np.float32),
        "state_cotangent": generator.normal(size=(3,)).astype(np.float32),
    }


def _reference(values, *, cap: float | None):
    raw = values["left"] @ values["right"]
    if cap is None:
        score = raw
        derivative = np.ones_like(raw)
    else:
        normalized = raw / np.float32(cap)
        score = np.float32(cap) * np.tanh(normalized)
        derivative = 1.0 - np.tanh(normalized) ** 2
    restricted = np.where(values["fold_validity"][None, :], score, -np.inf)
    maximum = np.max(restricted, axis=1)
    exponential = np.exp(restricted - maximum[:, None])
    log_normalizer = np.log(np.sum(exponential, axis=1)) + maximum
    selected = score[np.arange(score.shape[0]), values["selected_indices"]]
    row_validity = values["row_validity"]
    output = np.where(row_validity, log_normalizer - selected, 0.0)
    visible_log_normalizer = np.where(row_validity, log_normalizer, 0.0)
    effective_output_cotangent = np.where(row_validity, values["output_cotangent"], 0.0)
    effective_state_cotangent = np.where(row_validity, values["state_cotangent"], 0.0)
    probability = np.exp(restricted - log_normalizer[:, None])
    score_cotangent = probability * (effective_output_cotangent + effective_state_cotangent)[:, None]
    score_cotangent[np.arange(score.shape[0]), values["selected_indices"]] -= effective_output_cotangent
    score_cotangent *= derivative
    return (
        output,
        visible_log_normalizer,
        score_cotangent @ values["right"].T,
        values["left"].T @ score_cotangent,
        score_cotangent,
    )


@pytest.mark.parametrize("cap", [None, 1.3])
def test_generic_contract_normalized_exp_forward_and_reverse_match_reference(cap: float | None) -> None:
    score_expression = tanh_soft_cap_score_expression("score.raw", cap) if cap is not None else None
    program = build_normalized_exp_contract_training_program(
        rows=3,
        reduction=4,
        fold_extent=5,
        score_expression=score_expression,
    )
    values = _inputs()
    actual = execute_normalized_exp_contract_training(program, **values)
    expected = _reference(values, cap=cap)

    np.testing.assert_allclose(actual.output, expected[0], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual.log_normalizer, expected[1], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual.input_cotangent, expected[2], rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(actual.operand_cotangent, expected[3], rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(actual.score_cotangent, expected[4], rtol=3e-6, atol=3e-6)

    assert sum(isinstance(operation, ContractPrimitive) for operation in program.forward.operations) == 1
    assert sum(isinstance(operation, FoldPrimitive) for operation in program.forward.operations) == 2
    assert sum(isinstance(operation, MapPrimitive) for operation in program.reverse.operations) == 1
    assert sum(isinstance(operation, ContractPrimitive) for operation in program.reverse.operations) == 2
    assert all(
        "cross" not in operation.name and "loss" not in operation.name for operation in program.forward.operations
    )


def test_score_map_mutation_changes_scalar_reverse_without_changing_physical_family() -> None:
    identity = build_normalized_exp_contract_training_program(rows=3, reduction=4, fold_extent=5)
    capped = build_normalized_exp_contract_training_program(
        rows=3,
        reduction=4,
        fold_extent=5,
        score_expression=tanh_soft_cap_score_expression("score.raw", 1.7),
    )

    assert identity.score_contract == capped.score_contract
    assert identity.input_reverse_contract.output.shape == capped.input_reverse_contract.output.shape
    assert identity.operand_reverse_contract.output.shape == capped.operand_reverse_contract.output.shape
    assert identity.score_map.expression != capped.score_map.expression
    assert identity.reverse_score_map.expression != capped.reverse_score_map.expression


def test_indexed_selection_rejects_restricted_or_out_of_domain_coordinates() -> None:
    program = build_normalized_exp_contract_training_program(rows=3, reduction=4, fold_extent=5)
    values = _inputs()
    values["selected_indices"] = np.asarray([1, 4, 0], dtype=np.int32)
    with pytest.raises(ValueError, match="restricted Fold coordinate"):
        execute_normalized_exp_contract_training(program, **values)

    values["selected_indices"] = np.asarray([1, 5, 0], dtype=np.int32)
    with pytest.raises(ValueError, match="out-of-domain selected Fold coordinate"):
        execute_normalized_exp_contract_training(program, **values)
