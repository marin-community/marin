# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from tile_lifetime import (
    RowNormalizationSavePolicy,
    RowStatisticKind,
    RowStatisticScalePlacement,
    build_row_normalized_contract_program,
    compile_row_normalization_training,
    execute_tensor_program,
)
from tile_lifetime.plan import NumericalEquivalence


def _reference(
    x: np.ndarray,
    gamma: np.ndarray,
    weight: np.ndarray,
    output_cotangent: np.ndarray,
    statistic_kind: RowStatisticKind,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if statistic_kind is RowStatisticKind.CENTERED_SECOND_MOMENT:
        local = x - np.mean(x, axis=1, keepdims=True)
    else:
        local = x
    inverse = np.reciprocal(np.sqrt(np.mean(local * local, axis=1, keepdims=True) + epsilon))
    standardized = local * inverse
    scaled = standardized * gamma
    output = scaled @ weight
    projected = output_cotangent @ weight.T
    local_cotangent = projected * gamma
    correlation = np.mean(local_cotangent * standardized, axis=1, keepdims=True)
    centered_cotangent = local_cotangent - standardized * correlation
    if statistic_kind is RowStatisticKind.CENTERED_SECOND_MOMENT:
        centered_cotangent -= np.mean(local_cotangent, axis=1, keepdims=True)
    input_gradient = inverse * centered_cotangent
    gamma_gradient = np.sum(projected * standardized, axis=0)
    weight_gradient = scaled.T @ output_cotangent
    return output, input_gradient, gamma_gradient, weight_gradient, inverse[:, 0]


@pytest.mark.parametrize("statistic_kind", list(RowStatisticKind))
@pytest.mark.parametrize("save_policy", list(RowNormalizationSavePolicy))
def test_generated_row_statistic_backward_matches_independent_reference(
    statistic_kind: RowStatisticKind,
    save_policy: RowNormalizationSavePolicy,
) -> None:
    epsilon = 3e-4
    source = build_row_normalized_contract_program(
        rows=4,
        hidden=7,
        features=5,
        statistic_kind=statistic_kind,
        epsilon=epsilon,
    )
    plan = compile_row_normalization_training(
        source,
        save_policy=save_policy,
        scale_placement=RowStatisticScalePlacement.SOURCE_ORDERED_PREPARATION,
    )
    rng = np.random.default_rng(31)
    x = rng.normal(size=(4, 7)).astype(np.float32)
    gamma = rng.normal(size=(7,)).astype(np.float32)
    weight = rng.normal(size=(7, 5)).astype(np.float32)
    output_cotangent = rng.normal(size=(4, 5)).astype(np.float32)
    output, expected_x, expected_gamma, expected_weight, inverse = _reference(
        x,
        gamma,
        weight,
        output_cotangent,
        statistic_kind,
        epsilon,
    )
    standardized = (
        x - np.mean(x, axis=1, keepdims=True) if statistic_kind is RowStatisticKind.CENTERED_SECOND_MOMENT else x
    ) * inverse[:, None]
    backward_inputs = {
        "feature_scale": gamma,
        "weight": weight,
        "cotangent.output": output_cotangent,
    }
    if save_policy is RowNormalizationSavePolicy.SAVE_NORMALIZED:
        backward_inputs.update({"standardized": standardized, "inverse_scale": inverse})
    elif save_policy is RowNormalizationSavePolicy.SAVE_INPUT_AND_INVERSE:
        backward_inputs.update({"input": x, "inverse_scale": inverse})
    else:
        backward_inputs["input"] = x

    actual_forward = execute_tensor_program(
        source,
        {"input": x, "feature_scale": gamma, "weight": weight},
    )["output"]
    automatic = execute_tensor_program(
        plan.automatic_adjoint.program,
        {
            "input": x,
            "feature_scale": gamma,
            "weight": weight,
            "cotangent.output": output_cotangent,
        },
    )
    actual_backward = execute_tensor_program(plan.backward, backward_inputs)

    np.testing.assert_allclose(actual_forward, output, rtol=3e-6, atol=3e-6)
    automatic_x, automatic_gamma, automatic_weight = plan.automatic_adjoint.input_gradients
    np.testing.assert_allclose(automatic[automatic_x.name], expected_x, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(automatic[automatic_gamma.name], expected_gamma, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(automatic[automatic_weight.name], expected_weight, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual_backward["cotangent.input"], expected_x, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual_backward["cotangent.feature_scale"], expected_gamma, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual_backward["cotangent.weight"], expected_weight, rtol=2e-5, atol=2e-5)


def test_save_policies_make_activation_recomputation_visible() -> None:
    source = build_row_normalized_contract_program(
        rows=8,
        hidden=16,
        features=24,
        statistic_kind=RowStatisticKind.UNCENTERED_SECOND_MOMENT,
    )
    saved = compile_row_normalization_training(
        source,
        save_policy=RowNormalizationSavePolicy.SAVE_NORMALIZED,
        scale_placement=RowStatisticScalePlacement.SOURCE_ORDERED_PREPARATION,
    )
    input_and_inverse = compile_row_normalization_training(
        source,
        save_policy=RowNormalizationSavePolicy.SAVE_INPUT_AND_INVERSE,
        scale_placement=RowStatisticScalePlacement.SOURCE_ORDERED_PREPARATION,
    )
    recompute = compile_row_normalization_training(
        source,
        save_policy=RowNormalizationSavePolicy.RECOMPUTE_STATISTIC,
        scale_placement=RowStatisticScalePlacement.SOURCE_ORDERED_PREPARATION,
    )

    assert saved.saved_values == ("standardized", "inverse_scale")
    assert saved.recomputed_values == ()
    assert input_and_inverse.saved_values == ("input", "inverse_scale")
    assert input_and_inverse.recomputed_values == ("standardized",)
    assert recompute.saved_values == ("input",)
    assert recompute.recomputed_values == (
        "squared",
        "sum_square",
        "mean_square",
        "inverse_scale",
        "standardized",
    )
    assert len(saved.contracts) == len(input_and_inverse.contracts) == len(recompute.contracts) == 2
    assert len(saved.folds) < len(recompute.folds)


def test_source_ordered_and_delayed_scale_are_explicit_physical_candidates() -> None:
    source = build_row_normalized_contract_program(
        rows=128,
        hidden=512,
        features=768,
        statistic_kind=RowStatisticKind.UNCENTERED_SECOND_MOMENT,
    )
    source_ordered = compile_row_normalization_training(
        source,
        save_policy=RowNormalizationSavePolicy.SAVE_INPUT_AND_INVERSE,
        scale_placement=RowStatisticScalePlacement.SOURCE_ORDERED_PREPARATION,
    )
    delayed = compile_row_normalization_training(
        source,
        save_policy=RowNormalizationSavePolicy.SAVE_INPUT_AND_INVERSE,
        scale_placement=RowStatisticScalePlacement.REAL_ALGEBRA_EQUIVALENT_FINALIZATION,
    )

    assert source_ordered.numerical_equivalence is NumericalEquivalence.BITWISE_EXACT
    assert delayed.numerical_equivalence is NumericalEquivalence.ALGEBRAICALLY_EXACT
    assert tuple(attachment.site.value for attachment in source_ordered.forward_contract.prologue) == (
        "gemm_prologue",
        "gemm_prologue",
    )
    assert source_ordered.forward_contract.epilogue == ()
    assert len(delayed.forward_contract.prologue) == 1
    assert tuple(attachment.site.value for attachment in delayed.forward_contract.epilogue) == ("gemm_epilogue",)
    assert source_ordered.backward == delayed.backward


def test_centering_mutation_reuses_contract_map_fold_lowering() -> None:
    uncentered = compile_row_normalization_training(
        build_row_normalized_contract_program(
            rows=16,
            hidden=32,
            features=48,
            statistic_kind=RowStatisticKind.UNCENTERED_SECOND_MOMENT,
        ),
        save_policy=RowNormalizationSavePolicy.SAVE_NORMALIZED,
        scale_placement=RowStatisticScalePlacement.SOURCE_ORDERED_PREPARATION,
    )
    centered = compile_row_normalization_training(
        build_row_normalized_contract_program(
            rows=16,
            hidden=32,
            features=48,
            statistic_kind=RowStatisticKind.CENTERED_SECOND_MOMENT,
        ),
        save_policy=RowNormalizationSavePolicy.SAVE_NORMALIZED,
        scale_placement=RowStatisticScalePlacement.SOURCE_ORDERED_PREPARATION,
    )

    assert tuple(contract.skeleton.shape for contract in uncentered.contracts) == tuple(
        contract.skeleton.shape for contract in centered.contracts
    )
    assert len(centered.folds) == len(uncentered.folds) + 1
    assert centered.folds[-1].reducer.value == "sum"
    assert centered.maps[-1].expression != uncentered.maps[-1].expression
