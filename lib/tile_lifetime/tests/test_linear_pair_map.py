# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import pytest

from tile_lifetime import (
    PairMapSavePolicy,
    build_linear_pair_map_program,
    compile_gemm_program,
    compile_linear_pair_map_training,
    execute_tensor_program,
    pair_silu_product_expression,
    pair_tanh_product_expression,
)
from tile_lifetime.cuda_map_fold_codegen import evaluate_scalar_expression
from tile_lifetime.cute_pair_map_codegen import generate_cute_pair_map_vjp
from tile_lifetime.quack_gemm_codegen import generate_quack_gemm


def test_natural_linear_pair_map_matches_independent_reference() -> None:
    source = build_linear_pair_map_program(
        rows=3,
        reduction=5,
        features=4,
        pair_expression=pair_silu_product_expression(),
    )
    rng = np.random.default_rng(19)
    activation = rng.normal(size=(3, 5)).astype(np.float32)
    left_weight = rng.normal(size=(5, 4)).astype(np.float32)
    right_weight = rng.normal(size=(5, 4)).astype(np.float32)

    actual = execute_tensor_program(
        source,
        {
            "activation": activation,
            "left_weight": left_weight,
            "right_weight": right_weight,
        },
    )["pair_map_output"]

    left = activation @ left_weight
    right = activation @ right_weight
    expected = left / (1.0 + np.exp(-left)) * right
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_saved_preactivation_is_a_generated_dual_output_epilogue() -> None:
    source = build_linear_pair_map_program(
        rows=128,
        reduction=256,
        features=512,
        pair_expression=pair_silu_product_expression(),
    )
    training = compile_linear_pair_map_training(
        source,
        save_policy=PairMapSavePolicy.SAVE_PREACTIVATION,
    )

    generated = generate_quack_gemm(compile_gemm_program(training.forward))

    assert generated.writes_main_output
    assert tuple(output.destination for output in generated.outputs) == ("pair_map_output",)
    assert "'D': acc" in generated.source
    assert "'output_0': value_1" in generated.source
    assert "swiglu" not in generated.source.lower()
    assert training.recompute is None


def test_pair_map_vjp_and_semantic_mutation_use_the_same_physical_skeleton() -> None:
    silu = compile_linear_pair_map_training(
        build_linear_pair_map_program(
            rows=8,
            reduction=16,
            features=32,
            pair_expression=pair_silu_product_expression(),
        ),
        save_policy=PairMapSavePolicy.SAVE_PREACTIVATION,
    )
    tanh = compile_linear_pair_map_training(
        build_linear_pair_map_program(
            rows=8,
            reduction=16,
            features=32,
            pair_expression=pair_tanh_product_expression(),
        ),
        save_policy=PairMapSavePolicy.SAVE_PREACTIVATION,
    )
    silu_source = generate_cute_pair_map_vjp(silu.pair_vjp)
    tanh_source = generate_cute_pair_map_vjp(tanh.pair_vjp)

    inputs = {"pair.left": 0.7, "pair.right": -1.2, "cotangent": 0.3}
    sigmoid = 1.0 / (1.0 + math.exp(-inputs["pair.left"]))
    expected_silu_left = (
        inputs["cotangent"] * inputs["pair.right"] * sigmoid * (1.0 + inputs["pair.left"] * (1.0 - sigmoid))
    )
    expected_tanh_left = inputs["cotangent"] * inputs["pair.right"] * (1.0 - math.tanh(inputs["pair.left"]) ** 2)
    assert evaluate_scalar_expression(silu.pair_vjp.left, inputs) == pytest.approx(expected_silu_left)
    assert evaluate_scalar_expression(tanh.pair_vjp.left, inputs) == pytest.approx(expected_tanh_left)
    assert silu_source.digest != tanh_source.digest
    assert "cute.exp" in silu_source.source
    assert "cute.tanh" in tanh_source.source
    assert "swiglu" not in silu_source.source.lower()
    assert silu.input_gradient.shape == tanh.input_gradient.shape
    assert silu.weight_gradient.shape == tanh.weight_gradient.shape


def test_recompute_policy_adds_a_plain_generated_contract() -> None:
    training = compile_linear_pair_map_training(
        build_linear_pair_map_program(
            rows=32,
            reduction=64,
            features=128,
            pair_expression=pair_silu_product_expression(),
        ),
        save_policy=PairMapSavePolicy.RECOMPUTE_PREACTIVATION,
    )

    forward = generate_quack_gemm(compile_gemm_program(training.forward))

    assert not forward.writes_main_output
    assert training.recompute is not None
    assert training.recompute.epilogue == ()
    assert training.recompute.shape == (32, 256, 64)
