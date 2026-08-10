# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np

from shuttle.ir import DType
from tile_lifetime import (
    ContractPrimitive,
    FoldPrimitive,
    FoldReducer,
    MapPrimitive,
    ProgramValue,
    ScalarExpressionKind,
    TensorAxis,
    TensorProgram,
    differentiate_tensor_program,
    execute_tensor_program,
    extract_backward_tensor_program,
    scalar_expression_vjp,
)
from tile_lifetime.cuda_map_fold_codegen import evaluate_scalar_expression
from tile_lifetime.tensor_program import scalar_binary, scalar_constant, scalar_input, scalar_unary


def _silu_product(gate_name: str, up_name: str):
    gate = scalar_input(gate_name)
    sigmoid = scalar_binary(
        ScalarExpressionKind.DIVIDE,
        scalar_constant(1.0),
        scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_constant(1.0),
            scalar_unary(
                ScalarExpressionKind.EXP,
                scalar_binary(ScalarExpressionKind.MULTIPLY, scalar_constant(-1.0), gate),
            ),
        ),
    )
    return scalar_binary(
        ScalarExpressionKind.MULTIPLY,
        scalar_binary(ScalarExpressionKind.MULTIPLY, gate, sigmoid),
        scalar_input(up_name),
    )


def test_scalar_ast_vjp_derives_swiglu_and_semantic_mutation() -> None:
    expression = _silu_product("gate", "up")
    gate_vjp = scalar_expression_vjp(expression, input_name="gate", cotangent_name="dz")
    up_vjp = scalar_expression_vjp(expression, input_name="up", cotangent_name="dz")
    inputs = {"gate": 0.7, "up": -1.2, "dz": 0.3}
    sigmoid = 1.0 / (1.0 + math.exp(-inputs["gate"]))
    expected_gate = inputs["dz"] * inputs["up"] * sigmoid * (1.0 + inputs["gate"] * (1.0 - sigmoid))
    expected_up = inputs["dz"] * inputs["gate"] * sigmoid
    assert math.isclose(float(evaluate_scalar_expression(gate_vjp, inputs)), expected_gate, rel_tol=1e-6)
    assert math.isclose(float(evaluate_scalar_expression(up_vjp, inputs)), expected_up, rel_tol=1e-6)

    tanh_product = scalar_binary(
        ScalarExpressionKind.MULTIPLY,
        scalar_unary(ScalarExpressionKind.TANH, scalar_input("gate")),
        scalar_input("up"),
    )
    mutated_vjp = scalar_expression_vjp(tanh_product, input_name="gate", cotangent_name="dz")
    expected_mutated = inputs["dz"] * inputs["up"] * (1.0 - math.tanh(inputs["gate"]) ** 2)
    assert math.isclose(float(evaluate_scalar_expression(mutated_vjp, inputs)), expected_mutated, rel_tol=1e-6)
    assert mutated_vjp != gate_vjp


def test_scalar_ast_log_vjp_remains_generic() -> None:
    expression = scalar_unary(ScalarExpressionKind.LOG, scalar_input("state"))
    vjp = scalar_expression_vjp(expression, input_name="state", cotangent_name="cotangent")

    assert math.isclose(
        float(evaluate_scalar_expression(vjp, {"state": 2.5, "cotangent": 0.75})),
        0.75 / 2.5,
        rel_tol=1e-7,
    )


def _rms_contract_program() -> TensorProgram:
    row = TensorAxis(1, 2, "row")
    hidden = TensorAxis(2, 3, "hidden")
    feature = TensorAxis(3, 2, "feature")
    x = ProgramValue("x", (row, hidden), DType.FP32)
    gamma = ProgramValue("gamma", (hidden,), DType.FP32)
    weight = ProgramValue("weight", (hidden, feature), DType.FP32)
    squared = ProgramValue("squared", (row, hidden), DType.FP32)
    sum_square = ProgramValue("sum_square", (row,), DType.FP32)
    mean_square = ProgramValue("mean_square", (row,), DType.FP32)
    inverse_rms = ProgramValue("inverse_rms", (row,), DType.FP32)
    normalized = ProgramValue("normalized", (row, hidden), DType.FP32)
    output = ProgramValue("output", (row, feature), DType.FP32)
    return TensorProgram(
        inputs=(x, gamma, weight),
        operations=(
            MapPrimitive(
                "square",
                (x,),
                squared,
                scalar_binary(ScalarExpressionKind.MULTIPLY, scalar_input("x"), scalar_input("x")),
            ),
            FoldPrimitive("sum square", squared, sum_square, (hidden,), FoldReducer.SUM, DType.FP32),
            MapPrimitive(
                "mean square",
                (sum_square,),
                mean_square,
                scalar_binary(ScalarExpressionKind.DIVIDE, scalar_input("sum_square"), scalar_constant(3.0)),
            ),
            MapPrimitive(
                "inverse RMS",
                (mean_square,),
                inverse_rms,
                scalar_unary(
                    ScalarExpressionKind.RSQRT,
                    scalar_binary(ScalarExpressionKind.ADD, scalar_input("mean_square"), scalar_constant(1e-5)),
                ),
            ),
            MapPrimitive(
                "normalize",
                (x, inverse_rms, gamma),
                normalized,
                scalar_binary(
                    ScalarExpressionKind.MULTIPLY,
                    scalar_binary(
                        ScalarExpressionKind.MULTIPLY,
                        scalar_input("x"),
                        scalar_input("inverse_rms"),
                    ),
                    scalar_input("gamma"),
                ),
            ),
            ContractPrimitive(
                "projection",
                (normalized, weight),
                output,
                (hidden,),
                DType.FP32,
            ),
        ),
        outputs=(output,),
    )


def test_generic_contract_map_fold_reverse_mode_matches_rmsnorm_gemm_adjoint() -> None:
    source = _rms_contract_program()
    differentiated = differentiate_tensor_program(source, with_respect_to=("x", "gamma", "weight"))
    rng = np.random.default_rng(7)
    x = rng.normal(size=(2, 3)).astype(np.float32)
    gamma = rng.normal(size=(3,)).astype(np.float32)
    weight = rng.normal(size=(3, 2)).astype(np.float32)
    dy = rng.normal(size=(2, 2)).astype(np.float32)
    actual = execute_tensor_program(
        differentiated.program,
        {"x": x, "gamma": gamma, "weight": weight, "cotangent.output": dy},
    )

    inverse_rms = np.reciprocal(np.sqrt(np.mean(x * x, axis=1, keepdims=True) + 1e-5))
    normalized = x * inverse_rms * gamma
    dn = dy @ weight.T
    expected_weight = normalized.T @ dy
    expected_gamma = np.sum(dn * x * inverse_rms, axis=0)
    row_dot = np.sum(dn * gamma * x, axis=1, keepdims=True)
    expected_x = inverse_rms * gamma * dn - x * (inverse_rms**3 / x.shape[1]) * row_dot
    x_gradient, gamma_gradient, weight_gradient = differentiated.input_gradients
    np.testing.assert_allclose(actual[x_gradient.name], expected_x, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual[gamma_gradient.name], expected_gamma, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual[weight_gradient.name], expected_weight, rtol=2e-5, atol=2e-5)

    generated_backward = differentiated.program.operations[len(source.operations) :]
    assert sum(isinstance(operation, ContractPrimitive) for operation in generated_backward) == 2
    assert any(isinstance(operation, FoldPrimitive) for operation in generated_backward)


def _linear_swiglu_program() -> TensorProgram:
    row = TensorAxis(1, 2, "row")
    hidden = TensorAxis(2, 3, "hidden")
    intermediate = TensorAxis(3, 4, "intermediate")
    x = ProgramValue("x", (row, hidden), DType.FP32)
    gate_weight = ProgramValue("gate_weight", (hidden, intermediate), DType.FP32)
    up_weight = ProgramValue("up_weight", (hidden, intermediate), DType.FP32)
    gate = ProgramValue("gate", (row, intermediate), DType.FP32)
    up = ProgramValue("up", (row, intermediate), DType.FP32)
    output = ProgramValue("output", (row, intermediate), DType.FP32)
    return TensorProgram(
        inputs=(x, gate_weight, up_weight),
        operations=(
            ContractPrimitive("gate projection", (x, gate_weight), gate, (hidden,), DType.FP32),
            ContractPrimitive("up projection", (x, up_weight), up, (hidden,), DType.FP32),
            MapPrimitive("SwiGLU scalar map", (gate, up), output, _silu_product("gate", "up")),
        ),
        outputs=(output,),
    )


def test_backward_extraction_makes_swiglu_save_or_recompute_policy_explicit() -> None:
    source = _linear_swiglu_program()
    differentiated = differentiate_tensor_program(
        source,
        with_respect_to=("x", "gate_weight", "up_weight"),
    )
    saved = extract_backward_tensor_program(differentiated, saved_values=("gate", "up"))
    recomputed = extract_backward_tensor_program(differentiated)

    assert saved.recomputed_operations == ()
    assert tuple(value.name for value in saved.saved_values) == ("gate", "up")
    assert tuple(operation.name for operation in recomputed.recomputed_operations) == (
        "gate projection",
        "up projection",
    )

    rng = np.random.default_rng(19)
    x = rng.normal(size=(2, 3)).astype(np.float32)
    gate_weight = rng.normal(size=(3, 4)).astype(np.float32)
    up_weight = rng.normal(size=(3, 4)).astype(np.float32)
    cotangent = rng.normal(size=(2, 4)).astype(np.float32)
    gate = x @ gate_weight
    up = x @ up_weight
    common = {
        "x": x,
        "gate_weight": gate_weight,
        "up_weight": up_weight,
        "cotangent.output": cotangent,
    }
    saved_result = execute_tensor_program(saved.program, {**common, "gate": gate, "up": up})
    recomputed_result = execute_tensor_program(recomputed.program, common)
    for gradient in differentiated.input_gradients:
        np.testing.assert_allclose(saved_result[gradient.name], recomputed_result[gradient.name], rtol=1e-6, atol=1e-6)


def test_backward_extraction_rejects_unknown_or_duplicate_saved_values() -> None:
    differentiated = differentiate_tensor_program(
        _linear_swiglu_program(),
        with_respect_to=("x",),
    )
    with np.testing.assert_raises_regex(ValueError, "forward intermediates"):
        extract_backward_tensor_program(differentiated, saved_values=("not-a-value",))
    with np.testing.assert_raises_regex(ValueError, "must be unique"):
        extract_backward_tensor_program(differentiated, saved_values=("gate", "gate"))
