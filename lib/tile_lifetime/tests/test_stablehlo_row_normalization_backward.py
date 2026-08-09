# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tile_lifetime import (
    RowStatisticKind,
    compile_stablehlo_row_normalization_backward,
)
from tile_lifetime.cuda_axis_fold_codegen import evaluate_axis_fold_program, generate_cuda_axis_fold
from tile_lifetime.stablehlo_import import import_stablehlo


def _natural_jax_vjp(*, centered: bool, rows: int = 4, hidden: int = 8):
    def normalization(x, feature_scale):
        local = x.astype(jnp.float32)
        if centered:
            local -= jnp.mean(local, axis=-1, keepdims=True)
        inverse = jax.lax.rsqrt(jnp.mean(jnp.square(local), axis=-1, keepdims=True) + 1e-5)
        return (local * inverse * feature_scale.astype(jnp.float32)).astype(jnp.bfloat16)

    def reverse(x, feature_scale, cotangent):
        _, pullback = jax.vjp(normalization, x, feature_scale)
        return pullback(cotangent)

    arguments = (
        jnp.zeros((rows, hidden), dtype=jnp.bfloat16),
        jnp.ones((hidden,), dtype=jnp.bfloat16),
        jnp.zeros((rows, hidden), dtype=jnp.bfloat16),
    )
    exported = jax.export.export(jax.jit(reverse))(*arguments)
    graph = import_stablehlo(
        exported.mlir_module_serialized,
        input_names=("arbitrary_matrix_a", "arbitrary_vector", "arbitrary_matrix_b"),
    )
    return reverse, graph


@pytest.mark.parametrize(
    ("centered", "statistic_kind", "input_reduction_count"),
    [
        (False, RowStatisticKind.UNCENTERED_SECOND_MOMENT, 1),
        (True, RowStatisticKind.CENTERED_SECOND_MOMENT, 2),
    ],
)
def test_jax_vjp_recovers_generic_row_folds_and_matches_bf16_outputs(
    centered: bool,
    statistic_kind: RowStatisticKind,
    input_reduction_count: int,
) -> None:
    reverse, graph = _natural_jax_vjp(centered=centered)
    compilation = compile_stablehlo_row_normalization_backward(graph, threads=8)
    recovered = compilation.recovered
    programs = compilation.programs

    assert recovered.statistic_kind is statistic_kind
    assert recovered.rows == 4
    assert recovered.hidden == 8
    assert len(programs.input_cotangent.reductions) == input_reduction_count

    rng = np.random.default_rng(19)
    x = rng.normal(size=(4, 8)).astype(np.float32)
    feature_scale = rng.normal(size=(8,)).astype(np.float32)
    cotangent = rng.normal(size=(4, 8)).astype(np.float32)
    x_bf16 = jnp.asarray(x, dtype=jnp.bfloat16)
    scale_bf16 = jnp.asarray(feature_scale, dtype=jnp.bfloat16)
    cotangent_bf16 = jnp.asarray(cotangent, dtype=jnp.bfloat16)
    expected_input, expected_scale = reverse(x_bf16, scale_bf16, cotangent_bf16)

    source = np.asarray(x_bf16, dtype=np.float32)
    local = source - np.mean(source, axis=1, keepdims=True) if centered else source
    inverse = np.reciprocal(np.sqrt(np.mean(local * local, axis=1) + 1e-5))
    standardized = local * inverse[:, None]
    projected = np.asarray(cotangent_bf16, dtype=np.float32)
    actual_input = evaluate_axis_fold_program(
        programs.input_cotangent,
        {
            "projected": projected,
            "feature_scale": np.asarray(scale_bf16, dtype=np.float32),
            "standardized": standardized,
            "inverse_scale": inverse,
        },
    )
    actual_scale = evaluate_axis_fold_program(
        programs.feature_scale_cotangent,
        {"projected": projected, "standardized": standardized},
    )
    actual_input_bf16 = np.asarray(jnp.asarray(actual_input, dtype=jnp.bfloat16), dtype=np.float32)
    actual_scale_bf16 = np.asarray(jnp.asarray(actual_scale, dtype=jnp.bfloat16), dtype=np.float32)
    expected_input_array = np.asarray(expected_input, dtype=np.float32)
    expected_scale_array = np.asarray(expected_scale, dtype=np.float32)
    input_error = np.abs(actual_input_bf16 - expected_input_array)
    scale_error = np.abs(actual_scale_bf16 - expected_scale_array)

    assert float(input_error.max()) <= 0.015625
    assert float(input_error.mean()) <= 0.001
    assert float(scale_error.max()) <= 0.03125
    assert float(scale_error.mean()) <= 0.004
    generated = generate_cuda_axis_fold(programs.input_cotangent).source.lower()
    assert "rms" not in generated
    assert "layernorm" not in generated
