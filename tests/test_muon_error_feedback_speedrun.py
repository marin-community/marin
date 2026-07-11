# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from marin.execution.lazy import materialized_config

from experiments.speedrun.prism_berkeley_qwen3_scaling.materialize_muon_error_feedback_results import (
    SweepRun,
    build_payload,
)
from experiments.speedrun.prism_berkeley_qwen3_scaling.muon_error_feedback_optimizer import (
    ErrorAwareMuonConfig,
    clipped_nuclear_hessian,
    cubic_newton_schulz,
    error_aware_muon_step,
    quintic_newton_schulz,
    scale_with_error_aware_muon,
)
from experiments.speedrun.prism_berkeley_qwen3_scaling.muon_error_feedback_sweep import (
    ADAM_LR_RATIO,
    FEEDBACK_VARIANTS,
    LEARNING_RATES,
    build_sweep_configs,
)
from experiments.speedrun.prism_berkeley_qwen3_scaling.submission_support import default_speedrun

RESULTS_PATH = (
    Path(__file__).parents[1] / "experiments/speedrun/prism_berkeley_qwen3_scaling/muon_error_feedback_results.json"
)


def _matrix_sign_svd(matrix: np.ndarray) -> np.ndarray:
    left, _, right_t = np.linalg.svd(matrix, full_matrices=False)
    return left @ right_t


def _nuclear_hessian_svd(matrix: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < matrix.shape[1]:
        return _nuclear_hessian_svd(matrix.T, tangent.T).T

    left, singular_values, right_t = np.linalg.svd(matrix, full_matrices=False)
    right = right_t.T
    coordinates = left.T @ tangent @ right
    skew_coordinates = 0.5 * (coordinates - coordinates.T)
    pairwise_gain = np.zeros((len(singular_values), len(singular_values)))
    for row in range(len(singular_values)):
        for column in range(len(singular_values)):
            if row != column:
                pairwise_gain[row, column] = 2.0 / (singular_values[row] + singular_values[column])
    in_frame = left @ (pairwise_gain * skew_coordinates) @ right_t
    kernel = (np.eye(matrix.shape[0]) - left @ left.T) @ tangent @ right @ np.diag(1.0 / singular_values) @ right_t
    return in_frame + kernel


def _constant_quintic_reference(matrix: np.ndarray, *, steps: int = 5, eps: float = 1e-12) -> np.ndarray:
    a, b, c = 3.4445, -4.7750, 2.0315
    working = matrix / (np.linalg.norm(matrix) + eps)
    transposed = working.shape[0] < working.shape[1]
    if transposed:
        working = working.T
    for _ in range(steps):
        gram = working @ working.T
        working = a * working + (b * gram + c * gram @ gram) @ working
    return working.T if transposed else working


@pytest.mark.parametrize("shape", [(8, 5), (5, 8)])
def test_cubic_newton_schulz_value_and_jvp_match_svd_reference(shape):
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal(shape)
    tangent = rng.standard_normal(shape)
    expected_sign = _matrix_sign_svd(matrix)
    expected_hessian = _nuclear_hessian_svd(matrix, tangent)

    with jax.enable_x64(True):
        matrix_jax = jnp.asarray(matrix)
        tangent_jax = jnp.asarray(tangent)
        actual_sign, actual_hessian = jax.jvp(cubic_newton_schulz, (matrix_jax,), (tangent_jax,))

    np.testing.assert_allclose(np.asarray(actual_sign), expected_sign, atol=1e-10, rtol=1e-10)
    relative_error = np.linalg.norm(np.asarray(actual_hessian) - expected_hessian) / np.linalg.norm(expected_hessian)
    assert relative_error < 1e-10


@pytest.mark.parametrize("shape", [(8, 5), (5, 8)])
def test_quintic_newton_schulz_matches_constant_coefficient_reference(shape):
    matrix = np.random.default_rng(1).standard_normal(shape)
    expected = _constant_quintic_reference(matrix)

    with jax.enable_x64(True):
        actual = quintic_newton_schulz(jnp.asarray(matrix))

    np.testing.assert_allclose(np.asarray(actual), expected, atol=1e-12, rtol=1e-12)


def test_error_aware_policies_reduce_exactly_to_muon_at_zero_gain():
    key_matrix, key_gradient = jax.random.split(jax.random.key(3))
    matrix = jax.random.normal(key_matrix, (8, 5))
    gradient = jax.random.normal(key_gradient, (8, 5))
    muon = error_aware_muon_step(matrix, gradient, policy="muon")
    blend = error_aware_muon_step(matrix, gradient, policy="blend", blend_gain=0.0)
    hesscorr = error_aware_muon_step(matrix, gradient, policy="hesscorr", correction_gain=0.0)

    assert jnp.array_equal(blend, muon)
    assert jnp.array_equal(hesscorr, muon)


@pytest.mark.parametrize("shape", [(8, 5), (5, 8)])
def test_blend_policy_matches_independent_quintic_reference(shape):
    key_matrix, key_gradient = jax.random.split(jax.random.key(4))
    matrix = jax.random.normal(key_matrix, shape)
    gradient = jax.random.normal(key_gradient, shape)
    gain = 0.3
    blended = np.asarray(matrix + gain * (gradient - matrix))
    expected = _constant_quintic_reference(blended)

    actual = error_aware_muon_step(matrix, gradient, policy="blend", blend_gain=gain)

    np.testing.assert_allclose(np.asarray(actual), expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("shape", [(8, 5), (5, 8)])
def test_hesscorr_policy_matches_clipped_svd_oracle_in_float32(shape):
    rng = np.random.default_rng(5)
    matrix = rng.standard_normal(shape)
    gradient = matrix + 0.4 * rng.standard_normal(shape)
    correction = _nuclear_hessian_svd(matrix, gradient - matrix)
    cap = np.sqrt(min(matrix.shape))
    correction *= min(1.0, cap / np.linalg.norm(correction))
    expected = _constant_quintic_reference(matrix) + 0.3 * correction

    actual = error_aware_muon_step(
        jnp.asarray(matrix, dtype=jnp.float32),
        jnp.asarray(gradient, dtype=jnp.float32),
        policy="hesscorr",
        correction_gain=0.3,
        cubic_steps=30,
    )

    np.testing.assert_allclose(np.asarray(actual), expected, atol=2e-5, rtol=2e-5)


def test_clipped_nuclear_hessian_matches_unclipped_oracle_and_activates_cap():
    rng = np.random.default_rng(9)
    well_conditioned = rng.standard_normal((8, 5))
    tangent = 1e-3 * rng.standard_normal((8, 5))
    expected = _nuclear_hessian_svd(well_conditioned, tangent)
    apply_correction = jax.jit(lambda matrix, direction: clipped_nuclear_hessian(matrix, direction, steps=30))

    actual = apply_correction(jnp.asarray(well_conditioned), jnp.asarray(tangent))

    np.testing.assert_allclose(np.asarray(actual), expected, atol=2e-5, rtol=2e-5)

    matrix = jnp.diag(jnp.asarray([1.0, 0.1, 0.01, 0.001, 1e-6]))
    large_tangent = jax.random.normal(jax.random.key(9), matrix.shape)

    correction = apply_correction(matrix, large_tangent)
    zero_matrix_correction = apply_correction(jnp.zeros((8, 5)), jnp.ones((8, 5)))

    assert jnp.all(jnp.isfinite(correction))
    np.testing.assert_allclose(jnp.linalg.norm(correction, ord="fro"), jnp.sqrt(5.0), atol=1e-5, rtol=1e-5)
    assert jnp.all(jnp.isfinite(zero_matrix_correction))
    assert jnp.linalg.norm(zero_matrix_correction, ord="fro") <= jnp.sqrt(5.0) + 1e-6


def test_scale_transform_uses_normalized_float32_momentum_and_preserves_update_dtype():
    in_axis = hax.Axis("in", 8)
    out_axis = hax.Axis("out", 16)
    params = hax.nn.Linear.init(in_axis, out_axis, key=jax.random.key(6), out_first=True)
    first_gradient = jax.tree.map(lambda value: jnp.ones_like(value, dtype=jnp.bfloat16), params)
    second_gradient = jax.tree.map(lambda value: jnp.full_like(value, 3.0, dtype=jnp.bfloat16), params)
    transform = scale_with_error_aware_muon(momentum=0.95, nesterov=False, policy="muon")

    state = transform.init(params)
    _, state = transform.update(first_gradient, state)
    updates, state = transform.update(second_gradient, state)

    expected_momentum = 0.95 * 0.05 + 0.05 * 3.0
    np.testing.assert_allclose(
        np.asarray(state.momentum_buffer.weight.array),
        expected_momentum,
        atol=1e-6,
        rtol=1e-6,
    )
    assert state.momentum_buffer.weight.array.dtype == jnp.float32
    assert updates.weight.array.dtype == jnp.bfloat16
    assert jnp.all(jnp.isfinite(updates.weight.array))

    expected = _constant_quintic_reference(np.full((16, 8), expected_momentum, dtype=np.float32))
    expected *= np.sqrt(16 / 8)
    np.testing.assert_allclose(np.asarray(updates.weight.array, dtype=np.float32), expected, atol=2e-2, rtol=2e-2)


def test_optimizer_config_jit_update_matches_policy_and_adam_branches():
    in_axis = hax.Axis("config_in", 8)
    out_axis = hax.Axis("config_out", 16)
    linear = hax.nn.Linear.init(in_axis, out_axis, key=jax.random.key(12), out_first=True)
    params = {"hidden": linear, "norm": jnp.ones((16,), dtype=jnp.float32)}
    gradients = jax.tree.map(jnp.ones_like, params)
    config = ErrorAwareMuonConfig(
        learning_rate=0.02,
        adam_lr=0.001,
        momentum=0.0,
        policy="muon",
        weight_decay=0.0,
        adam_weight_decay=0.0,
        max_grad_norm=0.0,
        lr_schedule="constant",
        warmup=0,
    )
    optimizer = config.build(num_train_steps=10)
    state = optimizer.init(params)

    updates, _ = jax.jit(lambda grad, opt_state: optimizer.update(grad, opt_state, params))(gradients, state)

    raw_weight_gradient = gradients["hidden"].weight.array
    expected_weight = error_aware_muon_step(
        raw_weight_gradient,
        raw_weight_gradient,
        policy="muon",
    )
    expected_weight *= jnp.sqrt(raw_weight_gradient.shape[0] / raw_weight_gradient.shape[1])
    expected_weight *= -config.learning_rate
    np.testing.assert_allclose(updates["hidden"].weight.array, expected_weight, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(updates["norm"], -config.adam_lr, atol=1e-6, rtol=1e-6)


def test_optimizer_config_applies_nonzero_hessian_feedback_after_normalized_ema():
    linear = hax.nn.Linear.init(
        hax.Axis("feedback_in", 8),
        hax.Axis("feedback_out", 16),
        key=jax.random.key(13),
        out_first=True,
    )
    params = {"hidden": linear}

    def random_gradient(key):
        return jax.tree.map(
            lambda value: jax.random.normal(jax.random.fold_in(key, value.size), value.shape),
            params,
        )

    first_gradient = random_gradient(jax.random.key(14))
    second_gradient = random_gradient(jax.random.key(15))
    config = ErrorAwareMuonConfig(
        learning_rate=0.02,
        adam_lr=0.001,
        momentum=0.5,
        policy="hesscorr",
        correction_gain=0.3,
        cubic_steps=30,
        weight_decay=0.0,
        adam_weight_decay=0.0,
        max_grad_norm=0.0,
        lr_schedule="constant",
        warmup=0,
    )
    optimizer = config.build(num_train_steps=10)
    state = optimizer.init(params)
    _, state = optimizer.update(first_gradient, state, params)

    updates, _ = jax.jit(lambda gradient, opt_state: optimizer.update(gradient, opt_state, params))(
        second_gradient, state
    )

    first_array = np.asarray(first_gradient["hidden"].weight.array)
    second_array = np.asarray(second_gradient["hidden"].weight.array)
    momentum = 0.25 * first_array + 0.5 * second_array
    correction = _nuclear_hessian_svd(momentum, second_array - momentum)
    cap = np.sqrt(min(momentum.shape))
    correction *= min(1.0, cap / np.linalg.norm(correction))
    expected = _constant_quintic_reference(momentum) + config.correction_gain * correction
    expected *= np.sqrt(momentum.shape[0] / momentum.shape[1])
    expected *= -config.learning_rate

    np.testing.assert_allclose(updates["hidden"].weight.array, expected, atol=3e-5, rtol=3e-5)


def test_optimizer_mask_routes_only_large_hidden_linear_weights_to_muon():
    large = hax.nn.Linear.init(
        hax.Axis("large_in", 8),
        hax.Axis("large_out", 16),
        key=jax.random.key(7),
        out_first=True,
    )
    small = hax.nn.Linear.init(
        hax.Axis("small_in", 4),
        hax.Axis("small_out", 8),
        key=jax.random.key(8),
        out_first=True,
    )
    params = {"hidden": large, "small": small, "Embedding": large, "lm_head": large}

    mask = ErrorAwareMuonConfig().create_mask(params, use_kimi_scaling=False)

    assert mask["hidden"].weight == "error_aware_muon"
    assert mask["hidden"].bias == "adamw"
    assert mask["small"].weight == "adamw"
    assert mask["small"].bias == "adamw"
    assert mask["Embedding"] == "adamw"
    assert mask["lm_head"] == "adamw"


def test_130m_sweep_crosses_archived_learning_rates_with_deduplicated_gain_grid():
    sweep = build_sweep_configs()
    names = [name for name, _ in sweep]
    expected_variants = {
        ("muon", 0.0),
        ("blend", 0.05),
        ("blend", 0.15),
        ("blend", 0.3),
        ("blend", 0.5),
        ("hesscorr", 0.1),
        ("hesscorr", 0.3),
        ("hesscorr", 1.0),
    }

    assert {(variant.policy, variant.gain) for variant in FEEDBACK_VARIANTS} == expected_variants
    assert LEARNING_RATES == (0.008, 0.012, 0.016, 0.020, 0.024)
    assert len(sweep) == 40
    assert len(set(names)) == len(names)
    assert {config.train_config.learning_rate for _, config in sweep} == set(LEARNING_RATES)

    configured_cells = set()
    for _, config in sweep:
        optimizer = config.train_config.optimizer_config
        assert isinstance(optimizer, ErrorAwareMuonConfig)
        gain = optimizer.blend_gain if optimizer.policy == "blend" else optimizer.correction_gain
        if optimizer.policy == "muon":
            gain = 0.0
        configured_cells.add((optimizer.policy, gain, optimizer.learning_rate))
        assert optimizer.learning_rate == config.train_config.learning_rate
        assert optimizer.adam_lr == pytest.approx(ADAM_LR_RATIO * optimizer.learning_rate)
        assert optimizer.nesterov is False
        assert optimizer.cubic_steps == 30
        assert config.train_config.train_batch_size == 128
        assert config.train_config.num_train_steps == 4959

    assert configured_cells == {
        (policy, gain, learning_rate) for policy, gain in expected_variants for learning_rate in LEARNING_RATES
    }

    train_step, result_step = default_speedrun(*sweep[0], version="2026.07.11")
    assert result_step.deps == (train_step,)
    assert result_step.path("gs://test-prefix") != train_step.path("gs://test-prefix")
    results_config = materialized_config(result_step, "gs://test-prefix")
    assert results_config.output_path == f"{train_step.path('gs://test-prefix')}/speedrun_results.json"


def test_checked_in_results_cover_the_completed_grid_and_recompute_selection():
    results = json.loads(RESULTS_PATH.read_text())
    runs = [SweepRun(**run) for run in results["runs"]]

    rebuilt = build_payload(runs)

    assert len(runs) == 40
    assert all(run.state == "finished" for run in runs)
    assert rebuilt["best_observed_run"] == results["best_observed_run"]
    assert rebuilt["paired_summary"] == results["paired_summary"]
    assert rebuilt["best_observed_run"]["run_name"] == ("qwen3_130m_error_aware_muon_hesscorr-g0p1_lr0p02-72a859")
    assert rebuilt["best_observed_run"]["c4_en_bpb"] == pytest.approx(1.164665699005127)
    assert all(run.source_results_path.endswith("/speedrun_results.json") for run in runs)
