# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).with_name("benchmark_hellinger_krr_delphi_3e18_20260727.py")
SPEC = importlib.util.spec_from_file_location("hellinger_krr_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_squared_hellinger_respects_phase_fraction() -> None:
    histograms = np.asarray(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )
    features = benchmark.sqrt_features(histograms, np.asarray([0.8, 0.2]))
    distance = benchmark.squared_hellinger(features, features)
    np.testing.assert_allclose(np.diag(distance), 0.0, atol=1e-12)
    np.testing.assert_allclose(distance[0, 1], 0.8, atol=1e-12)
    np.testing.assert_allclose(distance, distance.T, atol=1e-12)


def test_prediction_gradient_matches_finite_difference() -> None:
    basis = np.asarray(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.2, 0.6],
        ]
    )
    train_weights = np.asarray(
        [
            [[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]],
            [[0.2, 0.6, 0.2], [0.4, 0.4, 0.2]],
            [[0.1, 0.2, 0.7], [0.6, 0.2, 0.2]],
            [[0.4, 0.1, 0.5], [0.2, 0.7, 0.1]],
        ]
    )
    phase_fractions = np.asarray([0.8, 0.2])
    train_histograms = np.stack(
        [train_weights[:, 0] @ basis, train_weights[:, 1] @ basis],
        axis=1,
    )
    train_features = benchmark.sqrt_features(train_histograms, phase_fractions)
    distance = benchmark.squared_hellinger(train_features, train_features)
    gamma = 2.3
    ridge_alpha = 0.1
    target = np.asarray([1.0, 1.1, 0.95, 1.04])
    kernel = np.exp(-gamma * distance)
    system = kernel + ridge_alpha * np.eye(len(target))
    fit = benchmark.KernelFit(
        kernel_space="content",
        target="test",
        gamma=gamma,
        ridge_alpha=ridge_alpha,
        target_mean=float(target.mean()),
        dual=np.linalg.solve(system, target - target.mean()),
        train_sqrt_features=train_features,
        train_phase_histograms=train_histograms,
        basis=basis,
        phase_fractions=phase_fractions,
        oof_predictions=np.zeros(len(target)),
        oof_rmse=0.0,
        cv_sse=0.0,
        gram_inverse=np.linalg.inv(system),
    )
    query = np.asarray([[0.3, 0.4, 0.3], [0.5, 0.2, 0.3]])
    logits = benchmark.weights_to_free_logits(query)
    value, analytic = benchmark.prediction_and_gradient(fit, logits)
    finite = np.empty_like(logits)
    epsilon = 1e-6
    for index in range(len(logits)):
        step = np.zeros_like(logits)
        step[index] = epsilon
        plus = benchmark.prediction_and_gradient(fit, logits + step)[0]
        minus = benchmark.prediction_and_gradient(fit, logits - step)[0]
        finite[index] = (plus - minus) / (2.0 * epsilon)
    assert np.isfinite(value)
    np.testing.assert_allclose(analytic, finite, rtol=2e-5, atol=2e-7)
