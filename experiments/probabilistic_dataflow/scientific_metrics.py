# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Metric = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class ScientificScore:
    name: str
    value: float


def field_rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def spectral_error(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction_spectrum = np.abs(np.fft.rfft(prediction))
    truth_spectrum = np.abs(np.fft.rfft(truth))
    return field_rmse(prediction_spectrum, truth_spectrum)


def score(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    metrics: tuple[Metric, ...],
) -> tuple[ScientificScore, ...]:
    if prediction.shape != truth.shape:
        raise ValueError(f"Prediction shape {prediction.shape} does not match truth shape {truth.shape}")
    return tuple(ScientificScore(metric.__name__, metric(prediction, truth)) for metric in metrics)
