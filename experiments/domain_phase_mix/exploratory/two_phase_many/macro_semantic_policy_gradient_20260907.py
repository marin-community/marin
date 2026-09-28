# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Vectorized macro semantic predictions and exact softmax-logit gradients.

``load_surface(...).value_logit_gradient(weights)`` returns values of shape
(n,) and gradients of shape (n,2,39), evaluated at the supplied simplex
weights. Gradients are with respect to separate phase softmax logits, not raw
weights. For a tied policy controlled by one shared logit vector, sum the two
phase gradients. Exposure-times-derivative formulas preserve finite zero-weight
limits even when the Weibull derivative with respect to weight is singular.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fit_two_phase_macro_semantic_20260907 as macro
import numpy as np
from scipy.special import expit, softmax


@dataclass(frozen=True)
class Surface:
    record: dict[str, Any]
    rate: np.ndarray
    power: np.ndarray
    threshold: np.ndarray
    acquisition: np.ndarray
    damage: np.ndarray
    intercept: np.ndarray
    floors: np.ndarray
    epochs: np.ndarray
    weights: np.ndarray

    def semantic_state(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return eta and weight_i*d eta/d weight_i for every component."""
        exposure = weights[:, None, :] * self.epochs[None, None, :]
        z = (self.rate[None, :, None] * exposure) ** self.power[None, :, None]
        benefit = -np.expm1(-z)
        log_exposure = np.log1p(exposure) - self.threshold[None, :, None]
        positive = np.logaddexp(log_exposure, 0)
        eta = self.intercept + np.sum(-benefit * self.acquisition + positive**2 * self.damage, axis=2)
        scaled_benefit_derivative = self.power[None, :, None] * z * np.exp(-z)
        scaled_harm_derivative = 2 * positive * expit(log_exposure) * exposure / (1 + exposure)
        scaled_derivative = -scaled_benefit_derivative * self.acquisition + scaled_harm_derivative * self.damage
        return eta, scaled_derivative

    def value_logit_gradient(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return objective BPB and its exact independent-phase logit gradients."""
        weights = np.asarray(weights, dtype=float)
        assert weights.ndim == 3 and weights.shape[1:] == (2, 39)
        alpha = self.record["alpha"]
        aggregate = alpha * weights[:, 0] + (1 - alpha) * weights[:, 1]
        eta_a, scaled_a = self.semantic_state(aggregate)
        deficit_components = np.exp(np.clip(eta_a, -30, 30))
        q = deficit_components @ self.weights
        floor = float(self.floors @ self.weights)
        baseline = floor + q
        active_deficit = deficit_components * ((eta_a > -30) & (eta_a < 30)) * self.weights
        baseline_gradient = np.empty_like(weights)
        for phase, duration in enumerate((alpha, 1 - alpha)):
            share = np.divide(duration * weights[:, phase], aggregate, out=np.zeros_like(aggregate), where=aggregate > 0)
            scaled = scaled_a * share[:, None, :]
            centered = scaled - weights[:, phase, None, :] * scaled.sum(axis=2, keepdims=True)
            baseline_gradient[:, phase] = np.einsum("nti,nt->ni", centered, active_deficit)
        if self.record["config"]["zero"]:
            return baseline, baseline_gradient
        eta0, scaled0 = self.semantic_state(weights[:, 0])
        eta1, scaled1 = self.semantic_state(weights[:, 1])
        delta = eta1 - eta0
        coefficient = np.asarray(self.record["coefficients"]) / np.asarray(self.record["scale"])
        width = len(self.weights)
        odd, even = coefficient[:width], coefficient[width:]
        correction = delta @ odd + delta**2 @ even
        derivative = odd + 2 * delta * even
        j0 = scaled0 - weights[:, 0, None, :] * scaled0.sum(axis=2, keepdims=True)
        j1 = scaled1 - weights[:, 1, None, :] * scaled1.sum(axis=2, keepdims=True)
        correction_gradient = np.stack(
            [-np.einsum("nti,nt->ni", j0, derivative), np.einsum("nti,nt->ni", j1, derivative)], axis=1
        )
        if self.record["model"] == "CRE2-019":
            return baseline + correction, baseline_gradient + correction_gradient
        with np.errstate(over="raise", invalid="raise"):
            ratio = np.exp(correction)
            return floor + q * ratio, ratio[:, None, None] * (baseline_gradient + q[:, None, None] * correction_gradient)

    def predict(self, weights: np.ndarray) -> np.ndarray:
        """Return vectorized objective values, preserving the unclipped phase link."""
        return self.value_logit_gradient(weights)[0]


def load_surface(output: Path, objective: str, context: str, model: str) -> Surface:
    """Load a frozen fit without changing its source, coefficients or predictions."""
    record = json.loads((output / "fits" / model / objective / context / "fit.json").read_text())
    spines = record["spines"]
    coefficients = np.array([item["coefficients"] for item in spines])
    return Surface(
        record,
        np.array([item["shape"]["rate"] for item in spines]),
        np.array([item["shape"]["power"] for item in spines]),
        np.array([item["shape"]["threshold"] for item in spines]),
        coefficients[:, :39],
        coefficients[:, 39:],
        np.array([item["intercept"] for item in spines]),
        np.array([item["floor"] for item in spines]),
        np.asarray(record["c_total"]),
        np.asarray(record["aggregation_weights"]),
    )


def checks(output: Path) -> None:
    _, panel, _ = macro.previous.inputs(str(macro.SOURCE))
    rng = np.random.default_rng(20260907)
    random_weights = rng.dirichlet(np.full(39, 0.7), size=(8, 2))
    logits = np.log(random_weights)
    records = []
    for model in macro.MODELS:
        for objective in macro.previous.OBJECTIVES:
            surface = load_surface(output, objective, "final", model)
            original = macro.load_predictor(output, objective, "final", model)
            predicted, gradients = surface.value_logit_gradient(panel["weights"])
            parity = float(np.max(np.abs(predicted - original.predict(panel["weights"]))))
            assert parity < 1e-12
            assert np.isfinite(gradients).all()
            value, gradient = surface.value_logit_gradient(random_weights)
            assert np.max(np.abs(value - original.predict(random_weights))) < 1e-12
            direction = rng.normal(size=logits.shape)
            direction /= np.linalg.norm(direction, axis=(1, 2), keepdims=True)
            step = 1e-6
            numeric = (
                surface.predict(softmax(logits + step * direction, axis=2))
                - surface.predict(softmax(logits - step * direction, axis=2))
            ) / (2 * step)
            analytic = np.sum(gradient * direction, axis=(1, 2))
            error = float(np.max(np.abs(numeric - analytic)))
            assert error < 1e-7
            tied_weights = np.repeat(random_weights[:, :1], 2, axis=1)
            _, tied_gradient = surface.value_logit_gradient(tied_weights)
            tied_logits = np.log(random_weights[:, 0])
            tied_direction = direction[:, 0]
            plus = np.repeat(softmax(tied_logits + step * tied_direction, axis=1)[:, None], 2, axis=1)
            minus = np.repeat(softmax(tied_logits - step * tied_direction, axis=1)[:, None], 2, axis=1)
            tied_numeric = (surface.predict(plus) - surface.predict(minus)) / (2 * step)
            tied_analytic = np.sum(tied_gradient.sum(axis=1) * tied_direction, axis=1)
            tied_error = float(np.max(np.abs(tied_numeric - tied_analytic)))
            assert tied_error < 1e-7
            records.append(
                {
                    "model": model,
                    "objective": objective,
                    "prediction_parity": parity,
                    "directional_gradient_error": error,
                    "tied_gradient_error": tied_error,
                    "maximum_phase_gradient_sum": float(np.max(np.abs(gradient.sum(axis=2)))),
                }
            )
    macro.previous.write_json(
        output / "policy_gradient_checks.json",
        {
            "checks": records,
            "source_sha256": macro.previous.file_hash(Path(__file__)),
            "fitter_source_sha256": macro.previous.file_hash(Path(macro.__file__)),
        },
    )
    print(json.dumps(records, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=macro.OUTPUT)
    args = parser.parse_args()
    checks(args.output)


if __name__ == "__main__":
    main()
