# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "joblib"]
# ///

"""Stable affine-simplex evaluation of unchanged geometry-finalist coefficients.

Retains the separate strict canonical failure; no prediction-parity tolerance is
relaxed. Decimal arithmetic removes a redundant common linear slope before
double-precision optimization. No fit or coefficient is changed.
"""

import argparse
import dataclasses
import decimal
import functools
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy import optimize, special

from experiments.domain_phase_mix.exploratory.two_phase_many import optimize_mariner_convex_round2_20260917 as original

D = decimal.Decimal.from_float


@dataclasses.dataclass(frozen=True)
class StableObjective(original.RoundTwoObjective):
    @functools.cached_property
    def centered(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        intercepts, slopes, removed = [], [], []
        count = len(self.inventory)
        with decimal.localcontext() as context:
            context.prec = 70
            for fit in self.fits:
                own = [
                    D(float(b)) * D(float(q)) / 32
                    for b, q in zip(fit.head.coefficients[count:], self.inventory, strict=True)
                ]
                center = sum(own, start=decimal.Decimal(0)) / decimal.Decimal(count)
                intercepts.append(float(D(fit.head.intercept) + center))
                slopes.append([float(s - center) for s in own])
                removed.append(float(center))
        return np.asarray(intercepts), np.asarray(slopes), np.asarray(removed)

    def logits(self, mixtures: np.ndarray) -> np.ndarray:
        weights = np.atleast_2d(mixtures)
        epochs = weights * self.inventory
        intercepts, slopes, _ = self.centered
        values = []
        for i, fit in enumerate(self.fits):
            benefit = original.geometry.benefit(
                epochs, fit.shape["rate"], fit.shape.get("power", 1.0), original.GEOMETRY[self.spec.model_id].benefit
            )
            values.append(intercepts[i] + weights @ slopes[i] - benefit @ fit.head.coefficients[: len(self.inventory)])
        return np.stack(values, axis=1)

    def log_value_gradient(self, mixture: np.ndarray) -> tuple[float, np.ndarray]:
        logits = self.logits(mixture)[0]
        _, slopes, _ = self.centered
        gradients = []
        for i, fit in enumerate(self.fits):
            rate = fit.shape["rate"]
            denominator = 1 + rate * mixture * self.inventory
            derivative = rate / denominator
            if self.spec.model_id == "cv2_shifted_power":
                power = fit.shape["power"]
                derivative = power * rate * denominator ** (power - 1)
            gradient = slopes[i] - fit.head.coefficients[: len(self.inventory)] * derivative * self.inventory
            if logits[i] < -original.base.LINK_CLIP:
                gradient[:] = 0.0
            gradients.append(gradient)
        terms = np.maximum(logits, -original.base.LINK_CLIP) + np.log(self.panel.group(self.target).aggregation_weights)
        return float(special.logsumexp(terms)), special.softmax(terms) @ np.asarray(gradients)


def decimal_logits(objective: StableObjective, point: np.ndarray) -> np.ndarray:
    """Evaluate the original formula at a mathematically exact simplex point."""
    values = []
    with decimal.localcontext() as context:
        context.prec = 70
        weights = [D(float(w)) for w in point[:-1]]
        weights.append(decimal.Decimal(1) - sum(weights))
        for fit in objective.fits:
            rate = D(float(fit.shape["rate"]))
            power = D(float(fit.shape.get("power", 1.0)))
            value = D(float(fit.head.intercept))
            for i, (weight, inventory) in enumerate(zip(weights, objective.inventory, strict=True)):
                epoch = weight * D(float(inventory))
                logged = (1 + rate * epoch).ln()
                benefit = logged if objective.spec.model_id == "cv2_log_benefit" else (power * logged).exp() - 1
                value += -D(float(fit.head.coefficients[i])) * benefit
                value += D(float(fit.head.coefficients[i + len(weights)])) * epoch / 32
            values.append(float(value))
    return np.asarray(values)


def verify(objective: StableObjective) -> dict:
    if any(np.any(f.head.coefficients < 0) or np.isfinite(f.head.cap) for f in objective.fits):
        raise ValueError("Expected nonnegative original amplitudes and infinite head caps")
    point = np.ones(len(objective.inventory)) / len(objective.inventory)
    reference = decimal_logits(objective, point)
    replay = objective.logits(point)[0]
    np.testing.assert_allclose(replay, reference, rtol=2e-12, atol=2e-12)
    _, derivative = objective.log_value_gradient(point)
    numerical = optimize._numdiff.approx_derivative(
        lambda w: objective.log_value_gradient(w)[0], point, method="3-point", abs_step=1e-7
    ).ravel()
    np.testing.assert_allclose(derivative, numerical, rtol=2e-5, atol=2e-7)
    points = np.vstack([objective.panel.features.weights[::17], np.eye(len(point))])
    stable = objective.logits(points)
    canonical = original.RoundTwoObjective.logits(objective, points)
    floors = np.asarray([f.head.floor for f in objective.fits])
    stable_loss = floors + np.exp(np.clip(stable, -30, 30))
    canonical_loss = floors + np.exp(np.clip(canonical, -30, 30))
    vertices = stable[-len(point) :]
    return {
        "decimal70_logit_replay_max_error": float(np.max(np.abs(replay - reference))),
        "gradient_max_error": float(np.max(np.abs(derivative - numerical))),
        "strict_canonical_logit_max_difference": float(np.max(np.abs(stable - canonical))),
        "strict_canonical_component_prediction_max_difference": float(np.max(np.abs(stable_loss - canonical_loss))),
        "strict_canonical_aggregate_prediction_max_difference": float(
            np.max(np.abs((stable_loss - canonical_loss) @ objective.panel.group(objective.target).aggregation_weights))
        ),
        "max_removed_simplex_normal_slope": float(np.max(np.abs(objective.centered[2]))),
        "upper_clipped_components_anywhere_on_simplex": int(np.sum(np.any(vertices > 30, axis=0))),
        "max_vertex_logit": float(vertices.max()),
        "canonical_float_implementation_certified": False,
        "mathematical_simplex_function_unchanged": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("cv2_shifted_power", "cv2_log_benefit"), required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    panel = original.canonical.bench.load_panel(args.panel)
    paths = [
        original.banks.prior.fit_path(args.bank_root, args.model, args.panel, args.target, i)
        for i in range(len(panel.group(args.target).components))
    ]
    hashes = {str(p): original.banks.prior.sha256(p) for p in paths}
    manifest = {
        "model": args.model,
        "panel": args.panel,
        "target": args.target,
        "fit_hashes": hashes,
        "sources": {
            str(Path(m.__file__).resolve()): hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
            for m in (original, original.geometry, original.canonical)
        },
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "method": (
            "Original five-start optimizer and sampler; exact common linear-slope recentering "
            "on simplex with 70-digit coefficient arithmetic"
        ),
        "bank_outcomes_used": False,
    }
    path = args.output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Stable evaluation identity changed")
    original.canonical.shared.write_json(path, manifest)
    initial = original.objective_from_fits(args.model, panel, args.target, [joblib.load(p) for p in paths])
    objective = StableObjective(initial.spec, panel, args.target, initial.fits)
    audit = verify(objective)
    record = original.canonical.solve(objective)
    record.update(audit=audit, physical=original.physical_diagnostics(objective, record), fit_hashes=hashes)
    record["scope"] = (
        "Same mathematical coefficients/function on the exact simplex; stable arithmetic. "
        "Strict canonical float replay discrepancy retained, not certified as unchanged floating implementation."
    )
    original.canonical.shared.write_json(args.output / "result.json", record)
    print(json.dumps({k: record[k] for k in ("model", "panel", "target", "value", "supporting_plane_gap_bpb")}))


if __name__ == "__main__":
    main()
