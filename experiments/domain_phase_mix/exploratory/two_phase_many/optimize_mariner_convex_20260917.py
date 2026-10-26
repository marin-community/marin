# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pandas", "joblib"]
# ///

"""Optimize frozen convex finalist fits and compute supporting-plane gaps."""

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy import optimize, special

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_single_phase_observatory_20260902 as bench,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    mariner_convex_candidates_20260917 as candidates,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    validate_mariner_optimizer_20260917 as shared,
)

INTERIOR_EPSILON = 1e-12


@dataclasses.dataclass(frozen=True)
class ConvexObjective:
    spec: candidates.CandidateSpec
    panel: bench.BenchPanel
    target: str
    fits: list[base.Fitted]

    @property
    def inventory(self) -> np.ndarray:
        return self.panel.features.inventory

    @property
    def floor(self) -> float:
        return float(
            np.asarray([fit.head.floor for fit in self.fits]) @ self.panel.group(self.target).aggregation_weights
        )

    def features(self, mixtures: np.ndarray) -> base.Features:
        return base.features_from_panel(
            np.atleast_2d(mixtures), self.inventory, self.panel.buckets, early_fraction=None, label="convex_optimizer"
        )

    def logits(self, mixtures: np.ndarray) -> np.ndarray:
        features = self.features(mixtures)
        values = []
        for fitted in self.fits:
            design = self.spec.design_function(features, fitted.shape, self.spec.params)
            values.append(fitted.head.intercept + design.values @ fitted.head.coefficients)
        return np.stack(values, axis=1)

    def log_value_gradient(self, mixture: np.ndarray) -> tuple[float, np.ndarray]:
        features = self.features(mixture)
        logits = []
        derivatives = []
        for fitted in self.fits:
            design = self.spec.design_function(features, fitted.shape, self.spec.params)
            value = float(fitted.head.intercept + design.values[0] @ fitted.head.coefficients)
            derivative, bucket_indices = candidates.design_epoch_derivative(self.spec, features, fitted.shape)
            weighted = np.zeros_like(fitted.head.coefficients)
            active = fitted.head.coefficients != 0.0
            weighted[active] = derivative[0, active] * fitted.head.coefficients[active]
            gradient = np.bincount(bucket_indices, weights=weighted, minlength=len(self.inventory)) * self.inventory
            if value < -base.LINK_CLIP:
                gradient[:] = 0.0
            logits.append(max(value, -base.LINK_CLIP))
            derivatives.append(gradient)
        log_terms = np.asarray(logits) + np.log(self.panel.group(self.target).aggregation_weights)
        return float(special.logsumexp(log_terms)), special.softmax(log_terms) @ np.asarray(derivatives)

    def predict(self, mixtures: np.ndarray) -> np.ndarray:
        logits = np.maximum(self.logits(mixtures), -base.LINK_CLIP)
        log_terms = logits + np.log(self.panel.group(self.target).aggregation_weights)
        return self.floor + np.exp(special.logsumexp(log_terms, axis=1))

    def certificate(self, mixture: np.ndarray) -> dict:
        log_value, log_gradient = self.log_value_gradient(mixture)
        gradient = np.exp(log_value) * log_gradient
        value = self.floor + np.exp(log_value)
        gap = float(gradient @ mixture - gradient.min())
        return {"value": float(value), "supporting_plane_gap_bpb": gap, "lower_bound": float(value - gap)}


def verify(objective: ConvexObjective) -> dict:
    panel = objective.panel
    if any(np.any(fit.head.coefficients < 0) or np.isfinite(fit.head.cap) for fit in objective.fits):
        raise ValueError("Convex proof requires nonnegative coefficients and infinite fitted caps")
    if any(fit.diagnostics["link"] != str(base.LinkKind.KAPPA_FLOOR) for fit in objective.fits):
        raise ValueError("Convex optimizer requires the fixed exponential floor link")
    points = np.vstack([panel.features.weights[::17], np.eye(len(panel.buckets))])
    features = objective.features(points)
    canonical = []
    for component, fitted in zip(panel.group(objective.target).components, objective.fits, strict=True):
        query = dataclasses.replace(features, component=str(component))
        model = candidates.build_model(objective.spec, query)
        canonical.append(model.predict(fitted, query, np.arange(len(points))))
    expected = np.stack(canonical, axis=1)
    logits = objective.logits(points)
    replay = np.asarray([fit.head.floor for fit in objective.fits]) + np.exp(np.clip(logits, -30.0, 30.0))
    np.testing.assert_allclose(replay, expected, rtol=2e-10, atol=2e-10)
    point = np.ones(len(panel.buckets)) / len(panel.buckets)
    _, derivative = objective.log_value_gradient(point)
    numerical = optimize._numdiff.approx_derivative(
        lambda w: objective.log_value_gradient(w)[0], point, method="3-point", abs_step=1e-7
    )
    np.testing.assert_allclose(derivative, numerical.ravel(), rtol=2e-5, atol=2e-7)
    vertices = logits[-len(panel.buckets) :]
    return {
        "max_vertex_logit": float(vertices.max()),
        "upper_clipped_components_anywhere_on_simplex": int(np.sum(np.any(vertices > base.LINK_CLIP, axis=0))),
        "gradient_max_error": float(np.max(np.abs(derivative - numerical.ravel()))),
        "canonical_replay_max_relative_error": float(
            np.max(np.abs(replay - expected) / np.maximum(np.abs(expected), 1.0))
        ),
    }


def solve(objective: ConvexObjective) -> dict:
    natural = 1 / objective.inventory
    natural /= natural.sum()
    indices = np.random.default_rng(shared.reference.START_SEED).choice(
        len(objective.panel.features.weights), shared.reference.PANEL_STARTS, replace=False
    )
    starts = [natural, *objective.panel.features.weights[indices]]
    records = []
    for index, start in enumerate(starts):
        start = (1 - len(natural) * INTERIOR_EPSILON) * start + INTERIOR_EPSILON
        result = optimize.minimize(
            objective.log_value_gradient,
            start,
            jac=True,
            method="SLSQP",
            bounds=[(INTERIOR_EPSILON, 1.0)] * len(natural),
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)},
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        mixture = np.maximum(result.x, INTERIOR_EPSILON)
        mixture /= mixture.sum()
        records.append(
            {
                "start": index,
                "success": bool(result.success),
                "message": str(result.message),
                "iterations": int(result.nit),
                "weights": mixture.tolist(),
                "raw_feasibility_error": float(max(abs(result.x.sum() - 1), -result.x.min())),
                **objective.certificate(mixture),
            }
        )
    best = min(records, key=lambda row: row["value"])
    weights = np.asarray(best["weights"])
    counts, runtime = shared.reference.runtime_policy(objective, weights)
    return {
        "model": objective.spec.model_id,
        "panel": objective.panel.name,
        "target": objective.target,
        "starts": records,
        "weights": weights.tolist(),
        **objective.certificate(weights),
        "runtime_weights": (counts / shared.reference.MIXTURE_BLOCK_SIZE).tolist(),
        **runtime,
        "numerical_interior_floor": INTERIOR_EPSILON,
        "certificate_domain": "Original whole simplex, from analytic convex supporting plane at returned interior point",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=shared.ROOT)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
    output = args.root / "optimizer_convex"
    sources = [Path(__file__), Path(candidates.__file__), Path(base.__file__), Path(shared.reference.__file__)]
    manifest = {
        "sources": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "models": args.models,
        "method": "Log-domain SLSQP, analytic derivatives, supporting-plane simplex gap",
        "interior_epsilon": INTERIOR_EPSILON,
    }
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Convex optimizer source or configuration changed")
    shared.write_json(path, manifest)
    specs = {s.model_id: s for s in candidates.candidate_specs()}
    bench.HELDOUT_DIR = shared.HERE / "reference_outputs/single_phase_heldout_benchmark_expanded_20260913"
    for panel_name in bench.THIRTY_NINE_BUCKET_PANELS:
        panel = bench.load_panel(panel_name)
        for target in ("uncheatable", "table9"):
            for model_id in args.models:
                if not specs[model_id].guaranteed_convex:
                    raise ValueError("Supporting-plane bound requires a proven convex family")
                fit_paths = [
                    args.root / "banks/convex/fits" / model_id / panel_name / target / f"component_{i:03d}.joblib"
                    for i in range(len(panel.group(target).components))
                ]
                fit_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in fit_paths}
                fits = [joblib.load(p) for p in fit_paths]
                objective = ConvexObjective(specs[model_id], panel, target, fits)
                path = output / panel_name / target / f"{model_id}.json"
                if path.exists():
                    if json.loads(path.read_text())["fit_hashes"] != fit_hashes:
                        raise ValueError("Convex optimizer input heads changed")
                    continue
                audit = verify(objective)
                record = solve(objective)
                _, bank_features = bench.heldout_features(panel, target)
                record.update(shared.geometric_metrics(np.asarray(record["weights"]), panel, bank_features.weights))
                record.update(audit=audit, fit_hashes=fit_hashes)
                shared.write_json(path, record)
                print(
                    json.dumps(
                        {k: record[k] for k in ("panel", "target", "model", "value", "supporting_plane_gap_bpb")}
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
