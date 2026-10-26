# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pandas", "joblib"]
# ///

"""Certify fixed round-two convex mixture fits, without consulting bank labels."""

import argparse
import dataclasses
import functools
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy import special

from experiments.domain_phase_mix.exploratory.two_phase_many import mariner_convex_candidates_20260917 as prior
from experiments.domain_phase_mix.exploratory.two_phase_many import mariner_convex_envelopes_round2_20260917 as envelopes
from experiments.domain_phase_mix.exploratory.two_phase_many import mariner_convex_fitting_round2_20260917 as fitting
from experiments.domain_phase_mix.exploratory.two_phase_many import mariner_convex_round2_models_20260917 as geometry
from experiments.domain_phase_mix.exploratory.two_phase_many import optimize_mariner_convex_20260917 as canonical
from experiments.domain_phase_mix.exploratory.two_phase_many import single_phase_observatory_models_20260902 as base
from experiments.domain_phase_mix.exploratory.two_phase_many import validate_convex_round2_banks_20260917 as banks

ALLOWED = (
    "cv2_shifted_power",
    "cv2_log_benefit",
    "cf2_huber_d8__selection",
    "cf2_huber_d8__bounded_rms",
    "ce2_majorant",
    "ce2_envelope",
)
GEOMETRY = {spec.model_id: spec for spec in geometry.candidate_specs()}
FITTING = {spec.model_id: spec for spec in fitting.candidate_specs()}
ENVELOPES = {spec.model_id: spec for spec in envelopes.candidate_specs()}
PRIOR = {spec.model_id: spec for spec in prior.candidate_specs()}


def design(model_id: str, features: base.Features, shape: base.Shape, params: tuple[float, ...]) -> base.Design:
    del params
    if model_id.startswith("cv2_"):
        return geometry.design(GEOMETRY[model_id], features, shape)
    if model_id.startswith("cf2_"):
        spec = PRIOR[FITTING[model_id].family_id]
        return spec.design_function(features, shape, spec.params)
    return envelopes.design(
        features, shape, model_id, tuple(map(float, features.inventory)), envelopes.MODULE_SOURCE_HASH
    )


def logit_gradient(model_id: str, fitted: base.Fitted, features: base.Features) -> tuple[float, np.ndarray]:
    if model_id.startswith("cv2_"):
        value, derivative = geometry.linear_and_weight_gradient(GEOMETRY[model_id], fitted, features)
        return float(value[0]), derivative[0]
    matrix = design(model_id, features, fitted.shape, ())
    if model_id.startswith("cf2_"):
        spec = PRIOR[FITTING[model_id].family_id]
        derivative, indices = prior.design_epoch_derivative(spec, features, fitted.shape)
    else:
        derivative, indices = envelopes.design_epoch_derivative(ENVELOPES[model_id], features, fitted.shape)
    active = fitted.head.coefficients != 0
    weighted = np.zeros_like(fitted.head.coefficients)
    weighted[active] = derivative[0, active] * fitted.head.coefficients[active]
    gradient = np.bincount(indices, weights=weighted, minlength=features.buckets) * features.inventory
    return float(fitted.head.intercept + matrix.values[0] @ fitted.head.coefficients), gradient


@dataclasses.dataclass(frozen=True)
class RoundTwoObjective(canonical.ConvexObjective):
    def log_value_gradient(self, mixture: np.ndarray) -> tuple[float, np.ndarray]:
        features = self.features(mixture)
        logits, gradients = [], []
        for fitted in self.fits:
            value, gradient = logit_gradient(self.spec.model_id, fitted, features)
            if value < -base.LINK_CLIP:
                gradient[:] = 0.0
            logits.append(max(value, -base.LINK_CLIP))
            gradients.append(gradient)
        terms = np.asarray(logits) + np.log(self.panel.group(self.target).aggregation_weights)
        return float(special.logsumexp(terms)), special.softmax(terms) @ np.asarray(gradients)


def objective_from_fits(model_id: str, panel, target: str, fits: list[base.Fitted]) -> RoundTwoObjective:
    if model_id not in ALLOWED:
        raise ValueError("Only explicitly advanced, proven convex models may receive certificates")
    proxy = prior.CandidateSpec(
        model_id,
        model_id,
        "round2",
        "Frozen registered convex response",
        "Negative concave benefit plus convex harm; exponential outer link",
        (),
        ("rate", "power", "threshold"),
        functools.partial(design, model_id),
    )
    return RoundTwoObjective(proxy, panel, target, fits)


def verify(objective: RoundTwoObjective) -> dict:
    audit = canonical.verify(objective)
    points = np.vstack([objective.panel.features.weights[::17], np.eye(len(objective.inventory))])
    query = objective.features(points)
    expected = []
    for component, fitted in zip(objective.panel.group(objective.target).components, objective.fits, strict=True):
        features = dataclasses.replace(query, component=component)
        model = banks.make_model(objective.spec.model_id, features)
        expected.append(model.predict(fitted, features, np.arange(len(points))))
    actual = np.asarray([fit.head.floor for fit in objective.fits]) + np.exp(
        np.clip(objective.logits(points), -base.LINK_CLIP, base.LINK_CLIP)
    )
    np.testing.assert_allclose(actual, np.stack(expected, axis=1), rtol=2e-10, atol=2e-10)
    audit["registered_model_replay_relative_error"] = float(
        np.max(np.abs(actual - np.stack(expected, axis=1)) / np.maximum(np.abs(actual), 1.0))
    )
    audit["canonical_global_convexity_certified"] = audit["upper_clipped_components_anywhere_on_simplex"] == 0
    audit["uncapped_model_convexity_certified"] = True
    return audit


def physical_diagnostics(objective: RoundTwoObjective, record: dict) -> dict:
    points = np.asarray([record["weights"], record["runtime_weights"]])
    logits = objective.logits(points)
    losses = np.asarray([fit.head.floor for fit in objective.fits]) + np.exp(np.maximum(logits, -base.LINK_CLIP))
    if not np.isfinite(losses).all():
        raise ValueError("Nonfinite physical predictions at returned policies")
    return {
        "component_floor_minimum": float(min(fit.head.floor for fit in objective.fits)),
        "aggregate_floor": objective.floor,
        "continuous_min_component_prediction": float(losses[0].min()),
        "runtime_min_component_prediction": float(losses[1].min()),
        "continuous_negative_components": int(np.sum(losses[0] < 0)),
        "runtime_negative_components": int(np.sum(losses[1] < 0)),
        "max_returned_logit": float(logits.max()),
        "returned_policies_upper_clipped": bool(np.any(logits > base.LINK_CLIP)),
        "component_positivity_elsewhere_on_simplex": (
            "Not established by positive endpoints or a positive aggregate bound"
        ),
    }


def solve_one(model_id: str, panel_name: str, target: str, bank_root: Path, output: Path) -> dict:
    panel = canonical.bench.load_panel(panel_name)
    paths = [
        banks.prior.fit_path(bank_root, model_id, panel_name, target, index)
        for index in range(len(panel.group(target).components))
    ]
    hashes = {str(path): banks.prior.sha256(path) for path in paths}
    path = output / panel_name / target / f"{model_id}.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result["fit_hashes"] != hashes:
            raise ValueError("Frozen input heads changed")
        return result
    objective = objective_from_fits(model_id, panel, target, [joblib.load(path) for path in paths])
    audit = verify(objective)
    result = canonical.solve(objective)
    result.update(audit=audit, physical=physical_diagnostics(objective, result), fit_hashes=hashes)
    result["bank_outcomes_used"] = False
    result["certificate_scope"] = (
        "Uncapped learned model; also canonical predictor if vertex audit excludes upper clipping. "
        "Floating-point analytic supporting-plane bound; no true-loss optimality claim."
    )
    canonical.shared.write_json(path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    if set(args.models) - set(ALLOWED):
        raise ValueError("Unsupported convex model")
    source_paths = [
        Path(module.__file__)
        for module in (canonical, prior, geometry, fitting, envelopes, base, banks, canonical.shared.reference)
    ] + [Path(__file__)]
    manifest = {
        "models": args.models,
        "bank_manifest_hash": banks.prior.sha256(args.bank_root / "manifest.json"),
        "sources": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "method": "Unchanged five-start analytic log-domain SLSQP, 2048-block quantization/exchange",
        "certificate": "Full original simplex supporting-plane gap at returned interior point",
        "interior_floor": canonical.INTERIOR_EPSILON,
        "bank_outcomes_used": False,
    }
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Optimizer source or protocol changed")
    canonical.shared.write_json(manifest_path, manifest)
    tasks = [
        (model, panel, target)
        for model in args.models
        for panel in canonical.bench.THIRTY_NINE_BUCKET_PANELS
        for target in ("uncheatable", "table9")
    ]
    with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
        records = joblib.Parallel(n_jobs=args.workers, return_as="generator_unordered")(
            joblib.delayed(solve_one)(*task, args.bank_root, args.output) for task in tasks
        )
        for record in records:
            print(
                json.dumps(
                    {key: record[key] for key in ("model", "panel", "target", "value", "supporting_plane_gap_bpb")}
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
