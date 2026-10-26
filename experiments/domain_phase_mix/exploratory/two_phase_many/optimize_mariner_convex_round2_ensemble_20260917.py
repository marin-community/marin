# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "joblib"]
# ///

"""Use the unchanged bounded optimizer on the predeclared convex model average."""

import dataclasses
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy import optimize, special

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    optimize_mariner_convex_round2_stable_20260917 as stable,
)

MEMBERS = (
    ("geometry", "cv2_shifted_power"),
    ("geometry", "cv2_log_benefit"),
    ("fitting", "cf2_huber_d8__selection"),
    ("fitting", "cf2_huber_d8__bounded_rms"),
    ("envelope", "ce2_envelope"),
)


@dataclasses.dataclass(frozen=True)
class EnsembleObjective(stable.original.canonical.ConvexObjective):
    members: tuple[stable.original.RoundTwoObjective, ...]

    @property
    def floor(self) -> float:
        return float(np.mean([member.floor for member in self.members]))

    def log_value_gradient(self, mixture: np.ndarray) -> tuple[float, np.ndarray]:
        records = [member.log_value_gradient(mixture) for member in self.members]
        values, gradients = np.asarray([r[0] for r in records]), np.asarray([r[1] for r in records])
        return float(special.logsumexp(values) - np.log(len(values))), special.softmax(values) @ gradients

    def predict(self, mixtures: np.ndarray) -> np.ndarray:
        return np.mean([member.predict(mixtures) for member in self.members], axis=0)


def solve_one(panel_name: str, target: str, output: Path) -> dict:
    m = stable.original
    panel = m.canonical.bench.load_panel(panel_name)
    members, hashes = [], {}
    for track, model in MEMBERS:
        paths = [
            m.banks.prior.fit_path(m.banks.OUTPUT / track, model, panel_name, target, i)
            for i in range(len(panel.group(target).components))
        ]
        hashes.update({str(path): m.banks.prior.sha256(path) for path in paths})
        member = m.objective_from_fits(model, panel, target, [joblib.load(path) for path in paths])
        if (model, panel_name, target) == ("cv2_shifted_power", "delphi_3e18_39bucket", "table9"):
            member = stable.StableObjective(member.spec, panel, target, member.fits)
        members.append(member)
    path = output / panel_name / f"{target}.json"
    if path.exists():
        record = json.loads(path.read_text())
        if record["fit_hashes"] != hashes:
            raise ValueError("Ensemble fitted heads changed")
        return record
    spec = dataclasses.replace(members[0].spec, model_id="round2_convex_equal_average")
    objective = EnsembleObjective(spec, panel, target, [], tuple(members))
    point = np.ones(len(panel.buckets)) / len(panel.buckets)
    value, gradient = objective.log_value_gradient(point)
    numerical = optimize._numdiff.approx_derivative(
        lambda w: objective.log_value_gradient(w)[0], point, method="3-point", abs_step=1e-7
    ).ravel()
    np.testing.assert_allclose(gradient, numerical, rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(objective.floor + np.exp(value), objective.predict(point)[0], rtol=2e-12, atol=2e-12)
    record = m.canonical.solve(objective)
    lower = max(start["lower_bound"] for start in record["starts"])
    points = np.asarray([record["weights"], record["runtime_weights"]])
    component_predictions = []
    for member in members:
        component_predictions.append(
            np.asarray([fit.head.floor for fit in member.fits]) + np.exp(np.maximum(member.logits(points), -30))
        )
    component_predictions = np.asarray(component_predictions)
    clip_bound = objective.floor + np.min(panel.group(target).aggregation_weights) / len(members) * np.exp(30)
    record.update(
        fit_hashes=hashes,
        gradient_max_error=float(np.max(np.abs(gradient - numerical))),
        best_joint_lower_bound=lower,
        best_joint_gap_bpb=min(record["value"], record["runtime_prediction"]) - lower,
        clip_region_lower_bound=float(clip_bound),
        clip_region_excluded=bool(clip_bound > record["value"]),
        min_individual_component_prediction=float(component_predictions.min()),
        min_average_component_prediction=float(component_predictions.mean(axis=0).min()),
        stable_geometry_sensitivity=panel_name == "delphi_3e18_39bucket" and target == "table9",
        scope=(
            "Predeclared equal positive average of fixed learned convex functions; same five starts "
            "and 2048-block sampler. Qwen suite uses separately audited stable-gauge mathematical "
            "evaluation. No true-loss optimality claim."
        ),
    )
    m.canonical.shared.write_json(path, record)
    return record


def main() -> None:
    m = stable.original
    output = m.banks.OUTPUT.parent / "convex_round2_optimizer/ensemble"
    manifest = {
        "members": MEMBERS,
        "sources": {
            str(Path(x.__file__).resolve()): hashlib.sha256(Path(x.__file__).read_bytes()).hexdigest()
            for x in (stable, m, m.canonical)
        },
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "method": "Unchanged five-start log-domain SLSQP; equal1/5 average; supporting-plane bounds;2048-block sampler",
        "bank_outcomes_used": False,
        "mathematical_convexity": "Positive average of convex learned objectives; each constituent proof retained",
    }
    manifest = json.loads(json.dumps(manifest))
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Ensemble optimizer identity changed")
    m.canonical.shared.write_json(path, manifest)
    with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
        results = joblib.Parallel(n_jobs=2, return_as="generator_unordered")(
            joblib.delayed(solve_one)(panel, target, output)
            for panel in m.canonical.bench.THIRTY_NINE_BUCKET_PANELS
            for target in ("uncheatable", "table9")
        )
        for r in results:
            print(json.dumps({k: r[k] for k in ("model", "panel", "target", "value", "best_joint_gap_bpb")}), flush=True)


if __name__ == "__main__":
    main()
