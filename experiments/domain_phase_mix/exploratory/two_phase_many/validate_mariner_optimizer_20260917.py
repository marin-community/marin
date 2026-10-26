# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pandas", "joblib"]
# ///
"""Check optimized-mixture stability of frozen archived MARINER challenger fits.

Run with the sibling mixture-selection package on PYTHONPATH. This reuses its
SLSQP and sampler-grid exchange procedure; no training is launched.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path

import joblib
import mixture_selection as reference
import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_single_phase_observatory_20260902 as bench,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    mariner_harm_screen_models_20260917 as candidates,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    validate_mariner_refit_robustness_20260917 as refits,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE / "reference_outputs/mariner_local_validation_20260917"
MODEL_IDS = ("hs17_mariner", "hs17_log1", "hs17_bounded_exp_c1")
SPECS = {spec.model_id: spec for spec in candidates.candidate_specs()}
EXTRA_START_SEED = 20260917


@dataclasses.dataclass(frozen=True)
class FrozenObjective:
    model_id: str
    inventory: np.ndarray
    rates: np.ndarray
    powers: np.ndarray
    thresholds: np.ndarray
    coefficients: np.ndarray
    intercepts: np.ndarray
    floors: np.ndarray
    caps: np.ndarray
    weights: np.ndarray

    def logits(self, mixtures: np.ndarray) -> np.ndarray:
        epochs = np.atleast_2d(mixtures)[:, None, :] * self.inventory[None, None, :]
        benefit = -np.expm1(-(np.maximum(epochs * self.rates[None, :, None], 0) ** self.powers[None, :, None]))
        s = np.logaddexp(0, np.log1p(np.maximum(epochs, 0)) - self.thresholds[None, :, None])
        origin = np.logaddexp(0, -self.thresholds)[None, :, None]
        if self.model_id == "hs17_log1":
            harm = s - origin
        else:
            harm = (s - origin) * (s + origin)
            if self.model_id == "hs17_bounded_exp_c1":
                harm = -np.expm1(-harm)
        buckets = len(self.inventory)
        return self.intercepts + np.sum(
            -benefit * self.coefficients[None, :, :buckets] + harm * self.coefficients[None, :, buckets:],
            axis=2,
        )

    def predict_tasks(self, mixtures: np.ndarray) -> np.ndarray:
        logits = self.logits(mixtures)
        return self.floors + np.exp(np.clip(logits, -base.LINK_CLIP, np.minimum(base.LINK_CLIP, self.caps)))

    def predict(self, mixtures: np.ndarray) -> np.ndarray:
        return self.predict_tasks(mixtures) @ self.weights


def objective_from_fits(panel: bench.BenchPanel, model_id: str, target: str, fits: list[base.Fitted]) -> FrozenObjective:
    if model_id not in MODEL_IDS:
        raise ValueError(f"Unsupported harm model: {model_id}")
    group = panel.group(target)
    if len(fits) != len(group.components):
        raise ValueError("Expected every component of the objective")
    if any(fitted.diagnostics["link"] != str(base.LinkKind.KAPPA_FLOOR) for fitted in fits):
        raise ValueError("This predictor requires the frozen exponential floor link")
    coefficients = np.stack([fitted.head.coefficients for fitted in fits])
    if coefficients.shape[1] != 2 * len(panel.buckets):
        raise ValueError("Expected one benefit and one harm coefficient per bucket")
    return FrozenObjective(
        model_id,
        panel.features.inventory,
        np.asarray([fit.shape["rate"] for fit in fits]),
        np.asarray([fit.shape["power"] for fit in fits]),
        np.asarray([fit.shape["threshold"] for fit in fits]),
        coefficients,
        np.asarray([fit.head.intercept for fit in fits]),
        np.asarray([fit.head.floor for fit in fits]),
        np.asarray([fit.head.cap for fit in fits]),
        group.aggregation_weights,
    )


def verify_predictor(objective: FrozenObjective, panel: bench.BenchPanel, target: str, fits: list[base.Fitted]) -> float:
    rows = np.vstack([panel.features.weights[::17], np.eye(len(panel.buckets))])
    features = base.features_from_panel(
        rows, panel.features.inventory, panel.buckets, early_fraction=None, label="optimizer_parity"
    )
    expected = []
    for component, fit in zip(panel.group(target).components, fits, strict=True):
        query = dataclasses.replace(features, component=str(component))
        model = candidates.build_model(SPECS[objective.model_id], query)
        expected.append(model.predict(fit, query, np.arange(len(rows))))
    expected = np.stack(expected, axis=1)
    actual = objective.predict_tasks(rows)
    np.testing.assert_allclose(actual, expected, rtol=3e-11, atol=3e-11)
    return float(np.max(np.abs(actual - expected) / np.maximum(np.abs(expected), 1.0)))


def full_fit_paths(root: Path, panel: str, model_id: str, target: str, count: int) -> list[Path]:
    folder = root / "banks/fits" / model_id / panel / target
    return [folder / f"component_{index:03d}.joblib" for index in range(count)]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f".partial-{os.getpid()}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    partial.replace(path)


def optimize_objective(
    objective: FrozenObjective, panel: bench.BenchPanel, *, extra_starts: bool, runtime_grid: bool
) -> tuple[np.ndarray, dict]:
    natural = 1 / objective.inventory
    natural /= natural.sum()
    indices = np.random.default_rng(reference.START_SEED).choice(
        len(panel.features.weights), reference.PANEL_STARTS, replace=False
    )
    starts = [natural, *panel.features.weights[indices]]
    if extra_starts:
        starts.extend(
            [
                np.ones(len(natural)) / len(natural),
                *np.random.default_rng(EXTRA_START_SEED).dirichlet(np.ones(len(natural)), size=4),
            ]
        )
    records = []
    solutions = []
    for index, start in enumerate(starts):
        start = reference.project_start(start, np.ones(len(natural)), natural)
        solution, record = reference.optimize_start(
            objective.predict, start, natural, np.ones(len(natural)), 0.0, reference.OPTIMIZER_OPTIONS
        )
        polished_solution, polished = reference.optimize_start(
            objective.predict, solution, natural, np.ones(len(natural)), 0.0, reference.POLISH_OPTIONS
        )
        if polished["objective"] <= record["objective"]:
            solution, record = polished_solution, {**polished, "polished": True}
        record.update(start=index, initial_weights=start.tolist(), endpoint_weights=solution.tolist())
        records.append(record)
        solutions.append(solution)
    feasible = [
        i
        for i, row in enumerate(records)
        if row["raw_feasibility_violation"] <= reference.ENDPOINT_FEASIBILITY_TOLERANCE
    ]
    if not feasible:
        raise ValueError("No feasible optimizer start")
    best = min(feasible, key=lambda i: records[i]["objective"])
    weights = solutions[best]
    best_value = records[best]["objective"]
    for i, record in enumerate(records):
        record.update(selected=i == best, tv_to_selected=float(np.abs(solutions[i] - weights).sum() / 2))
    logits = objective.logits(weights)[0]
    payload = {
        "starts": records,
        "continuous_prediction": best_value,
        "selected_success": records[best]["success"],
        "successful_starts": sum(record["success"] for record in records),
        "start_objective_range": max(records[i]["objective"] for i in feasible) - best_value,
        "start_tv_max": max(records[i]["tv_to_selected"] for i in feasible),
        "upper_clipped_components_at_solution": int(np.sum(logits > np.minimum(base.LINK_CLIP, objective.caps))),
        "lower_clipped_components_at_solution": int(np.sum(logits < -base.LINK_CLIP)),
        "continuous_weights": weights.tolist(),
    }
    if runtime_grid:
        counts, diagnostics = reference.runtime_policy(objective, weights)
        payload.update(
            runtime_counts=counts.tolist(),
            runtime_weights=(counts / reference.MIXTURE_BLOCK_SIZE).tolist(),
            **diagnostics,
        )
    return weights, payload


def geometric_metrics(weights: np.ndarray, panel: bench.BenchPanel, bank_weights: np.ndarray) -> dict:
    natural = 1 / panel.features.inventory
    natural /= natural.sum()
    return {
        "max_epoch": float(np.max(weights * panel.features.inventory)),
        "max_weight": float(weights.max()),
        "near_zero_buckets": int(np.sum(weights < 1e-6)),
        "tv_to_proportional": float(np.abs(weights - natural).sum() / 2),
        "nearest_fit_tv": float(np.min(np.abs(panel.features.weights - weights).sum(axis=1) / 2)),
        "nearest_bank_tv": float(np.min(np.abs(bank_weights - weights).sum(axis=1) / 2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--panels", nargs="*", default=list(bench.THIRTY_NINE_BUCKET_PANELS))
    parser.add_argument("--targets", nargs="*", default=["uncheatable", "table9"])
    parser.add_argument("--models", nargs="*", default=list(MODEL_IDS))
    parser.add_argument("--folds", nargs="*", type=int, default=list(range(5)))
    args = parser.parse_args()
    output = args.root / "optimizer"
    sources = [
        Path(__file__),
        Path(candidates.__file__),
        Path(base.__file__),
        Path(bench.__file__),
        Path(reference.__file__),
    ]
    manifest = {
        "sources": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "variant": "fullfit_ten_starts_outerfold_five_starts_reference_slsqp",
        "runtime_blocks": reference.MIXTURE_BLOCK_SIZE,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Optimizer source identity changed; preserve old results and use a new generation")
    write_json(manifest_path, manifest)
    bench.HELDOUT_DIR = HERE / "reference_outputs/single_phase_heldout_benchmark_expanded_20260913"
    rows = []
    for panel_name in args.panels:
        panel = bench.load_panel(panel_name)
        for target in args.targets:
            _, bank_features = bench.heldout_features(panel, target)
            objectives = {}
            solutions = {}
            for model_id in args.models:
                paths = full_fit_paths(args.root, panel_name, model_id, target, len(panel.group(target).components))
                missing = [str(path) for path in paths if not path.exists()]
                if missing:
                    raise FileNotFoundError(f"Full fits not ready: {missing[:2]}")
                fits = [joblib.load(path) for path in paths]
                fit_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
                objective = objective_from_fits(panel, model_id, target, fits)
                objectives[model_id] = objective
                parity = verify_predictor(objective, panel, target, fits)
                path = output / "fullfit" / panel_name / target / f"{model_id}.json"
                if path.exists():
                    record = json.loads(path.read_text())
                    if record["fit_hashes"] != fit_hashes:
                        raise ValueError("Frozen optimizer fits changed")
                    weights = np.asarray(record["continuous_weights"])
                else:
                    weights, record = optimize_objective(objective, panel, extra_starts=True, runtime_grid=True)
                    record.update(
                        panel=panel_name, target=target, model=model_id, fit_hashes=fit_hashes, predictor_parity=parity
                    )
                    record.update(geometric_metrics(weights, panel, bank_features.weights))
                    write_json(path, record)
                solutions[model_id] = weights
                rows.append({key: value for key, value in record.items() if not isinstance(value, (list, dict))})
                print(
                    json.dumps(
                        {
                            "panel": panel_name,
                            "target": target,
                            "model": model_id,
                            "prediction": record["continuous_prediction"],
                            "max_epoch": record["max_epoch"],
                        }
                    ),
                    flush=True,
                )
            cross = []
            for proposing, weights in solutions.items():
                for scoring, objective in objectives.items():
                    cross.append(
                        {
                            "proposal_model": proposing,
                            "scoring_model": scoring,
                            "prediction": float(objective.predict(weights)[0]),
                            "tv_to_scoring_optimum": float(np.abs(weights - solutions[scoring]).sum() / 2),
                        }
                    )
            pd.DataFrame(cross).to_csv(output / "fullfit" / panel_name / target / "cross_predictions.csv", index=False)
            fold_rows = []
            for model_id in args.models:
                full = objectives[model_id]
                full_solution = solutions[model_id]
                for fold in args.folds:
                    tasks = [
                        bench.FitTask(model_id, panel_name, target, i, str(component), 0, fold)
                        for i, component in enumerate(panel.group(target).components)
                    ]
                    paths = [refits.task_head_path(args.root / "refits", task) for task in tasks]
                    fit_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
                    fits = [refits.load_fitted(path) for path in paths]
                    objective = objective_from_fits(panel, model_id, target, fits)
                    parity = verify_predictor(objective, panel, target, fits)
                    path = output / "outerfold" / panel_name / target / model_id / f"fold_{fold:02d}.json"
                    if path.exists():
                        record = json.loads(path.read_text())
                        if record["fit_hashes"] != fit_hashes:
                            raise ValueError("Optimizer outerfold fits changed")
                    else:
                        weights, record = optimize_objective(objective, panel, extra_starts=False, runtime_grid=False)
                        record.update(
                            panel=panel_name,
                            target=target,
                            model=model_id,
                            fold=fold,
                            fit_hashes=fit_hashes,
                            predictor_parity=parity,
                            tv_to_fullfit=float(np.abs(weights - full_solution).sum() / 2),
                            fullfit_prediction=float(full.predict(weights)[0]),
                            fold_prediction_of_fullfit=float(objective.predict(full_solution)[0]),
                        )
                        record.update(geometric_metrics(weights, panel, bank_features.weights))
                        write_json(path, record)
                    fold_rows.append(
                        {key: value for key, value in record.items() if not isinstance(value, (list, dict))}
                    )
                    print(
                        json.dumps(
                            {
                                "panel": panel_name,
                                "target": target,
                                "model": model_id,
                                "fold": fold,
                                "tv_to_fullfit": record["tv_to_fullfit"],
                            }
                        ),
                        flush=True,
                    )
            pd.DataFrame(fold_rows).to_csv(output / "fullfit" / panel_name / target / "fold_stability.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "fullfit_summary.csv", index=False)


if __name__ == "__main__":
    main()
