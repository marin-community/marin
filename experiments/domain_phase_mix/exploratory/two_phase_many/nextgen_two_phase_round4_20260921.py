# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Registered shared boundary-benefit law crossed with conditional mean fitting."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import benchmark_two_phase_20260921 as benchmark
import nextgen_two_phase_round3_20260921 as previous
import numpy as np
import pandas as pd
import round3_log_head_20260921 as log_head
import round4_boundary_benefit_20260921 as boundary
import round4_mean_head_20260921 as mean_head
from nextgen_two_phase_surrogate_20260921 import check_geometry
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits

mariner = benchmark.mariner
ROOT = Path(__file__).resolve().parent / "reference_outputs/two_phase_nextgen_20260921/round4"
MODELS = {"NG2-011": "mean", "NG2-012": "log", "NG2-013": "mean"}
BOUNDARY_STATES = ((0.5, 0.0), *((q, kappa) for kappa in (0.5, 2.0) for q in (0.0, 0.5, 1.0)))


def source_hashes() -> dict[str, str]:
    paths = (Path(__file__), Path(boundary.__file__), Path(mean_head.__file__), ROOT / "registration.json")
    return {**previous.source_hashes(), **{str(path.resolve()): benchmark.sha256(path) for path in paths}}


def matrices(
    panel: benchmark.data.BenchPanel, indices: np.ndarray, shape: dict[str, float], q: float, kappa: float
) -> np.ndarray:
    check_geometry(panel)
    return boundary.design_matrix(panel, shape, q, kappa)[indices]


def conditional_head(
    matrix: np.ndarray, response: np.ndarray, ridge: float, gamma: float, model: str
) -> log_head.Head | mean_head.Head:
    phase = np.empty((len(response), 0))
    spec = mariner.FloorSpec(float("nan"), 0.0, gamma)
    solver = mean_head.fit_mean_head if MODELS[model] == "mean" else log_head.fit_log_head
    return solver(matrix, phase, response, ridge, spec, np.ones(matrix.shape[1]), np.empty(0))


@dataclasses.dataclass(frozen=True)
class Fit:
    model: str
    shape: dict[str, float]
    q: float
    kappa: float
    ridge: float
    gamma: float
    flat_profile: bool
    inner_rmse: float
    head: log_head.Head | mean_head.Head

    def predict(self, panel: benchmark.data.BenchPanel, indices: np.ndarray) -> np.ndarray:
        matrix = matrices(panel, indices, self.shape, self.q, self.kappa)
        return self.head.predict(matrix, np.empty((len(indices), 0)))

    def to_json(self) -> dict[str, Any]:
        result = {
            "model": self.model,
            "loss": MODELS[self.model],
            "shape": self.shape,
            "q": self.q,
            "kappa": self.kappa,
            "ridge": self.ridge,
            "gamma": self.gamma,
            "flat_profile": self.flat_profile,
            "inner_rmse": self.inner_rmse,
            "floor": self.head.floor,
            "intercept": self.head.intercept,
            "aggregate_coefficients": self.head.aggregate_coefficients.tolist(),
            "phase_coefficients": self.head.phase_coefficients.tolist(),
        }
        if isinstance(self.head, mean_head.Head):
            result["solver"] = {
                "objective": self.head.objective,
                "iterations": self.head.iterations,
                "certified_kkt_residual": self.head.certified_kkt_residual,
            }
        return result


def cross_validation(
    matrix: np.ndarray, response: np.ndarray, ridge: float, gamma: float, model: str, folds: mariner.Folds
) -> float:
    squared_error, count = 0.0, 0
    for train, valid in folds:
        head = conditional_head(matrix[train], response[train], ridge, gamma, model)
        prediction = head.predict(matrix[valid], np.empty((len(valid), 0)))
        if not np.isfinite(prediction).all():
            return float("inf")
        squared_error += float(np.sum((prediction - response[valid]) ** 2))
        count += len(valid)
    return math.sqrt(squared_error / count)


def fit_model(
    panel: benchmark.data.BenchPanel, response: np.ndarray, train: np.ndarray, inner: np.ndarray, model: str
) -> Fit:
    if model not in MODELS or panel.tied[train].any():
        raise ValueError("Unknown model or tied labels used for fitting")
    y = np.asarray(response)[train]
    if not np.isfinite(y).all() or min(y) <= 0 or len(train) != len(inner) or min(inner) < 0:
        raise ValueError("Invalid fitting data")
    folds = tuple((np.flatnonzero(inner != fold), np.flatnonzero(inner == fold)) for fold in sorted(set(inner)))
    if len(folds) < 2:
        raise ValueError("Need at least two inner folds")
    states = BOUNDARY_STATES[:1] if model == "NG2-011" else BOUNDARY_STATES
    best: tuple[float, int, int, int] | None = None
    for shape_index, shape in enumerate(mariner.SHAPES):
        for state_index, (q, kappa) in enumerate(states):
            matrix = matrices(panel, train, shape, q, kappa)
            for ridge_index, ridge in enumerate(mariner.RIDGE_GRID):
                score = cross_validation(matrix, y, ridge, mariner.KAPPA_PROVISIONAL, model, folds)
                candidate = (score, shape_index, state_index, ridge_index)
                if best is None or candidate < best:
                    best = candidate
    if best is None or not math.isfinite(best[0]):
        raise ValueError("No finite complete-grid candidate")
    _, shape_index, state_index, ridge_index = best
    shape, (q, kappa), ridge = dict(mariner.SHAPES[shape_index]), states[state_index], mariner.RIDGE_GRID[ridge_index]
    matrix = matrices(panel, train, shape, q, kappa)

    def objective(log_gamma: float) -> float:
        score = cross_validation(matrix, y, ridge, math.exp(log_gamma), model, folds)
        return score if math.isfinite(score) else mariner.INFINITE_CV_PENALTY

    low, high = mariner.KAPPA_BOUNDS
    search = minimize_scalar(
        objective,
        bounds=(math.log(low), math.log(high)),
        method="bounded",
        options={"maxiter": mariner.KAPPA_SEARCH_EVALUATIONS, "xatol": mariner.KAPPA_SEARCH_XATOL},
    )
    gamma, score = math.exp(search.x), float(search.fun)
    flat = math.log(gamma) >= (1 - mariner.FLAT_PROFILE_FRACTION) * math.log(high)
    if flat:
        gamma = mariner.FLAT_PROFILE_KAPPA
        score = objective(math.log(gamma))
    head = conditional_head(matrix, y, ridge, gamma, model)
    return Fit(model, shape, q, kappa, ridge, gamma, flat, score, head)


def fit_shard(arguments: tuple[str, str, str, int, str, str]) -> dict[str, Any]:
    prepared_name, panel_name, target, outer, model, output_name = arguments
    prepared, output = Path(prepared_name), Path(output_name)
    key = hashlib.sha256(json.dumps([panel_name, target, outer, model]).encode()).hexdigest()[:20]
    folder = output / "shards" / key
    identity = {
        "sources": source_hashes(),
        "environment": benchmark.environment_identity(),
        "panel": panel_name,
        "target": target,
        "outer_fold": outer,
        "model": model,
        "prepared_manifest": benchmark.sha256(prepared / "manifest.json"),
    }
    complete = folder / "complete.json"
    if complete.exists():
        record = json.loads(complete.read_text())
        if record["identity"] != identity:
            raise ValueError("Source/configuration drift; use a new output folder")
        benchmark.verify_hashes(record["outputs"], folder)
        return {"folder": str(folder), "cached": True, "model": model, "target": target, "fold": outer}
    panel = benchmark.read_panel(prepared, panel_name)
    table = pd.read_csv(prepared / panel_name / "folds.csv", dtype={"row_id": str})
    table = table[(table.repeat == 0) & (table.outer_fold == outer)]
    index = pd.Series(np.arange(len(panel.rows)), index=panel.rows.row_id)
    table = table.loc[np.isfinite(panel.targets[target][index.loc[table.row_id].to_numpy(int)])]
    training, query = table[table.role.eq("train")], table[table.role.ne("train")]
    train, indices = index.loc[training.row_id].to_numpy(int), index.loc[query.row_id].to_numpy(int)
    started = time.monotonic()
    with threadpool_limits(limits=1):
        fitted = fit_model(panel, panel.targets[target], train, training.inner_fold.to_numpy(int), model)
        predicted = fitted.predict(panel, indices)
    if not np.isfinite(predicted).all():
        raise ValueError("Nonfinite held-out predictions")
    frame = panel.rows.iloc[indices][["row_id", "policy_id", "group_id", "source", "region"]].copy()
    frame["model"], frame["panel"], frame["target"] = model, panel_name, target
    frame["repeat"], frame["outer_fold"], frame["role"] = 0, outer, query.role.to_numpy()
    frame["observed"], frame["predicted"] = panel.targets[target][indices], predicted
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / "predictions.csv", index=False)
    record = fitted.to_json() | {
        "training_rows": training.row_id.tolist(),
        "inner_folds": training.inner_fold.tolist(),
        "runtime_seconds": time.monotonic() - started,
    }
    benchmark.write_json(folder / "fit.json", record)
    benchmark.write_json(
        complete,
        {
            "identity": identity,
            "outputs": {name: benchmark.sha256(folder / name) for name in ("fit.json", "predictions.csv")},
        },
    )
    return {"folder": str(folder), "cached": False, "model": model, "target": target, "fold": outer}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=benchmark.DEFAULT_OUTPUT / "v3/prepared")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", default="starcoder_wsd80_dense")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--targets", nargs="+", default=list(benchmark.QUICK_TASKS))
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1 or not set(args.folds).issubset(range(5)):
        raise ValueError("Invalid worker or fold selection")
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    benchmark.verify_hashes(manifest["sources"])
    benchmark.verify_hashes(manifest["identity"]["code"])
    benchmark.verify_hashes(manifest["outputs"], args.prepared)
    if manifest["identity"]["code"] != benchmark.code_identity():
        raise ValueError("Benchmark code differs from the prepared manifest")
    if manifest["identity"]["environment"] != benchmark.environment_identity():
        raise ValueError("Benchmark environment differs from the prepared manifest")
    if manifest["identity"]["data_contract_sha256"] != benchmark.sha256(benchmark.PROTOCOL_DIR / "data_contract.json"):
        raise ValueError("Benchmark data contract differs from the prepared manifest")
    panel = benchmark.read_panel(args.prepared, args.panel)
    targets = benchmark.selected_targets(panel, args.targets, "screen")
    tasks = [
        (str(args.prepared), args.panel, target, fold, model, str(args.output))
        for model in args.models
        for target in targets
        for fold in args.folds
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(fit_shard, task) for task in tasks]):
            result = future.result()
            records.append(result)
            print(json.dumps(result), flush=True)
    frames = [
        pd.read_csv(Path(r["folder"]) / "predictions.csv", keep_default_na=False, float_precision="round_trip")
        for r in records
    ]
    predictions = pd.concat(frames, ignore_index=True).sort_values(
        ["model", "panel", "target", "repeat", "outer_fold", "row_id"]
    )
    predictions.to_csv(args.output / "predictions.csv", index=False)
    benchmark.scoring.score_predictions(predictions).to_csv(args.output / "metrics.csv", index=False)
    benchmark.scoring.summarize_predictions(predictions).to_csv(args.output / "summary.csv", index=False)

    benchmark.write_json(
        args.output / "run.json",
        {
            "sources": source_hashes(),
            "environment": benchmark.environment_identity(),
            "prepared_manifest": benchmark.sha256(args.prepared / "manifest.json"),
            "models": args.models,
            "targets": targets,
            "folds": args.folds,
            "completed_fits": len(records),
            "cached_fits": sum(r["cached"] for r in records),
            "outputs": {
                name: benchmark.sha256(args.output / name) for name in ("predictions.csv", "metrics.csv", "summary.csv")
            },
        },
    )


if __name__ == "__main__":
    main()
