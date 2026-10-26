# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Grouped validation selection of a common absolute floor with native log fitting."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import benchmark_two_phase_20260921 as benchmark
import nextgen_two_phase_round5_20260921 as previous
import numpy as np
import pandas as pd
import round3_log_head_20260921 as log_head
import round6_floor_cv_20260921 as floor_cv
from threadpoolctl import threadpool_limits

mariner = benchmark.mariner
ROOT = Path(__file__).resolve().parent / "reference_outputs/two_phase_nextgen_20260921/round6"
BOUNDARY_STATES = previous.BOUNDARY_STATES


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    states: tuple[tuple[float, float], ...]


MODELS = {
    "NG2-016": ModelSpec(BOUNDARY_STATES[:1]),
    "NG2-017": ModelSpec(BOUNDARY_STATES),
}
CANDIDATES = tuple(MODELS)


def source_hashes() -> dict[str, str]:
    paths = (Path(__file__), Path(floor_cv.__file__), ROOT / "registration.json")
    return {**previous.source_hashes(), **{str(path.resolve()): benchmark.sha256(path) for path in paths}}


def matrices(
    panel: benchmark.data.BenchPanel, indices: np.ndarray, shape: dict[str, float], q: float, kappa: float
) -> np.ndarray:
    return previous.matrices(panel, indices, shape, q, kappa)


@dataclasses.dataclass(frozen=True)
class Fit:
    model: str
    shape: dict[str, float]
    q: float
    kappa: float
    ridge: float
    inner_rmse: float
    head: log_head.Head
    floor_search: dict[str, Any]

    def predict(self, panel: benchmark.data.BenchPanel, indices: np.ndarray) -> np.ndarray:
        matrix = matrices(panel, indices, self.shape, self.q, self.kappa)
        return self.head.predict(matrix, np.empty((len(indices), 0)))

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "loss": "log",
            "floor_estimator": "common_absolute_floor_grouped_cv",
            "shape": self.shape,
            "q": self.q,
            "kappa": self.kappa,
            "ridge": self.ridge,
            "inner_rmse": self.inner_rmse,
            "floor": self.head.floor,
            "intercept": self.head.intercept,
            "aggregate_coefficients": self.head.aggregate_coefficients.tolist(),
            "phase_coefficients": self.head.phase_coefficients.tolist(),
            "floor_search": self.floor_search,
        }


def search_shape(
    panel: benchmark.data.BenchPanel,
    y: np.ndarray,
    train: np.ndarray,
    folds: mariner.Folds,
    states: tuple[tuple[float, float], ...],
    indexed_shape: tuple[int, dict[str, float]],
) -> tuple[tuple[float, int, int, int, float], floor_cv.FloorCVResult, list[dict[str, Any]]]:
    """Search every state and ridge for one shape using one BLAS thread."""
    shape_index, shape = indexed_shape
    best: tuple[float, int, int, int, float] | None = None
    best_search = None
    configurations = []
    with threadpool_limits(limits=1):
        for state_index, (q, kappa) in enumerate(states):
            matrix = matrices(panel, train, shape, q, kappa)
            for ridge_index, ridge in enumerate(mariner.RIDGE_GRID):
                searched = floor_cv.search_floor_cv(matrix, y, ridge, folds)
                score = searched.inner_rmse
                candidate = (score, shape_index, state_index, ridge_index, searched.floor)
                configurations.append(
                    {
                        "shape_index": shape_index,
                        "state_index": state_index,
                        "ridge_index": ridge_index,
                        "floor": searched.floor,
                        "inner_rmse": score,
                    }
                )
                if best is None or candidate < best:
                    best, best_search = candidate, searched
    if best is None or best_search is None:
        raise ValueError("No complete state/ridge candidate for this shape")
    return best, best_search, configurations


def fit_model(
    panel: benchmark.data.BenchPanel,
    response: np.ndarray,
    train: np.ndarray,
    inner: np.ndarray,
    model: str,
    *,
    workers: int = 1,
) -> Fit:
    if model not in MODELS or panel.tied[train].any():
        raise ValueError("Unknown model or tied labels used for fitting")
    if workers < 1:
        raise ValueError("Shape search workers must be positive")
    y = np.asarray(response)[train]
    if not np.isfinite(y).all() or min(y) <= 0 or len(train) != len(inner) or min(inner) < 0:
        raise ValueError("Invalid fitting data")
    folds = tuple((np.flatnonzero(inner != fold), np.flatnonzero(inner == fold)) for fold in sorted(set(inner)))
    if len(folds) < 2:
        raise ValueError("Need at least two inner folds")
    spec = MODELS[model]
    best: tuple[float, int, int, int, float] | None = None
    best_search = None
    configurations = []
    search = functools.partial(search_shape, panel, y, train, folds, spec.states)
    with ProcessPoolExecutor(max_workers=workers) if workers > 1 else nullcontext() as pool:
        indexed_shapes = enumerate(mariner.SHAPES)
        results = map(search, indexed_shapes) if pool is None else pool.map(search, indexed_shapes)
        for candidate, searched, shape_configurations in results:
            configurations.extend(shape_configurations)
            if best is None or candidate < best:
                best, best_search = candidate, searched
    if best is None or best_search is None or not math.isfinite(best[0]):
        raise ValueError("No finite complete-grid candidate")
    score, shape_index, state_index, ridge_index, selected_floor = best
    shape = dict(mariner.SHAPES[shape_index])
    q, kappa = spec.states[state_index]
    ridge = mariner.RIDGE_GRID[ridge_index]
    matrix = matrices(panel, train, shape, q, kappa)
    head = floor_cv.fixed_floor_head(matrix, y, ridge, selected_floor)
    inner_heads = []
    for fitting, validation in folds:
        conditional = floor_cv.fixed_floor_head(matrix[fitting], y[fitting], ridge, selected_floor)
        inner_heads.append(
            {
                "training_rows": panel.rows.iloc[train[fitting]].row_id.tolist(),
                "validation_rows": panel.rows.iloc[train[validation]].row_id.tolist(),
                "training_minimum": float(np.min(y[fitting])),
                "floor": conditional.floor,
                "intercept": conditional.intercept,
                "aggregate_coefficients": conditional.aggregate_coefficients.tolist(),
                "phase_coefficients": conditional.phase_coefficients.tolist(),
            }
        )
    search_record = {
        "selected_configuration_search": best_search.to_json(),
        "configuration_minima": configurations,
        "selected_configuration_inner_heads": inner_heads,
        "final_floor_unchanged": head.floor == selected_floor,
    }
    return Fit(model, shape, q, kappa, ridge, score, head, search_record)


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
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(CANDIDATES))
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
