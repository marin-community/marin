# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Registered native-floor penalty and independently signed phase comparisons."""

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
import causal_two_phase_flow_20260921_v2 as causal
import nextgen_two_phase_round2_20260921 as previous
import numpy as np
import pandas as pd
import round3_log_head_20260921 as log_head
from nextgen_two_phase_surrogate_20260921 import check_geometry
from scipy.linalg import helmert
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits

mariner = benchmark.mariner
ROOT = Path(__file__).resolve().parent / "reference_outputs/two_phase_nextgen_20260921/round3"
MODELS = {"NG2-007": "none", "NG2-008": "causal", "NG2-009": "contrast", "NG2-010": "increment"}
FORGETTING = (0.0, 1.0, 4.0, 16.0, 64.0)
SHAPE_PAIRS = previous.SHAPE_PAIRS


def source_hashes() -> dict[str, str]:
    paths = [Path(__file__), Path(causal.__file__), Path(log_head.__file__), ROOT / "registration.json"]
    return {**previous.source_hashes(), **{str(path.resolve()): benchmark.sha256(path) for path in paths}}


def causal_benefit(panel: benchmark.data.BenchPanel, shape: dict[str, float], mu: float) -> np.ndarray:
    return causal.causal_phase_benefit(
        panel.phase0_epochs,
        panel.phase1_epochs,
        panel.phase0,
        panel.phase1,
        panel.alpha,
        shape["rate"],
        shape["power"],
        mu,
        0.0,
        mode="additive",
    )


def geometry_identity(panel: benchmark.data.BenchPanel) -> dict[str, Any]:
    return {
        "panel": panel.name,
        "alpha": panel.alpha,
        "arrays": {
            name: hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
            for name, value in [
                ("phase0", panel.phase0),
                ("phase1", panel.phase1),
                ("early", panel.phase0_epochs),
                ("late", panel.phase1_epochs),
            ]
        },
    }


def geometry_cache(panel: benchmark.data.BenchPanel, folder: Path) -> np.ndarray:
    """Materialize outcome-free causal features, bound to code and geometry."""
    folder.mkdir(parents=True, exist_ok=True)
    receipt, archive = folder / f"{panel.name}.json", folder / f"{panel.name}.npz"
    identity = {"sources": source_hashes(), **geometry_identity(panel)}
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if record["identity"] != identity or record["sha256"] != benchmark.sha256(archive):
            raise ValueError("Causal geometry cache drift")
        with np.load(archive) as arrays:
            return arrays["benefits"]
    values = np.asarray(
        [[causal_benefit(panel, {"rate": rate, "power": power}, mu) for mu in FORGETTING] for rate, power in SHAPE_PAIRS]
    )
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite causal geometry")
    np.savez_compressed(archive, benefits=values)
    benchmark.write_json(receipt, {"identity": identity, "sha256": benchmark.sha256(archive)})
    return values


def phase_design(panel: benchmark.data.BenchPanel, shape: dict[str, float], model: str) -> np.ndarray:
    """Return tied-zero signed features; no outcome or inherited head is used."""
    kind = MODELS[model]
    if kind in ("none", "causal"):
        return np.empty((len(panel.rows), 0))
    if kind == "contrast":
        return panel.contrast @ helmert(len(panel.buckets), full=False).T
    values = mariner.benefit(panel.alpha * panel.epochs, shape["rate"], shape["power"]) - mariner.benefit(
        panel.phase0_epochs, shape["rate"], shape["power"]
    )
    # These coordinates have identical actual and counterfactual doses by the
    # constant-exposure contract. Preserve their algebraic zero in floating point.
    return np.where(panel.phase0 == panel.phase1, 0.0, values)


def matrices(
    panel: benchmark.data.BenchPanel,
    indices: np.ndarray,
    shape: dict[str, float],
    model: str,
    mu: float,
    benefits: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    check_geometry(panel)
    aggregate = mariner.design_matrix(panel.epochs[indices], shape)
    if MODELS[model] == "causal":
        if benefits is None:
            selected = dataclasses.replace(
                panel,
                phase0=panel.phase0[indices],
                phase1=panel.phase1[indices],
                rows=panel.rows.iloc[indices].reset_index(drop=True),
                targets={},
                metadata={
                    **panel.metadata,
                    "realized_phase0_epochs": panel.phase0_epochs[indices],
                    "realized_phase1_epochs": panel.phase1_epochs[indices],
                },
            )
            value = causal_benefit(selected, shape, mu)
        else:
            pair = SHAPE_PAIRS.index((shape["rate"], shape["power"]))
            value = benefits[pair, FORGETTING.index(mu), indices]
        aggregate[:, : len(panel.buckets)] = -value
    phase = phase_design(panel, shape, model)[indices]
    aggregate_penalty = (
        previous.feature_spans(panel, shape) if model in ("NG2-007", "NG2-008") else np.ones(aggregate.shape[1])
    )
    return aggregate, phase, aggregate_penalty, np.ones(phase.shape[1])


@dataclasses.dataclass(frozen=True)
class Fit:
    model: str
    shape: dict[str, float]
    forgetting: float
    ridge: float
    gamma: float
    flat_profile: bool
    inner_rmse: float
    head: log_head.Head

    def predict(self, panel: benchmark.data.BenchPanel, indices: np.ndarray) -> np.ndarray:
        aggregate, phase, _, _ = matrices(panel, indices, self.shape, self.model, self.forgetting)
        return self.head.predict(aggregate, phase)

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "shape": self.shape,
            "forgetting": self.forgetting,
            "ridge": self.ridge,
            "gamma": self.gamma,
            "flat_profile": self.flat_profile,
            "inner_rmse": self.inner_rmse,
            "floor": self.head.floor,
            "intercept": self.head.intercept,
            "aggregate_coefficients": self.head.aggregate_coefficients.tolist(),
            "phase_coefficients": self.head.phase_coefficients.tolist(),
        }


def cross_validation(
    aggregate: np.ndarray,
    phase: np.ndarray,
    response: np.ndarray,
    ridge: float,
    gamma: float,
    penalties: tuple[np.ndarray, np.ndarray],
    folds: mariner.Folds,
) -> float:
    squared_error, count = 0.0, 0
    for train, valid in folds:
        head = log_head.fit_log_head(
            aggregate[train],
            phase[train],
            response[train],
            ridge,
            mariner.FloorSpec(float("nan"), 0.0, gamma),
            *penalties,
        )
        predicted = head.predict(aggregate[valid], phase[valid])
        if not np.isfinite(predicted).all():
            return float("inf")
        squared_error += float(np.sum((predicted - response[valid]) ** 2))
        count += len(valid)
    return math.sqrt(squared_error / count)


def fit_model(
    panel: benchmark.data.BenchPanel,
    response: np.ndarray,
    train: np.ndarray,
    inner: np.ndarray,
    model: str,
    benefits: np.ndarray | None = None,
) -> Fit:
    if model not in MODELS or panel.tied[train].any():
        raise ValueError("Unknown model or tied labels used for fitting")
    y = np.asarray(response)[train]
    if not np.isfinite(y).all() or min(y) <= 0 or len(train) != len(inner) or min(inner) < 0:
        raise ValueError("Invalid fitting data")
    folds = tuple((np.flatnonzero(inner != fold), np.flatnonzero(inner == fold)) for fold in sorted(set(inner)))
    if len(folds) < 2:
        raise ValueError("Need at least two inner folds")
    if model == "NG2-008" and benefits is None:
        raise ValueError("Causal fitting requires a precomputed geometry cache")
    states = FORGETTING if model == "NG2-008" else (0.0,)
    best: tuple[float, int, int, int] | None = None
    for shape_index, shape in enumerate(mariner.SHAPES):
        for state_index, mu in enumerate(states):
            agg, phase, agg_penalty, phase_penalty = matrices(panel, train, shape, model, mu, benefits)
            for ridge_index, ridge in enumerate(mariner.RIDGE_GRID):
                score = cross_validation(
                    agg, phase, y, ridge, mariner.KAPPA_PROVISIONAL, (agg_penalty, phase_penalty), folds
                )
                candidate = (score, shape_index, state_index, ridge_index)
                if best is None or candidate < best:
                    best = candidate
    if best is None or not math.isfinite(best[0]):
        raise ValueError("No finite complete-grid candidate")
    _, shape_index, state_index, ridge_index = best
    shape, mu, ridge = dict(mariner.SHAPES[shape_index]), states[state_index], mariner.RIDGE_GRID[ridge_index]
    agg, phase, agg_penalty, phase_penalty = matrices(panel, train, shape, model, mu, benefits)

    def objective(log_gamma: float) -> float:
        score = cross_validation(agg, phase, y, ridge, math.exp(log_gamma), (agg_penalty, phase_penalty), folds)
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
    head = log_head.fit_log_head(
        agg, phase, y, ridge, mariner.FloorSpec(float("nan"), 0.0, gamma), agg_penalty, phase_penalty
    )
    return Fit(model, shape, mu, ridge, gamma, flat, score, head)


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
    benefits = geometry_cache(panel, output / "geometry") if model == "NG2-008" else None
    started = time.monotonic()
    with threadpool_limits(limits=1):
        fitted = fit_model(panel, panel.targets[target], train, training.inner_fold.to_numpy(int), model, benefits)
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
    if "NG2-008" in args.models:
        geometry_cache(panel, args.output / "geometry")
    tasks = [
        (str(args.prepared), args.panel, target, fold, model, str(args.output))
        for model in args.models
        for target in targets
        for fold in args.folds
    ]
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


if __name__ == "__main__":
    main()
