# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Registered clock-by-estimator comparison using asymmetric endpoints only."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import benchmark_two_phase_20260921 as benchmark
import causal_two_phase_flow_20260921 as causal
import nextgen_two_phase_flow_20260921 as endpoint
import nextgen_two_phase_surrogate_20260921 as previous
import numpy as np
import pandas as pd
import profiled_two_phase_head_20260921 as profile
from threadpoolctl import threadpool_limits

mariner = benchmark.mariner
ROOT = Path(__file__).resolve().parent / "reference_outputs/two_phase_nextgen_20260921/round2"
MODELS = {"NG2-004": ("endpoint", "profile"), "NG2-005": ("causal", "native"), "NG2-006": ("causal", "profile")}
FORGETTING = (0.0, 1.0, 4.0, 16.0, 64.0)
FLOORS = (-2.0, -1.0, 0.0, 0.25, 0.5, 0.75, 0.9, 0.97)
SHAPE_PAIRS = tuple((rate, power) for rate in mariner.SHAPE_RATES for power in mariner.SHAPE_POWERS)


def source_hashes() -> dict[str, str]:
    paths = [
        Path(__file__),
        Path(causal.__file__),
        Path(endpoint.__file__),
        Path(profile.__file__),
        Path(previous.__file__),
        ROOT / "registration.json",
    ]
    return {**benchmark.code_identity(), **{str(p.resolve()): benchmark.sha256(p) for p in paths}}


def benefit(
    panel: benchmark.data.BenchPanel, indices: np.ndarray, rate: float, power: float, forgetting: float, clock: str
) -> np.ndarray:
    previous.check_geometry(panel)
    function = causal.causal_phase_benefit if clock == "causal" else endpoint.phase_benefit
    return function(
        panel.phase0_epochs[indices],
        panel.phase1_epochs[indices],
        panel.phase0[indices],
        panel.phase1[indices],
        panel.alpha,
        rate,
        power,
        forgetting,
        0.0,
        mode="additive",
    )


def geometry_cache(panel: benchmark.data.BenchPanel, folder: Path, clock: str) -> np.ndarray:
    """Cache outcome-independent features; fit subsets select rows afterwards."""
    folder.mkdir(parents=True, exist_ok=True)
    identity = {
        "sources": source_hashes(),
        "panel": panel.name,
        "clock": clock,
        "geometry": {
            name: hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
            for name, value in [
                ("early", panel.phase0_epochs),
                ("late", panel.phase1_epochs),
                ("phase0", panel.phase0),
                ("phase1", panel.phase1),
            ]
        },
        "alpha": panel.alpha,
    }
    receipt_path, array_path = folder / f"{panel.name}_{clock}.json", folder / f"{panel.name}_{clock}.npz"
    if receipt_path.exists():
        record = json.loads(receipt_path.read_text())
        if record["identity"] != identity or record["sha256"] != benchmark.sha256(array_path):
            raise ValueError("Geometry cache drift; use a fresh output directory")
        with np.load(array_path) as arrays:
            return arrays["benefits"]
    indices = np.arange(len(panel.rows))
    values = np.asarray(
        [[benefit(panel, indices, rate, power, mu, clock) for mu in FORGETTING] for rate, power in SHAPE_PAIRS]
    )
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite causal or endpoint features")
    np.savez_compressed(array_path, benefits=values)
    benchmark.write_json(receipt_path, {"identity": identity, "sha256": benchmark.sha256(array_path)})
    return values


def design(
    panel: benchmark.data.BenchPanel, indices: np.ndarray, shape: dict[str, float], forgetting: float, clock: str
) -> np.ndarray:
    values = benefit(panel, indices, shape["rate"], shape["power"], forgetting, clock)
    return np.hstack([-values, mariner.harm(panel.epochs[indices], shape["threshold"])])


def feature_spans(panel: benchmark.data.BenchPanel, shape: dict[str, float]) -> np.ndarray:
    maximum = panel.c0 + panel.c1
    return np.concatenate(
        [
            2 * mariner.benefit(maximum, shape["rate"], shape["power"]),
            mariner.harm(maximum, shape["threshold"]) - mariner.harm(np.zeros_like(maximum), shape["threshold"]),
        ]
    )


@dataclasses.dataclass(frozen=True)
class Fit:
    model: str
    shape: dict[str, float]
    forgetting: float
    ridge: float
    rho: float | None
    gamma: float | None
    flat_profile: bool
    inner_rmse: float
    head: mariner.Head

    def predict(self, panel: benchmark.data.BenchPanel, indices: np.ndarray) -> np.ndarray:
        return self.head.predict(design(panel, indices, self.shape, self.forgetting, MODELS[self.model][0]))

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "shape": self.shape,
            "forgetting": self.forgetting,
            "ridge": self.ridge,
            "rho": self.rho,
            "gamma": self.gamma,
            "flat_profile": self.flat_profile,
            "inner_rmse": self.inner_rmse,
            "floor": self.head.floor,
            "intercept": self.head.intercept,
            "coefficients": self.head.coefficients.tolist(),
        }


def fit_model(
    panel: benchmark.data.BenchPanel,
    response: np.ndarray,
    train: np.ndarray,
    inner: np.ndarray,
    model: str,
    benefits: np.ndarray,
) -> Fit:
    """Select the complete registered grid with actual inner-training-only heads."""
    previous.check_geometry(panel)
    if model not in MODELS or panel.tied[train].any():
        raise ValueError("Unknown model or tied labels in training")
    y = np.asarray(response)[train]
    if not np.isfinite(y).all() or np.any(y <= 0) or len(train) != len(inner) or min(inner) < 0:
        raise ValueError("Invalid fitting rows or losses")
    folds = tuple((np.flatnonzero(inner != fold), np.flatnonzero(inner == fold)) for fold in sorted(set(inner)))
    if len(folds) < 2:
        raise ValueError("At least two inner folds are required")
    native = MODELS[model][1] == "native"
    floor_spec = mariner.FloorSpec(float("nan"), 0.0, mariner.KAPPA_PROVISIONAL)
    winner: tuple[float, int, int, int, int] | None = None
    for shape_index, shape in enumerate(mariner.SHAPES):
        pair = SHAPE_PAIRS.index((shape["rate"], shape["power"]))
        harm = mariner.harm(panel.epochs[train], shape["threshold"])
        spans = feature_spans(panel, shape)
        for state, _mu in enumerate(FORGETTING):
            matrix = np.hstack([-benefits[pair, state, train], harm])
            for ridge_index, ridge in enumerate(mariner.RIDGE_GRID):
                for floor_index, rho in enumerate((0.0,) if native else FLOORS):
                    score = (
                        mariner.cv_rmse(matrix, y, ridge, floor_spec, folds)
                        if native
                        else profile.profile_cv_rmse(matrix, y, ridge, rho, spans, folds)
                    )
                    value = (score, shape_index, state, ridge_index, floor_index)
                    if winner is None or value < winner:
                        winner = value
    if winner is None or not np.isfinite(winner[0]):
        raise ValueError("No finite complete-grid candidate")
    score, shape_index, state, ridge_index, floor_index = winner
    shape, mu, ridge = dict(mariner.SHAPES[shape_index]), FORGETTING[state], mariner.RIDGE_GRID[ridge_index]
    pair = SHAPE_PAIRS.index((shape["rate"], shape["power"]))
    matrix = np.hstack([-benefits[pair, state, train], mariner.harm(panel.epochs[train], shape["threshold"])])
    gamma, rho, flat = None, None, False
    if native:
        gamma, flat, score = mariner.search_kappa(matrix, y, ridge, mariner.Anchor(float("nan"), 0.0), folds)
        head = mariner.fit_head(matrix, y, ridge, mariner.FloorSpec(float("nan"), 0.0, gamma))
    else:
        rho = FLOORS[floor_index]
        head = profile.fit_profile_head(matrix, y, ridge, rho, feature_spans(panel, shape))
    return Fit(model, shape, mu, ridge, rho, gamma, flat, score, head)


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
            raise ValueError("Candidate source/config drift")
        benchmark.verify_hashes(record["outputs"], folder)
        return {"folder": str(folder), "cached": True}
    panel = benchmark.read_panel(prepared, panel_name)
    table = pd.read_csv(prepared / panel_name / "folds.csv", dtype={"row_id": str})
    table = table[(table.repeat == 0) & (table.outer_fold == outer)]
    index = pd.Series(np.arange(len(panel.rows)), index=panel.rows.row_id)
    table = table.loc[np.isfinite(panel.targets[target][index.loc[table.row_id].to_numpy(int)])]
    training, query = table[table.role.eq("train")], table[table.role.ne("train")]
    train, indices = index.loc[training.row_id].to_numpy(int), index.loc[query.row_id].to_numpy(int)
    values = geometry_cache(panel, output / "geometry", MODELS[model][0])
    started = time.monotonic()
    with threadpool_limits(limits=1):
        fitted = fit_model(panel, panel.targets[target], train, training.inner_fold.to_numpy(int), model, values)
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
            "outputs": {name: benchmark.sha256(folder / name) for name in ["fit.json", "predictions.csv"]},
        },
    )
    return {
        "folder": str(folder),
        "cached": False,
        "model": model,
        "target": target,
        "fold": outer,
        "runtime_seconds": record["runtime_seconds"],
    }


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
        raise ValueError("Prepared benchmark code differs from the current benchmark")
    if manifest["identity"]["environment"] != benchmark.environment_identity():
        raise ValueError("Prepared benchmark environment differs from the current environment")
    if manifest["identity"]["data_contract_sha256"] != benchmark.sha256(benchmark.PROTOCOL_DIR / "data_contract.json"):
        raise ValueError("Prepared benchmark data contract differs from the current contract")
    panel = benchmark.read_panel(args.prepared, args.panel)
    targets = benchmark.selected_targets(panel, args.targets, "screen")
    for clock in sorted({MODELS[model][0] for model in args.models}):
        geometry_cache(panel, args.output / "geometry", clock)
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
    benchmark.write_json(
        args.output / "run.json",
        {
            "sources": source_hashes(),
            "models": args.models,
            "panel": args.panel,
            "targets": targets,
            "folds": args.folds,
            "fits": len(records),
            "claim": "adaptive_offline_development_not_qualified",
            "lm_jobs": 0,
        },
    )


if __name__ == "__main__":
    main()
