# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Joint asymmetric-only MARINER fits with conserved acquisition-memory flow.

The flow changes benefit features; physical cumulative harm stays unchanged.
The entire native shape/ridge/kinetics search is evaluated inside the supplied
inner folds at provisional gamma, followed by the native scalar gamma search.
No tied labels, inherited one-phase fit, or matched gain enters fitting.
"""

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
import nextgen_two_phase_flow_20260921 as flow
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

mariner = benchmark.mariner
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "reference_outputs/two_phase_nextgen_20260921"
PREPARED = benchmark.DEFAULT_OUTPUT / "v3/prepared"
MODELS = ("NG2-001", "NG2-002", "NG2-003")
FORGETTING = (1.0, 4.0, 16.0, 64.0)
CONSOLIDATION = (0.0, 1.0, 4.0, 16.0)


def kinetics(model: str) -> tuple[tuple[float, float], ...]:
    if model not in MODELS:
        raise ValueError(f"Unknown registered model: {model}")
    maturation = (0.0,) if model == "NG2-001" else CONSOLIDATION
    return ((0.0, 0.0), *((forget, consolidate) for consolidate in maturation for forget in FORGETTING))


def check_geometry(panel: benchmark.data.BenchPanel) -> None:
    if not (
        np.allclose(panel.phase0_epochs, panel.phase0 * panel.c0, rtol=1e-10, atol=1e-12)
        and np.allclose(panel.phase1_epochs, panel.phase1 * panel.c1, rtol=1e-10, atol=1e-12)
        and np.allclose(panel.c0 / (panel.c0 + panel.c1), panel.alpha, rtol=1e-9, atol=1e-12)
    ):
        raise ValueError(
            "This registered flow adapter requires constant exposure geometry; discrete support not qualified"
        )


def design(
    panel: benchmark.data.BenchPanel,
    indices: np.ndarray,
    shape: dict[str, float],
    model: str,
    forgetting: float,
    consolidation: float,
) -> np.ndarray:
    """Return shared benefit/harm columns with an exact MARINER diagonal."""
    check_geometry(panel)
    if model not in MODELS or (model == "NG2-001" and consolidation != 0):
        raise ValueError("Unknown model or non-scalar kinetics for scalar ablation")
    early, late = panel.phase0_epochs[indices], panel.phase1_epochs[indices]
    benefit = flow.phase_benefit(
        early,
        late,
        panel.phase0[indices],
        panel.phase1[indices],
        panel.alpha,
        shape["rate"],
        shape["power"],
        forgetting,
        consolidation,
        mode="ratio" if model == "NG2-003" else "additive",
    )
    return np.hstack([-benefit, mariner.harm(early + late, shape["threshold"])])


@dataclasses.dataclass(frozen=True)
class Fit:
    model: str
    shape: dict[str, float]
    ridge: float
    kappa: float
    forgetting: float
    consolidation: float
    flat_profile: bool
    inner_rmse: float
    head: mariner.Head

    def predict(self, panel: benchmark.data.BenchPanel, indices: np.ndarray) -> np.ndarray:
        return self.head.predict(design(panel, indices, self.shape, self.model, self.forgetting, self.consolidation))

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "shape": self.shape,
            "ridge": self.ridge,
            "kappa": self.kappa,
            "forgetting": self.forgetting,
            "consolidation": self.consolidation,
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
) -> Fit:
    """Fit every parameter using only asymmetric train rows and supplied inner folds."""
    check_geometry(panel)
    if panel.rows.iloc[train].tied.any():
        raise ValueError("Tied labels reached candidate fitting")
    y = np.asarray(response, dtype=float)[train]
    if not np.isfinite(y).all() or len(inner) != len(train) or min(inner) < 0 or len(set(inner)) < 2:
        raise ValueError("Candidate needs finite train labels and at least two inner folds")
    splits = tuple((np.flatnonzero(inner != fold), np.flatnonzero(inner == fold)) for fold in sorted(set(inner)))
    floor_spec = mariner.FloorSpec(float("nan"), 0.0, mariner.KAPPA_PROVISIONAL)
    early, late = panel.phase0_epochs[train], panel.phase1_epochs[train]
    total = early + late
    # The kinetics cache is geometry-only; floors and all fitted amplitudes stay
    # local to each actual inner fit. It avoids six identical flow evaluations
    # across the harm-threshold grid without pruning the estimator.
    benefits = {}
    for rate in mariner.SHAPE_RATES:
        for power in mariner.SHAPE_POWERS:
            for forgetting, consolidation in kinetics(model):
                benefits[rate, power, forgetting, consolidation] = flow.phase_benefit(
                    early,
                    late,
                    panel.phase0[train],
                    panel.phase1[train],
                    panel.alpha,
                    rate,
                    power,
                    forgetting,
                    consolidation,
                    mode="ratio" if model == "NG2-003" else "additive",
                )
    harms = {threshold: mariner.harm(total, threshold) for threshold in mariner.SHAPE_THRESHOLDS}
    winner: tuple[float, int, int, int] | None = None
    for shape_index, shape in enumerate(mariner.SHAPES):
        for state_index, (forgetting, consolidation) in enumerate(kinetics(model)):
            matrix = np.hstack(
                [-benefits[shape["rate"], shape["power"], forgetting, consolidation], harms[shape["threshold"]]]
            )
            for ridge_index, ridge in enumerate(mariner.RIDGE_GRID):
                score = mariner.cv_rmse(matrix, y, ridge, floor_spec, splits)
                current = score, shape_index, state_index, ridge_index
                if winner is None or current < winner:
                    winner = current
    if winner is None or not np.isfinite(winner[0]):
        raise ValueError("No finite inner-CV candidate")
    _, shape_index, state_index, ridge_index = winner
    shape = dict(mariner.SHAPES[shape_index])
    forgetting, consolidation = kinetics(model)[state_index]
    ridge = mariner.RIDGE_GRID[ridge_index]
    matrix = design(panel, train, shape, model, forgetting, consolidation)
    anchor = mariner.Anchor(float("nan"), 0.0)
    kappa, flat, score = mariner.search_kappa(matrix, y, ridge, anchor, splits)
    head = mariner.fit_head(matrix, y, ridge, mariner.FloorSpec(float("nan"), 0.0, kappa))
    return Fit(model, shape, ridge, kappa, forgetting, consolidation, flat, score, head)


def source_hashes() -> dict[str, str]:
    return {
        **benchmark.code_identity(),
        **{
            str(path.resolve()): benchmark.sha256(path)
            for path in (Path(__file__), Path(flow.__file__), OUTPUT / "registration.json")
        },
    }


def fit_shard(arguments: tuple[str, str, str, int, str, str]) -> dict[str, Any]:
    prepared_name, panel_name, target, outer, model, output_name = arguments
    prepared, output = Path(prepared_name), Path(output_name)
    key = hashlib.sha256(json.dumps([panel_name, target, outer, model]).encode()).hexdigest()[:20]
    folder = output / "shards" / key
    identity = {
        "prepared_manifest": benchmark.sha256(prepared / "manifest.json"),
        "sources": source_hashes(),
        "environment": benchmark.environment_identity(),
        "panel": panel_name,
        "target": target,
        "outer_fold": outer,
        "repeat": 0,
        "model": model,
    }
    complete = folder / "complete.json"
    if complete.exists():
        previous = json.loads(complete.read_text())
        if previous["identity"] != identity:
            raise ValueError(f"Candidate shard source/config drift: {folder}")
        benchmark.verify_hashes(previous["outputs"], folder)
        return {"folder": str(folder), "cached": True}
    panel = benchmark.read_panel(prepared, panel_name)
    fold_table = pd.read_csv(prepared / panel_name / "folds.csv", dtype={"row_id": str})
    fold_table = fold_table[(fold_table.repeat == 0) & (fold_table.outer_fold == outer)]
    index = pd.Series(np.arange(len(panel.rows)), index=panel.rows.row_id)
    available = np.isfinite(panel.targets[target][index.loc[fold_table.row_id].to_numpy(int)])
    fold_table = fold_table.loc[available]
    training, query = fold_table[fold_table.role.eq("train")], fold_table[fold_table.role.ne("train")]
    train_indices = index.loc[training.row_id].to_numpy(int)
    query_indices = index.loc[query.row_id].to_numpy(int)
    started = time.monotonic()
    with threadpool_limits(limits=1):
        fitted = fit_model(panel, panel.targets[target], train_indices, training.inner_fold.to_numpy(int), model)
        predicted = fitted.predict(panel, query_indices)
    if not np.isfinite(predicted).all():
        raise ValueError("Nonfinite held-out candidate predictions")
    frame = panel.rows.iloc[query_indices][["row_id", "policy_id", "group_id", "source", "region"]].copy()
    frame["model"], frame["panel"], frame["target"] = model, panel_name, target
    frame["repeat"], frame["outer_fold"] = 0, outer
    frame["role"] = query.role.to_numpy()
    frame["observed"], frame["predicted"] = panel.targets[target][query_indices], predicted
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / "predictions.csv", index=False)
    fit_record = fitted.to_json() | {
        "training_rows": training.row_id.tolist(),
        "inner_folds": training.inner_fold.tolist(),
        "runtime_seconds": time.monotonic() - started,
    }
    benchmark.write_json(folder / "fit.json", fit_record)
    benchmark.write_json(
        complete,
        {
            "identity": identity,
            "outputs": {path.name: benchmark.sha256(path) for path in (folder / "predictions.csv", folder / "fit.json")},
        },
    )
    return {
        "folder": str(folder),
        "cached": False,
        "model": model,
        "target": target,
        "fold": outer,
        "runtime_seconds": fit_record["runtime_seconds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=PREPARED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", default="starcoder_wsd80_dense")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--targets", nargs="+", default=list(benchmark.QUICK_TASKS))
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1 or not set(args.folds).issubset(range(5)):
        raise ValueError("Invalid workers or frozen outer folds")
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    benchmark.verify_hashes(manifest["sources"])
    benchmark.verify_hashes(manifest["identity"]["code"])
    benchmark.verify_hashes(manifest["outputs"], args.prepared)
    panel = benchmark.read_panel(args.prepared, args.panel)
    targets = benchmark.selected_targets(panel, args.targets, "screen")
    arguments = [
        (str(args.prepared), args.panel, target, fold, model, str(args.output))
        for model in args.models
        for target in targets
        for fold in args.folds
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(fit_shard, argument) for argument in arguments]):
            result = future.result()
            rows.append(result)
            print(json.dumps(result), flush=True)
    frames = [
        pd.read_csv(Path(row["folder"]) / "predictions.csv", keep_default_na=False, float_precision="round_trip")
        for row in rows
    ]
    predictions = pd.concat(frames, ignore_index=True).sort_values(
        ["model", "panel", "target", "repeat", "outer_fold", "row_id"]
    )
    args.output.mkdir(parents=True, exist_ok=True)
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
            "outer_folds": args.folds,
            "candidate_fits": len(rows),
            "claim": "adaptive_offline_development_not_qualified",
            "lm_jobs": 0,
            "prepared_manifest": benchmark.sha256(args.prepared / "manifest.json"),
        },
    )


if __name__ == "__main__":
    main()
