# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Run the registered, local-only complete-TPP40 temporal diagnostic round."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import benchmark_two_phase_20260921 as benchmark
import numpy as np
import pandas as pd
import tpp40_temporal_data_20261014 as data
import tpp40_temporal_models_20261014 as models
import two_phase_benchmark_scoring_20260921 as scoring
from scipy.optimize import minimize
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reference_outputs/tpp40_temporal_round_20261014"
SEED = 20261014
SYNTHETIC_WORLDS = ("aggregate_null", "benefit_clock_r4", "joint_persistence_half", "pool_order_positive")


def identity() -> dict:
    paths = [
        Path(__file__),
        Path(data.__file__),
        Path(models.__file__),
        Path(models.mariner.__file__),
        Path(scoring.__file__),
        OUTPUT / "registration.json",
        OUTPUT / "audit_addendum.json",
        OUTPUT / "data/manifest.json",
    ]
    return {str(path.resolve()): benchmark.sha256(path) for path in paths}


def partitions(panel: data.TemporalData, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if fold == -1:
        train = np.flatnonzero(~panel.panel.tied)
        return train, np.arange(len(panel.rows)), panel.full_data_inner_labels[train]
    frame = panel.folds[panel.folds.outer_fold.eq(fold)]
    train = frame[frame.role.eq("train")]
    query = frame[frame.role.ne("train")]
    return train.row_index.to_numpy(int), query.row_index.to_numpy(int), train.inner_fold.to_numpy(int)


def verify_fit(folder: Path) -> None:
    receipt = json.loads((folder / "complete.json").read_text())
    if receipt["identity"] != identity():
        raise ValueError(f"Source drift in {folder}")
    benchmark.verify_hashes(receipt["outputs"], folder)


def real_shard(args: tuple[int, int]) -> dict:
    fold, task = args
    folder = OUTPUT / "fits" / f"fold{fold}_task{task}"
    receipt = folder / "complete.json"
    frozen = identity()
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous["identity"] != frozen:
            raise ValueError(f"Source drift in {folder}")
        benchmark.verify_hashes(previous["outputs"], folder)
        return {"fold": fold, "task": task, "cached": True}
    panel = data.load_complete_tpp40()
    train, query, labels = partitions(panel, fold)
    early, late = panel.panel.phase0_epochs, panel.panel.phase1_epochs
    start = time.monotonic()
    with threadpool_limits(limits=1):
        fits = models.fit_bundle(
            early[train],
            late[train],
            panel.final_targets[train, task],
            labels,
            panel.panel.alpha,
            np.asarray(panel.panel.metadata["pool_tokens"]),
        )
    records = []
    for name, fit in fits.items():
        predicted = fit.predict(early[query], late[query])
        total = early[query] + late[query]
        tied = fit.predict(panel.panel.alpha * total, (1 - panel.panel.alpha) * total)
        for index, pred, diagonal in zip(query, predicted, tied, strict=True):
            row = panel.rows.iloc[index]
            records.append(
                {
                    "model": name,
                    "panel": "tpp40_complete",
                    "target": panel.task_names[task],
                    "task_index": task,
                    "repeat": 0,
                    "outer_fold": fold,
                    "row_index": int(index),
                    "row_id": row.row_id,
                    "policy_id": row.policy_id,
                    "group_id": row.group_id,
                    "source": row.source,
                    "region": row.region,
                    "role": (
                        ("tied_diagnostic" if row.tied else "train")
                        if fold == -1
                        else ("tied_test" if row.tied else "test")
                    ),
                    "observed": float(panel.final_targets[index, task]),
                    "predicted": float(pred),
                    "predicted_tied_counterfactual": float(diagonal),
                }
            )
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(folder / "predictions.csv", index=False)
    benchmark.write_json(folder / "models.json", {name: fit.to_json() for name, fit in fits.items()})
    elapsed = time.monotonic() - start
    benchmark.write_json(
        receipt,
        {
            "identity": frozen,
            "fold": fold,
            "task": task,
            "train_rows": train.tolist(),
            "query_rows": query.tolist(),
            "outputs": {name: benchmark.sha256(folder / name) for name in ("predictions.csv", "models.json")},
            "elapsed_seconds": elapsed,
        },
    )
    return {"fold": fold, "task": task, "seconds": round(elapsed, 2), "cached": False}


def synthetic_truth(panel: data.TemporalData, world: str, floor_compatible: bool = False) -> models.TemporalFit:
    rng = np.random.default_rng(SEED)
    count = len(panel.panel.buckets)
    pools = np.asarray(panel.panel.metadata["pool_tokens"])
    coefficients = np.r_[rng.uniform(0.03, 0.09, count), rng.uniform(0.002, 0.008, count)]
    if floor_compatible:
        coefficients *= 4.0
    names = {
        "aggregate_null": ("aggregate_mariner", {}),
        "benefit_clock_r4": ("benefit_clock", {"multiplier": 4.0}),
        "joint_persistence_half": ("joint_persistence", {"benefit_loss": 0.5, "harm_loss": 0.5}),
        "pool_order_positive": ("pool_order", {"order_coefficient": 1.0}),
    }
    name, parameters = names[world]
    large = np.log(pools) >= np.median(np.log(pools))
    shape = {"rate": 0.25, "power": 0.7, "threshold": 3.0}
    intercept = 0.3
    if floor_compatible:
        matrix, offset = models.design_and_offset(
            panel.panel.phase0_epochs, panel.panel.phase1_epochs, panel.panel.alpha, shape, name, parameters, large
        )
        intercept = float(np.log(0.4 / np.median(np.exp(matrix @ coefficients + offset))))
    return models.TemporalFit(
        name,
        panel.panel.alpha,
        large,
        shape,
        0.01,
        2.5,
        False,
        0.0,
        0.0,
        parameters,
        models.mariner.Head(intercept, coefficients, 0.6),
        (),
    )


def synthetic_shard(args: tuple[str, float, bool]) -> dict:
    world, noise, floor_compatible = args
    generator = "floor_compatible" if floor_compatible else "floor_stress"
    folder = OUTPUT / "synthetic" / generator / f"{world}_noise{noise:g}"
    receipt = folder / "complete.json"
    frozen = identity()
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous["identity"] != frozen:
            raise ValueError(f"Source drift in {folder}")
        benchmark.verify_hashes(previous["outputs"], folder)
        return {"world": world, "noise": noise, "generator": generator, "cached": True}
    panel = data.load_complete_tpp40()
    train, outer_query, labels = partitions(panel, 0)
    asymmetric_query = outer_query[~panel.panel.tied[outer_query]]
    tied_query = np.flatnonzero(panel.panel.tied)
    query = np.r_[asymmetric_query, tied_query]
    asymmetric = np.arange(len(asymmetric_query))
    tied_diagnostic = np.arange(len(asymmetric_query), len(query))
    early, late = panel.panel.phase0_epochs, panel.panel.phase1_epochs
    truth = synthetic_truth(panel, world, floor_compatible)
    clean = truth.predict(early, late)
    measured = clean + np.random.default_rng(SEED + 1).normal(0, noise, len(clean))
    with threadpool_limits(limits=1):
        fits = models.fit_bundle(
            early[train],
            late[train],
            measured[train],
            labels,
            panel.panel.alpha,
            np.asarray(panel.panel.metadata["pool_tokens"]),
        )
    total = early[query] + late[query]
    e0, e1 = panel.panel.alpha * total, (1 - panel.panel.alpha) * total
    true_tied = truth.predict(e0, e1)
    summaries, predictions = [], []
    for name, fit in fits.items():
        pred = fit.predict(early[query], late[query])
        diagonal = fit.predict(e0, e1)
        truth_gain = true_tied - clean[query]
        pred_gain = diagonal - pred
        selected = int(asymmetric[np.argmin(pred[asymmetric])])
        joint_selected = int(np.argmin(pred))
        summaries.append(
            {
                "world": world,
                "generator": generator,
                "noise_sd": noise,
                "model": name,
                "asymmetric_test_n": len(asymmetric),
                "actual_tied_diagnostic_n": len(tied_diagnostic),
                "heldout_rmse": float(np.sqrt(np.mean((pred[asymmetric] - clean[asymmetric_query]) ** 2))),
                "actual_tied_rmse": float(np.sqrt(np.mean((pred[tied_diagnostic] - clean[tied_query]) ** 2))),
                "counterfactual_tied_rmse": float(np.sqrt(np.mean((diagonal[asymmetric] - true_tied[asymmetric]) ** 2))),
                "phase_gain_rmse": float(np.sqrt(np.mean((pred_gain[asymmetric] - truth_gain[asymmetric]) ** 2))),
                "max_abs_predicted_phase_gain": float(np.max(np.abs(pred_gain[asymmetric]))),
                "max_true_phase_gain": float(np.max(truth_gain[asymmetric])),
                "regret_at1": float(clean[query[selected]] - np.min(clean[asymmetric_query])),
                "joint_diagnostic_regret_at1": float(clean[query[joint_selected]] - np.min(clean[query])),
                "selected_optimism": float(clean[query[selected]] - pred[selected]),
                "selected_predicted_phase_gain": float(pred_gain[selected]),
                "selected_true_phase_gain": float(truth_gain[selected]),
            }
        )
        for index, p, d, gt in zip(query, pred, diagonal, true_tied, strict=True):
            predictions.append(
                {
                    "model": name,
                    "generator": generator,
                    "row_index": int(index),
                    "role": "tied_diagnostic" if panel.panel.tied[index] else "test",
                    "truth": float(clean[index]),
                    "predicted": float(p),
                    "tied_truth": float(gt),
                    "tied_predicted": float(d),
                }
            )
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(folder / "scores.csv", index=False)
    pd.DataFrame(predictions).to_csv(folder / "predictions.csv", index=False)
    benchmark.write_json(folder / "models.json", {name: fit.to_json() for name, fit in fits.items()})
    benchmark.write_json(folder / "truth.json", truth.to_json())
    benchmark.write_json(
        receipt,
        {
            "identity": frozen,
            "generator": generator,
            "train": train.tolist(),
            "query": query.tolist(),
            "primary_asymmetric_query": asymmetric_query.tolist(),
            "additional_tied_diagnostic_query": tied_query.tolist(),
            "query_scope": (
                "Primary metrics use 66 asymmetric outer-fold0 policies; "
                "42 actual tied policies are separate evaluation-only diagnostics."
            ),
            "outputs": {
                name: benchmark.sha256(folder / name)
                for name in ("scores.csv", "predictions.csv", "models.json", "truth.json")
            },
        },
    )
    return {"world": world, "noise": noise, "generator": generator, "cached": False}


def objective_predictions(frame: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    keys = [
        "model",
        "panel",
        "repeat",
        "outer_fold",
        "row_index",
        "row_id",
        "policy_id",
        "group_id",
        "source",
        "region",
        "role",
    ]
    result = []
    for _, group in frame.groupby(keys, sort=True):
        group = group.sort_values("task_index")
        if group.task_index.tolist() != list(range(len(weights))):
            raise ValueError("Objective lacks a component")
        row = group.iloc[0][keys].to_dict()
        row.update(target="uncheatable_bpb", task_index=-1)
        for value in ("observed", "predicted", "predicted_tied_counterfactual"):
            row[value] = float(group[value].to_numpy() @ weights)
        result.append(row)
    return pd.DataFrame(result)


def summarize() -> None:
    panel = data.load_complete_tpp40()
    paths = [OUTPUT / "fits" / f"fold{fold}_task{task}" / "predictions.csv" for fold in range(5) for task in range(7)]
    for path in paths:
        verify_fit(path.parent)
    atomic = pd.concat([pd.read_csv(path, float_precision="round_trip") for path in paths], ignore_index=True)
    objective = objective_predictions(atomic, panel.objective_weights)
    combined = pd.concat([atomic, objective], ignore_index=True)
    scores = scoring.summarize_predictions(combined)
    ranks = []
    for _, row in scores.iterrows():
        menu = combined[(combined.model == row.model) & (combined.target == row.target)]
        if row.role != "joint_test":
            menu = menu[menu.role.eq(row.role)]
        ranks.append(float(spearmanr(menu.observed, menu.predicted).statistic))
    scores["spearman"] = ranks
    combined.to_csv(OUTPUT / "predictions.csv", index=False)
    scores.to_csv(OUTPUT / "scores.csv", index=False)
    scoring.score_predictions(combined).to_csv(OUTPUT / "fold_scores.csv", index=False)
    regional = (
        combined.assign(error=lambda x: x.predicted - x.observed)
        .groupby(["model", "target", "region", "role"])
        .error.agg(["count", "mean", lambda x: np.sqrt(np.mean(x**2))])
    )
    regional.rename(columns={"mean": "bias", "<lambda_0>": "rmse"}).to_csv(OUTPUT / "regional_scores.csv")
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(panel.rows), size=(3000, len(panel.rows)))
    bootstraps = []
    for name in models.MODEL_NAMES[1:]:
        candidate = objective[objective.model.eq(name)].sort_values("row_index")
        control = objective[objective.model.eq("aggregate_mariner")].sort_values("row_index")
        np.testing.assert_array_equal(candidate.row_index, control.row_index)
        err = (candidate.predicted.to_numpy() - candidate.observed.to_numpy()) ** 2
        ref = (control.predicted.to_numpy() - control.observed.to_numpy()) ** 2
        delta = np.sqrt(err[draws].mean(axis=1)) - np.sqrt(ref[draws].mean(axis=1))
        bootstraps.append(
            {
                "model": name,
                "metric": "joint_oof_rmse_difference",
                "difference": float(np.sqrt(err.mean()) - np.sqrt(ref.mean())),
                "conditional_group_bootstrap_lower": float(np.quantile(delta, 0.025)),
                "conditional_group_bootstrap_upper": float(np.quantile(delta, 0.975)),
            }
        )
    pd.DataFrame(bootstraps).to_csv(OUTPUT / "conditional_intervals.csv", index=False)
    print(
        scores.query("target == 'uncheatable_bpb' and role == 'joint_test'")[
            ["model", "rmse", "regret_at1", "regret_at3", "frontier_optimism", "selected_optimism"]
        ].to_string(index=False)
    )


def optimize() -> None:
    panel = data.load_complete_tpp40()
    coefficients = panel.panel.c0 + panel.panel.c1
    alpha, weights = panel.panel.alpha, panel.objective_weights
    by_model = {name: [] for name in models.MODEL_NAMES}
    for task in range(7):
        folder = OUTPUT / "fits" / f"fold-1_task{task}"
        verify_fit(folder)
        stored = json.loads((folder / "models.json").read_text())
        for name in models.MODEL_NAMES:
            by_model[name].append(models.TemporalFit.from_json(stored[name]))
    outer_models = []
    for fold in range(5):
        bundle = {name: [] for name in models.MODEL_NAMES}
        for task in range(7):
            folder = OUTPUT / "fits" / f"fold{fold}_task{task}"
            verify_fit(folder)
            stored = json.loads((folder / "models.json").read_text())
            for name in models.MODEL_NAMES:
                bundle[name].append(models.TemporalFit.from_json(stored[name]))
        outer_models.append(bundle)
    natural = np.asarray(panel.panel.metadata["pool_tokens"])
    natural = natural / natural.sum()
    rng = np.random.default_rng(SEED)
    records, solutions = [], {}
    for name, fits in by_model.items():

        def predict(first: np.ndarray, second: np.ndarray, fitted_heads=fits) -> np.ndarray:
            return (
                np.column_stack(
                    [
                        fit.predict(first * coefficients * alpha, second * coefficients * (1 - alpha))
                        for fit in fitted_heads
                    ]
                )
                @ weights
            )

        observed_predictions = predict(panel.panel.phase0, panel.panel.phase1)
        best_rows = np.argsort(observed_predictions, kind="stable")[:3]
        tied_starts = [
            natural,
            np.full(len(natural), 1 / len(natural)),
            *(alpha * panel.panel.phase0[i] + (1 - alpha) * panel.panel.phase1[i] for i in best_rows),
        ]
        candidates = []

        def execute(start: np.ndarray, tied: bool, start_index: int, model_name=name, destinations=candidates) -> None:
            width = len(natural)

            def objective(x: np.ndarray) -> float:
                return float(predict(x[None, :width], x[None, :width] if tied else x[None, width:])[0])

            constraints = [{"type": "eq", "fun": lambda x: np.sum(x[:width]) - 1}]
            if not tied:
                constraints.append({"type": "eq", "fun": lambda x: np.sum(x[width:]) - 1})
            result = minimize(
                objective,
                start,
                method="SLSQP",
                bounds=[(0, 1)] * len(start),
                constraints=constraints,
                options={"maxiter": 600, "ftol": 1e-10},
            )
            first = result.x[:width]
            second = first if tied else result.x[width:]
            feasible = abs(first.sum() - 1) < 1e-6 and abs(second.sum() - 1) < 1e-6 and min(result.x) >= -1e-10
            records.append(
                {
                    "model": model_name,
                    "tied": tied,
                    "start": start_index,
                    "predicted": float(result.fun),
                    "success": bool(result.success),
                    "feasible": bool(feasible),
                    "iterations": int(result.nit),
                    "message": str(result.message),
                }
            )
            if feasible and np.isfinite(result.fun):
                destinations.append((float(result.fun), first, second, bool(result.success), tied))

        def include_static(weights_static: np.ndarray, reason: str, model_name=name, destinations=candidates) -> None:
            value = float(predict(weights_static[None], weights_static[None])[0])
            if not np.isfinite(value):
                raise ValueError(f"Nonfinite direct static candidate for {model_name}")
            records.append(
                {
                    "model": model_name,
                    "tied": True,
                    "start": reason,
                    "predicted": value,
                    "success": False,
                    "feasible": True,
                    "iterations": 0,
                    "message": "Direct feasible policy evaluation; no optimizer convergence claim",
                }
            )
            destinations.append((value, weights_static, weights_static, False, True))

        with threadpool_limits(limits=1):
            for i, start in enumerate(tied_starts):
                execute(start, True, i)
            tied_best = min(candidates, key=lambda x: x[0])
            full_starts = [
                np.r_[tied_best[1], tied_best[1]],
                *(np.r_[panel.panel.phase0[i], panel.panel.phase1[i]] for i in best_rows),
                np.r_[rng.dirichlet(np.ones(len(natural))), rng.dirichlet(np.ones(len(natural)))],
            ]
            for i, start in enumerate(full_starts):
                execute(start, False, i)
            free_best = min(candidates, key=lambda x: x[0])
            free_aggregate = alpha * free_best[1] + (1 - alpha) * free_best[2]
            include_static(free_aggregate, "aggregate_of_best_free_solution")
            execute(free_aggregate, True, len(tied_starts))
            if name == "shared_clock":
                multipliers = np.asarray([fit.parameters["multiplier"] for fit in fits])
                if np.all(multipliers == multipliers[0]):
                    multiplier = float(multipliers[0])
                    common_mimic = (alpha * free_best[1] + multiplier * (1 - alpha) * free_best[2]) / (
                        alpha + multiplier * (1 - alpha)
                    )
                    include_static(common_mimic, "exact_common_shared_clock_mimic")
                    execute(common_mimic, True, len(tied_starts) + 1)
        tied_best = min((candidate for candidate in candidates if candidate[4]), key=lambda x: x[0])
        best = min(candidates, key=lambda x: x[0])
        aggregate = alpha * best[1] + (1 - alpha) * best[2]
        distances = np.linalg.norm(
            np.c_[np.sqrt(panel.panel.phase0), np.sqrt(panel.panel.phase1)] - np.r_[np.sqrt(best[1]), np.sqrt(best[2])],
            axis=1,
        )
        atomic_mimics = []
        if name in ("shared_clock", "benefit_clock"):
            for task, fit in enumerate(fits):
                r = fit.parameters["multiplier"]
                mimic = (alpha * best[1] + r * (1 - alpha) * best[2]) / (alpha + r * (1 - alpha))
                gap = fit.predict(alpha * coefficients * best[1][None], (1 - alpha) * coefficients * best[2][None])
                same = fit.predict(alpha * coefficients * mimic[None], (1 - alpha) * coefficients * mimic[None])
                atomic_mimics.append(
                    {
                        "task": panel.task_names[task],
                        "kind": (
                            "exact_static_equivalence" if name == "shared_clock" else "benefit_matching_static_policy"
                        ),
                        "multiplier": r,
                        "prediction_difference": float(gap[0] - same[0]),
                        "two_phase_prediction": float(gap[0]),
                        "static_prediction": float(same[0]),
                        "static_weights": mimic.tolist(),
                    }
                )
        crossfold = []
        for fold, bundle in enumerate(outer_models):
            query_first = np.asarray([best[1], tied_best[1], aggregate])
            query_second = np.asarray([best[2], tied_best[1], aggregate])
            atomic = np.column_stack(
                [
                    fit.predict(alpha * coefficients * query_first, (1 - alpha) * coefficients * query_second)
                    for fit in bundle[name]
                ]
            )
            objective_values = atomic @ weights
            crossfold.append(
                {
                    "outer_fold": fold,
                    "selected_two_phase_prediction": float(objective_values[0]),
                    "selected_static_prediction": float(objective_values[1]),
                    "selected_two_phase_aggregate_tied_prediction": float(objective_values[2]),
                    "gain_vs_selected_static": float(objective_values[1] - objective_values[0]),
                    "gain_at_same_aggregate": float(objective_values[2] - objective_values[0]),
                    "selected_two_phase_atomic": atomic[0].tolist(),
                    "selected_static_atomic": atomic[1].tolist(),
                }
            )
        solutions[name] = {
            "predicted_tied_optimum": tied_best[0],
            "predicted_two_phase_optimum": best[0],
            "predicted_gain": tied_best[0] - best[0],
            "best_success": best[3],
            "tied_best_success": tied_best[3],
            "best_is_tied": best[4],
            "phase0": best[1].tolist(),
            "phase1": best[2].tolist(),
            "tied_policy": tied_best[1].tolist(),
            "max_total_epochs": float(np.max(coefficients * aggregate)),
            "phase_tv": float(0.5 * np.sum(np.abs(best[1] - best[2]))),
            "nearest_policy_sqrt_distance": float(np.min(distances)),
            "atomic_static_mimics": atomic_mimics,
            "crossfold_predictions": crossfold,
            "crossfold_claim": (
                "Sensitivity of full-data-selected policies to outer-fold fits; "
                "not held-out policy validation or trainer-seed uncertainty."
            ),
            "optimization_claim": (
                "Best feasible policies found numerically; success fields distinguish converged solves "
                "from direct evaluations and nonconverged feasible iterates."
            ),
        }
        print(
            json.dumps(
                {
                    "model": name,
                    **{
                        k: v
                        for k, v in solutions[name].items()
                        if k not in ("phase0", "phase1", "tied_policy", "atomic_static_mimics", "crossfold_predictions")
                    },
                }
            ),
            flush=True,
        )
        pd.DataFrame(records).to_csv(OUTPUT / "optimization_traces.csv", index=False)
        benchmark.write_json(OUTPUT / "optima.json", solutions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("synthetic", "fit", "fullfit", "summarize", "optimize"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.stage in ("summarize", "optimize"):
        {"summarize": summarize, "optimize": optimize}[args.stage]()
        return
    if args.stage == "synthetic":
        work = [
            (world, noise, floor_compatible)
            for floor_compatible in (False, True)
            for world in SYNTHETIC_WORLDS
            for noise in (0.0, 0.003)
        ]
        function = synthetic_shard
    else:
        work = [(fold, task) for fold in (range(5) if args.stage == "fit" else [-1]) for task in range(7)]
        function = real_shard
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(function, item) for item in work]):
            print(json.dumps(future.result()), flush=True)


if __name__ == "__main__":
    main()
