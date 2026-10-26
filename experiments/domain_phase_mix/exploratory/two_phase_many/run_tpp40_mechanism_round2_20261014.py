# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Registered local-only mechanism and estimator comparisons on complete TPP40.

Synthetic oracle checks establish numerical representation, not deployable
recovery. Real outcomes are fitted only by explicit fit/fullfit commands.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import benchmark_two_phase_20260921 as benchmark
import numpy as np
import pandas as pd
import run_tpp40_temporal_round_20261014 as prior
import tpp40_mechanism_models_20261014 as mechanisms
import tpp40_temporal_data_20261014 as data
import two_phase_benchmark_scoring_20260921 as scoring
from scipy.optimize import minimize
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_temporal_models_20261014 as temporal

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reference_outputs/tpp40_mechanism_round2_20261014"
ESTIMATORS = ("native", "ridge_gamma_refined")
MODEL_NAMES = tuple(
    f"{estimator}__{family}" for estimator in ESTIMATORS for family in ("aggregate_mariner", *mechanisms.MODEL_NAMES)
)
SYNTHETIC_WORLDS = ("aggregate_null", "old_persistence", "odds", "rate", "joint", "hazard", "contrast_magnitude")
SEED = 20261015
ORACLE_TOLERANCE = 1e-8
Fit = temporal.TemporalFit | mechanisms.MechanismFit


def identity() -> dict[str, str]:
    paths = (
        Path(__file__),
        Path(mechanisms.__file__),
        Path(temporal.__file__),
        Path(temporal.mariner.__file__),
        Path(prior.__file__),
        Path(data.__file__),
        Path(scoring.__file__),
        Path(benchmark.__file__),
        OUTPUT / "registration.json",
        prior.OUTPUT / "result_manifest.json",
        prior.OUTPUT / "data/manifest.json",
    )
    return {str(path.resolve()): benchmark.sha256(path) for path in paths}


def completed(folder: Path, expected: dict[str, Any]) -> bool:
    receipt = folder / "complete.json"
    if not receipt.exists():
        return False
    record = json.loads(receipt.read_text())
    if record["identity"] != identity() or record["context"] != expected:
        raise ValueError(f"Stale round-two shard: {folder}")
    benchmark.verify_hashes(record["outputs"], folder)
    return True


def finish(
    folder: Path,
    context: dict[str, Any],
    files: tuple[str, ...],
    extra: dict | None = None,
    *,
    expected_identity: dict[str, str],
) -> None:
    if identity() != expected_identity:
        raise ValueError("Sources changed while computing the shard")
    record = {
        "identity": expected_identity,
        "context": context,
        "outputs": {name: benchmark.sha256(folder / name) for name in files},
    }
    record.update(extra or {})
    benchmark.write_json(folder / "complete.json", record)


def restore(value: dict) -> Fit:
    if value["model"] == "aggregate_mariner":
        return temporal.TemporalFit.from_json(value)
    return mechanisms.MechanismFit.from_json(value)


def bundle(
    early: np.ndarray,
    late: np.ndarray,
    observed: np.ndarray,
    labels: np.ndarray,
    aggregate: temporal.TemporalFit,
) -> dict[str, Fit]:
    aggregates = {
        "native": aggregate,
        "ridge_gamma_refined": mechanisms.refine_ridge_gamma(early, late, observed, labels, aggregate),
    }
    fitted: dict[str, Fit] = {}
    for estimator, base in aggregates.items():
        fitted[f"{estimator}__aggregate_mariner"] = base
        for family in mechanisms.MODEL_NAMES:
            fitted[f"{estimator}__{family}"] = mechanisms.fit_extension(early, late, observed, labels, base, family)
    return fitted


def real_shard(arguments: tuple[int, int]) -> dict:
    fold, task = arguments
    context = {"stage": "real", "fold": fold, "task": task}
    folder = OUTPUT / "fits" / f"fold{fold}_task{task}"
    if completed(folder, context):
        return {**context, "cached": True}
    frozen = identity()
    panel = data.load_complete_tpp40()
    train, query, labels = prior.partitions(panel, fold)
    source = prior.OUTPUT / "fits" / f"fold{fold}_task{task}"
    prior.verify_fit(source)
    source_receipt = json.loads((source / "complete.json").read_text())
    if source_receipt["train_rows"] != train.tolist() or source_receipt["query_rows"] != query.tolist():
        raise ValueError("Cached native aggregate fit uses different rows")
    aggregate = temporal.TemporalFit.from_json(json.loads((source / "models.json").read_text())["aggregate_mariner"])
    early, late = panel.panel.phase0_epochs, panel.panel.phase1_epochs
    started = time.monotonic()
    fitted = bundle(early[train], late[train], panel.final_targets[train, task], labels, aggregate)
    total = early[query] + late[query]
    records = []
    for name, fit in fitted.items():
        predicted = fit.predict(early[query], late[query])
        tied = fit.predict(panel.panel.alpha * total, (1 - panel.panel.alpha) * total)
        for index, prediction, counterfactual in zip(query, predicted, tied, strict=True):
            row = panel.rows.iloc[index]
            role = ("tied_diagnostic" if row.tied else "train") if fold == -1 else ("tied_test" if row.tied else "test")
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
                    "role": role,
                    "observed": float(panel.final_targets[index, task]),
                    "predicted": float(prediction),
                    "predicted_tied_counterfactual": float(counterfactual),
                }
            )
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(folder / "predictions.csv", index=False)
    benchmark.write_json(folder / "models.json", {name: fit.to_json() for name, fit in fitted.items()})
    finish(
        folder,
        context,
        ("predictions.csv", "models.json"),
        {
            "train_rows": train.tolist(),
            "query_rows": query.tolist(),
            "native_source_receipt": {
                "path": str(source / "complete.json"),
                "sha256": benchmark.sha256(source / "complete.json"),
            },
            "elapsed_seconds": time.monotonic() - started,
        },
        expected_identity=frozen,
    )
    return {**context, "cached": False, "seconds": round(time.monotonic() - started, 2)}


def synthetic_truth(panel: data.TemporalData, world: str) -> Fit:
    source = prior.OUTPUT / "synthetic/floor_compatible/aggregate_null_noise0"
    prior.verify_fit(source)
    original = temporal.TemporalFit.from_json(json.loads((source / "truth.json").read_text()))
    configurations = {
        "odds": ("bounded_benefit_odds", {"kappa": 4.0}),
        "rate": ("rate_damage", {"nu": 1.0}),
        "joint": ("bounded_benefit_rate", {"kappa": 4.0, "nu": 1.0}),
        "hazard": ("hazard_recency", {"rho": 0.5}),
        "contrast_magnitude": ("contrast_magnitude", {"theta": -0.1}),
    }
    if world == "aggregate_null":
        truth: Fit = original
    elif world == "old_persistence":
        truth = dataclasses.replace(
            original, model="joint_persistence", parameters={"benefit_loss": 0.5, "harm_loss": 0.5}
        )
    else:
        family, parameters = configurations[world]
        truth = mechanisms.MechanismFit(
            family,
            original.alpha,
            original.shape,
            original.ridge,
            original.kappa,
            False,
            0.0,
            0.0,
            parameters,
            original.head,
            (),
        )
    zero_intercept = dataclasses.replace(truth, head=dataclasses.replace(truth.head, intercept=0.0))
    raw_deficit = zero_intercept.predict(panel.panel.phase0_epochs, panel.panel.phase1_epochs) - truth.head.floor
    intercept = float(np.log(0.4 / np.median(raw_deficit)))
    return dataclasses.replace(truth, head=dataclasses.replace(truth.head, intercept=intercept))


def synthetic_context(panel: data.TemporalData) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train, query, labels = prior.partitions(panel, 0)
    asymmetric = query[~panel.panel.tied[query]]
    tied = np.flatnonzero(panel.panel.tied)
    return train, asymmetric, tied, labels


def synthetic_scores(
    panel: data.TemporalData,
    truth: Fit,
    fitted: dict[str, Fit],
    asymmetric: np.ndarray,
    tied: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    early, late = panel.panel.phase0_epochs, panel.panel.phase1_epochs
    query = np.r_[asymmetric, tied]
    clean = truth.predict(early, late)
    total = early[query] + late[query]
    diagonal_early, diagonal_late = panel.panel.alpha * total, (1 - panel.panel.alpha) * total
    true_diagonal = truth.predict(diagonal_early, diagonal_late)
    a = np.arange(len(asymmetric))
    t = np.arange(len(asymmetric), len(query))
    scores, rows = [], []
    for name, fit in fitted.items():
        predicted = fit.predict(early[query], late[query])
        diagonal = fit.predict(diagonal_early, diagonal_late)
        gain = diagonal - predicted
        true_gain = true_diagonal - clean[query]
        selected = int(a[np.argmin(predicted[a])])
        joint_selected = int(np.argmin(predicted))
        scores.append(
            {
                "model": name,
                "asymmetric_test_n": len(a),
                "actual_tied_diagnostic_n": len(t),
                "heldout_rmse": float(np.sqrt(np.mean((predicted[a] - clean[asymmetric]) ** 2))),
                "actual_tied_rmse": float(np.sqrt(np.mean((predicted[t] - clean[tied]) ** 2))),
                "counterfactual_tied_rmse": float(np.sqrt(np.mean((diagonal[a] - true_diagonal[a]) ** 2))),
                "phase_gain_rmse": float(np.sqrt(np.mean((gain[a] - true_gain[a]) ** 2))),
                "max_abs_predicted_phase_gain": float(np.max(abs(gain[a]))),
                "max_true_phase_gain": float(np.max(true_gain[a])),
                "regret_at1": float(clean[query[selected]] - np.min(clean[asymmetric])),
                "joint_diagnostic_regret_at1": float(clean[query[joint_selected]] - np.min(clean[query])),
                "selected_optimism": float(clean[query[selected]] - predicted[selected]),
                "selected_predicted_phase_gain": float(gain[selected]),
                "selected_true_phase_gain": float(true_gain[selected]),
            }
        )
        rows.extend(
            {
                "model": name,
                "row_index": int(index),
                "role": "tied_diagnostic" if panel.panel.tied[index] else "test",
                "truth": float(clean[index]),
                "predicted": float(predicted[position]),
                "tied_truth": float(true_diagonal[position]),
                "tied_predicted": float(diagonal[position]),
            }
            for position, index in enumerate(query)
        )
    return pd.DataFrame(scores), pd.DataFrame(rows)


def synthetic_shard(arguments: tuple[str, float]) -> dict:
    world, noise = arguments
    context = {"stage": "synthetic", "world": world, "noise": noise}
    folder = OUTPUT / "synthetic" / f"{world}_noise{noise:g}"
    if completed(folder, context):
        return {**context, "cached": True}
    frozen = identity()
    panel = data.load_complete_tpp40()
    train, asymmetric, tied, labels = synthetic_context(panel)
    early, late = panel.panel.phase0_epochs, panel.panel.phase1_epochs
    truth = synthetic_truth(panel, world)
    clean = truth.predict(early, late)
    measured = clean + np.random.default_rng(SEED).normal(0, noise, len(clean))
    aggregate = temporal.fit_aggregate(
        early[train],
        late[train],
        measured[train],
        labels,
        panel.panel.alpha,
        np.asarray(panel.panel.metadata["pool_tokens"]),
    )
    fitted = bundle(early[train], late[train], measured[train], labels, aggregate)
    scores, predictions = synthetic_scores(panel, truth, fitted, asymmetric, tied)
    scores.insert(0, "world", world)
    scores.insert(1, "noise_sd", noise)
    folder.mkdir(parents=True, exist_ok=True)
    scores.to_csv(folder / "scores.csv", index=False)
    predictions.to_csv(folder / "predictions.csv", index=False)
    benchmark.write_json(folder / "truth.json", truth.to_json())
    benchmark.write_json(folder / "models.json", {name: fit.to_json() for name, fit in fitted.items()})
    finish(
        folder,
        context,
        ("scores.csv", "predictions.csv", "truth.json", "models.json"),
        {
            "train_rows": train.tolist(),
            "asymmetric_query": asymmetric.tolist(),
            "tied_query": tied.tolist(),
            "claim": "Whole-estimator synthetic diagnostic; no oracle parameters supplied to these fits.",
        },
        expected_identity=frozen,
    )
    return {**context, "cached": False}


def oracle_head(matrix: np.ndarray, offset: np.ndarray, observed: np.ndarray, floor: float) -> temporal.mariner.Head:
    if not np.all(observed > floor):
        raise ValueError("Known synthetic floor must be below all labels")
    intercept, coefficients = temporal.mariner.nonnegative_solve(matrix, np.log(observed - floor) - offset, 0.0)
    return temporal.mariner.Head(intercept, coefficients, floor)


def oracle_shard(world: str) -> dict:
    context = {"stage": "oracle", "world": world}
    folder = OUTPUT / "oracle" / world
    if completed(folder, context):
        return {**context, "cached": True}
    frozen = identity()
    if world == "old_persistence":
        raise ValueError("Old persistence is a deliberate misspecification stress, not an in-family oracle check")
    panel = data.load_complete_tpp40()
    train, asymmetric, tied, labels = synthetic_context(panel)
    early, late = panel.panel.phase0_epochs, panel.panel.phase1_epochs
    truth = synthetic_truth(panel, world)
    y = truth.predict(early, late)
    splits = temporal.inner_splits(labels)
    families = ("aggregate_mariner", *mechanisms.MODEL_NAMES) if world == "aggregate_null" else (truth.model,)
    fitted: dict[str, Fit] = {}
    profiles = {}
    with threadpool_limits(limits=1):
        for family in families:
            grid = ({},) if family == "aggregate_mariner" else mechanisms.parameter_grid(family)
            best = None
            profile = []
            for index, parameters in enumerate(grid):
                if family == "aggregate_mariner":
                    matrix = temporal.mariner.design_matrix(early + late, truth.shape)
                    offset = np.zeros(len(early))
                else:
                    matrix, offset = mechanisms.design_and_offset(
                        early, late, truth.alpha, truth.shape, family, parameters
                    )
                errors = []
                for left, right in splits:
                    head = oracle_head(matrix[train[left]], offset[train[left]], y[train[left]], truth.head.floor)
                    predicted = temporal.predict_head(head, matrix[train[right]], offset[train[right]])
                    errors.extend((predicted - y[train[right]]).tolist())
                score = float(np.sqrt(np.mean(np.square(errors))))
                profile.append({"parameters": parameters, "inner_rmse": score})
                if best is None or (score, index) < best:
                    best = score, index
            assert best is not None
            parameters = grid[best[1]]
            if family == "aggregate_mariner":
                matrix = temporal.mariner.design_matrix(early + late, truth.shape)
                offset = np.zeros(len(early))
            else:
                matrix, offset = mechanisms.design_and_offset(early, late, truth.alpha, truth.shape, family, parameters)
            head = oracle_head(matrix[train], offset[train], y[train], truth.head.floor)
            if family == "aggregate_mariner":
                assert isinstance(truth, temporal.TemporalFit)
                fitted[family] = dataclasses.replace(truth, head=head, ridge=0.0)
            else:
                fitted[family] = mechanisms.MechanismFit(
                    family,
                    truth.alpha,
                    truth.shape,
                    0.0,
                    truth.kappa,
                    False,
                    best[0],
                    best[0],
                    parameters,
                    head,
                    tuple(profile),
                )
            profiles[family] = profile
    scores, predictions = synthetic_scores(panel, truth, fitted, asymmetric, tied)
    scores["selected_null"] = [not any(fitted[name].parameters.values()) for name in scores.model]
    scores["passed"] = (
        scores.heldout_rmse.lt(ORACLE_TOLERANCE)
        & scores.actual_tied_rmse.lt(ORACLE_TOLERANCE)
        & scores.phase_gain_rmse.lt(ORACLE_TOLERANCE)
    )
    if world == "aggregate_null":
        scores["passed"] &= scores.selected_null
    folder.mkdir(parents=True, exist_ok=True)
    scores.to_csv(folder / "scores.csv", index=False)
    predictions.to_csv(folder / "predictions.csv", index=False)
    benchmark.write_json(folder / "truth.json", truth.to_json())
    benchmark.write_json(folder / "models.json", {name: fit.to_json() for name, fit in fitted.items()})
    benchmark.write_json(folder / "profiles.json", profiles)
    finish(
        folder,
        context,
        ("scores.csv", "predictions.csv", "truth.json", "models.json", "profiles.json"),
        {
            "passed": bool(scores.passed.all()),
            "tolerance": ORACLE_TOLERANCE,
            "claim": (
                "Known-shape/floor, zero-ridge representation and numerical check; "
                "not deployable-estimator qualification."
            ),
        },
        expected_identity=frozen,
    )
    return {**context, "cached": False, "passed": bool(scores.passed.all())}


def load_fit_bundle(fold: int, task: int) -> dict[str, Fit]:
    folder = OUTPUT / "fits" / f"fold{fold}_task{task}"
    if not completed(folder, {"stage": "real", "fold": fold, "task": task}):
        raise ValueError(f"Missing fit: {folder}")
    return {name: restore(value) for name, value in json.loads((folder / "models.json").read_text()).items()}


def require_oracle_pass() -> None:
    """Require current-source numerical representation checks before real fits."""
    for world in SYNTHETIC_WORLDS:
        if world == "old_persistence":
            continue
        folder = OUTPUT / "oracle" / world
        if not completed(folder, {"stage": "oracle", "world": world}):
            raise ValueError(f"Missing required oracle check: {world}")
        if not json.loads((folder / "complete.json").read_text())["passed"]:
            raise ValueError(f"Numerical representation check failed: {world}")


def conditional_intervals(objective: pd.DataFrame) -> pd.DataFrame:
    records = []
    comparisons = [
        (name, name.split("__")[0] + "__aggregate_mariner")
        for name in MODEL_NAMES
        if not name.endswith("__aggregate_mariner")
    ]
    comparisons.append(("ridge_gamma_refined__aggregate_mariner", "native__aggregate_mariner"))
    for name, reference in comparisons:
        for role in ("joint_test", "test", "tied_test"):
            candidate = objective[objective.model.eq(name)]
            control = objective[objective.model.eq(reference)]
            if role != "joint_test":
                candidate = candidate[candidate.role.eq(role)]
                control = control[control.role.eq(role)]
            joined = candidate.merge(
                control[["row_id", "predicted"]], on="row_id", validate="one_to_one", suffixes=("", "_reference")
            )
            joined["candidate_sq"] = (joined.predicted - joined.observed) ** 2
            joined["reference_sq"] = (joined.predicted_reference - joined.observed) ** 2
            grouped = joined.groupby("group_id").agg(
                candidate_sq=("candidate_sq", "sum"), reference_sq=("reference_sq", "sum"), count=("row_id", "size")
            )
            draws = np.random.default_rng(SEED).integers(0, len(grouped), size=(3000, len(grouped)))
            count = grouped["count"].to_numpy()[draws].sum(axis=1)
            delta = np.sqrt(grouped.candidate_sq.to_numpy()[draws].sum(axis=1) / count)
            delta -= np.sqrt(grouped.reference_sq.to_numpy()[draws].sum(axis=1) / count)
            records.append(
                {
                    "model": name,
                    "reference": reference,
                    "role": role,
                    "metric": "rmse_difference",
                    "difference": float(np.sqrt(joined.candidate_sq.mean()) - np.sqrt(joined.reference_sq.mean())),
                    "conditional_group_bootstrap_lower": float(np.quantile(delta, 0.025)),
                    "conditional_group_bootstrap_upper": float(np.quantile(delta, 0.975)),
                }
            )
    return pd.DataFrame(records)


def summarize() -> None:
    frozen = identity()
    panel = data.load_complete_tpp40()
    frames = []
    for fold in range(5):
        for task in range(7):
            load_fit_bundle(fold, task)
            frames.append(
                pd.read_csv(OUTPUT / "fits" / f"fold{fold}_task{task}/predictions.csv", float_precision="round_trip")
            )
    atomic = pd.concat(frames, ignore_index=True)
    if atomic.duplicated(["model", "target", "row_id"]).any() or len(atomic) != len(MODEL_NAMES) * 7 * 280:
        raise ValueError("Incomplete or duplicate outer predictions")
    objective = prior.objective_predictions(atomic, panel.objective_weights)
    combined = pd.concat([atomic, objective], ignore_index=True)
    scores = scoring.summarize_predictions(combined)
    ranks = []
    for row in scores.itertuples(index=False):
        frame = combined[combined.model.eq(row.model) & combined.target.eq(row.target)]
        if row.role != "joint_test":
            frame = frame[frame.role.eq(row.role)]
        ranks.append(float(spearmanr(frame.observed, frame.predicted).statistic))
    scores["spearman"] = ranks
    combined.to_csv(OUTPUT / "predictions.csv", index=False)
    scores.to_csv(OUTPUT / "scores.csv", index=False)
    scoring.score_predictions(combined).to_csv(OUTPUT / "fold_scores.csv", index=False)
    regional = (
        combined.assign(error=lambda frame: frame.predicted - frame.observed)
        .groupby(["model", "target", "region", "role"])
        .error.agg(count="size", bias="mean", rmse=lambda errors: np.sqrt(np.mean(errors**2)))
    )
    regional.to_csv(OUTPUT / "regional_scores.csv")
    conditional_intervals(objective).to_csv(OUTPUT / "conditional_intervals.csv", index=False)
    finish(
        OUTPUT / "summary",
        {"stage": "summary"},
        (),
        {
            "tables": {
                name: benchmark.sha256(OUTPUT / name)
                for name in (
                    "predictions.csv",
                    "scores.csv",
                    "fold_scores.csv",
                    "regional_scores.csv",
                    "conditional_intervals.csv",
                )
            }
        },
        expected_identity=frozen,
    )
    print(
        scores.query("target == 'uncheatable_bpb' and role == 'joint_test'")[
            ["model", "rmse", "regret_at1", "frontier_optimism"]
        ].to_string(index=False)
    )


def optimize(selected_models: tuple[str, ...]) -> None:
    panel = data.load_complete_tpp40()
    full = [load_fit_bundle(-1, task) for task in range(7)]
    outer = [[load_fit_bundle(fold, task) for task in range(7)] for fold in range(5)]
    alpha = panel.panel.alpha
    scales = panel.panel.c0 + panel.panel.c1
    weights = panel.objective_weights
    natural = np.asarray(panel.panel.metadata["pool_tokens"])
    natural = natural / natural.sum()
    width = len(natural)
    for name in selected_models:
        folder = OUTPUT / "optimization" / name
        context = {"stage": "optimization", "model": name}
        if completed(folder, context):
            print(json.dumps({**context, "cached": True}), flush=True)
            continue
        frozen = identity()
        fits = [task[name] for task in full]
        candidates = []
        traces = []

        def atomic(first: np.ndarray, second: np.ndarray, fitted=fits) -> np.ndarray:
            return np.column_stack(
                [fit.predict(alpha * scales * first, (1 - alpha) * scales * second) for fit in fitted]
            )

        def consider(
            first: np.ndarray,
            second: np.ndarray,
            tied: bool,
            source: str,
            success: bool = False,
            trace_rows=traces,
            choices=candidates,
            predict_atomic=atomic,
        ) -> None:
            value = float(predict_atomic(first[None], second[None])[0] @ weights)
            feasible = (
                max(abs(first.sum() - 1), abs(second.sum() - 1)) < 1e-6 and min(first.min(), second.min()) >= -1e-10
            )
            trace_rows.append(
                {"source": source, "tied": tied, "predicted": value, "feasible": feasible, "success": success}
            )
            if feasible and np.isfinite(value):
                choices.append((value, first.copy(), second.copy(), success, tied))

        def execute(
            start: np.ndarray,
            tied: bool,
            source: str,
            trace_rows=traces,
            consider_policy=consider,
            predict_atomic=atomic,
        ) -> None:
            first, second = start[:width], start[:width] if tied else start[width:]
            consider_policy(first, second, tied, source + ":start")
            constraints = [{"type": "eq", "fun": lambda x: x[:width].sum() - 1}]
            if not tied:
                constraints.append({"type": "eq", "fun": lambda x: x[width:].sum() - 1})
            result = minimize(
                lambda x: float(
                    predict_atomic(x[None, :width], x[None, :width] if tied else x[None, width:])[0] @ weights
                ),
                start,
                method="SLSQP",
                bounds=[(0, 1)] * len(start),
                constraints=constraints,
                options={"maxiter": 600, "ftol": 1e-10},
            )
            consider_policy(
                result.x[:width],
                result.x[:width] if tied else result.x[width:],
                tied,
                source + ":result",
                bool(result.success),
            )
            trace_rows[-1].update(iterations=int(result.nit), message=str(result.message))

        observed = atomic(panel.panel.phase0, panel.panel.phase1) @ weights
        best_rows = np.argsort(observed, kind="stable")[:3]
        starts = [
            natural,
            np.full(width, 1 / width),
            *(alpha * panel.panel.phase0[i] + (1 - alpha) * panel.panel.phase1[i] for i in best_rows),
        ]
        with threadpool_limits(limits=1):
            for index, start in enumerate(starts):
                execute(start, True, f"tied{index}")
            tied_best = min(candidates, key=lambda value: value[0])
            if not name.endswith("__aggregate_mariner"):
                rng = np.random.default_rng(SEED)
                free_starts = [
                    np.r_[tied_best[1], tied_best[1]],
                    *(np.r_[panel.panel.phase0[i], panel.panel.phase1[i]] for i in best_rows),
                    np.r_[rng.dirichlet(np.ones(width)), rng.dirichlet(np.ones(width))],
                ]
                for index, start in enumerate(free_starts):
                    execute(start, False, f"free{index}")
                best = min(candidates, key=lambda value: value[0])
                execute(alpha * best[1] + (1 - alpha) * best[2], True, "free_aggregate")
        tied_best = min((candidate for candidate in candidates if candidate[4]), key=lambda value: value[0])
        best = min(candidates, key=lambda value: value[0])
        aggregate = alpha * best[1] + (1 - alpha) * best[2]
        aggregate_tied_prediction = float(atomic(aggregate[None], aggregate[None])[0] @ weights)
        crossfold = []
        for fold, task_models in enumerate(outer):
            first = np.asarray([best[1], tied_best[1], aggregate])
            second = np.asarray([best[2], tied_best[1], aggregate])
            values = atomic(first, second, [task[name] for task in task_models]) @ weights
            crossfold.append(
                {
                    "outer_fold": fold,
                    "two_phase": float(values[0]),
                    "static": float(values[1]),
                    "aggregate_tied": float(values[2]),
                    "gain_vs_selected_static": float(values[1] - values[0]),
                    "gain_same_aggregate": float(values[2] - values[0]),
                }
            )
        distance = np.linalg.norm(
            np.c_[np.sqrt(panel.panel.phase0), np.sqrt(panel.panel.phase1)] - np.r_[np.sqrt(best[1]), np.sqrt(best[2])],
            axis=1,
        )
        result = {
            "model": name,
            "predicted_tied_optimum": tied_best[0],
            "predicted_two_phase_optimum": best[0],
            "predicted_gain": tied_best[0] - best[0],
            "predicted_aggregate_tied": aggregate_tied_prediction,
            "predicted_same_aggregate_gain": aggregate_tied_prediction - best[0],
            "phase0": best[1].tolist(),
            "phase1": best[2].tolist(),
            "tied_policy": tied_best[1].tolist(),
            "best_success": best[3],
            "tied_success": tied_best[3],
            "best_is_tied": best[4],
            "phase_tv": float(0.5 * abs(best[1] - best[2]).sum()),
            "max_total_epochs": float(np.max(scales * aggregate)),
            "nearest_policy_sqrt_distance": float(distance.min()),
            "crossfold": crossfold,
            "claim": (
                "Exploratory numerical policies and refit sensitivity, not observed gain, "
                "global optimality, or held-out policy validation."
            ),
        }
        folder.mkdir(parents=True, exist_ok=True)
        benchmark.write_json(folder / "optimum.json", result)
        pd.DataFrame(traces).to_csv(folder / "traces.csv", index=False)
        finish(folder, context, ("optimum.json", "traces.csv"), expected_identity=frozen)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in result.items()
                    if key not in ("phase0", "phase1", "tied_policy", "crossfold")
                }
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("synthetic", "oracle", "fit", "fullfit", "summarize", "optimize"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES)
    args = parser.parse_args()
    identity()  # Registration and frozen sources must exist before any stage.
    if args.stage == "summarize":
        summarize()
        return
    if args.stage == "optimize":
        optimize(tuple(args.models or [name for name in MODEL_NAMES if name.startswith("ridge_gamma_refined__")]))
        return
    if args.stage == "synthetic":
        work = [(world, noise) for world in SYNTHETIC_WORLDS for noise in (0.0, 0.003)]
        function = synthetic_shard
    elif args.stage == "oracle":
        work = [world for world in SYNTHETIC_WORLDS if world != "old_persistence"]
        function = oracle_shard
    else:
        require_oracle_pass()
        work = [(fold, task) for fold in (range(5) if args.stage == "fit" else [-1]) for task in range(7)]
        function = real_shard
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(function, item) for item in work]):
            print(json.dumps(future.result()), flush=True)


if __name__ == "__main__":
    main()
