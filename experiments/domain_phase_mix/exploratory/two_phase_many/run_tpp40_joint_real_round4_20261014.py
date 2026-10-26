# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Registered real-data diagnostic of joint estimation, using frozen evaluations.

This runner fits only three J2 families. Historical predictions, objective
aggregation, scoring, and the numerical policy optimizer are reused. Nothing
here promotes a model or supplies prospective confirmation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_two_phase_20260921 as benchmark
from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_mechanism_round2_20261014 as round2
from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_temporal_round_20261014 as prior
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_joint_estimation_20261014 as joint
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_temporal_data_20261014 as data
from experiments.domain_phase_mix.exploratory.two_phase_many import two_phase_benchmark_scoring_20260921 as scoring

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reference_outputs/tpp40_joint_real_round4_20261014"
SYNTHETIC = ROOT / "reference_outputs/tpp40_joint_estimation_round3_20261014"
ESTIMATOR = "j2_joint_shape"
MODEL_NAMES = tuple(f"{ESTIMATOR}__{family}" for family in joint.MODEL_NAMES)
SEED = 20261015
HISTORICAL = (
    "prior__joint_persistence",
    "ridge_gamma_refined__aggregate_mariner",
    "ridge_gamma_refined__bounded_benefit_odds",
    "ridge_gamma_refined__contrast_magnitude",
)


def source_identity() -> dict[str, str]:
    """Pin loaded local dependencies and the completed synthetic evidence."""
    paths = {
        Path(module.__file__).resolve()
        for module in tuple(sys.modules.values())
        if getattr(module, "__file__", None)
        and Path(module.__file__).resolve().is_relative_to(ROOT)
        and Path(module.__file__).suffix == ".py"
    }
    paths.update(
        {
            Path(__file__).resolve(),
            Path(joint.mariner.__file__).resolve(),
            ROOT / "two_phase_benchmark_data_20260921.py",
            OUTPUT / "registration.json",
            OUTPUT / "pre_execution_notes.json",
            SYNTHETIC / "summary_manifest.json",
            SYNTHETIC / "result_manifest.json",
            prior.OUTPUT / "result_manifest.json",
            prior.OUTPUT / "data/manifest.json",
            prior.OUTPUT / "data/arrays.npz",
            prior.OUTPUT / "data/folds.csv",
            prior.OUTPUT / "data/rows.csv",
            prior.OUTPUT / "predictions.csv",
            round2.OUTPUT / "summary/complete.json",
            round2.OUTPUT / "predictions.csv",
        }
    )
    data_manifest = json.loads((prior.OUTPUT / "data/manifest.json").read_text())
    paths.update(Path(path) for path in data_manifest["source_hashes"])
    paths.update(Path(path) for path in data_manifest["implementation_hashes"])
    return {str(path): benchmark.sha256(path) for path in sorted(paths)}


def identity() -> dict[str, str]:
    frozen = json.loads((OUTPUT / "execution_identity.json").read_text())
    if source_identity() != frozen:
        raise ValueError("Frozen sources or data changed; retain outputs and use a new protocol version")
    return frozen


def completed(folder: Path, context: dict[str, Any]) -> bool:
    if not (folder / "complete.json").exists():
        return False
    record = json.loads((folder / "complete.json").read_text())
    if record["identity"] != identity() or record["context"] != context:
        raise ValueError(f"Stale result: {folder}")
    benchmark.verify_hashes(record["outputs"], folder)
    for family, digest in record.get("model_receipts", {}).items():
        if benchmark.sha256(folder / family / "complete.json") != digest:
            raise ValueError(f"A model receipt changed after prediction: {folder / family}")
    return True


def finish(folder: Path, context: dict[str, Any], files: tuple[str, ...], frozen: dict[str, str], **extra: Any) -> None:
    if source_identity() != frozen:
        raise ValueError("Sources changed during execution")
    benchmark.write_json(
        folder / "complete.json",
        {
            "identity": frozen,
            "context": context,
            "outputs": {name: benchmark.sha256(folder / name) for name in files},
            **extra,
        },
    )


def prepare() -> None:
    """Verify source provenance and exact split replay before freezing execution."""
    manifest = json.loads((prior.OUTPUT / "data/manifest.json").read_text())
    benchmark.verify_hashes(manifest["files"], prior.OUTPUT / "data")
    for path, digest in {**manifest["implementation_hashes"], **manifest["source_hashes"]}.items():
        if benchmark.sha256(Path(path)) != digest:
            raise ValueError(f"Frozen data/source drift: {path}")
    synthetic = json.loads((SYNTHETIC / "summary_manifest.json").read_text())
    sealed_synthetic = json.loads((SYNTHETIC / "result_manifest.json").read_text())
    if sealed_synthetic["cells"] != 96 or sealed_synthetic["status"] != "complete":
        raise ValueError("Expected the complete 96-cell synthetic diagnostic")
    benchmark.verify_hashes(sealed_synthetic["files"], SYNTHETIC)
    benchmark.verify_hashes(synthetic["outputs"], SYNTHETIC)
    for path, digest in {**synthetic["identity"], **synthetic["cell_receipts"]}.items():
        if benchmark.sha256(Path(path)) != digest:
            raise ValueError(f"Synthetic evidence drift: {path}")
    prior_manifest = json.loads((prior.OUTPUT / "result_manifest.json").read_text())
    if benchmark.sha256(prior.OUTPUT / "predictions.csv") != prior_manifest["sha256"]["predictions.csv"]:
        raise ValueError("Prior prediction cache changed")
    round2_summary = json.loads((round2.OUTPUT / "summary/complete.json").read_text())
    if benchmark.sha256(round2.OUTPUT / "predictions.csv") != round2_summary["tables"]["predictions.csv"]:
        raise ValueError("Round-two prediction cache changed")
    panel = data.load_complete_tpp40()
    folds = pd.read_csv(prior.OUTPUT / "data/folds.csv")
    pd.testing.assert_frame_equal(panel.folds.reset_index(drop=True), folds.reset_index(drop=True), check_exact=True)
    with np.load(prior.OUTPUT / "data/arrays.npz") as arrays:
        np.testing.assert_array_equal(panel.full_data_inner_labels, arrays["full_data_inner_labels"])
        np.testing.assert_array_equal(panel.rows.row_id.to_numpy(str), arrays["row_ids"])
        np.testing.assert_array_equal(panel.panel.tied, arrays["tied"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen = source_identity()
    target = OUTPUT / "execution_identity.json"
    if target.exists() and json.loads(target.read_text()) != frozen:
        raise ValueError("Existing execution identity differs")
    benchmark.write_json(target, frozen)
    benchmark.write_json(
        OUTPUT / "split_replay.json",
        {
            "frozen_fold_records_identical": len(folds),
            "full_data_inner_labels_identical": len(panel.rows),
            "all_tied_rows_excluded_from_fit": not np.any(panel.full_data_inner_labels[panel.panel.tied] >= 0),
            "scope": "Read-only verification; frozen arrays and splits retained.",
        },
    )
    print("Prepared source identity and exact frozen split replay", flush=True)


def model_context(fold: int, task: int, family: str) -> dict[str, Any]:
    return {"stage": "fit", "fold": fold, "task": task, "family": family}


def model_folder(fold: int, task: int, family: str) -> Path:
    return OUTPUT / "fits" / f"fold{fold}_task{task}" / family


def load_fit_bundle(fold: int, task: int) -> dict[str, joint.JointFit]:
    result = {}
    for family in joint.MODEL_NAMES:
        folder = model_folder(fold, task, family)
        if not completed(folder, model_context(fold, task, family)):
            raise ValueError(f"Incomplete model: {folder}")
        result[f"{ESTIMATOR}__{family}"] = joint.JointFit.from_json(json.loads((folder / "model.json").read_text()))
    return result


def fit_context(arguments: tuple[int, int]) -> dict[str, Any]:
    fold, task = arguments
    folder = OUTPUT / "fits" / f"fold{fold}_task{task}"
    context = {"stage": "predictions", "fold": fold, "task": task}
    if completed(folder, context):
        return {**context, "cached": True}
    frozen = identity()
    panel = data.load_complete_tpp40()
    train, query, labels = prior.partitions(panel, fold)
    early, late = panel.panel.phase0_epochs, panel.panel.phase1_epochs
    observed = panel.final_targets[train, task]
    start = time.monotonic()
    fits: dict[str, joint.JointFit] = {}
    for family in joint.MODEL_NAMES:
        cell = model_folder(fold, task, family)
        cell_context = model_context(fold, task, family)
        if completed(cell, cell_context):
            fit = joint.JointFit.from_json(json.loads((cell / "model.json").read_text()))
        else:
            identity()
            fit = joint.fit_joint(
                early[train],
                late[train],
                observed,
                labels,
                panel.panel.alpha,
                family,
                matched_null_fit=fits.get("aggregate_mariner"),
            )
            benchmark.write_json(cell / "model.json", fit.to_json())
            finish(cell, cell_context, ("model.json",), frozen, train_rows=train.tolist(), inner_labels=labels.tolist())
        fits[family] = fit
    records = []
    total = early[query] + late[query]
    for family, fit in fits.items():
        predicted = fit.predict(early[query], late[query])
        diagonal = fit.predict(panel.panel.alpha * total, (1 - panel.panel.alpha) * total)
        for index, value, tied in zip(query, predicted, diagonal, strict=True):
            row = panel.rows.iloc[index]
            records.append(
                {
                    "model": f"{ESTIMATOR}__{family}",
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
                    "predicted": float(value),
                    "predicted_tied_counterfactual": float(tied),
                }
            )
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(folder / "predictions.csv", index=False)
    finish(
        folder,
        context,
        ("predictions.csv",),
        frozen,
        train_rows=train.tolist(),
        query_rows=query.tolist(),
        inner_labels=labels.tolist(),
        model_receipts={
            family: benchmark.sha256(model_folder(fold, task, family) / "complete.json") for family in joint.MODEL_NAMES
        },
        elapsed_seconds=time.monotonic() - start,
    )
    return {**context, "cached": False, "seconds": time.monotonic() - start}


def paired_intervals(objective: pd.DataFrame) -> pd.DataFrame:
    """Use the prior fixed-fit group bootstrap for the preregistered contrasts."""
    aggregate, odds, magnitude = MODEL_NAMES
    comparisons = (
        (odds, aggregate),
        (odds, magnitude),
        (aggregate, "ridge_gamma_refined__aggregate_mariner"),
        (odds, "ridge_gamma_refined__bounded_benefit_odds"),
        (odds, "prior__joint_persistence"),
    )
    records = []
    for name, reference in comparisons:
        for role in ("test", "tied_test", "joint_test"):
            eligible = objective if role == "joint_test" else objective[objective.role.eq(role)]
            candidate = eligible[eligible.model.eq(name)].set_index("row_id").sort_index()
            control = eligible[eligible.model.eq(reference)].set_index("row_id").loc[candidate.index]
            np.testing.assert_array_equal(candidate.observed, control.observed)
            np.testing.assert_array_equal(candidate.group_id, control.group_id)
            grouped = (
                pd.DataFrame(
                    {
                        "group": candidate.group_id,
                        "candidate": (candidate.predicted - candidate.observed) ** 2,
                        "reference": (control.predicted - control.observed) ** 2,
                        "count": 1,
                    }
                )
                .groupby("group")
                .sum()
            )
            draws = np.random.default_rng(SEED).integers(0, len(grouped), size=(3000, len(grouped)))
            totals = grouped.to_numpy()[draws].sum(axis=1)
            delta = np.sqrt(totals[:, 0] / totals[:, 2]) - np.sqrt(totals[:, 1] / totals[:, 2])
            records.append(
                {
                    "model": name,
                    "reference": reference,
                    "role": role,
                    "rmse_difference": float(
                        np.sqrt(grouped.candidate.sum() / grouped["count"].sum())
                        - np.sqrt(grouped.reference.sum() / grouped["count"].sum())
                    ),
                    "conditional_group_bootstrap_lower": float(np.quantile(delta, 0.025)),
                    "conditional_group_bootstrap_upper": float(np.quantile(delta, 0.975)),
                }
            )
    return pd.DataFrame(records)


def summarize() -> None:
    frozen = identity()
    panel = data.load_complete_tpp40()
    frames, parameters = [], []
    for fold in range(5):
        for task in range(7):
            fits = load_fit_bundle(fold, task)
            parameters.extend(parameter_rows(fits, fold, task))
            folder = OUTPUT / "fits" / f"fold{fold}_task{task}"
            if not completed(folder, {"stage": "predictions", "fold": fold, "task": task}):
                raise ValueError(f"Incomplete predictions: {folder}")
            frames.append(pd.read_csv(folder / "predictions.csv", float_precision="round_trip"))
    fresh = pd.concat(frames, ignore_index=True)
    if len(fresh) != 3 * 7 * 280 or fresh.duplicated(["model", "task_index", "row_id"]).any():
        raise ValueError("Incomplete or duplicate J2 outer predictions")
    historical = pd.read_csv(prior.OUTPUT / "predictions.csv", float_precision="round_trip")
    historical = historical[historical.model.eq("joint_persistence") & historical.task_index.ge(0)].copy()
    historical["model"] = "prior__joint_persistence"
    old = pd.read_csv(round2.OUTPUT / "predictions.csv", float_precision="round_trip")
    old = old[old.model.isin(HISTORICAL) & old.task_index.ge(0)]
    atomic = pd.concat([fresh, historical, old], ignore_index=True)
    if len(atomic) != 7 * 7 * 280 or atomic.duplicated(["model", "task_index", "row_id"]).any():
        raise ValueError("Historical comparison coverage differs")
    objective = prior.objective_predictions(atomic, panel.objective_weights)
    combined = pd.concat([atomic, objective], ignore_index=True)
    combined.to_csv(OUTPUT / "predictions.csv", index=False)
    folds = scoring.score_predictions(combined)
    ranks, tied_counts = [], []
    for row in folds.itertuples(index=False):
        menu = combined[
            combined.model.eq(row.model) & combined.target.eq(row.target) & combined.outer_fold.eq(row.outer_fold)
        ]
        if row.role != "joint_test":
            menu = menu[menu.role.eq(row.role)]
        ranks.append(int(1 + np.sum(menu.observed < row.selected_observed)))
        tied_counts.append(int(np.sum(menu.observed == row.selected_observed)))
    folds["selected_observed_rank_min"] = ranks
    folds["selected_observed_rank_tie_count"] = tied_counts
    folds.to_csv(OUTPUT / "fold_scores.csv", index=False)
    scores = scoring.summarize_predictions(combined)
    keys = ["model", "panel", "target", "repeat", "role"]
    rank_summary = folds.groupby(keys).selected_observed_rank_min.mean().rename("fold_mean_selected_observed_rank_min")
    scores = scores.merge(rank_summary, on=keys, validate="one_to_one")
    correlations = []
    for row in scores.itertuples(index=False):
        menu = combined[combined.model.eq(row.model) & combined.target.eq(row.target)]
        if row.role != "joint_test":
            menu = menu[menu.role.eq(row.role)]
        correlations.append(float(spearmanr(menu.observed, menu.predicted).statistic))
    scores["spearman"] = correlations
    scores.to_csv(OUTPUT / "scores.csv", index=False)
    for task in range(7):
        parameters.extend(parameter_rows(load_fit_bundle(-1, task), -1, task))
    pd.DataFrame(parameters).to_csv(OUTPUT / "selected_parameters.csv", index=False)
    paired_intervals(objective).to_csv(OUTPUT / "conditional_intervals.csv", index=False)
    combined.assign(error=lambda frame: frame.predicted - frame.observed).groupby(
        ["model", "target", "region", "role"]
    ).error.agg(count="size", bias="mean", rmse=lambda error: np.sqrt(np.mean(error**2))).to_csv(
        OUTPUT / "regional_scores.csv"
    )
    finish(
        OUTPUT,
        {"stage": "summary"},
        (
            "predictions.csv",
            "scores.csv",
            "fold_scores.csv",
            "conditional_intervals.csv",
            "regional_scores.csv",
            "selected_parameters.csv",
        ),
        frozen,
        claim=(
            "Adaptive development comparison; fixed-fit bootstrap excludes fitting/training uncertainty. "
            "Tied policies diagnose aggregate extrapolation. Selected rank is one plus the number of strictly "
            "better measured outcomes in the same fold menu; ties share the minimum rank."
        ),
    )


def parameter_rows(fits: dict[str, joint.JointFit], fold: int, task: int) -> list[dict[str, Any]]:
    return [
        {
            "model": name,
            "outer_fold": fold,
            "task_index": task,
            **fit.shape,
            "parameters": json.dumps(fit.parameters, sort_keys=True),
            "ridge": fit.ridge,
            "gamma": fit.kappa,
            "absolute_floor": fit.head.floor,
            "gamma_at_domain_boundary": fit.kappa in joint.mariner.KAPPA_BOUNDS,
            "ridge_at_grid_boundary": fit.ridge in (min(joint.mariner.RIDGE_GRID), max(joint.mariner.RIDGE_GRID)),
            "non_null": any(fit.parameters.values()),
            "selected_inner_rmse": fit.inner_rmse,
            "selected_from_failed_refinement": fit.search_space["selected_from_failed_refinement"],
        }
        for name, fit in fits.items()
    ]


def optimizer_backend(
    output: Path,
    identity_function: Callable[[], dict[str, str]],
    loader: Callable[[int, int], dict[str, joint.JointFit]],
) -> ModuleType:
    """Configure an isolated instance of the frozen optimizer without changing its singleton."""
    spec = importlib.util.spec_from_file_location("_tpp40_round4_optimizer_backend", Path(round2.__file__))
    assert spec is not None and spec.loader is not None
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    backend.OUTPUT = output
    backend.identity = identity_function
    backend.load_fit_bundle = loader
    return backend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fit", "fullfit", "summarize", "optimize"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
        return
    identity()
    if args.stage == "summarize":
        summarize()
        return
    if args.stage == "optimize":
        optimizer_backend(OUTPUT, identity, load_fit_bundle).optimize(MODEL_NAMES)
        return
    work = [(fold, task) for fold in (range(5) if args.stage == "fit" else [-1]) for task in range(7)]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(fit_context, item) for item in work]):
            print(json.dumps(future.result()), flush=True)


if __name__ == "__main__":
    main()
