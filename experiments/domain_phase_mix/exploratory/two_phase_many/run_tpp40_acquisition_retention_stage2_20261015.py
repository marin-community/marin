# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Noiseless unknown-floor/shape J2 smoke on the four sealed synthetic worlds.

Only asymmetric outer-zero training labels enter fitting. The native aggregate
is shared between clock hypotheses; all ties remain evaluation-only. This
stage has no optimizer, noisy fitting, empirical outcomes, or automatic advance.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    run_tpp40_acquisition_retention_oracle_20261015 as oracle,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    tpp40_acquisition_retention_joint_models_20261015 as models,
)

benchmark = oracle.benchmark
ROOT = Path(__file__).resolve().parent
PRIOR = oracle.OUTPUT
OUTPUT = oracle.PARENT / "stage2"
SCOPE = "tp034_outer_zero_noiseless_full_j2_smoke"
CONTRACT = {
    "scope": SCOPE,
    "outer_fold": 0,
    "repeat": 0,
    "noise_sd": 0.0,
    "worlds": list(oracle.WORLDS),
    "strengths": list(oracle.STRENGTHS),
    "quadrature": oracle.CONTRACT["quadrature"],
    "cells": 10,
    "shapes_override": None,
    "test_grid_override": None,
    "known_floor": None,
    "raw_optimization": False,
    "unresolved_numerical_fields": [
        "query_clipping_count",
        "training_deficit_clip_count",
        "inner_training_deficit_clip_count",
        "inner_prediction_clip_count",
        "selected_from_failed_refinement",
        "failed_refinements",
    ],
    "frontier_fraction": 0.2,
    "gates": {
        "exact_null_maximum_absolute_gain": 1e-8,
        "null_maximum_positive_gain": 0.003,
        "null_gain_rmse": 0.003,
        "positive_gain_rmse": 0.001,
        "positive_gain_zero_baseline_ratio": 0.5,
        "correct_family_rmse": 0.003,
        "frontier_absolute_mean_optimism": 0.003,
        "regret_at_1": 0.001,
        "misspecified_null_native_rmse_allowance": 0.003,
    },
}


def prerequisite_paths() -> set[Path]:
    return {
        PRIOR / name
        for name in (
            "registration.json",
            "execution_identity.json",
            "result_manifest.json",
            "audit/independent_replay_report_v1.json",
            "oracle_summary.json",
            "complete.json",
            "prepared/complete.json",
            "prepared/clean.npz",
            "prepared/matched_tied.npz",
            "prepared/preflight.json",
        )
    }


def source_paths() -> set[Path]:
    return oracle.source_paths() | {
        Path(__file__),
        ROOT / "test_tpp40_acquisition_retention_stage2_20261015.py",
        oracle.PARENT / "reviews/synthetic_protocol_fable_v1/adjudication_v1.md",
    }


def verify_prerequisite() -> dict[str, Any]:
    """Require the complete independently replayed prior stage, including failures."""
    seal = json.loads((PRIOR / "result_manifest.json").read_text())
    benchmark.verify_hashes({name: entry["sha256"] for name, entry in seal["files"].items()}, PRIOR)
    audit = json.loads((PRIOR / "audit/independent_replay_report_v1.json").read_text())
    summary = json.loads((PRIOR / "oracle_summary.json").read_text())
    if not audit["passed"] or not summary["oracle_passed"] or not summary["learned_stage_ready"]:
        raise ValueError("Preserve failed prior qualification; learned fitting is blocked")
    identity = oracle.execution_identity()
    if not oracle.completed(PRIOR, {"stage": "oracle_summary"}, identity):
        raise ValueError("The sealed oracle summary receipt is missing")
    if not oracle.completed(PRIOR / "prepared", {"stage": "prepare", "contract": oracle.CONTRACT}, identity):
        raise ValueError("The sealed truth receipt is missing")
    return {"seal_sha256": benchmark.sha256(PRIOR / "result_manifest.json"), "summary": summary}


def registration() -> dict[str, Any]:
    value = json.loads((OUTPUT / "registration.json").read_text())
    if value.get("status") != "registered" or value.get("candidate_id") != "TP40-034":
        raise ValueError("Register the reviewed learned-stage contract before execution")
    if value.get("contract") != CONTRACT or value.get("environment") != benchmark.environment_identity():
        raise ValueError("Registered learned-stage contract or runtime changed")
    groups = (
        ("source_hashes", source_paths()),
        ("input_hashes", oracle.input_paths()),
        ("prerequisite_hashes", prerequisite_paths()),
    )
    for key, paths in groups:
        if set(value[key]) != {str(path.resolve()) for path in paths}:
            raise ValueError("The registered source, covariate, or prerequisite pin set differs")
        benchmark.verify_hashes(value[key])
    verify_prerequisite()
    return value


def source_identity() -> dict[str, Any]:
    value = registration()
    return {
        "registration_sha256": benchmark.sha256(OUTPUT / "registration.json"),
        **{key: value[key] for key in ("source_hashes", "input_hashes", "prerequisite_hashes", "environment")},
    }


def freeze() -> None:
    identity = source_identity()
    path = OUTPUT / "execution_identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Preserve the prior execution identity")
    benchmark.write_json(path, identity)


def execution_identity() -> dict[str, Any]:
    identity = source_identity()
    if json.loads((OUTPUT / "execution_identity.json").read_text()) != identity:
        raise ValueError("Frozen learned-stage identity changed")
    approval = json.loads((OUTPUT / "execution_approval.json").read_text())
    required = {
        "scope": SCOPE,
        "registration_sha256": identity["registration_sha256"],
        "execution_identity_sha256": benchmark.sha256(OUTPUT / "execution_identity.json"),
    }
    if any(approval.get(key) != value for key, value in required.items()):
        raise ValueError("Exact learned-stage approval is required")
    return identity


def native_for_clock(fit: models.JointFit, config: models.FeatureConfig) -> models.JointFit:
    """Rebind only the irrelevant native clock metadata, preserving the entire fit."""
    if fit.model != "aggregate_mariner" or fit.parameters != {}:
        raise ValueError("Only a native aggregate can share its fit across clocks")
    old = fit.config
    if (old.scales, old.relative_tolerance, old.max_panels) != (
        config.scales,
        config.relative_tolerance,
        config.max_panels,
    ):
        raise ValueError("Shared native control must retain exposure and numerical controls")
    return replace(fit, config=config)


def fit_candidate(
    early: np.ndarray,
    late: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    candidate: str,
    config: models.FeatureConfig,
    cache: models.BenefitCache,
    aggregate: models.JointFit | None,
) -> models.JointFit:
    """Use the full frozen J2 procedure, with no truth-shape/floor information."""
    native = candidate == "aggregate_mariner"
    if native != (aggregate is None):
        raise ValueError("Fit native once before the matching temporal family")
    controls = () if aggregate is None else (native_for_clock(aggregate, config),)
    with threadpool_limits(limits=1):
        return models.fit_joint(
            early,
            late,
            values,
            labels,
            alpha,
            "aggregate_mariner" if native else models.MODEL,
            config=config,
            strengths=oracle.STRENGTHS,
            cache=cache,
            controls=controls,
            shapes=None,
            test_grid=None,
        )


def selected_inner(
    fit: models.JointFit,
    early: np.ndarray,
    late: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    cache: models.BenefitCache | None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    matrix, offset = models.design_and_offset(
        early, late, fit.alpha, fit.shape, fit.model, fit.parameters, fit.config, cache
    )
    predictions, records = np.empty(len(values)), []
    for label in np.unique(labels):
        train, held = np.flatnonzero(labels != label), np.flatnonzero(labels == label)
        head = models.frozen.fit_fixed_head(matrix[train], offset[train], values[train], fit.ridge, fit.kappa, None)
        predictions[held] = models.frozen.temporal.predict_head(head, matrix[held], offset[held])
        linear = head.intercept + matrix[held] @ head.coefficients + offset[held]
        records.append(
            {
                "label": int(label),
                "train": train.tolist(),
                "held": held.tolist(),
                "head": oracle.head_record(head),
                "training_deficit_clip_count": int(np.sum(values[train] - head.floor <= models.mariner.DEFICIT_FLOOR)),
                "held_prediction_clip_count": int(np.sum(abs(linear) >= models.mariner.LOG_CLIP)),
            }
        )
    return predictions, records


def menu_scores(predicted: np.ndarray, truth: np.ndarray, query: pd.DataFrame) -> dict[str, Any]:
    """Order only by prediction and stable role/row identity; positive optimism is favorable bias."""
    order = np.lexsort((query.row_id.to_numpy(dtype=str), query.role.to_numpy(dtype=str), predicted))
    frontier = order[: max(1, int(np.ceil(CONTRACT["frontier_fraction"] * len(order))))]
    optimism = truth - predicted
    result = {
        "calibration_bias": float(np.mean(predicted - truth)),
        "mean_optimism": float(np.mean(optimism)),
        "frontier_mean_optimism": float(np.mean(optimism[frontier])),
        "frontier_maximum_optimism": float(np.max(optimism[frontier])),
        "frontier_rmse": float(np.sqrt(np.mean(optimism[frontier] ** 2))),
        "selected_optimism": float(optimism[order[0]]),
        "selected_role": str(query.iloc[order[0]].role),
        "selected_row_id": str(query.iloc[order[0]].row_id),
        "frontier_indices": frontier.tolist(),
    }
    for k in (1, 3, 5):
        result[f"regret_at_{k}"] = float(np.min(truth[order[:k]]) - np.min(truth))
    return result


def diagnostic_scores(
    fit: models.JointFit, predicted: np.ndarray, truth: np.ndarray, query: pd.DataFrame
) -> dict[str, Any]:
    count = int(np.sum(query.role == "asymmetric"))
    gain, true_gain = predicted[count : 2 * count] - predicted[:count], truth[count : 2 * count] - truth[:count]
    rank = None
    if np.ptp(gain) > 0 and np.ptp(true_gain) > 1e-14:
        rank = float(spearmanr(gain, true_gain).statistic)
    defined = abs(true_gain) > 1e-12
    return {
        **oracle.prediction_scores(predicted, truth, count),
        **menu_scores(predicted, truth, query),
        "maximum_positive_predicted_gain": float(max(0, np.max(gain))),
        "gain_spearman": rank,
        "gain_sign_accuracy": (
            float(np.mean(np.sign(gain[defined]) == np.sign(true_gain[defined]))) if np.any(defined) else None
        ),
        "selected_mu": float(fit.parameters.get("mu", 0.0)),
        "selected_upper_boundary": fit.parameters.get("mu") == oracle.STRENGTHS[-1],
        "selected_from_failed_refinement": fit.search_space["selected_from_failed_refinement"],
        "failed_refinements": sum(not item["success"] for item in fit.search_space["refinement_results"]),
        "gamma": fit.kappa,
        "floor": fit.head.floor,
        "shape": fit.shape,
        "ridge": fit.ridge,
    }


def calibration_flags(
    fit: models.JointFit, early: np.ndarray, late: np.ndarray, cache: models.BenefitCache
) -> np.ndarray:
    matrix, offset = models.design_and_offset(
        early, late, fit.alpha, fit.shape, fit.model, fit.parameters, fit.config, cache
    )
    linear = fit.head.intercept + matrix @ fit.head.coefficients + offset
    return abs(linear) >= models.mariner.LOG_CLIP


def support_distance(
    early: np.ndarray,
    late: np.ndarray,
    query_early: np.ndarray,
    query_late: np.ndarray,
    scales: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Nearest training distance in the two named phase-mixture simplexes."""
    training = np.hstack((early / (alpha * scales), late / ((1 - alpha) * scales)))
    query = np.hstack((query_early / (alpha * scales), query_late / ((1 - alpha) * scales)))
    return np.min(np.linalg.norm(query[:, None] - training[None], axis=2), axis=1)


def prepare() -> dict[str, Any]:
    identity = execution_identity()
    folder, spec = OUTPUT / "prepared", {"stage": "prepare", "contract": CONTRACT}
    if oracle.completed(folder, spec, identity):
        return json.loads((folder / "preparation.json").read_text())
    value = {
        "prerequisite": verify_prerequisite(),
        "cache_policy": "New stage-local cache; sealed stage1 SQLite is never opened for writing",
        "truth_policy": "Read exact sealed clean and matched arrays, no truth regeneration or measured labels",
    }
    benchmark.write_json(folder / "preparation.json", value)
    oracle.finish(folder, spec, identity, ("preparation.json",), (PRIOR / "result_manifest.json",))
    return value


def run() -> None:
    identity = execution_identity()
    prepare()
    data, folds = oracle.inputs()
    split = oracle.split_indices(data.rows, folds, 0)
    train, labels = split["train"], split["inner"]
    query_early, query_late, query = oracle.query_frame(data, split)
    query["nearest_training_phase_weight_distance"] = support_distance(
        data.early[train], data.late[train], query_early, query_late, data.total_scales, data.alpha
    )
    cache = models.BenefitCache(OUTPUT / "features.sqlite")
    with np.load(PRIOR / "prepared/clean.npz") as original, np.load(PRIOR / "prepared/matched_tied.npz") as tied:
        for world in oracle.WORLDS:
            aggregate = None
            values = original[world["name"]]
            truth = np.r_[values[split["test"]], tied[world["name"]][split["test"]], values[split["tied_test"]]]
            for candidate in oracle.candidates(world):
                spec = {"world": world["name"], "candidate": candidate, "outer_fold": 0, "noise_sd": 0.0}
                folder = OUTPUT / "cells" / world["name"] / candidate
                if oracle.completed(folder, spec, identity):
                    if candidate == "aggregate_mariner":
                        aggregate = models.JointFit.from_json(json.loads((folder / "model.json").read_text()))
                    continue
                config = oracle.feature_config(
                    data.total_scales, oracle.CLOCKS[0] if candidate == "aggregate_mariner" else candidate
                )
                fit = fit_candidate(
                    data.early[train],
                    data.late[train],
                    values[train],
                    labels,
                    data.alpha,
                    candidate,
                    config,
                    cache,
                    aggregate,
                )
                predicted = fit.predict(query_early, query_late, cache)
                inner, heads = selected_inner(fit, data.early[train], data.late[train], values[train], labels, cache)
                scores = diagnostic_scores(fit, predicted, truth, query)
                scores["inner_rmse_replay"] = float(np.sqrt(np.mean((inner - values[train]) ** 2)))
                clips = calibration_flags(fit, query_early, query_late, cache)
                scores["query_clipping_count"] = int(np.sum(clips))
                scores["training_deficit_clip_count"] = int(
                    np.sum(values[train] - fit.head.floor <= models.mariner.DEFICIT_FLOOR)
                )
                scores["inner_training_deficit_clip_count"] = sum(head["training_deficit_clip_count"] for head in heads)
                scores["inner_prediction_clip_count"] = sum(head["held_prediction_clip_count"] for head in heads)
                folder.mkdir(parents=True, exist_ok=True)
                benchmark.write_json(folder / "model.json", fit.to_json())
                benchmark.write_json(folder / "inner_heads.json", heads)
                benchmark.write_json(folder / "scores.json", scores)
                pd.DataFrame(
                    {
                        "source_index": train,
                        "row_id": data.rows.iloc[train].row_id.to_numpy(),
                        "inner_fold": labels,
                        "truth": values[train],
                        "predicted": inner,
                    }
                ).to_csv(folder / "inner_predictions.csv", index=False)
                query.assign(truth=truth, predicted=predicted, optimism=truth - predicted, clipped=clips).to_csv(
                    folder / "predictions.csv", index=False
                )
                dependencies = [OUTPUT / "prepared/complete.json"]
                if candidate == "aggregate_mariner":
                    aggregate = fit
                else:
                    dependencies.append(OUTPUT / "cells" / world["name"] / "aggregate_mariner/complete.json")
                oracle.finish(
                    folder,
                    spec,
                    identity,
                    ("model.json", "inner_heads.json", "scores.json", "inner_predictions.csv", "predictions.csv"),
                    tuple(dependencies),
                )
    summarize()


def gate_checks(
    world: dict[str, Any], candidate: str, score: dict[str, Any], native: dict[str, Any]
) -> list[dict[str, Any]]:
    gates = CONTRACT["gates"]
    checks = []

    def check(name: str, value: float, limit: float, *, strict: bool = False) -> None:
        checks.append(
            {
                "name": name,
                "value": value,
                "limit": limit,
                "strict": strict,
                "passed": bool(value < limit if strict else value <= limit),
            }
        )

    correct = world["kind"] == "native" or (world["kind"] == "temporal" and candidate == world["clock"])
    if correct:
        for key in ("asymmetric_rmse", "actual_tied_rmse", "matched_tied_rmse"):
            check(key, score[key], gates["correct_family_rmse"])
        check(
            "frontier_absolute_mean_optimism",
            abs(score["frontier_mean_optimism"]),
            gates["frontier_absolute_mean_optimism"],
        )
        check("regret_at_1", score["regret_at_1"], gates["regret_at_1"])
    if world["kind"] == "native":
        check(
            "exact_null_absolute_gain",
            score["maximum_absolute_predicted_gain"],
            gates["exact_null_maximum_absolute_gain"],
        )
    if world["kind"] in ("native", "heterogeneous"):
        check(
            "null_maximum_positive_gain", score["maximum_positive_predicted_gain"], gates["null_maximum_positive_gain"]
        )
        check("null_gain_rmse", score["gain_rmse"], gates["null_gain_rmse"])
    if world["kind"] == "temporal" and correct:
        check("positive_gain_rmse", score["gain_rmse"], gates["positive_gain_rmse"])
        check(
            "positive_gain_zero_baseline_ratio",
            score["gain_rmse"],
            gates["positive_gain_zero_baseline_ratio"] * score["true_gain_rms"],
            strict=True,
        )
    if world["kind"] == "heterogeneous" and candidate != "aggregate_mariner":
        for key in ("asymmetric_rmse", "actual_tied_rmse", "matched_tied_rmse"):
            check(f"misspecified_null_{key}", score[key], native[key] + gates["misspecified_null_native_rmse_allowance"])
    check("upper_boundary", float(score["selected_upper_boundary"]), 0.0)
    for key in CONTRACT["unresolved_numerical_fields"]:
        check(f"unresolved_{key}", float(score[key]), 0.0)
    return checks


def summarize() -> None:
    identity = execution_identity()
    rows, dependencies = [], []
    for world in oracle.WORLDS:
        native = json.loads((OUTPUT / "cells" / world["name"] / "aggregate_mariner/scores.json").read_text())
        for candidate in oracle.candidates(world):
            spec = {"world": world["name"], "candidate": candidate, "outer_fold": 0, "noise_sd": 0.0}
            folder = OUTPUT / "cells" / world["name"] / candidate
            if not oracle.completed(folder, spec, identity):
                raise ValueError("Preserve partial results; all ten cells must complete before qualification")
            score = json.loads((folder / "scores.json").read_text())
            checks = gate_checks(world, candidate, score, native)
            absolute = all(
                score[key] <= CONTRACT["gates"]["correct_family_rmse"]
                for key in ("asymmetric_rmse", "actual_tied_rmse", "matched_tied_rmse")
            )
            rows.append(
                {
                    **spec,
                    **score,
                    "checks": checks,
                    "passed": all(item["passed"] for item in checks),
                    "absolute_calibration_passed": absolute,
                    "native_null_estimator_failed": world["kind"] == "native" and not absolute,
                    "static_approximation_limited": (
                        world["kind"] == "heterogeneous"
                        and not all(
                            native[key] <= CONTRACT["gates"]["correct_family_rmse"]
                            for key in ("asymmetric_rmse", "actual_tied_rmse", "matched_tied_rmse")
                        )
                    ),
                }
            )
            dependencies.append(folder / "complete.json")
    if len(rows) != CONTRACT["cells"]:
        raise ValueError("Learned smoke coverage differs")
    summary = {
        "cells": len(rows),
        "passed": all(item["passed"] for item in rows),
        "native_null_estimator_failed": any(item["native_null_estimator_failed"] for item in rows),
        "results": rows,
        "raw_optimization_complete": False,
        "further_execution_authorized": False,
        "scope": (
            "Noiseless outer-zero fit/menu smoke only; raw policy, all-fold noisy qualification "
            "and independent replay remain separate"
        ),
    }
    benchmark.write_json(OUTPUT / "summary.json", summary)
    oracle.finish(OUTPUT, {"stage": "learned_smoke_summary"}, identity, ("summary.json",), tuple(dependencies))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "prepare", "run", "summarize"))
    command = parser.parse_args().command
    {"freeze": freeze, "prepare": prepare, "run": run, "summarize": summarize}[command]()


if __name__ == "__main__":
    main()
