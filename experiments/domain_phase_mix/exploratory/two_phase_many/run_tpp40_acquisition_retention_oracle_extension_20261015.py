# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Known-floor/shape noise/support diagnostics and separately approved truth search.

The learned estimator remains blocked by its prior floor-domain result. This
runner supplies neither a replacement estimator nor a qualification verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
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
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_two_bucket_optimization_20261015 as optimizer

models, benchmark = oracle.models, oracle.benchmark
ROOT = Path(__file__).resolve().parent
PRIOR = oracle.OUTPUT
OUTPUT = oracle.PARENT / "oracle_extension"
OPTIMIZER_SEAL = oracle.PARENT / "optimizer/result_manifest_v1.json"
SCOPES = {"oracle": "tp034_extended_known_floor_oracle", "raw_truth": "tp034_raw_truth_only"}
CONTRACT = {
    "worlds": list(oracle.WORLDS),
    "outer_folds": list(range(5)),
    "noise_names": ["clean", *map(str, oracle.SEEDS)],
    "noise_sd": oracle.CONTRACT["noise_sd"],
    "strengths": list(oracle.STRENGTHS),
    "known_shape": oracle.SHAPE,
    "known_floor": oracle.HEAD["floor"],
    "ridge": 0.0,
    "cells": 200,
    "conditional_nnls_solves": 3680,
    "duplicate_clean_outer_zero_cells": 10,
    "frontier_fraction": 0.2,
    "gain_sign_zero_tolerance": 1e-12,
    "search_numerics": {"relative_tolerance": 1e-11, "max_panels": 512},
    "replay_numerics": {"relative_tolerance": 1e-12, "max_panels": 2048},
    "raw_worlds": 4,
    "learned_fitting": False,
    "fitted_policy_optimization": False,
    "promotion": False,
}


def source_paths() -> set[Path]:
    return oracle.source_paths() | {
        Path(__file__),
        ROOT / "test_tpp40_acquisition_retention_oracle_extension_20261015.py",
        Path(optimizer.__file__),
        ROOT / "test_tpp40_two_bucket_optimization_20261015.py",
        ROOT / "tpp40_tangent_optimization_20261015.py",
        oracle.PARENT / "optimizer/independent_source_review_v1.json",
        oracle.PARENT / "optimizer/author_validation_v1.json",
        oracle.PARENT / "optimizer/README.md",
        OPTIMIZER_SEAL,
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
            "prepared/noise.npz",
            "prepared/preflight.json",
        )
    }


def verify_prerequisite() -> None:
    seal = json.loads((PRIOR / "result_manifest.json").read_text())
    benchmark.verify_hashes({name: entry["sha256"] for name, entry in seal["files"].items()}, PRIOR)
    audit = json.loads((PRIOR / "audit/independent_replay_report_v1.json").read_text())
    summary = json.loads((PRIOR / "oracle_summary.json").read_text())
    if not audit["passed"] or not summary["oracle_passed"]:
        raise ValueError("Require the independently replayed clean known-floor oracle")
    optimizer_seal = json.loads(OPTIMIZER_SEAL.read_text())
    if optimizer_seal["status"] != "source_and_toy_review_pass":
        raise ValueError("Require the independently reviewed optimizer")
    benchmark.verify_hashes({name: entry["sha256"] for name, entry in optimizer_seal["files"].items()})
    identity = oracle.execution_identity()
    if not oracle.completed(PRIOR, {"stage": "oracle_summary"}, identity):
        raise ValueError("Prior oracle receipt is missing")
    if not oracle.completed(PRIOR / "prepared", {"stage": "prepare", "contract": oracle.CONTRACT}, identity):
        raise ValueError("Prior truth receipt is missing")


def registration() -> dict[str, Any]:
    value = json.loads((OUTPUT / "registration.json").read_text())
    if value.get("status") != "registered" or value.get("candidate_id") != "TP40-034":
        raise ValueError("Register the reviewed diagnostic scope before execution")
    if value.get("contract") != CONTRACT or value.get("environment") != benchmark.environment_identity():
        raise ValueError("Registered diagnostic contract or runtime changed")
    for key, paths in (
        ("source_hashes", source_paths()),
        ("input_hashes", oracle.input_paths()),
        ("prerequisite_hashes", prerequisite_paths()),
    ):
        if set(value[key]) != {str(path.resolve()) for path in paths}:
            raise ValueError("Source, covariate, or prerequisite pin set differs")
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
        raise ValueError("Preserve the previous execution identity")
    benchmark.write_json(path, identity)


def execution_identity(scope: str) -> dict[str, Any]:
    identity = source_identity()
    if json.loads((OUTPUT / "execution_identity.json").read_text()) != identity:
        raise ValueError("Frozen diagnostic identity changed")
    approval = json.loads((OUTPUT / f"{scope}_approval.json").read_text())
    required = {
        "scope": SCOPES[scope],
        "registration_sha256": identity["registration_sha256"],
        "execution_identity_sha256": benchmark.sha256(OUTPUT / "execution_identity.json"),
    }
    if any(approval.get(key) != value for key, value in required.items()):
        raise ValueError("Exact separate-scope approval is required")
    return identity


def contexts() -> list[dict[str, Any]]:
    return [
        {"world": world, "outer_fold": outer, "noise": noise}
        for outer in CONTRACT["outer_folds"]
        for noise in CONTRACT["noise_names"]
        for world in oracle.WORLDS
    ]


def context_folder(spec: dict[str, Any]) -> Path:
    return OUTPUT / "cells" / f"outer{spec['outer_fold']}" / spec["noise"] / spec["world"]["name"]


def noisy_training(clean: np.ndarray, noise: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Noise applies only to the current asymmetric fitting rows."""
    return clean[train] + noise[train]


def profile_diagnostics(fit: dict[str, Any]) -> dict[str, Any]:
    profiles = fit["profiles"]
    selected = profiles[fit["selected_index"]]
    ordered = sorted(profile["inner_rmse"] for profile in profiles)
    return {
        "selected_mu": fit["selected_mu"],
        "selected_upper_boundary": fit["selected_mu"] == oracle.STRENGTHS[-1],
        "selected_inner_rmse": selected["inner_rmse"],
        "runner_up_inner_rmse_gap": ordered[1] - ordered[0] if len(ordered) > 1 else None,
        "inner_rmse_improvement_over_zero": profiles[0]["inner_rmse"] - selected["inner_rmse"],
        "exact_best_score_ties": sum(profile["inner_rmse"] == selected["inner_rmse"] for profile in profiles),
    }


def scores(predicted: np.ndarray, truth: np.ndarray, query: pd.DataFrame) -> dict[str, Any]:
    count = int(np.sum(query.role == "asymmetric"))
    gain = predicted[count : 2 * count] - predicted[:count]
    true_gain = truth[count : 2 * count] - truth[:count]
    order = np.lexsort((query.row_id.to_numpy(dtype=str), query.role.to_numpy(dtype=str), predicted))
    frontier = order[: max(1, int(np.ceil(CONTRACT["frontier_fraction"] * len(order))))]
    optimism = truth - predicted
    tolerance = CONTRACT["gain_sign_zero_tolerance"]
    informative = abs(true_gain) > tolerance
    rank = (
        float(spearmanr(gain, true_gain).statistic)
        if np.ptp(gain) > tolerance and np.ptp(true_gain) > tolerance
        else None
    )
    result = {
        **oracle.prediction_scores(predicted, truth, count),
        "maximum_positive_predicted_gain": float(max(0, np.max(gain))),
        "gain_spearman": rank,
        "gain_sign_accuracy": (
            float(np.mean(np.sign(gain[informative]) == np.sign(true_gain[informative])))
            if np.any(informative)
            else None
        ),
        "gain_sign_count": int(np.sum(informative)),
        "frontier_rmse": float(np.sqrt(np.mean(optimism[frontier] ** 2))),
        "frontier_mean_optimism": float(np.mean(optimism[frontier])),
        "frontier_maximum_optimism": float(np.max(optimism[frontier])),
        "selected_role": str(query.iloc[order[0]].role),
        "selected_row_id": str(query.iloc[order[0]].row_id),
        "selected_optimism": float(optimism[order[0]]),
        "frontier_indices": frontier.tolist(),
    }
    for k in (1, 3, 5):
        result[f"regret_at_{k}"] = float(np.min(truth[order[:k]]) - np.min(truth))
    for role in ("asymmetric", "matched_tied", "actual_tied"):
        errors = optimism[query.role.to_numpy() == role]
        result[f"{role}_mean_optimism"] = float(np.mean(errors))
        result[f"{role}_maximum_absolute_error"] = float(np.max(abs(errors)))
    return result


def duplicate_parity(spec: dict[str, Any], candidate: str, fit: dict[str, Any]) -> dict[str, Any]:
    if spec["outer_fold"] != 0 or spec["noise"] != "clean":
        return {"duplicate_of_stage1": False}
    prior = PRIOR / "cells" / spec["world"]["name"] / candidate
    saved = json.loads((prior / "fit.json").read_text())
    equal = all(fit[key] == saved[key] for key in fit)
    if not equal:
        raise ValueError("Duplicated clean outer-zero oracle differs from its sealed original")
    return {
        "duplicate_of_stage1": True,
        "duplicate_complete_sha256": benchmark.sha256(prior / "complete.json"),
        "duplicate_numerical_equal": equal,
    }


def run_oracle() -> None:
    identity = execution_identity("oracle")
    data, folds = oracle.inputs()
    cache = models.BenefitCache(OUTPUT / "features.sqlite")
    with (
        np.load(PRIOR / "prepared/clean.npz") as clean,
        np.load(PRIOR / "prepared/matched_tied.npz") as ties,
        np.load(PRIOR / "prepared/noise.npz") as noises,
    ):
        for spec in contexts():
            split = oracle.split_indices(data.rows, folds, spec["outer_fold"])
            train, labels = split["train"], split["inner"]
            values = clean[spec["world"]["name"]]
            observed = noisy_training(values, noises[spec["noise"]], train)
            query0, query1, query = oracle.query_frame(data, split)
            train_weights = np.hstack(
                (
                    data.early[train] / (data.alpha * data.total_scales),
                    data.late[train] / ((1 - data.alpha) * data.total_scales),
                )
            )
            query_weights = np.hstack(
                (query0 / (data.alpha * data.total_scales), query1 / ((1 - data.alpha) * data.total_scales))
            )
            code = data.buckets.index("starcoder")
            query["early_starcoder_weight"] = query_weights[:, code]
            query["late_starcoder_weight"] = query_weights[:, code + 2]
            query["nearest_training_phase_weight_distance"] = np.min(
                np.linalg.norm(query_weights[:, None] - train_weights[None], axis=2), axis=1
            )
            truth = np.r_[values[split["test"]], ties[spec["world"]["name"]][split["test"]], values[split["tied_test"]]]
            for candidate in oracle.candidates(spec["world"]):
                cell_spec = {**spec, "candidate": candidate}
                folder = context_folder(spec) / candidate
                if oracle.completed(folder, cell_spec, identity):
                    continue
                config = oracle.feature_config(
                    data.total_scales, oracle.CLOCKS[0] if candidate == "aggregate_mariner" else candidate
                )
                with threadpool_limits(limits=1):
                    fit = oracle.oracle_fit(
                        data.early[train],
                        data.late[train],
                        observed,
                        labels,
                        query0,
                        query1,
                        data.alpha,
                        config,
                        (0.0,) if candidate == "aggregate_mariner" else oracle.STRENGTHS,
                        oracle.HEAD["floor"],
                        cache,
                    )
                selected = fit["profiles"][fit["selected_index"]]
                predicted = np.asarray(selected["query_predictions"])
                report = {
                    **scores(predicted, truth, query),
                    **profile_diagnostics(fit),
                    **duplicate_parity(spec, candidate, fit),
                    "conditional_nnls_solves": fit["conditional_nnls_solves"],
                    "diagnostic_only": True,
                }
                folder.mkdir(parents=True, exist_ok=True)
                benchmark.write_json(folder / "fit.json", {**fit, "spec": cell_spec, "config": config.to_json()})
                benchmark.write_json(folder / "scores.json", report)
                query.assign(truth=truth, predicted=predicted, optimism=truth - predicted).to_csv(
                    folder / "predictions.csv", index=False
                )
                pd.DataFrame(
                    {
                        "source_index": train,
                        "row_id": data.rows.iloc[train].row_id.to_numpy(),
                        "inner_fold": labels,
                        "clean": values[train],
                        "noise": noises[spec["noise"]][train],
                        "observed": observed,
                        "predicted": selected["inner_predictions"],
                    }
                ).to_csv(folder / "inner_predictions.csv", index=False)
                oracle.finish(
                    folder,
                    cell_spec,
                    identity,
                    ("fit.json", "scores.json", "predictions.csv", "inner_predictions.csv"),
                    (PRIOR / "result_manifest.json", PRIOR / "prepared/complete.json"),
                )
    summarize_oracle()


def summarize_oracle() -> None:
    identity = execution_identity("oracle")
    reports, dependencies = [], []
    for spec in contexts():
        for candidate in oracle.candidates(spec["world"]):
            folder = context_folder(spec) / candidate
            if not oracle.completed(folder, {**spec, "candidate": candidate}, identity):
                raise ValueError("Extended oracle coverage is incomplete")
            reports.append(
                {
                    "world": spec["world"]["name"],
                    "outer_fold": spec["outer_fold"],
                    "noise": spec["noise"],
                    "candidate": candidate,
                    **json.loads((folder / "scores.json").read_text()),
                }
            )
            dependencies.append(folder / "complete.json")
    solves = sum(item["conditional_nnls_solves"] for item in reports)
    if len(reports) != CONTRACT["cells"] or solves != CONTRACT["conditional_nnls_solves"]:
        raise ValueError("Extended oracle cell/solver count differs")
    summary = {
        "status": "diagnostic_complete",
        "cells": len(reports),
        "conditional_nnls_solves": solves,
        "recomputed_stage1_cells": sum(item["duplicate_of_stage1"] for item in reports),
        "results": reports,
        "learned_stage_remains_blocked": True,
        "promotion": False,
        "scope": "Known-floor/shape conditional noise/support baseline; no learned-estimator qualification",
    }
    benchmark.write_json(OUTPUT / "oracle_summary.json", summary)
    oracle.finish(
        OUTPUT / "oracle_completion",
        {"stage": "extended_oracle_summary"},
        identity,
        (),
        (OUTPUT / "oracle_summary.json", *dependencies),
    )


@dataclass(frozen=True)
class TruthPredictor:
    world: dict[str, Any]
    alpha: float
    scales: np.ndarray
    numerics: dict[str, Any]
    cache: models.BenefitCache | None

    def __call__(self, early: np.ndarray, late: np.ndarray) -> float:
        if self.world["kind"] != "temporal":
            values, _ = oracle.truth_values(early[None], late[None], self.alpha, self.scales, self.world, None)
            return float(values[0])
        config = replace(oracle.feature_config(self.scales, self.world["clock"]), **self.numerics)
        matrix, _ = models.design_and_offset(
            early[None], late[None], self.alpha, oracle.SHAPE, models.MODEL, {"mu": self.world["mu"]}, config, self.cache
        )
        linear = oracle.HEAD["intercept"] + matrix @ np.asarray(oracle.HEAD["coefficients"])
        result = oracle.HEAD["floor"] + np.exp(linear)
        if not np.isfinite(result).all():
            raise FloatingPointError("Unclipped hand-defined truth is nonfinite")
        return float(result[0])


def raw_truth() -> None:
    identity = execution_identity("raw_truth")
    data, _ = oracle.inputs()
    cache = models.BenefitCache(OUTPUT / "raw_features.sqlite")
    weights = np.stack(
        (data.early / (data.alpha * data.total_scales), data.late / ((1 - data.alpha) * data.total_scales)), axis=1
    )
    for world in oracle.WORLDS:
        spec = {"world": world, "stage": "raw_truth_only"}
        folder = OUTPUT / "raw_truth" / world["name"]
        if oracle.completed(folder, spec, identity):
            continue
        search = TruthPredictor(world, data.alpha, data.total_scales, CONTRACT["search_numerics"], cache)
        replay = replace(search, numerics=CONTRACT["replay_numerics"])
        predictors = optimizer.Predictors(
            search,
            replay,
            {
                "identity": identity,
                "world": world,
                "shape": oracle.SHAPE,
                "head": oracle.HEAD,
                "response": "unclipped hand-defined exponential truth",
            },
            CONTRACT["search_numerics"],
            CONTRACT["replay_numerics"],
        )
        result = optimizer.optimize_two_bucket(
            predictors,
            alpha=data.alpha,
            scales=data.total_scales,
            restriction=(
                optimizer.Restriction.TEMPORAL if world["kind"] == "temporal" else optimizer.Restriction.AGGREGATE
            ),
            observed_policies=weights,
        )
        benchmark.write_json(folder / "optimization.json", result)
        oracle.finish(folder, spec, identity, ("optimization.json",), (PRIOR / "result_manifest.json",))
    summarize_raw_truth()


def summarize_raw_truth() -> None:
    identity = execution_identity("raw_truth")
    rows, dependencies = [], []
    for world in oracle.WORLDS:
        folder = OUTPUT / "raw_truth" / world["name"]
        if not oracle.completed(folder, {"world": world, "stage": "raw_truth_only"}, identity):
            raise ValueError("Raw truth coverage is incomplete")
        result = json.loads((folder / "optimization.json").read_text())
        best, static = result["best"], result["best_static"]
        rows.append(
            {
                "world": world,
                "best_free": None if best is None else best["predicted"],
                "best_static": None if static is None else static["predicted"],
                "free_advantage": None if best is None else static["predicted"] - best["predicted"],
                "numerical_audit_unresolved": result["numerical_audit_unresolved"],
                "calls": result["calls"],
                "diagnostic_only": True,
            }
        )
        dependencies.append(folder / "complete.json")
    benchmark.write_json(
        OUTPUT / "raw_truth_summary.json",
        {
            "status": "diagnostic_complete",
            "worlds": rows,
            "global_optimum_certified": False,
            "fitted_policy_optimization": False,
            "promotion": False,
        },
    )
    oracle.finish(
        OUTPUT / "raw_truth_completion",
        {"stage": "raw_truth_summary"},
        identity,
        (),
        (OUTPUT / "raw_truth_summary.json", *dependencies),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "oracle", "summarize-oracle", "raw-truth", "summarize-raw-truth"))
    command = parser.parse_args().command
    {
        "freeze": freeze,
        "oracle": run_oracle,
        "summarize-oracle": summarize_oracle,
        "raw-truth": raw_truth,
        "summarize-raw-truth": summarize_raw_truth,
    }[command]()


if __name__ == "__main__":
    main()
