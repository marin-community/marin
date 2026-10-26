# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Conditional raw-versus-NG2-007 ridge diagnostic; no learned kinetics or floor.

Only approved synthetic labels are used. Known truth shape, kinetics and floor
are identical across arms. The old physical reference scales are not envelopes
for rate-amplified harm. This is estimator isolation, not model qualification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_two_phase_20260921 as benchmark
from experiments.domain_phase_mix.exploratory.two_phase_many import nextgen_two_phase_round2_20260921 as old_scales
from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_tangent_round5_20261015 as scoring
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_rate_reopening_20261015 as truth_source

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reference_outputs/tpp40_penalty_isolation_20261015"
OLD = ROOT / "reference_outputs/two_phase_nextgen_20260921/round3"
SEEDS = (20261015, 20261016, 20261017)
PENALTIES = ("raw", "ng2_007_reference")
STAGE1_SEAL = "d101b3c8eef79f61b1591e851f7a369a3ce8af5c67aa6a60c80939bcf4c3ce4c"
mariner = truth_source.mariner
CONTRACT = {
    "outer_fold": 0,
    "train_rows": 172,
    "asymmetric_rows": 66,
    "actual_tied_rows": 42,
    "matched_tied_rows": 66,
    "inner_folds": 3,
    "shape": {"rate": 0.25, "power": 0.7, "threshold": 3.0},
    "floor": 0.6,
    "ridge_grid": list(mariner.RIDGE_GRID),
    "penalties": list(PENALTIES),
    "noise_sd": 0.003,
    "noise_seeds": list(SEEDS),
    "selected_full_heads": 32,
    "inner_heads": 480,
    "selection": "Pooled inner BPB RMSE, then native ridge-grid index; no query labels",
    "calibration_strata": ["clean_asymmetric_rmse", "clean_actual_tied_rmse", "clean_matched_tied_rmse"],
    "descriptive_continuation": (
        "Strict decrease in unweighted mean squared gain error across the nine non-null noisy cells, "
        "with no increase in each of the three separately pooled calibration MSEs across all twelve noisy cells. "
        "Directional evidence only; no scientific qualification, significance or automatic promotion."
    ),
}


def context_specs() -> list[dict[str, Any]]:
    return [
        {
            **world,
            "truth_world": world["name"],
            "name": f"{world['name']}_noise{noise:g}_seed{seed}",
            "noise_sd": noise,
            "noise_seed": seed,
        }
        for world in truth_source.registration()["worlds"]
        for noise, seed in [(0.0, SEEDS[0]), *[(0.003, seed) for seed in SEEDS]]
    ]


def source_paths() -> set[Path]:
    paths = {Path(path) for path in truth_source.source_identity()["sources"]}
    paths.update(Path(path) for path in scoring.source_identity())
    paths.update(Path(path) for path in old_scales.source_hashes())
    paths.update(
        {
            Path(__file__).resolve(),
            ROOT / "test_tpp40_penalty_isolation_20261015.py",
            OLD / "registration.json",
            OLD / "REPORT.md",
            OLD / "promotion/screen_receipt.json",
            truth_source.OUTPUT / "result_manifest.json",
            truth_source.OUTPUT / "execution_identity.json",
            truth_source.OUTPUT / "audit/final_report_v1.json",
            OUTPUT / "registration.json",
        }
    )
    return {path.resolve() for path in paths}


def verify_adoption() -> None:
    """Verify sealed synthetic law/geometry and the original NG2-007 scale source."""
    if benchmark.sha256(truth_source.OUTPUT / "result_manifest.json") != STAGE1_SEAL:
        raise ValueError("TP026 stage-one seal changed")
    seal = json.loads((truth_source.OUTPUT / "result_manifest.json").read_text())
    benchmark.verify_hashes({path: item["sha256"] for path, item in seal["files"].items()}, truth_source.OUTPUT)
    frozen = json.loads((truth_source.OUTPUT / "execution_identity.json").read_text())
    if truth_source.source_identity() != frozen:
        raise ValueError("TP026 frozen implementation/input identity changed")
    old_receipt = json.loads((OLD / "promotion/screen_receipt.json").read_text())
    scale_path = Path(old_scales.__file__).resolve()
    if benchmark.sha256(scale_path) != old_receipt["sources"][str(scale_path)]:
        raise ValueError("NG2-007 scale implementation differs from its original screen receipt")


def registration() -> dict[str, Any]:
    value = json.loads((OUTPUT / "registration.json").read_text())
    expected = {
        "candidate_id": "TP40-030",
        "status": "registered",
        "execution_contract": CONTRACT,
        "contexts": context_specs(),
        "environment": benchmark.environment_identity(),
        "stage1_seal_sha256": STAGE1_SEAL,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError("Registration differs from conditional penalty isolation")
    paths = source_paths() - {OUTPUT / "registration.json"}
    if set(value["source_hashes"]) != {str(path) for path in paths}:
        raise ValueError("Registration does not pin the complete source/input set")
    benchmark.verify_hashes(value["source_hashes"])
    verify_adoption()
    return value


def source_identity() -> dict[str, Any]:
    return {
        "sources": {str(path): benchmark.sha256(path) for path in sorted(source_paths())},
        "environment": benchmark.environment_identity(),
    }


def freeze() -> None:
    registration()
    current = source_identity()
    destination = OUTPUT / "execution_identity.json"
    if destination.exists() and json.loads(destination.read_text()) != current:
        raise ValueError("Preserve previous identity; use a separately reviewed revision")
    benchmark.write_json(destination, current)


def execution_identity() -> dict[str, Any]:
    registration()
    current = source_identity()
    if json.loads((OUTPUT / "execution_identity.json").read_text()) != current:
        raise ValueError("Frozen conditional diagnostic changed")
    approval = json.loads((OUTPUT / "execution_approval.json").read_text())
    expected = {
        "scope": "conditional_penalty_isolation_32_heads",
        "registration_sha256": benchmark.sha256(OUTPUT / "registration.json"),
        "execution_identity_sha256": benchmark.sha256(OUTPUT / "execution_identity.json"),
    }
    if any(approval.get(key) != item for key, item in expected.items()):
        raise ValueError("Execution approval does not name this frozen diagnostic")
    return current


def completed(folder: Path, spec: dict[str, Any], identity: dict[str, Any], dependencies: dict[str, str]) -> bool:
    path = folder / "complete.json"
    if not path.exists():
        return False
    receipt = json.loads(path.read_text())
    if receipt["identity"] != identity or receipt["spec"] != spec or receipt["dependencies"] != dependencies:
        raise ValueError(f"Completed receipt changed: {folder}")
    benchmark.verify_hashes(dependencies)
    benchmark.verify_hashes(receipt["outputs"], folder)
    return True


def finish(folder: Path, spec: dict[str, Any], outputs: tuple[str, ...], dependencies: dict[str, str]) -> None:
    benchmark.verify_hashes(dependencies)
    benchmark.write_json(
        folder / "complete.json",
        {
            "identity": execution_identity(),
            "spec": spec,
            "dependencies": dependencies,
            "outputs": {name: benchmark.sha256(folder / name) for name in outputs},
        },
    )


def head_record(head: mariner.Head) -> dict[str, Any]:
    return {"floor": head.floor, "intercept": head.intercept, "coefficients": head.coefficients.tolist()}


def read_head(value: dict[str, Any]) -> mariner.Head:
    return mariner.Head(float(value["intercept"]), np.asarray(value["coefficients"]), float(value["floor"]))


def fixed_floor_head(
    matrix: np.ndarray, response: np.ndarray, ridge: float, floor: float, scales: np.ndarray
) -> mariner.Head:
    """Use native centered QR/NNLS with only its diagonal ridge rows changed."""
    deficit = response - floor
    if not np.isfinite(matrix).all() or not np.isfinite(deficit).all() or np.min(deficit) < mariner.DEFICIT_FLOOR:
        raise ValueError("Invalid design or clamped fixed-floor deficit; preserve failure")
    if scales.shape != (matrix.shape[1],) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Penalty scales must be finite positive column multipliers")
    target = np.log(deficit)
    if ridge == 0 or np.array_equal(scales, np.ones(matrix.shape[1])):
        intercept, coefficients = mariner.nonnegative_solve(matrix, target, ridge)
        return mariner.Head(intercept, coefficients, floor)
    width = matrix.shape[1]
    mean = matrix.mean(axis=0)
    target_mean = float(target.mean())
    rows = np.vstack((matrix - mean, np.diag(np.sqrt(ridge) * scales)))
    rhs = np.r_[target - target_mean, np.zeros(width)]
    if rows.shape[0] > 2 * width:
        orthogonal, triangular = np.linalg.qr(rows, mode="reduced")
        rows, rhs = triangular, orthogonal.T @ rhs
    coefficients, _ = nnls(rows, rhs, maxiter=mariner.NNLS_MAXITER_FACTOR * width)
    return mariner.Head(target_mean - float(mean @ coefficients), coefficients, floor)


def fit_conditional(
    matrix: np.ndarray,
    observed: np.ndarray,
    labels: np.ndarray,
    query_matrix: np.ndarray,
    floor: float,
    scales: np.ndarray,
) -> dict[str, Any]:
    """Select only native ridge strength; save every split-local head and OOF."""
    splits = truth_source.joint.temporal.inner_splits(labels)
    profile = []
    with threadpool_limits(limits=1):
        for index, ridge in enumerate(mariner.RIDGE_GRID):
            oof = np.full(len(observed), np.nan)
            heads = []
            for train, held in splits:
                head = fixed_floor_head(matrix[train], observed[train], ridge, floor, scales)
                oof[held] = head.predict(matrix[held])
                linear = head.intercept + matrix[held] @ head.coefficients
                heads.append(
                    {
                        "train_positions": train.tolist(),
                        "held_positions": held.tolist(),
                        "head": head_record(head),
                        "held_clip_count": int(np.sum(abs(linear) >= mariner.LOG_CLIP)),
                    }
                )
            if not np.isfinite(oof).all():
                raise ValueError("Incomplete or nonfinite inner predictions")
            profile.append(
                {
                    "ridge_index": index,
                    "ridge": ridge,
                    "inner_rmse": float(np.sqrt(np.mean((oof - observed) ** 2))),
                    "inner_predictions": oof.tolist(),
                    "inner_heads": heads,
                }
            )
        winner = min(profile, key=lambda row: (row["inner_rmse"], row["ridge_index"]))
        head = fixed_floor_head(matrix, observed, winner["ridge"], floor, scales)
    linear = head.intercept + query_matrix @ head.coefficients
    return {
        "selected_ridge_index": winner["ridge_index"],
        "selected_ridge": winner["ridge"],
        "selected_inner_rmse": winner["inner_rmse"],
        "profile": profile,
        "head": head_record(head),
        "penalty_scales": scales.tolist(),
        "query_predictions": head.predict(query_matrix).tolist(),
        "query_clip_count": int(np.sum(abs(linear) >= mariner.LOG_CLIP)),
    }


def reference_scales(geometry: scoring.Geometry, shape: dict[str, float], c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
    """Call the unchanged NG2-007 static-reference scale function with no targets."""
    panel = benchmark.data.BenchPanel(
        name="tpp40_conditional_covariates_only",
        buckets=tuple(str(i) for i in range(geometry.early.shape[1])),
        phase0=geometry.phase0,
        phase1=geometry.phase1,
        alpha=geometry.alpha,
        c0=c0,
        c1=c1,
        targets={},
        rows=pd.DataFrame({"tied": geometry.tied}),
        input_hashes={},
        metadata={},
    )
    return old_scales.feature_spans(panel, shape)


def prepare() -> None:
    identity = execution_identity()
    arrays, folds, truth, head = truth_source.inputs()
    geometry = scoring.load_geometry()
    if truth["shape"] != CONTRACT["shape"] or head.floor != CONTRACT["floor"]:
        raise ValueError("Frozen generating shape or floor differs")
    with np.load(truth_source.DATA / "arrays.npz", allow_pickle=False) as inventory:
        scales = reference_scales(geometry, truth["shape"], inventory["c0"], inventory["c1"])
    for spec in context_specs():
        folder = OUTPUT / "contexts" / spec["name"]
        if completed(folder, spec, identity, {}):
            continue
        data = truth_source.world_inputs(arrays, folds, truth, head, spec["parameters"])
        noise = np.random.default_rng(spec["noise_seed"]).normal(0, spec["noise_sd"], len(data["clean"]))
        observed = data["clean"].copy()
        observed[data["train"]] += noise[data["train"]]
        validity = {
            "finite_truth": bool(np.isfinite(data["clean"]).all() and np.isfinite(data["clean_tied"]).all()),
            "truth_unclipped": all(bool(np.all(abs(data[key]) < mariner.LOG_CLIP)) for key in ("linear", "tied_linear")),
            "minimum_training_deficit": float(np.min(observed[data["train"]] - head.floor)),
        }
        valid = (
            validity["finite_truth"]
            and validity["truth_unclipped"]
            and validity["minimum_training_deficit"] >= mariner.DEFICIT_FLOOR
        )
        folder.mkdir(parents=True, exist_ok=True)
        benchmark.write_json(folder / "validity.json", {**validity, "valid": valid, "identity": identity, "spec": spec})
        if not valid:
            raise ValueError(f"Invalid fixed truth or noisy deficit in {spec['name']}; preserve without rescue")
        np.savez(
            folder / "labels.npz",
            clean=data["clean"],
            clean_tied=data["clean_tied"],
            observed=observed,
            noise_vector=noise,
            train=data["train"],
            asymmetric=data["held"],
            tied=data["tied"],
            inner_labels=data["labels"],
            reference_scales=scales,
        )
        benchmark.write_json(folder / "truth.json", {**truth, "parameters": spec["parameters"]})
        finish(folder, spec, ("labels.npz", "truth.json", "validity.json"), {})


def load_context(spec: dict[str, Any], identity: dict[str, Any]) -> scoring.Context:
    folder = OUTPUT / "contexts" / spec["name"]
    if not completed(folder, spec, identity, {}):
        raise ValueError("Prepare the approved context first")
    with np.load(folder / "labels.npz", allow_pickle=False) as data:
        return scoring.Context(
            spec,
            scoring.load_geometry(),
            data["train"],
            data["asymmetric"],
            data["tied"],
            data["inner_labels"],
            data["clean"],
            data["observed"],
            data["clean_tied"],
            json.loads((folder / "truth.json").read_text()),
        )


def fit() -> None:
    for spec in context_specs():
        identity = execution_identity()
        context = load_context(spec, identity)
        geometry, shape = context.geometry, context.truth["shape"]
        query_early, query_late = scoring.query_coordinates(context)
        train = context.train
        matrix = truth_source.design(
            geometry.early[train], geometry.late[train], geometry.alpha, shape, spec["parameters"]
        )
        query = truth_source.design(query_early, query_late, geometry.alpha, shape, spec["parameters"])
        context_path = OUTPUT / "contexts" / spec["name"]
        dependencies = {str(context_path / "complete.json"): benchmark.sha256(context_path / "complete.json")}
        with np.load(context_path / "labels.npz", allow_pickle=False) as saved:
            physical_scales = saved["reference_scales"]
        for penalty in PENALTIES:
            identity = execution_identity()
            cell = {"context": spec, "penalty": penalty}
            folder = OUTPUT / "fits" / spec["name"] / penalty
            if completed(folder, cell, identity, dependencies):
                continue
            scales = np.ones(matrix.shape[1]) if penalty == "raw" else physical_scales
            fitted = fit_conditional(
                matrix, context.observed[train], context.inner_labels, query, CONTRACT["floor"], scales
            )
            predicted = np.asarray(fitted["query_predictions"])
            scores, rows = scoring.prediction_scores(context, predicted)
            benchmark.write_json(folder / "fit.json", fitted)
            benchmark.write_json(folder / "scores.json", scores)
            rows.to_csv(folder / "predictions.csv", index=False)
            np.savez(
                folder / "replay.npz",
                train_matrix=matrix,
                query_matrix=query,
                train_observed=context.observed[train],
                inner_labels=context.inner_labels,
                penalty_scales=scales,
            )
            finish(folder, cell, ("fit.json", "scores.json", "predictions.csv", "replay.npz"), dependencies)
            print(f"Completed {spec['name']} / {penalty}", flush=True)


def paired_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Use equal cell weights; keep all three calibration strata separate."""
    noisy = [row for row in records if row["noise_sd"] > 0]
    metrics = [*CONTRACT["calibration_strata"], "clean_gain_rmse"]
    pooled: dict[str, Any] = {}
    for metric in metrics:
        eligible = [row for row in noisy if metric != "clean_gain_rmse" or row["truth_world"] != "null"]
        by_arm = {
            arm: float(np.mean([row[metric] ** 2 for row in eligible if row["penalty"] == arm])) for arm in PENALTIES
        }
        pooled[metric] = {
            "cells_per_arm": len(eligible) // 2,
            "mse_bpb_squared": by_arm,
            "reference_over_raw_mse_ratio": (
                by_arm[PENALTIES[1]] / by_arm[PENALTIES[0]] if by_arm[PENALTIES[0]] else None
            ),
            "reference_minus_raw_mse": by_arm[PENALTIES[1]] - by_arm[PENALTIES[0]],
        }
    improvement = pooled["clean_gain_rmse"]["reference_minus_raw_mse"] < 0 and all(
        pooled[key]["reference_minus_raw_mse"] <= 0 for key in CONTRACT["calibration_strata"]
    )
    return {
        "pooled": pooled,
        "directional_continuation_signal": improvement,
        "interpretation": CONTRACT["descriptive_continuation"],
        "null_false_gain": {
            arm: max(
                (
                    row["max_abs_predicted_gain"]
                    for row in noisy
                    if row["truth_world"] == "null" and row["penalty"] == arm
                ),
                default=None,
            )
            for arm in PENALTIES
        },
        "null_scope": (
            "Kinetics are fixed to zero in null worlds; zero phase gain is structural "
            "and does not certify kinetic selection."
        ),
    }


def summarize() -> None:
    identity = execution_identity()
    records, dependencies = [], {}
    for spec in context_specs():
        context_path = OUTPUT / "contexts" / spec["name"] / "complete.json"
        context_dependencies = {str(context_path): benchmark.sha256(context_path)}
        load_context(spec, identity)
        for penalty in PENALTIES:
            folder = OUTPUT / "fits" / spec["name"] / penalty
            if not completed(folder, {"context": spec, "penalty": penalty}, identity, context_dependencies):
                raise ValueError("All 32 conditional heads are required before paired interpretation")
            dependencies[str(folder / "complete.json")] = benchmark.sha256(folder / "complete.json")
            fitted = json.loads((folder / "fit.json").read_text())
            records.append(
                {
                    **spec,
                    "penalty": penalty,
                    **json.loads((folder / "scores.json").read_text()),
                    "selected_ridge": fitted["selected_ridge"],
                    "query_clip_count": fitted["query_clip_count"],
                }
            )
    result = paired_summary(records)
    result["per_world"] = {
        world: paired_summary([row for row in records if row["truth_world"] == world])
        for world in ("benefit", "rate", "combined")
    }
    result["per_seed"] = {
        str(seed): paired_summary([row for row in records if row["noise_seed"] == seed and row["noise_sd"] > 0])
        for seed in SEEDS
    }
    benchmark.write_json(OUTPUT / "summary.json", {**result, "records": records, "qualification": None})
    pd.DataFrame(records).to_csv(OUTPUT / "scores.csv", index=False)
    finish(OUTPUT, {"scope": "32 conditional heads"}, ("summary.json", "scores.csv"), dependencies)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "prepare", "fit", "summarize"))
    args = parser.parse_args()
    {"freeze": freeze, "prepare": prepare, "fit": fit, "summarize": summarize}[args.command]()


if __name__ == "__main__":
    main()
