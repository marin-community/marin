# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Four fixed two-bucket truths, floor preflight, and a known-floor oracle only.

Reads whitelisted frozen policy covariates and partitions, never measured losses.
Gamma feasibility controls eligibility for a separately registered learned stage;
a clean known-floor oracle remains informative when that feasibility fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_two_phase_20260921 as benchmark
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    run_tpp40_acquisition_retention_geometry_20261015 as geometry,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    tpp40_acquisition_retention_joint_models_20261015 as models,
)

ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "reference_outputs/tpp40_acquisition_retention_20261015"
OUTPUT = PARENT / "stage1"
DATA = ROOT / "reference_outputs/two_phase_benchmark_20260921/v3/prepared/starcoder_wsd80_dense"
PURE_SEAL = PARENT / "pure_qualification/result_manifest_v1.json"
PURE_SHA = "2c54e3d4d140f4a202b9aa72c9bb9f560c33f752ccf39aec9816003812e047df"
ADAPTER_SEAL = PARENT / "adapter/result_manifest_v1.json"
ADAPTER_SHA = "8a449a632041caae70e95724a6f897b910added2504cb2d39c9a87b2f5f944bc"
ADAPTER_REVIEW = PARENT / "adapter/independent_source_review_v1.json"
ADAPTER_REVIEW_SHA = "98515d2cb4e1793e77063b39e81dbf05799c44d315baa8c9383ad8f91b356e8c"
SCOPE = "tp034_four_world_known_floor_oracle"
CLOCKS = ("equal_group", "global_pool")
STRENGTHS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
SEEDS = (20261015, 20261016, 20261017)
SHAPE = {"rate": 0.25, "power": 0.7, "threshold": 3.0}
HEAD = {"floor": 0.6, "intercept": 1.0, "coefficients": [1.0, 3.0, 0.1, 0.1]}
WORLDS: tuple[dict[str, Any], ...] = (
    {"name": "native_null", "kind": "native", "clock": None, "mu": 0.0},
    {"name": "heterogeneous_null", "kind": "heterogeneous", "clock": None, "mu": 0.0},
    {"name": "equal_positive", "kind": "temporal", "clock": "equal_group", "mu": 3.0},
    {"name": "pool_positive", "kind": "temporal", "clock": "global_pool", "mu": 3.0},
)
CONTRACT = {
    "scope": SCOPE,
    "worlds": list(WORLDS),
    "shape": SHAPE,
    "head": HEAD,
    "heterogeneous_log_rate_span": 0.25,
    "strengths": list(STRENGTHS),
    "quadrature": {"relative_tolerance": 1e-11, "max_panels": 512},
    "noise_sd": 0.003,
    "noise_seeds": list(SEEDS),
    "repeat": 0,
    "oracle_outer_fold": 0,
    "preflight_outer_folds": list(range(5)),
    "oracle_ridge": 0.0,
    "correct_family_query_max_abs_error_bpb": 1e-8,
    "oracle_cells": 10,
    "conditional_nnls_solves": 184,
    "floor_feasibility_records": 320,
    "learned_fitting": False,
    "raw_optimization": False,
}


def input_paths() -> set[Path]:
    return {DATA / name for name in ("policies.npz", "metadata.json", "rows.csv", "folds.csv")}


def source_paths() -> set[Path]:
    return {Path(name) for name in models.SOURCE_HASHES} | {
        Path(__file__),
        Path(benchmark.__file__),
        Path(geometry.__file__),
        ROOT / "test_tpp40_acquisition_retention_oracle_20261015.py",
        ROOT / "test_tpp40_acquisition_retention_joint_models_20261015.py",
        PURE_SEAL,
        ADAPTER_SEAL,
        ADAPTER_REVIEW,
        PARENT / "reviews/synthetic_qualification_proposal_v1.md",
        PARENT / "geometry/audit/interpretation_v1.md",
    }


def verify_pure() -> None:
    if benchmark.sha256(PURE_SEAL) != PURE_SHA:
        raise ValueError("The prerequisite pure-law seal changed")
    seal = json.loads(PURE_SEAL.read_text())
    if seal["status"] != "sealed_pass":
        raise ValueError("The pure-law prerequisite must pass")
    benchmark.verify_hashes({name: item["sha256"] for name, item in seal["files"].items()})
    if benchmark.sha256(ADAPTER_SEAL) != ADAPTER_SHA or benchmark.sha256(ADAPTER_REVIEW) != ADAPTER_REVIEW_SHA:
        raise ValueError("The prerequisite adapter seal or independent review changed")
    adapter = json.loads(ADAPTER_SEAL.read_text())
    if adapter["status"] != "source_review_pass" or json.loads(ADAPTER_REVIEW.read_text())["status"] != "PASS":
        raise ValueError("The adapter source prerequisite must pass")
    benchmark.verify_hashes({name: item["sha256"] for name, item in adapter["files"].items()})


def registration() -> dict[str, Any]:
    value = json.loads((OUTPUT / "registration.json").read_text())
    if value.get("status") != "registered" or value.get("candidate_id") != "TP40-034":
        raise ValueError("Must register the reviewed stage-one protocol before execution")
    if value.get("contract") != CONTRACT or value.get("environment") != benchmark.environment_identity():
        raise ValueError("Registered stage-one contract or runtime differs")
    for key, paths in (("source_hashes", source_paths()), ("input_hashes", input_paths())):
        if set(value[key]) != {str(path.resolve()) for path in paths}:
            raise ValueError("Registration must pin the complete source and whitelisted input sets")
        benchmark.verify_hashes(value[key])
    verify_pure()
    models.verify_sources()
    return value


def source_identity() -> dict[str, Any]:
    value = registration()
    return {
        "registration_sha256": benchmark.sha256(OUTPUT / "registration.json"),
        "source_hashes": value["source_hashes"],
        "input_hashes": value["input_hashes"],
        "environment": value["environment"],
    }


def freeze() -> None:
    """Freeze source/data identity without parsing covariates or generating labels."""
    identity = source_identity()
    path = OUTPUT / "execution_identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Preserve the previous identity; register a separate revision")
    benchmark.write_json(path, identity)


def execution_identity() -> dict[str, Any]:
    identity = source_identity()
    if json.loads((OUTPUT / "execution_identity.json").read_text()) != identity:
        raise ValueError("Frozen stage-one identity changed")
    approval = json.loads((OUTPUT / "execution_approval.json").read_text())
    required = {
        "scope": SCOPE,
        "registration_sha256": identity["registration_sha256"],
        "execution_identity_sha256": benchmark.sha256(OUTPUT / "execution_identity.json"),
    }
    if any(approval.get(key) != value for key, value in required.items()):
        raise ValueError("Exact stage-one execution approval is required")
    return identity


def completed(folder: Path, spec: dict[str, Any], identity: dict[str, Any]) -> bool:
    path = folder / "complete.json"
    if not path.exists():
        return False
    receipt = json.loads(path.read_text())
    if receipt["spec"] != spec or receipt["identity"] != identity:
        raise ValueError("Completed cell belongs to another frozen identity")
    benchmark.verify_hashes(receipt["files"], folder)
    benchmark.verify_hashes(receipt["dependencies"])
    return True


def finish(
    folder: Path,
    spec: dict[str, Any],
    identity: dict[str, Any],
    files: tuple[str, ...],
    dependencies: tuple[Path, ...] = (),
) -> None:
    benchmark.write_json(
        folder / "complete.json",
        {
            "spec": spec,
            "identity": identity,
            "files": {name: benchmark.sha256(folder / name) for name in files},
            "dependencies": {str(path.resolve()): benchmark.sha256(path) for path in dependencies},
        },
    )


def inputs() -> tuple[geometry.Geometry, pd.DataFrame]:
    """Reuse the audited metadata-only loader; no outcome archive is opened."""
    value = geometry.read_geometry(DATA)
    folds = pd.read_csv(DATA / "folds.csv", usecols=list(geometry.FOLD_COLUMNS))
    return value, folds.loc[folds["repeat"] == 0].copy()


def split_indices(rows: pd.DataFrame, folds: pd.DataFrame, outer: int) -> dict[str, np.ndarray]:
    part = folds.loc[folds.outer_fold == outer]
    lookup = {name: i for i, name in enumerate(rows.row_id)}
    result = {
        role: np.array([lookup[name] for name in part.loc[part.role == role, "row_id"]], dtype=int)
        for role in ("train", "test", "tied_test")
    }
    result["inner"] = part.loc[part.role == "train", "inner_fold"].to_numpy(dtype=int)
    if len(np.unique(result["inner"])) != 3 or rows.iloc[result["train"]].tied.any():
        raise ValueError("Expected three original inner groups and asymmetric training only")
    return result


def feature_config(scales: np.ndarray, clock: str) -> models.FeatureConfig:
    masses = geometry.clock_weights(scales)[clock]
    return models.FeatureConfig(tuple(scales), clock, tuple(masses), **CONTRACT["quadrature"])


def candidates(world: dict[str, Any]) -> tuple[str, ...]:
    return ("aggregate_mariner", world["clock"]) if world["kind"] == "temporal" else ("aggregate_mariner", *CLOCKS)


def matrix(
    early: np.ndarray,
    late: np.ndarray,
    alpha: float,
    config: models.FeatureConfig,
    mu: float,
    cache: models.BenefitCache | None,
) -> np.ndarray:
    return models.design_and_offset(early, late, alpha, SHAPE, models.MODEL, {"mu": mu}, config, cache)[0]


def truth_values(
    early: np.ndarray,
    late: np.ndarray,
    alpha: float,
    scales: np.ndarray,
    world: dict[str, Any],
    cache: models.BenefitCache | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Evaluate only the fixed hand-defined law, with no fitted or measured input."""
    if world["kind"] == "heterogeneous":
        pools = 1 / scales
        order = np.lexsort((np.array(["nemotron_full", "starcoder"]), np.log(pools)))
        coordinate = np.empty(2)
        coordinate[order] = [-1, 1]
        rates = SHAPE["rate"] * np.exp(CONTRACT["heterogeneous_log_rate_span"] * coordinate)
        total = early + late
        design = np.hstack(
            (-models.mariner.benefit(total, rates, SHAPE["power"]), models.mariner.harm(total, SHAPE["threshold"]))
        )
        extra = {"per_bucket_rates": rates.tolist(), "rank_coordinate": coordinate.tolist()}
    else:
        clock = world["clock"] or CLOCKS[0]
        config = feature_config(scales, clock)
        design = matrix(early, late, alpha, config, world["mu"], cache)
        extra = {"feature_config": config.to_json()}
    linear = HEAD["intercept"] + design @ np.asarray(HEAD["coefficients"])
    values = HEAD["floor"] + np.exp(linear)
    if not np.isfinite(values).all():
        raise FloatingPointError("Fixed synthetic truth is nonfinite")
    return values, {
        "world": world,
        "shape": SHAPE,
        "head": HEAD,
        **extra,
        "native_prediction_clip_count": int(np.sum(abs(linear) >= models.mariner.LOG_CLIP)),
    }


def floor_feasibility(values: np.ndarray, floor: float) -> dict[str, Any]:
    median, minimum = float(np.median(values)), float(np.min(values))
    gap = median - minimum
    gamma = (median - floor) / gap if gap > 0 else None
    finite = bool(np.isfinite(values).all())
    return {
        "count": len(values),
        "median": median,
        "minimum": minimum,
        "true_floor": floor,
        "gamma_required": gamma,
        "floor_below_all": finite and minimum > floor,
        "deficit_clipping_absent": finite and minimum - floor > models.mariner.DEFICIT_FLOOR,
        "gamma_feasible": gamma is not None and np.isfinite(gamma) and 1 <= gamma <= 6,
    }


def preflight_records(
    clean: dict[str, np.ndarray], noise: dict[str, np.ndarray], rows: pd.DataFrame, folds: pd.DataFrame
) -> list[dict[str, Any]]:
    records = []
    for world, values in clean.items():
        for noise_name, errors in noise.items():
            observed = values + errors
            for outer in range(5):
                split = split_indices(rows, folds, outer)
                subsets = {"full": split["train"]} | {
                    f"inner_train_{label}": split["train"][split["inner"] != label] for label in range(3)
                }
                for subset, indices in subsets.items():
                    records.append(
                        {
                            "world": world,
                            "noise": noise_name,
                            "outer_fold": outer,
                            "subset": subset,
                            "row_indices": indices.tolist(),
                            **floor_feasibility(observed[indices], HEAD["floor"]),
                        }
                    )
    return records


def prepare() -> dict[str, Any]:
    identity = execution_identity()
    folder, spec = OUTPUT / "prepared", {"stage": "prepare", "contract": CONTRACT}
    if completed(folder, spec, identity):
        return json.loads((folder / "preflight.json").read_text())
    data, folds = inputs()
    cache = models.BenefitCache(OUTPUT / "features.sqlite")
    folder.mkdir(parents=True, exist_ok=True)
    total = data.early + data.late
    clean, matched, truths = {}, {}, {}
    for world in WORLDS:
        try:
            clean[world["name"]], truths[world["name"]] = truth_values(
                data.early, data.late, data.alpha, data.total_scales, world, cache
            )
            matched[world["name"]], matched_record = truth_values(
                data.alpha * total, (1 - data.alpha) * total, data.alpha, data.total_scales, world, cache
            )
            truths[world["name"]]["matched_tied_prediction_clip_count"] = matched_record["native_prediction_clip_count"]
        except ArithmeticError as error:
            benchmark.write_json(
                folder / "numerical_failure.json",
                {"identity": identity, "world": world, "exception": type(error).__name__, "message": str(error)},
            )
            raise
    noise = {"clean": np.zeros(len(data.rows))} | {
        str(seed): np.random.default_rng(seed).normal(0, CONTRACT["noise_sd"], len(data.rows)) for seed in SEEDS
    }
    records = preflight_records(clean, noise, data.rows, folds)
    if len(records) != CONTRACT["floor_feasibility_records"]:
        raise ValueError("Preflight coverage differs from the frozen contract")
    oracle_ready = all(
        np.min(values) - HEAD["floor"] > models.mariner.DEFICIT_FLOOR for values in (*clean.values(), *matched.values())
    ) and all(
        truth["native_prediction_clip_count"] == truth["matched_tied_prediction_clip_count"] == 0
        for truth in truths.values()
    )
    floor_ready = all(
        record["floor_below_all"] and record["deficit_clipping_absent"] and record["gamma_feasible"]
        for record in records
    )
    report = {
        "oracle_ready": bool(oracle_ready),
        "learned_floor_domain_ready": bool(floor_ready),
        "learned_stage_ready": False,
        "reason": "Learned eligibility also requires completed oracle gates and a separate registration",
        "floor_checks": records,
        "truths": truths,
        "rows": len(data.rows),
        "scope": "Clean known-floor oracle is allowed despite gamma-domain failure; no learned fit is authorized",
    }
    np.savez_compressed(folder / "clean.npz", **clean)
    np.savez_compressed(folder / "matched_tied.npz", **matched)
    np.savez_compressed(folder / "noise.npz", **noise)
    benchmark.write_json(folder / "preflight.json", report)
    finish(folder, spec, identity, ("clean.npz", "matched_tied.npz", "noise.npz", "preflight.json"))
    return report


def head_record(head: models.mariner.Head) -> dict[str, Any]:
    return {"floor": head.floor, "intercept": head.intercept, "coefficients": head.coefficients.tolist()}


def conditional_head(design: np.ndarray, values: np.ndarray, floor: float) -> models.mariner.Head:
    if not np.isfinite(values).all() or np.min(values) - floor <= models.mariner.DEFICIT_FLOOR:
        raise ValueError("Known-floor oracle needs finite unclipped positive deficits")
    intercept, coefficients = models.mariner.nonnegative_solve(design, np.log(values - floor), 0.0)
    return models.mariner.Head(intercept, coefficients, floor)


def predict(head: models.mariner.Head, design: np.ndarray) -> np.ndarray:
    linear = head.intercept + design @ head.coefficients
    if not np.isfinite(linear).all() or np.any(abs(linear) >= models.mariner.LOG_CLIP):
        raise FloatingPointError("Oracle prediction would invoke native clipping")
    return head.predict(design)


def oracle_fit(
    early: np.ndarray,
    late: np.ndarray,
    observed: np.ndarray,
    labels: np.ndarray,
    query_early: np.ndarray,
    query_late: np.ndarray,
    alpha: float,
    config: models.FeatureConfig,
    strengths: tuple[float, ...],
    floor: float,
    cache: models.BenefitCache | None = None,
) -> dict[str, Any]:
    """Select by inner BPB error only and preserve every conditional head."""
    profiles = []
    for strength in strengths:
        design = matrix(early, late, alpha, config, strength, cache)
        inner, predictions = [], np.empty(len(observed))
        for label in np.unique(labels):
            train, held = np.flatnonzero(labels != label), np.flatnonzero(labels == label)
            head = conditional_head(design[train], observed[train], floor)
            predictions[held] = predict(head, design[held])
            inner.append(
                {"label": int(label), "train": train.tolist(), "held": held.tolist(), "head": head_record(head)}
            )
        head = conditional_head(design, observed, floor)
        query = predict(head, matrix(query_early, query_late, alpha, config, strength, cache))
        profiles.append(
            {
                "mu": strength,
                "inner_rmse": float(np.sqrt(np.mean((predictions - observed) ** 2))),
                "inner_predictions": predictions.tolist(),
                "inner_heads": inner,
                "head": head_record(head),
                "query_predictions": query.tolist(),
            }
        )
    selected = int(np.argmin([profile["inner_rmse"] for profile in profiles]))
    return {
        "profiles": profiles,
        "selected_index": selected,
        "selected_mu": profiles[selected]["mu"],
        "conditional_nnls_solves": sum(len(profile["inner_heads"]) + 1 for profile in profiles),
    }


def query_frame(data: geometry.Geometry, split: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    held, tied = split["test"], split["tied_test"]
    total = data.early[held] + data.late[held]
    early = np.vstack((data.early[held], data.alpha * total, data.early[tied]))
    late = np.vstack((data.late[held], (1 - data.alpha) * total, data.late[tied]))
    frame = pd.DataFrame(
        {
            "role": ["asymmetric"] * len(held) + ["matched_tied"] * len(held) + ["actual_tied"] * len(tied),
            "source_index": np.r_[held, held, tied],
            "row_id": data.rows.iloc[np.r_[held, held, tied]].row_id.to_numpy(),
        }
    )
    return early, late, frame


def prediction_scores(predicted: np.ndarray, truth: np.ndarray, held_count: int) -> dict[str, Any]:
    error = predicted - truth
    gain = predicted[held_count : 2 * held_count] - predicted[:held_count]
    true_gain = truth[held_count : 2 * held_count] - truth[:held_count]
    return {
        "query_max_abs_error": float(np.max(abs(error))),
        "query_rmse": float(np.sqrt(np.mean(error**2))),
        "asymmetric_rmse": float(np.sqrt(np.mean(error[:held_count] ** 2))),
        "matched_tied_rmse": float(np.sqrt(np.mean(error[held_count : 2 * held_count] ** 2))),
        "actual_tied_rmse": float(np.sqrt(np.mean(error[2 * held_count :] ** 2))),
        "gain_rmse": float(np.sqrt(np.mean((gain - true_gain) ** 2))),
        "true_gain_rms": float(np.sqrt(np.mean(true_gain**2))),
        "maximum_absolute_predicted_gain": float(np.max(abs(gain))),
    }


def oracle() -> None:
    identity = execution_identity()
    preflight = prepare()
    if not preflight["oracle_ready"]:
        raise ValueError("Preserved preflight blocks clean known-floor oracle")
    data, folds = inputs()
    split = split_indices(data.rows, folds, 0)
    train, labels = split["train"], split["inner"]
    query_early, query_late, query = query_frame(data, split)
    cache = models.BenefitCache(OUTPUT / "features.sqlite")
    with np.load(OUTPUT / "prepared/clean.npz") as original, np.load(OUTPUT / "prepared/matched_tied.npz") as tied:
        for world in WORLDS:
            values = original[world["name"]]
            truth = np.r_[values[split["test"]], tied[world["name"]][split["test"]], values[split["tied_test"]]]
            for candidate in candidates(world):
                spec = {"world": world, "candidate": candidate, "outer_fold": 0}
                folder = OUTPUT / "cells" / world["name"] / candidate
                if completed(folder, spec, identity):
                    continue
                config = feature_config(data.total_scales, CLOCKS[0] if candidate == "aggregate_mariner" else candidate)
                strengths = (0.0,) if candidate == "aggregate_mariner" else STRENGTHS
                with threadpool_limits(limits=1):
                    fit = oracle_fit(
                        data.early[train],
                        data.late[train],
                        values[train],
                        labels,
                        query_early,
                        query_late,
                        data.alpha,
                        config,
                        strengths,
                        HEAD["floor"],
                        cache,
                    )
                selected = fit["profiles"][fit["selected_index"]]
                predicted = np.asarray(selected["query_predictions"])
                scores = prediction_scores(predicted, truth, len(split["test"]))
                correct = world["kind"] == "native" or (world["kind"] == "temporal" and candidate == world["clock"])
                report = {
                    **scores,
                    "exact_matching_gate_applicable": correct,
                    "exact_matching_gate_passed": (
                        not correct
                        or scores["query_max_abs_error"] <= CONTRACT["correct_family_query_max_abs_error_bpb"]
                    ),
                    "conditional_nnls_solves": fit["conditional_nnls_solves"],
                    "selected_mu": fit["selected_mu"],
                    "selected_upper_boundary": fit["selected_mu"] == STRENGTHS[-1],
                    "scope": "Known-shape/floor response recovery; misspecified controls remain descriptive",
                }
                folder.mkdir(parents=True, exist_ok=True)
                benchmark.write_json(folder / "fit.json", {**fit, "spec": spec, "config": config.to_json()})
                benchmark.write_json(folder / "scores.json", report)
                query.assign(truth=truth, predicted=predicted).to_csv(folder / "predictions.csv", index=False)
                finish(
                    folder,
                    spec,
                    identity,
                    ("fit.json", "scores.json", "predictions.csv"),
                    (OUTPUT / "prepared/complete.json",),
                )
    summarize()


def summarize() -> None:
    identity = execution_identity()
    if not completed(OUTPUT / "prepared", {"stage": "prepare", "contract": CONTRACT}, identity):
        raise ValueError("Prepared truth receipt is missing")
    checks, dependencies = [], []
    for world in WORLDS:
        for candidate in candidates(world):
            folder = OUTPUT / "cells" / world["name"] / candidate
            spec = {"world": world, "candidate": candidate, "outer_fold": 0}
            if not completed(folder, spec, identity):
                raise ValueError("Oracle coverage is incomplete")
            checks.append({**spec, **json.loads((folder / "scores.json").read_text())})
            dependencies.append(folder / "complete.json")
    preflight = json.loads((OUTPUT / "prepared/preflight.json").read_text())
    solves = sum(check["conditional_nnls_solves"] for check in checks)
    if len(checks) != CONTRACT["oracle_cells"] or solves != CONTRACT["conditional_nnls_solves"]:
        raise ValueError("Oracle cell or conditional-solver coverage differs")
    passed = preflight["oracle_ready"] and all(check["exact_matching_gate_passed"] for check in checks)
    result = {
        "cells": len(checks),
        "conditional_nnls_solves": solves,
        "checks": checks,
        "oracle_passed": bool(passed),
        "learned_stage_ready": bool(passed and preflight["learned_floor_domain_ready"]),
        "learned_execution_authorized": False,
        "independent_replay_required_before_learned_registration": True,
        "scope": "Separate reviewed registration required for learned J2, noise fitting, or raw optimization",
    }
    benchmark.write_json(OUTPUT / "oracle_summary.json", result)
    finish(OUTPUT, {"stage": "oracle_summary"}, identity, ("oracle_summary.json",), tuple(dependencies))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "prepare", "oracle", "summarize"))
    command = parser.parse_args().command
    {"freeze": freeze, "prepare": prepare, "oracle": oracle, "summarize": summarize}[command]()


if __name__ == "__main__":
    main()
