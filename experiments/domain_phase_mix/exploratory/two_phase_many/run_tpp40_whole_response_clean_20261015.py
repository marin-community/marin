# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""TP043: ten clean response-space shape/kinetic CV heads, no empirical fitting."""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import shutil
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    run_tpp40_acquisition_retention_stage2_20261015 as stage,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_response_profile_20261015 as prior
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    tpp40_response_profile_trust_region_20261015 as solver,
)

BASE, PARENT = prior.BASE, prior.PARENT
OUTPUT = PARENT / "whole_response_clean"
KNOWN = PARENT / "known_floor_diagnostic"
CACHE = KNOWN / "features.sqlite"
HELPER = KNOWN / "audit/author_replay_v1.py"
HELPER_SHA = "082e3f361e6b1ec168f4d3f7a00a15444419f01c4e19e1e31539a5fdb9ed0ac1"
SEALS = {
    prior.STAGE1_SEAL: "a8c38029614a5df658743f2a551f38b97f44fe7fcc1590a430b3ac91762b6718",
    KNOWN / "result_manifest_v1.json": "8d6c3fafd825b432e76dc270c1f2c3dbb63c068bd1af0d3ef942f8637ae626e0",
    prior.OUTPUT / "result_manifest_v1.json": "c97acfbf0434a4a2fd818983d966781718d6b6388b828c73a6492e046783a4ed",
    PARENT / "response_solver_control/result_manifest_v1.json": (
        "fbdbcd654819d5ad84d4bfa6cf4dcec7a4d7ecb093439f79bb1918bc188c382f"
    ),
}
SCOPE = "tp043_ten_clean_outer0_whole_response_cv_ridge0"
SHAPES = tuple(dict(item) for item in stage.models.mariner.SHAPES)
STRENGTHS = tuple(stage.oracle.STRENGTHS)
WORLDS: tuple[dict[str, Any], ...] = stage.oracle.WORLDS
MINIMUM_FREE_BYTES = 4 * 1024**3
CONTRACT: dict[str, Any] = {
    "candidate_id": "TP40-043",
    "scope": SCOPE,
    "worlds": list(WORLDS),
    "outer_fold": 0,
    "repeat": 0,
    "noise": "clean",
    "heads": 10,
    "shapes": list(SHAPES),
    "strengths": list(STRENGTHS),
    "ridge": 0.0,
    "inner_folds": 3,
    "gammas_for_starts_only": list(solver.prior.GAMMAS),
    "solver_options": solver.SOLVER_OPTIONS,
    "evaluation_cap_per_start": solver.EVALUATION_CAP,
    "objective": "mean_squared_bpb_divided_by_0.003_squared",
    "floor_domain": "nonnegative absolute floor; no upper bound; no generating-floor start",
    "native_scope": "same-head tied law only; not the full MARINER estimator or unrestricted native floor family",
    "selection": "pooled inner BPB SSE; ties by strength index then native shape index",
    "native_reuse": "all native inner shape profiles and selected full fit reused exactly at mu0",
    "maximum_conditional_fits": 20170,
    "maximum_nonlinear_starts": 100850,
    "minimum_free_bytes_before_fresh_profile": MINIMUM_FREE_BYTES,
    "resource_stop": "preserve every completed trace; stop before next profile, resume only with unchanged identity",
    "checkpoint": "one gzip JSON per conditional five-start fit, all finite and failed trials retained",
    "historical_performance_gates": stage.CONTRACT["gates"],
    "clean_function_recovery_reference_bpb": 1e-8,
    "training_access": "sealed training-only TP036 arrays; ignore its supplied-shape starts",
    "held_truth_access": "separate score command after all ten full fits are durable",
    "features": "immutable sealed complete-shape cache; missing rows fail, no integration",
    "numerical_status": "unchanged TP040 first-order, active-face, solve and underflow flags",
    "old_j2_refinement_flags": "not applicable; no gamma refinement is performed",
    "noisy_expansion": False,
    "empirical_outcomes": False,
    "raw_optimization": False,
    "original_stage2_remains_blocked": True,
    "scientific_promotion": False,
}


def load_helper() -> Any:
    if prior.digest(HELPER) != HELPER_SHA:
        raise ValueError("The frozen immutable-cache helper changed")
    spec = importlib.util.spec_from_file_location("_tp043_readonly_cache", HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the frozen cache reader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = load_helper()


def source_paths() -> set[Path]:
    return (
        prior.source_paths()
        | stage.source_paths()
        | {
            Path(__file__),
            Path(solver.__file__),
            BASE / "test_tpp40_response_profile_trust_region_20261015.py",
            HELPER,
            BASE / "test_tpp40_whole_response_clean_20261015.py",
            OUTPUT / "proposal_v1.md",
            PARENT / "reviews/ordering_initialization_and_complete_response_scope_v1.md",
            PARENT / "reviews/whole_response_next_step_fable_v1/review.md",
            PARENT / "reviews/whole_response_next_step_fable_v1/adjudication_v1.md",
            PARENT / "reviews/whole_response_toy_resource_estimate_v1.json",
        }
    )


def input_paths() -> set[Path]:
    return (
        prior.input_paths()
        | set(SEALS)
        | {
            CACHE,
            KNOWN / "registration.json",
            prior.OUTPUT / "execution_identity.json",
            *[
                prior.OUTPUT / "cells/outer0/clean" / world["name"] / "initialization" / name
                for world in WORLDS
                for name in ("starts.json", "complete.json")
            ],
        }
    )


def identity() -> dict[str, Any]:
    registered = prior.read(OUTPUT / "registration.json")
    if registered.get("status") != "registered" or registered.get("contract") != CONTRACT:
        raise ValueError("Require the exact separately registered clean response scope")
    if registered["environment"] != prior.runtime():
        raise ValueError("Runtime changed")
    for name, paths in (("source_hashes", source_paths()), ("input_hashes", input_paths())):
        if set(registered[name]) != {str(path.resolve()) for path in paths}:
            raise ValueError("The exact source/input whitelist changed")
        prior.verify_hashes(registered[name])
    for path, expected in SEALS.items():
        if prior.digest(path) != expected:
            raise ValueError("A sealed prerequisite changed")
        prior.verify_hashes(
            {str(path.parent / name): item["sha256"] for name, item in prior.read(path)["files"].items()}
        )
    if prior.read(prior.STAGE1_SEAL)["learned_stage_ready"]:
        raise ValueError("Preserve the failed original learned-stage prerequisite")
    if any(Path(str(CACHE) + suffix).exists() for suffix in ("-wal", "-shm")):
        raise ValueError("Immutable feature cache must be closed")
    return {key: registered[key] for key in ("source_hashes", "input_hashes", "environment")} | {
        "registration_sha256": prior.digest(OUTPUT / "registration.json")
    }


def execution_identity() -> dict[str, Any]:
    frozen = identity()
    if prior.read(OUTPUT / "execution_identity.json") != frozen:
        raise ValueError("Frozen execution identity changed")
    expected = {
        "scope": SCOPE,
        "registration_sha256": frozen["registration_sha256"],
        "execution_identity_sha256": prior.digest(OUTPUT / "execution_identity.json"),
    }
    if any(prior.read(OUTPUT / "execution_approval.json").get(name) != value for name, value in expected.items()):
        raise ValueError("Exact parent approval is required")
    return frozen


def atomic_bytes(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Preserve differing checkpoint {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    if temporary.exists():
        raise ValueError(f"Preserve interrupted checkpoint {temporary}")
    with temporary.open("xb") as handle:
        handle.write(content)
    temporary.replace(path)


def save_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def require_disk_space(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path.parent).free
    if free < MINIMUM_FREE_BYTES:
        save_json(
            path.parent / f"resource_stop_{time.time_ns()}.json",
            {
                "free_bytes": free,
                "required_free_bytes": MINIMUM_FREE_BYTES,
                "next_profile": str(path),
                "scientific_failure": False,
                "action": "Stop before this profile; preserve all traces and resume unchanged after resources recover",
            },
        )
        raise RuntimeError("Insufficient disk for a fresh profile; scientific work preserved")


def training_only(world: dict[str, Any], train: np.ndarray) -> np.ndarray:
    folder = prior.OUTPUT / "cells/outer0/clean" / world["name"] / "initialization"
    receipt = prior.read(folder / "complete.json")
    if receipt["identity"] != prior.read(prior.OUTPUT / "execution_identity.json"):
        raise ValueError("Training-only receipt belongs to another procedure")
    if receipt["spec"] != {"world": world, "outer_fold": 0, "noise": "clean", "stage": "five_gamma_starts"}:
        raise ValueError("Training-only receipt has the wrong world, fold, or noise condition")
    prior.verify_hashes({str(folder / name): sha for name, sha in receipt["files"].items()})
    saved = prior.read(folder / "starts.json")
    if saved["train_indices"] != train.tolist():
        raise ValueError("Training row order changed")
    values = np.asarray(saved["observed"], float)
    if values.shape != train.shape or not np.isfinite(values).all():
        raise ValueError("Invalid saved training-only observations")
    return values


def conditional_fit(
    path: Path, matrix: np.ndarray, observed: np.ndarray, spec: dict[str, Any], identity_sha: str
) -> dict[str, Any]:
    """Resume one complete five-start solve without altering solver semantics."""
    header = {
        "spec": spec,
        "identity_sha256": identity_sha,
        "input_sha256": prior.canonical_hash({"matrix": matrix.tolist(), "observed": observed.tolist()}),
    }
    if path.exists():
        record = json.loads(gzip.decompress(path.read_bytes()))
        if record["header"] != header or record["fit_sha256"] != prior.canonical_hash(record["fit"]):
            raise ValueError("Conditional checkpoint identity or contents differ")
        return record["fit"]
    if path.with_name(path.name + ".writing").exists():
        raise ValueError(f"Preserve interrupted checkpoint before fitting {path}")
    require_disk_space(path)
    try:
        starts = solver.prior.gamma_starts(matrix, observed)
        fit = solver.fit_response(matrix, observed, None, starts, evaluation_cap=solver.EVALUATION_CAP)
    except (ArithmeticError, ValueError, np.linalg.LinAlgError) as error:
        fit = {
            "status": "initialization_or_solver_failure",
            "selected": None,
            "unresolved": True,
            "solves": [],
            "trials": [],
            "exception": type(error).__name__,
            "message": str(error),
        }
    record = {"header": header, "fit_sha256": prior.canonical_hash(fit), "fit": fit}
    atomic_bytes(path, gzip.compress(json.dumps(record, sort_keys=True, allow_nan=False).encode(), mtime=0))
    return fit


def feature_spec(shape: dict[str, float], strength: float, clock: str, alpha: float) -> dict[str, Any]:
    return {"shape": shape, "alpha": alpha, "parameters": {"mu": strength}, "feature_config": {"clock": clock}}


def profile_candidate(
    matrix: np.ndarray, observed: np.ndarray, labels: np.ndarray, folder: Path, spec: dict[str, Any], identity_sha: str
) -> dict[str, Any]:
    predictions = np.empty(len(observed))
    records, failures = [], []
    for label in np.unique(labels):
        train, held = np.flatnonzero(labels != label), np.flatnonzero(labels == label)
        path = folder / f"inner{label}.json.gz"
        fit = conditional_fit(path, matrix[train], observed[train], spec | {"inner_fold": int(label)}, identity_sha)
        diagnostics = prior.fit_diagnostics(fit)
        record = {
            "label": int(label),
            "path": str(path),
            "sha256": prior.digest(path),
            "train_indices": train.tolist(),
            "held_indices": held.tolist(),
            "unresolved": fit["unresolved"],
            "diagnostics": diagnostics,
        }
        try:
            if fit["selected"] is None:
                raise ArithmeticError("No finite conditional candidate")
            predictions[held] = solver.prior.predict(matrix[held], fit["selected"])
        except ArithmeticError as error:
            record["prediction_failure"] = str(error)
            failures.append(int(label))
        records.append(record)
    return {
        "spec": spec,
        "eligible_for_selection": not failures,
        "inner_mse": None if failures else float(np.mean((predictions - observed) ** 2)),
        "inner_predictions": None if failures else predictions.tolist(),
        "conditional_records": records,
        "prediction_failure_folds": failures,
        "any_unresolved_inner": any(record["unresolved"] for record in records),
    }


def profile_diagnostics(profiles: list[dict[str, Any]]) -> dict[str, int]:
    records = [record for item in profiles for record in item["conditional_records"]]
    return {
        "candidate_profiles": len(profiles),
        "invalid_candidates": sum(not item["eligible_for_selection"] for item in profiles),
        "unresolved_candidates": sum(item["any_unresolved_inner"] for item in profiles),
        "inner_profiles": len(records),
        "unresolved_inner_profiles": sum(record["unresolved"] for record in records),
        "failed_inner_starts": sum(record["diagnostics"]["failed_start_count"] for record in records),
        "inner_evaluation_caps": sum(record["diagnostics"]["evaluation_cap_count"] for record in records),
        "inner_numerical_trial_failures": sum(
            record["diagnostics"]["numerical_failure_evaluation_count"] for record in records
        ),
    }


def choose(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [item for item in profiles if item["eligible_for_selection"]]
    if not eligible:
        raise ArithmeticError("No finite complete inner-CV candidate; all failed profiles remain saved")
    return min(
        eligible, key=lambda item: (item["inner_mse"], item["spec"]["strength_index"], item["spec"]["shape_index"])
    )


def reuse_native_full(native_fit: dict[str, Any], selected: dict[str, Any], candidate: str) -> dict[str, Any]:
    """Retain the complete matched native head when a temporal menu selects zero."""
    if selected["spec"] != native_fit["selected_candidate"]:
        raise ValueError("Temporal null must select the exact complete native candidate")
    return dict(native_fit) | {"candidate": candidate, "native_full_fit_reused": True}


def completed_fit(folder: Path, identity_sha: str) -> bool:
    path = folder / "fit_complete.json"
    if not path.exists():
        return False
    receipt = prior.read(path)
    if receipt["identity_sha256"] != identity_sha:
        raise ValueError("Completed fit belongs to another identity")
    prior.verify_hashes({str(folder / name): value for name, value in receipt["files"].items()})
    for name, expected in receipt["conditional_files"].items():
        if prior.digest(Path(name)) != expected:
            raise ValueError("A conditional profile changed")
    return True


def fit_world(name: str) -> None:
    frozen = execution_identity()
    world = next(item for item in WORLDS if item["name"] == name)
    data, folds = stage.oracle.inputs()
    split = stage.oracle.split_indices(data.rows, folds, 0)
    train, labels = split["train"], split["inner"]
    observed = training_only(world, train)
    identity_sha = prior.digest(OUTPUT / "execution_identity.json")
    context = {"scales": data.total_scales, "alpha": data.alpha}
    with closing(sqlite3.connect(f"file:{CACHE}?mode=ro&immutable=1", uri=True)) as connection:
        cache = helper.Cache(connection, helper.Checks(), frozen, context)
        native_profiles, native_fit = None, None
        for candidate in stage.oracle.candidates(world):
            if execution_identity() != frozen:
                raise ValueError("Sources changed before a fresh family search")
            folder = OUTPUT / "cells" / name / candidate
            if completed_fit(folder, identity_sha):
                profiles, fitted = prior.read(folder / "profiles.json")["profiles"], prior.read(folder / "fit.json")
            else:
                clock = stage.oracle.CLOCKS[0] if candidate == "aggregate_mariner" else candidate
                profiles = []
                for strength_index, strength in enumerate((0.0,) if candidate == "aggregate_mariner" else STRENGTHS):
                    if strength == 0 and candidate != "aggregate_mariner":
                        if native_profiles is None:
                            raise ValueError("Native profiles must be fitted before temporal families")
                        profiles.extend(native_profiles)
                        continue
                    for shape_index, shape in enumerate(SHAPES):
                        spec = {
                            "world": name,
                            "candidate": candidate,
                            "shape_index": shape_index,
                            "strength_index": strength_index,
                            "shape": shape,
                            "mu": strength,
                            "clock": clock,
                        }
                        matrix = cache.matrix(
                            data.early[train], data.late[train], feature_spec(shape, strength, clock, data.alpha)
                        )
                        path = folder / "conditional" / f"mu{strength_index}_shape{shape_index:03d}"
                        with threadpool_limits(limits=1):
                            profile = profile_candidate(matrix, observed, labels, path, spec, identity_sha)
                        profiles.append(profile)
                save_json(folder / "profiles.json", {"identity_sha256": identity_sha, "profiles": profiles})
                selected = choose(profiles)
                spec = selected["spec"]
                if spec["mu"] == 0 and candidate != "aggregate_mariner":
                    if native_fit is None:
                        raise ValueError("A complete native head is required")
                    fitted = reuse_native_full(native_fit, selected, candidate)
                else:
                    matrix = cache.matrix(
                        data.early[train], data.late[train], feature_spec(spec["shape"], spec["mu"], clock, data.alpha)
                    )
                    with threadpool_limits(limits=1):
                        full = conditional_fit(
                            folder / "full.json.gz", matrix, observed, spec | {"inner_fold": None}, identity_sha
                        )
                    fitted = {
                        "identity_sha256": identity_sha,
                        "world": name,
                        "candidate": candidate,
                        "selected_candidate": spec,
                        "inner_mse": selected["inner_mse"],
                        "inner_predictions": selected["inner_predictions"],
                        "selected_inner_records": selected["conditional_records"],
                        "selected": full["selected"],
                        "full_diagnostics": prior.fit_diagnostics(full),
                        "full_unresolved": full["unresolved"],
                        "native_full_fit_reused": False,
                        "full_checkpoint": str(folder / "full.json.gz"),
                        "train_indices": train.tolist(),
                        "training_sha256": prior.canonical_hash(observed.tolist()),
                    }
                save_json(folder / "fit.json", fitted)
                conditional_files = {
                    record["path"]: record["sha256"] for item in profiles for record in item["conditional_records"]
                }
                conditional_files[fitted["full_checkpoint"]] = prior.digest(Path(fitted["full_checkpoint"]))
                save_json(
                    folder / "fit_complete.json",
                    {
                        "identity_sha256": identity_sha,
                        "files": {file: prior.digest(folder / file) for file in ("profiles.json", "fit.json")},
                        "conditional_files": conditional_files,
                    },
                )
            if candidate == "aggregate_mariner":
                native_profiles, native_fit = profiles, fitted
            print(
                json.dumps(
                    {
                        "world": name,
                        "candidate": candidate,
                        "selected": fitted["selected_candidate"],
                        "inner_mse": fitted["inner_mse"],
                        "full_unresolved": fitted["full_unresolved"],
                    }
                ),
                flush=True,
            )


def performance_checks(
    world: dict[str, Any], candidate: str, score: dict[str, Any], native: dict[str, Any] | None
) -> list[dict[str, Any]] | None:
    """Reuse the historical performance rule, excluding inapplicable J2 solve fields."""
    if native is None:
        return None
    supplied = score | {name: 0 for name in stage.CONTRACT["unresolved_numerical_fields"]}
    return [
        item
        for item in stage.gate_checks(world, candidate, supplied, native)
        if not item["name"].startswith("unresolved_")
    ]


def score_all() -> None:
    frozen = execution_identity()
    identity_sha = prior.digest(OUTPUT / "execution_identity.json")
    expected = [(world, candidate) for world in WORLDS for candidate in stage.oracle.candidates(world)]
    for world, candidate in expected:
        if not completed_fit(OUTPUT / "cells" / world["name"] / candidate, identity_sha):
            raise ValueError("Every full head must be durable before any held truth is opened")
    data, folds = stage.oracle.inputs()
    split = stage.oracle.split_indices(data.rows, folds, 0)
    query0, query1, query = stage.oracle.query_frame(data, split)
    rows, failures, output_receipts = [], [], []
    with closing(sqlite3.connect(f"file:{CACHE}?mode=ro&immutable=1", uri=True)) as connection:
        cache = helper.Cache(connection, helper.Checks(), frozen, {"scales": data.total_scales, "alpha": data.alpha})
        with (
            np.load(prior.STAGE1 / "prepared/clean.npz") as original,
            np.load(prior.STAGE1 / "prepared/matched_tied.npz") as tied,
        ):
            for world in WORLDS:
                truth = np.r_[
                    original[world["name"]][split["test"]],
                    tied[world["name"]][split["test"]],
                    original[world["name"]][split["tied_test"]],
                ]
                native = None
                for candidate in stage.oracle.candidates(world):
                    folder = OUTPUT / "cells" / world["name"] / candidate
                    fitted = prior.read(folder / "fit.json")
                    spec = fitted["selected_candidate"]
                    matrix = cache.matrix(
                        query0, query1, feature_spec(spec["shape"], spec["mu"], spec["clock"], data.alpha)
                    )
                    try:
                        if fitted["selected"] is None:
                            raise ArithmeticError("No finite full-training head")
                        predicted = solver.prior.predict(matrix, fitted["selected"])
                    except ArithmeticError as error:
                        failure = {"world": world["name"], "candidate": candidate, "error": str(error)}
                        save_json(folder / "query_failure.json", failure)
                        failures.append(failure)
                        output_receipts.append(folder / "query_failure.json")
                        continue
                    count = len(split["test"])
                    gain = predicted[count : 2 * count] - predicted[:count]
                    score = (
                        prior.score(predicted, truth, query)
                        | stage.menu_scores(predicted, truth, query)
                        | {
                            "maximum_positive_predicted_gain": float(max(0, np.max(gain))),
                            "selected_upper_boundary": spec["mu"] == STRENGTHS[-1],
                            "selected_mu": spec["mu"],
                            "shape": spec["shape"],
                            "floor": fitted["selected"]["floor"],
                            "ridge": 0.0,
                            "full_unresolved": fitted["full_unresolved"],
                            "full_diagnostics": fitted["full_diagnostics"],
                            "selected_inner_unresolved": any(
                                item["unresolved"] for item in fitted["selected_inner_records"]
                            ),
                            "inner_rmse": float(np.sqrt(fitted["inner_mse"])),
                            "native_full_fit_reused": fitted["native_full_fit_reused"],
                            "all_candidate_numerical_counts": profile_diagnostics(
                                prior.read(folder / "profiles.json")["profiles"]
                            ),
                            "clean_function_recovery_reference_met": bool(np.max(abs(predicted - truth)) <= 1e-8),
                        }
                    )
                    point = fitted["selected"]
                    query_logit = (
                        None
                        if point["scale"] == 0
                        else np.log(point["scale"]) + (matrix @ np.asarray(point["amplitudes"]) - point["normalization"])
                    )
                    score["query_exponential_underflow_count"] = (
                        0 if query_logit is None else int(np.sum(np.exp(query_logit) == 0))
                    )
                    score["zero_scale_closure"] = point["scale"] == 0
                    score["unresolved"] = bool(
                        fitted["full_unresolved"]
                        or score["selected_inner_unresolved"]
                        or score["query_exponential_underflow_count"]
                        or score["zero_scale_closure"]
                    )
                    if candidate == "aggregate_mariner":
                        native = score
                    checks = performance_checks(world, candidate, score, native)
                    score["historical_performance_checks"] = checks
                    score["historical_performance_status"] = (
                        "incomplete_native_query"
                        if checks is None
                        else "pass"
                        if all(item["passed"] for item in checks)
                        else "fail"
                    )
                    score["original_stage2_remains_blocked"] = True
                    save_json(folder / "scores.json", score)
                    predictions = query.assign(truth=truth, predicted=predicted)
                    atomic_bytes(folder / "predictions.csv", predictions.to_csv(index=False).encode())
                    save_json(
                        folder / "score_complete.json",
                        {
                            "identity_sha256": identity_sha,
                            "files": {
                                name: prior.digest(folder / name)
                                for name in ("fit_complete.json", "scores.json", "predictions.csv")
                            },
                        },
                    )
                    output_receipts.append(folder / "score_complete.json")
                    rows.append({"world": world["name"], "candidate": candidate, **score})
    save_json(
        OUTPUT / "summary.json",
        {
            "rows": rows,
            "query_failures": failures,
            "heads": len(expected),
            "finite_query_heads": len(rows),
            "unresolved_heads": sum(item["unresolved"] for item in rows),
            "historical_performance_failures": [
                {"world": item["world"], "candidate": item["candidate"], **check}
                for item in rows
                for check in item["historical_performance_checks"] or []
                if not check["passed"]
            ],
            "historical_performance_complete": (
                len(rows) == len(expected) and all(item["historical_performance_checks"] is not None for item in rows)
            ),
            "historical_performance_passed": (
                len(rows) == len(expected) and all(item["historical_performance_status"] == "pass" for item in rows)
            ),
            "original_stage2_remains_blocked": True,
            "scientific_promotion": False,
            "noisy_expansion_authorized": False,
        },
    )
    save_json(
        OUTPUT / "complete.json",
        {
            "identity_sha256": identity_sha,
            "files": {
                str(path.relative_to(OUTPUT)): prior.digest(path) for path in [OUTPUT / "summary.json", *output_receipts]
            },
            "completed_heads": len(rows) + len(failures),
            "original_stage2_remains_blocked": True,
            "scientific_promotion": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "fit", "score"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "freeze":
        save_json(OUTPUT / "execution_identity.json", identity())
    elif args.command == "fit":
        execution_identity()
        if args.workers < 1:
            raise ValueError("A positive local worker count is required")
        names = [world["name"] for world in WORLDS]
        if args.workers == 1:
            for name in names:
                fit_world(name)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(fit_world, names))
    else:
        score_all()


if __name__ == "__main__":
    main()
