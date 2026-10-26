# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""TP046: the unchanged response estimator on one preidentified noisy stress fold."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_response_profile_20261015 as prior

BASE, PARENT = prior.BASE, prior.PARENT
TEMPLATE = BASE / "run_tpp40_whole_response_clean_v2_20261015.py"
TEMPLATE_SHA = "bd8f4bf4449972915d96a45767decb319a54838d5be8fedb5c97886947626685"


def private_template() -> Any:
    if prior.digest(TEMPLATE) != TEMPLATE_SHA:
        raise ValueError("The reviewed clean response procedure changed")
    spec = importlib.util.spec_from_file_location("_tp046_response_template", TEMPLATE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the frozen response helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # No noisy head can adopt a clean conditional fit. Use the identical fresh
    # conditional solver, retaining the v2 overflow-safe profile scorer.
    module.conditional_fit = module._original_conditional_fit
    return module


template = private_template()
backend, stage, solver = template.backend, template.backend.stage, template.backend.solver
OUTPUT = PARENT / "whole_response_fold3_noise"
CLEAN = PARENT / "whole_response_clean_v2"
ALL_FOLDS = PARENT / "known_floor_all_folds"
CACHE = ALL_FOLDS / "features.sqlite"
ALL_FOLDS_SEAL_SHA = "3cf46cfc976d35a11993fbffb084d1e4ee7a5d925994a73db76536dfad0300bc"
OUTER = 3
NOISES = ("20261015", "20261016", "20261017")
SCOPE = "tp046_outer3_three_paired_noises_whole_response_stress"
CONTRACT = {
    k: v
    for k, v in template.CONTRACT.items()
    if k
    not in {
        "continuation_of",
        "interrupted_v1_sha256",
        "adoption",
        "noise",
        "scope",
        "candidate_id",
        "outer_fold",
        "heads",
        "maximum_conditional_fits",
        "maximum_nonlinear_starts",
        "held_truth_access",
    }
} | {
    "candidate_id": "TP40-046",
    "scope": SCOPE,
    "outer_fold": OUTER,
    "noise_seeds": list(NOISES),
    "noise_sd": 0.003,
    "noise_application": "paired saved full-vector noise, training entries only",
    "heads": 30,
    "contexts": 12,
    "maximum_conditional_fits": 60510,
    "maximum_nonlinear_starts": 302550,
    "targeted_stress_only": True,
    "five_fold_or_prospective_claim": False,
    "training_access": "sealed TP036 outer3/noise training-only arrays, exact row order and receipts",
    "held_truth_access": "separate score command after all thirty full heads are durable",
    "cache": "sealed all-fold cache, immutable read-only; missing features fail; no copying or integration",
    "score_overflow": (
        "preserve ordinary finite values; scale-safe affected aggregates; "
        "retain predictions and explicit failed cell if still nonfinite"
    ),
    "resource_stop": (
        "four-GiB floor before next conditional fit; preserve fixed-schedule partial scope; "
        "no outcome-selected omissions"
    ),
    "scientific_promotion": False,
    "noisy_expansion": "exactly this predeclared thirty-head stress test",
}


def contexts() -> list[tuple[str, str]]:
    return [(noise, world["name"]) for noise in NOISES for world in backend.WORLDS]


def cell(noise: str, world: str, candidate: str) -> Path:
    return OUTPUT / "cells" / f"outer{OUTER}" / noise / world / candidate


def source_paths() -> set[Path]:
    return template.source_paths() | {
        Path(__file__),
        BASE / "test_tpp40_whole_response_fold3_noise_20261015.py",
        OUTPUT / "proposal_v1.md",
    }


def input_paths() -> set[Path]:
    return template.input_paths() | {
        CACHE,
        ALL_FOLDS / "result_manifest_v1.json",
        ALL_FOLDS / "registration.json",
        CLEAN / "registration.json",
        CLEAN / "execution_identity.json",
        CLEAN / "complete.json",
        CLEAN / "audit/author_replay_report_v1.json",
        CLEAN / "audit/independent_author_replay_source_review_v1.json",
        *[
            prior.OUTPUT / "cells" / f"outer{OUTER}" / noise / world / "initialization" / name
            for noise, world in contexts()
            for name in ("starts.json", "complete.json")
        ],
    }


def identity() -> dict[str, Any]:
    registered = prior.read(OUTPUT / "registration.json")
    if registered.get("status") != "registered" or registered.get("contract") != CONTRACT:
        raise ValueError("Register the exact thirty-head stress scope")
    if registered["environment"] != prior.runtime():
        raise ValueError("Runtime changed")
    for key, paths in (("source_hashes", source_paths()), ("input_hashes", input_paths())):
        if set(registered[key]) != {str(p.resolve()) for p in paths}:
            raise ValueError("Exact source/input whitelist changed")
        prior.verify_hashes(registered[key])
    if prior.digest(ALL_FOLDS / "result_manifest_v1.json") != ALL_FOLDS_SEAL_SHA:
        raise ValueError("The audited all-fold cache source changed")
    seal = prior.read(ALL_FOLDS / "result_manifest_v1.json")
    prior.verify_hashes({str(ALL_FOLDS / name): item["sha256"] for name, item in seal["files"].items()})
    if any(Path(str(CACHE) + suffix).exists() for suffix in ("-wal", "-shm")):
        raise ValueError("The sealed feature cache must remain closed")
    clean_report = prior.read(CLEAN / "audit/author_replay_report_v1.json")
    if clean_report["status"] != "author_saved_output_replay_passed" or clean_report["scientific_promotion"]:
        raise ValueError("Require the clean numerical replay without claiming promotion")
    if prior.read(prior.STAGE1_SEAL)["learned_stage_ready"]:
        raise ValueError("Preserve the failed original learned-stage prerequisite")
    return {key: registered[key] for key in ("source_hashes", "input_hashes", "environment")} | {
        "registration_sha256": prior.digest(OUTPUT / "registration.json")
    }


def execution_identity() -> dict[str, Any]:
    frozen = identity()
    if prior.read(OUTPUT / "execution_identity.json") != frozen:
        raise ValueError("Frozen stress identity changed")
    expected = {
        "scope": SCOPE,
        "registration_sha256": frozen["registration_sha256"],
        "execution_identity_sha256": prior.digest(OUTPUT / "execution_identity.json"),
    }
    if any(prior.read(OUTPUT / "execution_approval.json").get(k) != v for k, v in expected.items()):
        raise ValueError("Exact parent stress-test approval required")
    return frozen


def training_only(world: dict[str, Any], noise: str, train: np.ndarray) -> np.ndarray:
    folder = prior.OUTPUT / "cells" / f"outer{OUTER}" / noise / world["name"] / "initialization"
    receipt = prior.read(folder / "complete.json")
    if receipt["identity"] != prior.read(prior.OUTPUT / "execution_identity.json"):
        raise ValueError("Training-only receipt identity differs")
    if receipt["spec"] != {"world": world, "outer_fold": OUTER, "noise": noise, "stage": "five_gamma_starts"}:
        raise ValueError("Training-only world/fold/noise differs")
    prior.verify_hashes({str(folder / name): sha for name, sha in receipt["files"].items()})
    saved = prior.read(folder / "starts.json")
    values = np.asarray(saved["observed"], float)
    if saved["train_indices"] != train.tolist() or values.shape != train.shape or not np.isfinite(values).all():
        raise ValueError("Training row order or observations differ")
    return values


def fit_context(context: tuple[str, str]) -> None:
    noise, name = context
    if context not in contexts():
        raise ValueError("Unregistered noise/world context")
    frozen = execution_identity()
    identity_sha = prior.digest(OUTPUT / "execution_identity.json")
    world = next(w for w in backend.WORLDS if w["name"] == name)
    data, folds = stage.oracle.inputs()
    split = stage.oracle.split_indices(data.rows, folds, OUTER)
    train, labels = split["train"], split["inner"]
    observed = training_only(world, noise, train)
    with closing(sqlite3.connect(f"file:{CACHE}?mode=ro&immutable=1", uri=True)) as connection:
        cache = backend.helper.Cache(
            connection, backend.helper.Checks(), frozen, {"scales": data.total_scales, "alpha": data.alpha}
        )
        native_profiles, native_fit = None, None
        for candidate in stage.oracle.candidates(world):
            if execution_identity() != frozen:
                raise ValueError("Sources changed before a family search")
            folder = cell(noise, name, candidate)
            if backend.completed_fit(folder, identity_sha):
                profiles, fitted = prior.read(folder / "profiles.json")["profiles"], prior.read(folder / "fit.json")
            else:
                clock = stage.oracle.CLOCKS[0] if candidate == "aggregate_mariner" else candidate
                profiles = []
                strengths = (0.0,) if candidate == "aggregate_mariner" else backend.STRENGTHS
                for strength_index, strength in enumerate(strengths):
                    if strength == 0 and candidate != "aggregate_mariner":
                        if native_profiles is None:
                            raise ValueError("Native profiles required first")
                        profiles.extend(native_profiles)
                        continue
                    for shape_index, shape in enumerate(backend.SHAPES):
                        spec = {
                            "world": name,
                            "noise": noise,
                            "outer_fold": OUTER,
                            "candidate": candidate,
                            "shape_index": shape_index,
                            "strength_index": strength_index,
                            "shape": shape,
                            "mu": strength,
                            "clock": clock,
                        }
                        matrix = cache.matrix(
                            data.early[train], data.late[train], backend.feature_spec(shape, strength, clock, data.alpha)
                        )
                        path = folder / "conditional" / f"mu{strength_index}_shape{shape_index:03d}"
                        with threadpool_limits(limits=1):
                            profiles.append(
                                template.profile_candidate(matrix, observed, labels, path, spec, identity_sha)
                            )
                backend.save_json(folder / "profiles.json", {"identity_sha256": identity_sha, "profiles": profiles})
                selected = backend.choose(profiles)
                spec = selected["spec"]
                if spec["mu"] == 0 and candidate != "aggregate_mariner":
                    if native_fit is None:
                        raise ValueError("Complete native head required")
                    fitted = backend.reuse_native_full(native_fit, selected, candidate)
                else:
                    matrix = cache.matrix(
                        data.early[train],
                        data.late[train],
                        backend.feature_spec(spec["shape"], spec["mu"], clock, data.alpha),
                    )
                    with threadpool_limits(limits=1):
                        full = template.conditional_fit(
                            folder / "full.json.gz", matrix, observed, spec | {"inner_fold": None}, identity_sha
                        )
                    fitted = {
                        "identity_sha256": identity_sha,
                        "world": name,
                        "noise": noise,
                        "outer_fold": OUTER,
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
                backend.save_json(folder / "fit.json", fitted)
                conditional = {r["path"]: r["sha256"] for p in profiles for r in p["conditional_records"]}
                conditional[fitted["full_checkpoint"]] = prior.digest(Path(fitted["full_checkpoint"]))
                backend.save_json(
                    folder / "fit_complete.json",
                    {
                        "identity_sha256": identity_sha,
                        "files": {f: prior.digest(folder / f) for f in ("profiles.json", "fit.json")},
                        "conditional_files": conditional,
                    },
                )
            if candidate == "aggregate_mariner":
                native_profiles, native_fit = profiles, fitted
            print(
                json.dumps(
                    {
                        "noise": noise,
                        "world": name,
                        "candidate": candidate,
                        "selected": fitted["selected_candidate"],
                        "inner_mse": fitted["inner_mse"],
                        "full_unresolved": fitted["full_unresolved"],
                    }
                ),
                flush=True,
            )


def stable_rms(values: np.ndarray) -> float:
    if not len(values) or not np.isfinite(values).all():
        raise ArithmeticError("Undefined or nonfinite residual vector")
    scale = float(np.max(abs(values), initial=0))
    return 0.0 if scale == 0 else scale * math.sqrt(float(np.mean((values / scale) ** 2)))


def stable_mean(values: np.ndarray) -> float:
    if not len(values) or not np.isfinite(values).all():
        raise ArithmeticError("Undefined or nonfinite residual vector")
    return math.fsum(float(x / len(values)) for x in values)


def finite_metrics(predicted: np.ndarray, truth: np.ndarray, query: pd.DataFrame) -> tuple[dict[str, Any], list[str]]:
    """Keep ordinary metrics exactly; repair only nonfinite aggregate arithmetic."""
    with np.errstate(over="ignore", invalid="ignore"):
        metrics = prior.score(predicted, truth, query) | stage.menu_scores(predicted, truth, query)
    affected = [k for k, v in metrics.items() if isinstance(v, (float, np.floating)) and not np.isfinite(v)]
    if not affected:
        return metrics, []
    count = int(np.sum(query.role == "asymmetric"))
    with np.errstate(over="ignore", invalid="ignore"):
        error = predicted - truth
        gain_error = predicted[count : 2 * count] - predicted[:count] - (truth[count : 2 * count] - truth[:count])
    frontier = metrics["frontier_indices"]
    replacements = {
        "query_rmse": (stable_rms, error),
        "gain_rmse": (stable_rms, gain_error),
        "calibration_bias": (stable_mean, error),
        "mean_optimism": (stable_mean, -error),
        "frontier_mean_optimism": (stable_mean, -error[frontier]),
        "frontier_rmse": (stable_rms, error[frontier]),
    }
    for role in ("asymmetric", "matched_tied", "actual_tied"):
        part = error[query.role.to_numpy() == role]
        replacements.update({f"{role}_rmse": (stable_rms, part), f"{role}_bias": (stable_mean, part)})
    for name in affected:
        metrics[name] = None
        if name in replacements:
            function, values = replacements[name]
            try:
                value = function(values)
                metrics[name] = value if np.isfinite(value) else None
            except ArithmeticError:
                # Retain the finite prediction table and an explicit failed
                # metric; never let one unrepresentable summary erase peers.
                pass
    return metrics, affected


def save_query_summary(
    folder: Path, predicted: np.ndarray, truth: np.ndarray, query: pd.DataFrame, context: dict[str, Any]
) -> tuple[dict[str, Any], list[str], dict[str, Any] | None]:
    """Persist finite predictions before attempting potentially extreme metrics."""
    backend.atomic_bytes(
        folder / "predictions.csv", query.assign(truth=truth, predicted=predicted).to_csv(index=False).encode()
    )
    metrics, affected = finite_metrics(predicted, truth, query)
    invalid = [
        k for k, v in metrics.items() if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v))
    ]
    if not invalid:
        return metrics, affected, None
    failure = context | {
        "error": "Nonrepresentable query summary",
        "invalid_metrics": invalid,
        "finite_predictions_saved": True,
        "predictions_sha256": prior.digest(folder / "predictions.csv"),
        "metrics": {k: None if k in invalid else v for k, v in metrics.items()},
    }
    backend.save_json(folder / "query_failure.json", failure)
    return metrics, affected, failure


def score_all() -> None:
    frozen = execution_identity()
    identity_sha = prior.digest(OUTPUT / "execution_identity.json")
    expected = [
        (noise, world, candidate)
        for noise in NOISES
        for world in backend.WORLDS
        for candidate in stage.oracle.candidates(world)
    ]
    if any(
        not backend.completed_fit(cell(noise, world["name"], candidate), identity_sha)
        for noise, world, candidate in expected
    ):
        raise ValueError("All thirty full fits must be durable before opening held truth")
    data, folds = stage.oracle.inputs()
    split = stage.oracle.split_indices(data.rows, folds, OUTER)
    query0, query1, query = stage.oracle.query_frame(data, split)
    rows, failures, receipts = [], [], []
    with closing(sqlite3.connect(f"file:{CACHE}?mode=ro&immutable=1", uri=True)) as connection:
        cache = backend.helper.Cache(
            connection, backend.helper.Checks(), frozen, {"scales": data.total_scales, "alpha": data.alpha}
        )
        with (
            np.load(prior.STAGE1 / "prepared/clean.npz") as original,
            np.load(prior.STAGE1 / "prepared/matched_tied.npz") as tied,
        ):
            for noise in NOISES:
                for world in backend.WORLDS:
                    name = world["name"]
                    truth = np.r_[
                        original[name][split["test"]], tied[name][split["test"]], original[name][split["tied_test"]]
                    ]
                    native = None
                    for candidate in stage.oracle.candidates(world):
                        folder = cell(noise, name, candidate)
                        fitted = prior.read(folder / "fit.json")
                        spec, head = fitted["selected_candidate"], fitted["selected"]
                        matrix = cache.matrix(
                            query0, query1, backend.feature_spec(spec["shape"], spec["mu"], spec["clock"], data.alpha)
                        )
                        try:
                            if head is None:
                                raise ArithmeticError("No finite full-training head")
                            predicted = solver.prior.predict(matrix, head)
                        except ArithmeticError as error:
                            failure = {
                                "world": name,
                                "noise": noise,
                                "outer_fold": OUTER,
                                "candidate": candidate,
                                "error": str(error),
                                "finite_predictions_saved": False,
                            }
                            backend.save_json(folder / "query_failure.json", failure)
                            failures.append(failure)
                            receipts.append(folder / "query_failure.json")
                            continue
                        metrics, affected, failure = save_query_summary(
                            folder,
                            predicted,
                            truth,
                            query,
                            {"world": name, "noise": noise, "outer_fold": OUTER, "candidate": candidate},
                        )
                        if failure is not None:
                            failures.append(failure)
                            receipts.append(folder / "query_failure.json")
                            continue
                        count = len(split["test"])
                        gain = predicted[count : 2 * count] - predicted[:count]
                        query_logit = (
                            None
                            if head["scale"] == 0
                            else np.log(head["scale"])
                            + (matrix @ np.asarray(head["amplitudes"]) - head["normalization"])
                        )
                        underflow = 0 if query_logit is None else int(np.sum(np.exp(query_logit) == 0))
                        inner_unresolved = any(r["unresolved"] for r in fitted["selected_inner_records"])
                        score = metrics | {
                            "maximum_positive_predicted_gain": float(max(0, np.max(gain))),
                            "selected_upper_boundary": spec["mu"] == backend.STRENGTHS[-1],
                            "selected_mu": spec["mu"],
                            "shape": spec["shape"],
                            "floor": head["floor"],
                            "ridge": 0.0,
                            "full_unresolved": fitted["full_unresolved"],
                            "full_diagnostics": fitted["full_diagnostics"],
                            "selected_inner_unresolved": inner_unresolved,
                            "inner_rmse": float(np.sqrt(fitted["inner_mse"])),
                            "native_full_fit_reused": fitted["native_full_fit_reused"],
                            "all_candidate_numerical_counts": backend.profile_diagnostics(
                                prior.read(folder / "profiles.json")["profiles"]
                            ),
                            "query_exponential_underflow_count": underflow,
                            "zero_scale_closure": head["scale"] == 0,
                            "unresolved": bool(
                                fitted["full_unresolved"] or inner_unresolved or underflow or head["scale"] == 0
                            ),
                            "metric_overflow_fallback_fields": affected,
                            "original_stage2_remains_blocked": True,
                        }
                        if candidate == "aggregate_mariner":
                            native = score
                        checks = backend.performance_checks(world, candidate, score, native)
                        score["historical_performance_checks"] = checks
                        score["historical_performance_status"] = (
                            "incomplete_native_query"
                            if checks is None
                            else "pass"
                            if all(c["passed"] for c in checks)
                            else "fail"
                        )
                        backend.save_json(folder / "scores.json", score)
                        backend.save_json(
                            folder / "score_complete.json",
                            {
                                "identity_sha256": identity_sha,
                                "files": {
                                    f: prior.digest(folder / f)
                                    for f in ("fit_complete.json", "scores.json", "predictions.csv")
                                },
                            },
                        )
                        receipts.append(folder / "score_complete.json")
                        rows.append(
                            {"world": name, "noise": noise, "outer_fold": OUTER, "candidate": candidate, **score}
                        )
    complete = len(rows) == 30 and all(r["historical_performance_checks"] is not None for r in rows)
    backend.save_json(
        OUTPUT / "summary.json",
        {
            "rows": rows,
            "query_failures": failures,
            "heads": 30,
            "finite_query_heads": len(rows),
            "unresolved_heads": sum(r["unresolved"] for r in rows),
            "historical_performance_complete": complete,
            "historical_performance_passed": (
                complete and all(r["historical_performance_status"] == "pass" for r in rows)
            ),
            "historical_performance_failures": [
                {"world": r["world"], "noise": r["noise"], "candidate": r["candidate"], **c}
                for r in rows
                for c in r["historical_performance_checks"] or []
                if not c["passed"]
            ],
            "original_stage2_remains_blocked": True,
            "scientific_promotion": False,
            "targeted_stress_only": True,
        },
    )
    backend.save_json(
        OUTPUT / "complete.json",
        {
            "identity_sha256": identity_sha,
            "files": {str(p.relative_to(OUTPUT)): prior.digest(p) for p in [OUTPUT / "summary.json", *receipts]},
            "completed_heads": len(rows) + len(failures),
            "scientific_promotion": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "fit", "score"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "freeze":
        backend.save_json(OUTPUT / "execution_identity.json", identity())
    elif args.command == "fit":
        execution_identity()
        if args.workers < 1:
            raise ValueError("A positive worker count is required")
        if args.workers == 1:
            for context in contexts():
                fit_context(context)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(fit_context, contexts()))
    else:
        score_all()


if __name__ == "__main__":
    main()
