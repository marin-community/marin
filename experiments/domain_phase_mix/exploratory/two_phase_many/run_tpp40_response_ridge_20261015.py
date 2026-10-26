# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""TP048: fixed-law ridge curves, with no ridge selection or model promotion."""

from __future__ import annotations

import argparse
import json
import shutil
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

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    run_tpp40_response_profile_trust_region_20261015 as old,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_response_ridge_20261015 as model

prior = old.prior
OUTPUT = prior.PARENT / "response_ridge_isolation"
SEAL = old.OUTPUT / "result_manifest_v1.json"
SEAL_SHA = "fbdbcd654819d5ad84d4bfa6cf4dcec7a4d7ecb093439f79bb1918bc188c382f"
SCOPE = "tp048_conditional_response_ridge_curves_no_selection"
RIDGES = (0.0, 1e-4, 0.01, 1.0, 100.0)
ARMS = old.ARMS
CONTRACT = {
    "id": "TP40-048",
    "scope": SCOPE,
    "contexts": 80,
    "heads": 800,
    "adopted_zero_ridge_heads": 160,
    "new_conditional_fits": 640,
    "new_starts": 3200,
    "arms": list(ARMS),
    "ridges": list(RIDGES),
    "worlds": list(prior.WORLDS),
    "noise_names": list(prior.NOISES),
    "outer_folds": list(range(5)),
    "objective": "mean((prediction-y)^2)/0.003^2 + lambda*sum(canonical_amplitudes^2)",
    "ridge_units": "lambda multiplies the mean noise-scaled objective, not a sum objective; no implicit n factor",
    "linear_profile": "same exact nonnegative floor/scale profile; penalty depends only on amplitudes",
    "shape_and_kinetics": "fixed TP036 shape/kinetics; heterogeneous world retains shared-shape misspecification",
    "parameter_bounds": old.CONTRACT["parameter_bounds"],
    "solver_options": model.frozen.SOLVER_OPTIONS,
    "evaluation_cap_per_start": 5000,
    "starts": "exact five saved TP036 training-local starts; no previous response incumbent added",
    "selection": (
        "per fixed lambda, smallest total training objective across all finite trials; no selection across lambdas"
    ),
    "numerical_flags": (
        "total-objective first-order flag plus original unpenalized flag; all failed and nonwinning starts retained"
    ),
    "zero_ridge": "byte-identical fit.json adoption from sealed TP040, never refitted",
    "held_truth": "read only after the corresponding fit has a durable fit_complete receipt",
    "features": "immutable TP036 cache and identical saved initialization matrix; no new integration",
    "new_empirical_labels": False,
    "new_policy_queries": False,
    "full_shape_kinetic_search": False,
    "ridge_selection": False,
    "scientific_promotion": False,
    "original_learned_stage_remains_blocked": True,
    "failures": "retain failed fits/queries/metrics with terminal receipts; continue the fixed inventory",
}


def source_paths() -> set[Path]:
    return old.source_paths() | {
        Path(__file__),
        Path(model.__file__),
        prior.BASE / "test_tpp40_response_ridge_20261015.py",
        OUTPUT / "proposal_v1.md",
    }


def input_paths() -> set[Path]:
    return old.input_paths() | {
        SEAL,
        old.OUTPUT / "registration.json",
        old.OUTPUT / "execution_identity.json",
        old.OUTPUT / "complete.json",
        old.OUTPUT / "summary.json",
        old.OUTPUT / "audit/independent_replay_report_v1.json",
    }


def registration() -> dict[str, Any]:
    value = prior.read(OUTPUT / "registration.json")
    if (
        value.get("status") != "registered"
        or value.get("contract") != CONTRACT
        or value["environment"] != prior.runtime()
    ):
        raise ValueError("Register the exact TP048 scope/runtime")
    historical = prior.read(old.OUTPUT / "registration.json")
    for field in ("source_hashes", "input_hashes"):
        prior.verify_hashes(historical[field])
    for field, paths in (("source_hashes", source_paths()), ("input_hashes", input_paths())):
        if set(value[field]) != {str(path.resolve()) for path in paths}:
            raise ValueError("Exact source/input whitelist changed")
        prior.verify_hashes(value[field])
    if prior.digest(SEAL) != SEAL_SHA:
        raise ValueError("Frozen TP040 seal changed")
    prior.verify_hashes({str(old.OUTPUT / name): item["sha256"] for name, item in prior.read(SEAL)["files"].items()})
    old.registration()
    return value


def identity() -> dict[str, Any]:
    registered = registration()
    return {key: registered[key] for key in ("source_hashes", "input_hashes", "environment")} | {
        "registration_sha256": prior.digest(OUTPUT / "registration.json")
    }


def execution_identity() -> dict[str, Any]:
    frozen = identity()
    if prior.read(OUTPUT / "execution_identity.json") != frozen:
        raise ValueError("Frozen identity changed")
    expected = {
        "scope": SCOPE,
        "registration_sha256": frozen["registration_sha256"],
        "execution_identity_sha256": prior.digest(OUTPUT / "execution_identity.json"),
    }
    if any(prior.read(OUTPUT / "execution_approval.json").get(key) != val for key, val in expected.items()):
        raise ValueError("Exact execution approval required")
    return frozen


def fit_saved_training(saved: dict[str, Any], arm: str, ridge: float) -> dict[str, Any]:
    if arm not in ARMS or ridge not in RIDGES[1:]:
        raise ValueError("Only registered positive ridge cells are freshly fitted")
    return model.fit_response(
        np.asarray(saved["matrix"]),
        np.asarray(saved["observed"]),
        0.6 if arm == ARMS[0] else None,
        saved["starts"],
        ridge=ridge,
        evaluation_cap=5000,
    )


def save_fit(
    folder: Path,
    saved: dict[str, Any],
    arm: str,
    ridge: float,
    spec: dict[str, Any],
    frozen: dict[str, Any],
    old_target: Path,
) -> dict[str, Any]:
    complete = folder / "fit_complete.json"
    if complete.exists():
        receipt = prior.read(complete)
        if receipt["spec"] != spec or receipt["identity"] != frozen:
            raise ValueError("Saved fit identity differs")
        prior.verify_hashes({str(folder / "fit.json"): receipt["fit_sha256"]})
        return prior.read(folder / "fit.json")
    if (folder / "fit.json").exists():
        raise ValueError("Preserve unreceipted fit before any scientific retry")
    folder.mkdir(parents=True, exist_ok=True)
    if ridge == 0:
        shutil.copyfile(old_target / "fit.json", folder / "fit.json")
        fit = prior.read(folder / "fit.json")
    else:
        with threadpool_limits(limits=1):
            fit = fit_saved_training(saved, arm, ridge)
        prior.write(folder / "fit.json", fit)
    prior.write(
        complete,
        {
            "spec": spec,
            "identity": frozen,
            "fit_sha256": prior.digest(folder / "fit.json"),
            "adopted_from": str(old_target / "fit.json") if ridge == 0 else None,
            "adopted_sha256": prior.digest(old_target / "fit.json") if ridge == 0 else None,
        },
    )
    return fit


def selected_readout(fit: dict[str, Any], ridge: float) -> dict[str, Any]:
    value = fit["selected"]
    report = prior.fit_diagnostics(fit) | {"unresolved": fit["unresolved"], "ridge": ridge}
    if value is None:
        return report | {"selected": None}
    data_objective = value["raw_mean_squared_bpb"] / model.prior.NOISE_SCALE**2
    penalty = 0.0 if ridge == 0 else float(ridge * np.sum(np.asarray(value["amplitudes"]) ** 2))
    return report | {
        "selected": value,
        "data_scaled_objective": data_objective,
        "data_mean_squared_bpb": value["raw_mean_squared_bpb"],
        "penalty": penalty,
        "total_scaled_objective": data_objective + penalty,
        "data_first_order_tolerance_met": value.get(
            "data_first_order_tolerance_met", value["first_order_tolerance_met"]
        ),
        "data_projected_gradient_inf": value.get("data_projected_gradient_inf", value["projected_gradient_inf"]),
        "total_first_order_tolerance_met": value["first_order_tolerance_met"],
        "total_projected_gradient_inf": value["projected_gradient_inf"],
        "floor": value["floor"],
        "scale": value["scale"],
        "actual_evaluation_calls": sum(item["actual_evaluation_calls"] for item in fit["solves"]),
    }


def score_saved_fit(
    target: Path,
    fit: dict[str, Any],
    saved: dict[str, Any],
    ridge: float,
    matrix: np.ndarray,
    query: pd.DataFrame,
    previous_csv: Path,
) -> tuple[str, ...]:
    """Only the persisted training-selected head reaches a held outcome reader."""
    if not (target / "fit_complete.json").exists():
        raise ValueError("Persist the fit receipt before reading held truth")
    readout = selected_readout(fit, ridge)
    if fit["selected"] is None:
        prior.write(target / "scores.json", {"status": "fit_failed", "readout": readout})
        return ("fit.json", "fit_complete.json", "scores.json")
    names = ["fit.json", "fit_complete.json", "scores.json"]
    try:
        previous = pd.read_csv(previous_csv, keep_default_na=False, float_precision="round_trip")
        if not previous[["role", "source_index", "row_id"]].equals(query):
            raise ValueError("Frozen query order changed")
        predicted = model.prior.predict(matrix, fit["selected"])
        truth = previous.truth.to_numpy()
        query.assign(truth=truth, predicted=predicted).to_csv(target / "predictions.csv", index=False)
        names.append("predictions.csv")
        training = model.prior.predict(np.asarray(saved["matrix"]), fit["selected"])
        pd.DataFrame(
            {"source_index": saved["train_indices"], "observed": saved["observed"], "predicted": training}
        ).to_csv(target / "training.csv", index=False)
        names.append("training.csv")
        with np.errstate(over="raise", invalid="raise"):
            metrics = prior.score(predicted, truth, query)
        result = {"status": "finite", "metrics": metrics, "readout": readout}
        json.dumps(result, allow_nan=False)
    except (ArithmeticError, ValueError) as error:
        result = {
            "status": "query_or_metric_failed",
            "readout": readout,
            "exception": type(error).__name__,
            "message": str(error),
        }
    prior.write(target / "scores.json", result)
    return tuple(names)


def run_context(spec: dict[str, Any], frozen: dict[str, Any]) -> None:
    relative = Path("cells") / f"outer{spec['outer_fold']}" / spec["noise"] / spec["world"]["name"]
    initialized = prior.OUTPUT / relative / "initialization"
    saved = old.saved_initialization(initialized, spec, prior.read(prior.OUTPUT / "execution_identity.json"))
    data = prior.geometry(spec["outer_fold"])
    if saved["train_indices"] != data["split"]["train"].tolist():
        raise ValueError("Frozen asymmetric training rows changed")
    with closing(sqlite3.connect(f"file:{prior.CACHE_PATH}?mode=ro&immutable=1", uri=True)) as connection:
        cache = prior.FrozenFeatures(connection, data, prior.read(prior.OUTPUT / "registration.json"))
        train = data["split"]["train"]
        matrix = cache.matrix(data["first"][train], data["last"][train], spec["world"])
        query_matrix = cache.matrix(data["query0"], data["query1"], spec["world"])
    np.testing.assert_array_equal(matrix, np.asarray(saved["matrix"]))
    old_identity = prior.read(old.OUTPUT / "execution_identity.json")
    for arm in ARMS:
        old_target = old.OUTPUT / relative / arm
        if not prior.receipt(old_target, spec | {"arm": arm}, old_identity):
            raise ValueError("Missing complete sealed zero-ridge reference")
        if prior.read(old_target / "fit.json")["starts"] != saved["starts"]:
            raise ValueError("Frozen starts differ from TP040")
        for index, ridge in enumerate(RIDGES):
            target = OUTPUT / relative / arm / f"ridge{index}"
            cell = spec | {"arm": arm, "ridge": ridge}
            if prior.receipt(target, cell, frozen):
                continue
            fit = save_fit(target, saved, arm, ridge, cell, frozen, old_target)
            names = score_saved_fit(
                target, fit, saved, ridge, query_matrix, data["query"], old_target / "predictions.csv"
            )
            prior.finish(
                target, cell, frozen, names, (old_target / "complete.json", initialized / "complete.json", SEAL)
            )
    print(json.dumps({"completed_context": spec}), flush=True)


def summarize(frozen: dict[str, Any]) -> None:
    records, dependencies = [], []
    for spec in prior.contexts():
        for arm in ARMS:
            for index, ridge in enumerate(RIDGES):
                target = (
                    OUTPUT
                    / "cells"
                    / f"outer{spec['outer_fold']}"
                    / spec["noise"]
                    / spec["world"]["name"]
                    / arm
                    / f"ridge{index}"
                )
                cell = spec | {"arm": arm, "ridge": ridge}
                if not prior.receipt(target, cell, frozen):
                    raise ValueError("All 800 finite or failed cell receipts are required")
                records.append(cell | prior.read(target / "scores.json"))
                dependencies.append(target / "complete.json")
    prior.write(
        OUTPUT / "summary.json",
        {
            "status": "descriptive_fixed_lambda_curves_only",
            "heads": len(records),
            "records": records,
            "failed_cells": sum(record["status"] != "finite" for record in records),
            "unresolved_heads": sum(record["readout"]["unresolved"] for record in records),
            "ridge_selection": False,
            "scientific_promotion": False,
            "original_learned_stage_remains_blocked": True,
        },
    )
    prior.finish(OUTPUT, {"stage": "summary"}, frozen, ("summary.json",), tuple(dependencies))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "fit", "summarize"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "freeze":
        current = identity()
        path = OUTPUT / "execution_identity.json"
        if path.exists() and prior.read(path) != current:
            raise ValueError("Preserve previous frozen identity")
        prior.write(path, current)
        return
    frozen = execution_identity()
    if args.command == "fit":
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(run_context, prior.contexts(), [frozen] * len(prior.contexts())))
    summarize(frozen)


if __name__ == "__main__":
    main()
