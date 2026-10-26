# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""TP040: saved-input solver control; no new design, law, or model selection."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_response_profile_20261015 as prior
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_response_profile_trust_region_20261015 as model

OUTPUT = prior.PARENT / "response_solver_control"
PREVIOUS = prior.OUTPUT
SEAL = PREVIOUS / "result_manifest_v1.json"
AUDIT = PREVIOUS / "audit/independent_replay_v1.json"
NOTE = prior.PARENT / "reviews/estimator_bottleneck_v1.md"
SCOPE = "tp040_identical_fixed_law_response_solver_control"
ARMS = prior.ARMS[1:]
CONTRACT = {
    "id": "TP40-040",
    "scope": SCOPE,
    "contexts": 80,
    "nonlinear_profiles": 160,
    "starts": 800,
    "arms": list(ARMS),
    "worlds": list(prior.WORLDS),
    "noise_names": list(prior.NOISES),
    "outer_folds": list(range(5)),
    "objective": prior.CONTRACT["objective"],
    "fixed_shape": prior.CONTRACT["fixed_shape"],
    "ridge": 0.0,
    "solver": model.SOLVER_OPTIONS,
    "actual_evaluations_per_start": model.EVALUATION_CAP,
    "original_projected_gradient_tolerance": model.prior.SOLVER_OPTIONS["gtol"],
    "linear_face_roundoff_multiplier": model.FACE_ROUNDOFF_MULTIPLIER,
    "starts_and_training_arrays": "exact saved TP036 initialization, no new NNLS or labels",
    "parameter_bounds": "unchanged nonnegative amplitudes and nonnegative learned floor; no upper bounds",
    "selection": "smallest training objective over every finite TRF trial; ties by trial order",
    "old_incumbent": "comparison only, never a new TRF start or candidate",
    "query_access": "held prediction/truth tables read only after the new fit is durably saved",
    "scientific_promotion": False,
    "new_acceptance_gate": False,
    "learned_stage_remains_blocked": True,
    "new_features_or_integration": False,
    "new_policy_queries_or_optimization": False,
    "empirical_outcomes": False,
    "shape_kinetic_ridge_selection": False,
    "regularity": "constant-rank strict active face; nonregular finite points retained without Jacobian certificate",
    "bound_status": "no amplitude snapping; TRF optimality and old exact-zero projected gradient remain separate",
}


def source_paths() -> set[Path]:
    return {Path(path) for path in prior.read(PREVIOUS / "registration.json")["source_hashes"]} | {
        Path(__file__),
        Path(model.__file__),
        NOTE,
        prior.BASE / "test_tpp40_response_profile_trust_region_20261015.py",
        prior.BASE / "test_run_tpp40_response_profile_trust_region_20261015.py",
    }


def input_paths() -> set[Path]:
    return {Path(path) for path in prior.read(PREVIOUS / "registration.json")["input_hashes"]} | {
        SEAL,
        AUDIT,
        PREVIOUS / "registration.json",
        PREVIOUS / "execution_identity.json",
        PREVIOUS / "summary.json",
        PREVIOUS / "complete.json",
    }


def registration() -> dict[str, Any]:
    value = prior.read(OUTPUT / "registration.json")
    if (
        value.get("status") != "registered"
        or value.get("contract") != CONTRACT
        or value["environment"] != prior.runtime()
    ):
        raise ValueError("Register the exact TP040 scope and runtime before execution")
    for name, paths in (("source_hashes", source_paths()), ("input_hashes", input_paths())):
        if set(value[name]) != {str(path.resolve()) for path in paths}:
            raise ValueError("The source/input whitelist changed")
        prior.verify_hashes(value[name])
    if (
        prior.digest(SEAL) != value["prerequisites"]["tp036_seal_sha256"]
        or prior.digest(AUDIT) != value["prerequisites"]["tp036_audit_sha256"]
    ):
        raise ValueError("The registered TP036 prerequisite differs")
    prior.verify_hashes({str(PREVIOUS / name): item["sha256"] for name, item in prior.read(SEAL)["files"].items()})
    if not prior.read(AUDIT)["passed"]:
        raise ValueError("TP036 needs a passing independent numerical replay")
    prior.registration()
    return value


def identity() -> dict[str, Any]:
    registered = registration()
    return {name: registered[name] for name in ("source_hashes", "input_hashes", "environment", "prerequisites")} | {
        "registration_sha256": prior.digest(OUTPUT / "registration.json")
    }


def freeze() -> None:
    current = identity()
    path = OUTPUT / "execution_identity.json"
    if path.exists() and prior.read(path) != current:
        raise ValueError("Preserve the old execution identity")
    prior.write(path, current)


def execution_identity() -> dict[str, Any]:
    current = identity()
    if prior.read(OUTPUT / "execution_identity.json") != current:
        raise ValueError("Frozen identity differs")
    approval = prior.read(OUTPUT / "execution_approval.json")
    expected = {
        "scope": SCOPE,
        "registration_sha256": current["registration_sha256"],
        "execution_identity_sha256": prior.digest(OUTPUT / "execution_identity.json"),
    }
    if any(approval.get(key) != value for key, value in expected.items()):
        raise ValueError("Exact frozen execution requires approval")
    return current


def saved_initialization(folder: Path, spec: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    if not prior.receipt(folder, {**spec, "stage": "five_gamma_starts"}, frozen):
        raise ValueError("Missing completed TP036 initialization")
    saved = prior.read(folder / "starts.json")
    if set(saved) != {"matrix", "observed", "train_indices", "starts"} or len(saved["starts"]) != 5:
        raise ValueError("The saved training-only initialization contract changed")
    return saved


def fit_saved_training(saved: dict[str, Any], arm: str) -> dict[str, Any]:
    if arm not in ARMS or set(saved) != {"matrix", "observed", "train_indices", "starts"}:
        raise ValueError("Only saved asymmetric training arrays may enter TP040")
    return model.fit_response(
        np.asarray(saved["matrix"]),
        np.asarray(saved["observed"]),
        0.6 if arm == ARMS[0] else None,
        saved["starts"],
        evaluation_cap=5000,
    )


def comparison(
    fit: dict[str, Any], old: dict[str, Any], predicted: np.ndarray, old_prediction: np.ndarray, query: pd.DataFrame
) -> dict[str, Any]:
    selected, previous = fit["selected"], old["selected"]
    held = int(np.sum(query.role == "asymmetric"))
    difference = predicted - old_prediction
    return {
        "old_scaled_objective": previous["scaled_objective"],
        "new_scaled_objective": selected["scaled_objective"],
        "scaled_objective_change": selected["scaled_objective"] - previous["scaled_objective"],
        "maximum_query_prediction_change": float(np.max(abs(difference))),
        "maximum_gain_prediction_change": float(np.max(abs(difference[held : 2 * held] - difference[:held]))),
        "old_unresolved": old["unresolved"],
        "new_unresolved": fit["unresolved"],
        "old_first_order_tolerance_met": previous["first_order_tolerance_met"],
        "new_first_order_tolerance_met": selected["first_order_tolerance_met"],
        "old_projected_gradient_inf": previous["projected_gradient_inf"],
        "new_projected_gradient_inf": selected["projected_gradient_inf"],
        "old_selected_from_successful_solve": old["selected_from_successful_solve"],
        "new_selected_from_successful_solve": fit["selected_from_successful_solve"],
        "old_floor": previous["floor"],
        "new_floor": selected["floor"],
        "new_jacobian_regular": selected["jacobian_regular"],
        "new_zero_scale_closure": selected["zero_scale_closure"],
        "minimum_amplitude_bound_distance": float(min(selected["amplitudes"])),
        "new_failed_starts": sum(not item["success"] for item in fit["solves"]),
        "old_failed_starts": sum(not item["success"] for item in old["solves"]),
        "actual_evaluation_calls": sum(item["actual_evaluation_calls"] for item in fit["solves"]),
        "scientific_promotion": False,
        "learned_stage_remains_blocked": True,
    }


def fit_and_score(
    saved: dict[str, Any],
    arm: str,
    target: Path,
    query_matrix: np.ndarray,
    query: pd.DataFrame,
    old_fit: dict[str, Any],
    previous_csv: Path,
) -> None:
    """Save a training-selected fit before opening the held-label comparison."""
    with threadpool_limits(limits=1):
        fit = fit_saved_training(saved, arm)
    prior.write(target / "fit.json", fit)
    if fit["selected"] is None:
        raise ArithmeticError("No finite new response-profile candidate")
    previous = pd.read_csv(previous_csv, keep_default_na=False)
    if not previous[["role", "source_index", "row_id"]].equals(query):
        raise ValueError("The old query order changed")
    predicted = model.prior.predict(query_matrix, fit["selected"])
    old_prediction = model.prior.predict(query_matrix, old_fit["selected"])
    np.testing.assert_allclose(old_prediction, previous.predicted.to_numpy(), atol=2e-12, rtol=0)
    truth = previous.truth.to_numpy()
    report = {
        "comparison": comparison(fit, old_fit, predicted, old_prediction, query),
        "new": prior.score(predicted, truth, query),
        "old": prior.score(old_prediction, truth, query),
    }
    prior.write(target / "scores.json", report)
    query.assign(truth=truth, predicted=predicted, old_predicted=old_prediction).to_csv(
        target / "predictions.csv", index=False
    )
    pd.DataFrame(
        {
            "source_index": saved["train_indices"],
            "observed": saved["observed"],
            "predicted": model.prior.predict(np.asarray(saved["matrix"]), fit["selected"]),
        }
    ).to_csv(target / "training.csv", index=False)


def run_context(spec: dict[str, Any], frozen: dict[str, Any]) -> None:
    old_identity = prior.read(PREVIOUS / "execution_identity.json")
    relative = Path("cells") / f"outer{spec['outer_fold']}" / spec["noise"] / spec["world"]["name"]
    old_folder = PREVIOUS / relative
    saved = saved_initialization(old_folder / "initialization", spec, old_identity)
    data = prior.geometry(spec["outer_fold"])
    if saved["train_indices"] != data["split"]["train"].tolist():
        raise ValueError("Saved training indices differ from frozen asymmetric folds")
    with closing(sqlite3.connect(f"file:{prior.CACHE_PATH}?mode=ro&immutable=1", uri=True)) as connection:
        cache = prior.FrozenFeatures(connection, data, prior.read(PREVIOUS / "registration.json"))
        train = data["split"]["train"]
        matrix = cache.matrix(data["first"][train], data["last"][train], spec["world"])
        query_matrix = cache.matrix(data["query0"], data["query1"], spec["world"])
    np.testing.assert_array_equal(matrix, np.asarray(saved["matrix"]))
    for arm in ARMS:
        arm_spec, target, old_target = {**spec, "arm": arm}, OUTPUT / relative / arm, old_folder / arm
        if prior.receipt(target, arm_spec, frozen):
            continue
        if (target / "execution_failure.json").exists():
            raise ValueError("Preserve the failed attempt; no automatic scientific retry")
        if not prior.receipt(old_target, arm_spec, old_identity):
            raise ValueError("Missing completed original response profile")
        old_fit = prior.read(old_target / "fit.json")
        if old_fit["starts"] != saved["starts"]:
            raise ValueError("Old response profile used different starts")
        try:
            fit_and_score(saved, arm, target, query_matrix, data["query"], old_fit, old_target / "predictions.csv")
            prior.finish(
                target,
                arm_spec,
                frozen,
                ("fit.json", "scores.json", "predictions.csv", "training.csv"),
                (old_target / "complete.json", old_folder / "initialization/complete.json", SEAL),
            )
        except (ArithmeticError, ValueError, RuntimeError) as error:
            prior.write(
                target / "execution_failure.json",
                {"identity": frozen, "spec": arm_spec, "exception": type(error).__name__, "message": str(error)},
            )
            raise


def summarize(frozen: dict[str, Any]) -> None:
    records, dependencies = [], []
    for spec in prior.contexts():
        for arm in ARMS:
            target = OUTPUT / "cells" / f"outer{spec['outer_fold']}" / spec["noise"] / spec["world"]["name"] / arm
            if not prior.receipt(target, {**spec, "arm": arm}, frozen):
                raise ValueError("All160 registered nonlinear profiles are required")
            records.append({**spec, "arm": arm, **prior.read(target / "scores.json")})
            dependencies.append(target / "complete.json")
    prior.write(
        OUTPUT / "summary.json",
        {
            "status": "solver_comparison_only",
            "records": records,
            "heads": len(records),
            "scientific_promotion": False,
            "new_acceptance_gate": False,
            "learned_stage_remains_blocked": True,
        },
    )
    prior.finish(OUTPUT, {"stage": "summary"}, frozen, ("summary.json",), tuple(dependencies))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "fit", "summarize"))
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
        return
    frozen = execution_identity()
    if args.command == "fit":
        for spec in prior.contexts():
            run_context(spec, frozen)
    summarize(frozen)


if __name__ == "__main__":
    main()
