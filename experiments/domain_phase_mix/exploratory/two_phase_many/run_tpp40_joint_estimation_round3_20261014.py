# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Synthetic-only isolation of phase-aware shape and floor estimation.

Prepare freezes sources and materializes labels without fitting. Run executes
individually resumable estimator cells; no actual TPP40 outcomes are accessed.
The common-floor cell is a true-shape diagnostic, not a promoted estimator.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_two_phase_20260921 as benchmark
from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_mechanism_round2_20261014 as round2
from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_temporal_round_20261014 as prior
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_joint_estimation_20261014 as joint
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_mechanism_models_20261014 as mechanisms
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_temporal_models_20261014 as temporal

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reference_outputs/tpp40_joint_estimation_round3_20261014"
FAMILIES = ("aggregate_mariner", "bounded_benefit_odds")
ESTIMATORS = (
    "native",
    "ridge_gamma_refined",
    "j1_fixed_shape",
    "j2_joint_shape",
    "oracle_true_shape_true_floor",
    "oracle_all_shape_true_floor",
    "oracle_true_shape_shared_gamma",
    "oracle_true_shape_common_floor",
)
FLOOR_BOUNDS = (-8.0, 8.0)
FLOOR_GRID = tuple(sorted((*FLOOR_BOUNDS, *((index + 0.5) / 4 for index in range(-32, 32)))))
TRANSITION_BISECTIONS = 20
REFINEMENT_SUBINTERVALS = 4
REFINEMENT_XATOL = 1e-8
REFINEMENT_MAXITER = 64
Fit = temporal.TemporalFit | mechanisms.MechanismFit | joint.JointFit


@dataclasses.dataclass(frozen=True)
class Context:
    name: str
    world: str
    noise_sd: float
    noise_seed: int
    early: np.ndarray
    late: np.ndarray
    alpha: float
    pool_tokens: np.ndarray
    train: np.ndarray
    asymmetric: np.ndarray
    tied: np.ndarray
    labels: np.ndarray
    clean: np.ndarray
    observed: np.ndarray
    truth: round2.Fit


def registered_contexts() -> list[dict[str, Any]]:
    return json.loads((OUTPUT / "registration.json").read_text())["contexts"]


def context_name(spec: dict[str, Any]) -> str:
    return f"{spec['world']}_noise{spec['noise_sd']:g}_seed{spec['noise_seed']}"


def source_identity() -> dict[str, str]:
    """Pin loaded local dependencies, transitive fold code and immutable inputs."""
    sources = {
        Path(module.__file__).resolve()
        for module in tuple(sys.modules.values())
        if getattr(module, "__file__", None)
        and Path(module.__file__).resolve().is_relative_to(ROOT)
        and Path(module.__file__).suffix == ".py"
    }
    sources.update(
        {
            Path(__file__).resolve(),
            Path(temporal.mariner.__file__).resolve(),
            ROOT / "two_phase_benchmark_data_20260921.py",
            ROOT / "round6_floor_cv_20260921.py",
            OUTPUT / "registration.json",
            OUTPUT / "common_floor_registration.json",
            OUTPUT / "pre_execution_notes.json",
            prior.OUTPUT / "result_manifest.json",
            prior.OUTPUT / "data/manifest.json",
            prior.OUTPUT / "data/arrays.npz",
            prior.OUTPUT / "data/rows.csv",
            prior.OUTPUT / "data/folds.csv",
        }
    )
    for world in ("aggregate_null", "odds"):
        for noise in (0.0, 0.003):
            folder = round2.OUTPUT / "synthetic" / f"{world}_noise{noise:g}"
            sources.update(folder / name for name in ("complete.json", "truth.json", "models.json"))
    return {str(path): benchmark.sha256(path) for path in sorted(sources)}


def execution_identity() -> dict[str, str]:
    expected = json.loads((OUTPUT / "execution_identity.json").read_text())
    if source_identity() != expected:
        raise ValueError("Round-three frozen sources or inputs changed; refusing to resume")
    return expected


def completed(folder: Path, context: dict[str, Any], frozen: dict[str, str]) -> bool:
    receipt = folder / "complete.json"
    if not receipt.exists():
        return False
    record = json.loads(receipt.read_text())
    if record["identity"] != frozen or record["context"] != context:
        raise ValueError(f"Stale round-three result: {folder}")
    benchmark.verify_hashes(record["outputs"], folder)
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


def saved_source(spec: dict[str, Any]) -> Path:
    return round2.OUTPUT / "synthetic" / f"{spec['world']}_noise{spec['noise_sd']:g}"


def verify_saved_source(spec: dict[str, Any]) -> dict[str, Any]:
    source = saved_source(spec)
    record = json.loads((source / "complete.json").read_text())
    for name, digest in record["identity"].items():
        if benchmark.sha256(Path(name)) != digest:
            raise ValueError(f"Historical source drift: {name}")
    benchmark.verify_hashes(record["outputs"], source)
    return record


def prepare() -> None:
    """Freeze inputs and synthetic labels without executing any fit."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data_manifest = json.loads((prior.OUTPUT / "data/manifest.json").read_text())
    benchmark.verify_hashes(data_manifest["files"], prior.OUTPUT / "data")
    for name, digest in data_manifest["implementation_hashes"].items():
        if benchmark.sha256(Path(name)) != digest:
            raise ValueError(f"Frozen data implementation drift: {name}")
    frozen = source_identity()
    identity_path = OUTPUT / "execution_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != frozen:
            raise ValueError("Existing execution identity differs; no automatic overwrite")
    else:
        benchmark.write_json(identity_path, frozen)
    fold_table = pd.read_csv(prior.OUTPUT / "data/folds.csv")
    outer = fold_table[fold_table.outer_fold.eq(0)]
    train_rows = outer[outer.role.eq("train")]
    train = train_rows.row_index.to_numpy(int)
    labels = train_rows.inner_fold.to_numpy(int)
    with np.load(prior.OUTPUT / "data/arrays.npz") as arrays:
        early, late = arrays["phase0_epochs"], arrays["phase1_epochs"]
        tied = np.flatnonzero(arrays["tied"])
        query = outer.loc[outer.role.ne("train"), "row_index"].to_numpy(int)
        asymmetric = query[~arrays["tied"][query]]
        if (len(train), len(asymmetric), len(tied)) != (172, 66, 42) or arrays["tied"][train].any():
            raise ValueError("Synthetic split differs from the registered outer-zero geometry")
        temporal.inner_splits(labels)
        for spec in registered_contexts():
            folder = OUTPUT / "contexts" / context_name(spec)
            if completed(folder, spec, frozen):
                continue
            old = verify_saved_source(spec)
            if old["train_rows"] != train.tolist() or old["asymmetric_query"] != asymmetric.tolist():
                raise ValueError("Historical synthetic rows differ")
            if old["tied_query"] != tied.tolist():
                raise ValueError("Historical actual tied query differs")
            truth_json = json.loads((saved_source(spec) / "truth.json").read_text())
            truth = round2.restore(truth_json)
            clean = truth.predict(early, late)
            observed = clean + np.random.default_rng(spec["noise_seed"]).normal(0, spec["noise_sd"], len(clean))
            folder.mkdir(parents=True, exist_ok=True)
            np.savez(
                folder / "labels.npz",
                clean=clean,
                observed=observed,
                train=train,
                asymmetric=asymmetric,
                tied=tied,
                inner_labels=labels,
            )
            roles = np.full(len(clean), "unused", dtype=object)
            roles[train], roles[asymmetric], roles[tied] = "train", "test", "tied_diagnostic"
            inner = np.full(len(clean), -1)
            inner[train] = labels
            pd.DataFrame(
                {
                    "row_index": np.arange(len(clean)),
                    "role": roles,
                    "inner_fold": inner,
                    "clean": clean,
                    "observed": observed,
                }
            ).to_csv(folder / "labels.csv", index=False)
            benchmark.write_json(folder / "truth.json", truth_json)
            finish(
                folder,
                spec,
                ("labels.npz", "labels.csv", "truth.json"),
                frozen,
                source_receipt=str(saved_source(spec) / "complete.json"),
            )
            print(f"Prepared {context_name(spec)}: 172 train / 66 asymmetric / 42 tied", flush=True)


def load_context(spec: dict[str, Any], frozen: dict[str, str]) -> Context:
    folder = OUTPUT / "contexts" / context_name(spec)
    if not completed(folder, spec, frozen):
        raise ValueError("Run prepare before scientific fits")
    with np.load(prior.OUTPUT / "data/arrays.npz") as geometry, np.load(folder / "labels.npz") as labels:
        return Context(
            context_name(spec),
            spec["world"],
            spec["noise_sd"],
            spec["noise_seed"],
            geometry["phase0_epochs"],
            geometry["phase1_epochs"],
            float(geometry["alpha"]),
            geometry["pool_tokens"],
            labels["train"],
            labels["asymmetric"],
            labels["tied"],
            labels["inner_labels"],
            labels["clean"],
            labels["observed"],
            round2.restore(json.loads((folder / "truth.json").read_text())),
        )


def search_absolute_floor(matrix: np.ndarray, observed: np.ndarray, labels: np.ndarray, ridge: float) -> dict[str, Any]:
    """Replay the NG2-016/017 search schedule with current native NNLS heads."""
    folds = temporal.inner_splits(labels)
    cache: dict[float, tuple[float, list[dict[str, Any]]]] = {}
    trace: list[dict[str, Any]] = []
    offset = np.zeros(len(observed))

    def evaluate(floor: float, stage: str, stage_index: int | None = None) -> float:
        floor = float(floor)
        if not FLOOR_BOUNDS[0] <= floor <= FLOOR_BOUNDS[1]:
            raise ValueError("Absolute-floor search left its registered fixed domain")
        cached = floor in cache
        if not cached:
            records = [
                {
                    "fold_index": index,
                    "training_rows": len(train),
                    "validation_rows": len(held),
                    "training_minimum": float(np.min(observed[train])),
                    "admissible": bool(floor <= np.min(observed[train])),
                    "deficit_clamp_count": int(
                        np.count_nonzero(observed[train] - floor < temporal.mariner.DEFICIT_FLOOR)
                    ),
                    "validation_sse": None,
                }
                for index, (train, held) in enumerate(folds)
            ]
            score = float("inf")
            if all(record["admissible"] for record in records):
                for record, (train, held) in zip(records, folds, strict=True):
                    head = joint.fit_fixed_head(matrix[train], offset[train], observed[train], ridge, None, floor)
                    prediction = head.predict(matrix[held])
                    error = float(np.sum(np.square(prediction - observed[held])))
                    if not math.isfinite(error):
                        raise FloatingPointError("Nonfinite conditional BPB error")
                    record["validation_sse"] = error
                score = math.sqrt(sum(record["validation_sse"] for record in records) / len(observed))
            cache[floor] = score, records
        score, records = cache[floor]
        trace.append(
            {
                "floor": floor,
                "inner_rmse": score if math.isfinite(score) else None,
                "folds": records,
                "stage": stage,
                "stage_index": stage_index,
                "cached": cached,
            }
        )
        return score

    coarse = [evaluate(floor, "coarse") for floor in FLOOR_GRID]
    transitions: list[dict[str, Any]] = []
    transition_bounds: dict[tuple[float, float], float] = {}
    for index in range(len(FLOOR_GRID) - 1):
        if math.isfinite(coarse[index]) and not math.isfinite(coarse[index + 1]):
            lower, upper = FLOOR_GRID[index : index + 2]
            finite, invalid = lower, upper
            for _ in range(TRANSITION_BISECTIONS):
                middle = (finite + invalid) / 2
                if math.isfinite(evaluate(middle, "transition", len(transitions))):
                    finite = middle
                else:
                    invalid = middle
            transitions.append(
                {
                    "coarse_lower": lower,
                    "coarse_upper": upper,
                    "final_finite": finite,
                    "final_inadmissible": invalid,
                    "bisections": TRANSITION_BISECTIONS,
                }
            )
            transition_bounds[lower, upper] = finite
    brackets = []
    for index in range(1, len(FLOOR_GRID) - 1):
        if math.isfinite(coarse[index]) and coarse[index] <= coarse[index - 1] and coarse[index] <= coarse[index + 1]:
            lower, upper = FLOOR_GRID[index - 1], FLOOR_GRID[index + 1]
            if not math.isfinite(coarse[index + 1]):
                upper = transition_bounds[FLOOR_GRID[index], upper]
            brackets.append((lower, upper))
    for left, right in ((0, 1), (len(FLOOR_GRID) - 2, len(FLOOR_GRID) - 1)):
        if math.isfinite(coarse[left]) and math.isfinite(coarse[right]):
            brackets.append((FLOOR_GRID[left], FLOOR_GRID[right]))
    brackets.extend((row["coarse_lower"], row["final_finite"]) for row in transitions)
    refinements = []
    seen: set[tuple[float, float]] = set()
    duplicates = 0
    for bracket_index, (lower, upper) in enumerate(dict.fromkeys(brackets)):
        if lower == upper:
            continue
        endpoints = np.linspace(lower, upper, REFINEMENT_SUBINTERVALS + 1)
        for endpoint in endpoints:
            evaluate(float(endpoint), "refinement_endpoint", bracket_index)
        for subinterval_index in range(REFINEMENT_SUBINTERVALS):
            start, stop = float(endpoints[subinterval_index]), float(endpoints[subinterval_index + 1])
            if (start, stop) in seen:
                duplicates += 1
                continue
            seen.add((start, stop))
            result = minimize_scalar(
                functools.partial(evaluate, stage="refinement", stage_index=len(refinements)),
                bounds=(start, stop),
                method="bounded",
                options={"xatol": REFINEMENT_XATOL, "maxiter": REFINEMENT_MAXITER},
            )
            refinements.append(
                {
                    "bracket_index": bracket_index,
                    "subinterval_index": subinterval_index,
                    "lower_floor": start,
                    "upper_floor": stop,
                    "floor": float(result.x),
                    "inner_rmse": float(result.fun),
                    "success": bool(result.success),
                    "status": int(result.status),
                    "message": str(result.message),
                    "evaluations": int(result.nfev),
                }
            )
    floor = min(cache, key=lambda value: (cache[value][0], value))
    if not math.isfinite(cache[floor][0]):
        raise ValueError("No admissible finite absolute floor was evaluated")
    return {
        "floor": floor,
        "inner_rmse": cache[floor][0],
        "trace": trace,
        "transitions": transitions,
        "refinements": refinements,
        "duplicate_subintervals_skipped": duplicates,
        "unique_evaluations": len(cache),
        "floor_bounds": FLOOR_BOUNDS,
        "coarse_grid": FLOOR_GRID,
        "bound_hits": [name for name, value in zip(("lower", "upper"), FLOOR_BOUNDS, strict=True) if floor == value],
    }


def fit_common_floor(context: Context, model: str) -> joint.JointFit:
    """Diagnose one learned absolute floor with the true shape supplied."""
    train, shape = context.train, context.truth.shape
    profile = []
    best = None
    for parameter_index, parameters in enumerate(joint.parameter_grid(model)):
        matrix, offset = joint.design_and_offset(
            context.early[train], context.late[train], context.alpha, shape, model, parameters
        )
        if np.any(offset):
            raise ValueError("This registered absolute-floor diagnostic only supports zero-offset laws")
        for ridge_index, ridge in enumerate(temporal.mariner.RIDGE_GRID):
            result = search_absolute_floor(matrix, context.observed[train], context.labels, ridge)
            profile.append(
                {
                    "stage": "common_absolute_floor",
                    "shape_index": 0,
                    "parameter_index": parameter_index,
                    "ridge_index": ridge_index,
                    "parameters": parameters,
                    "ridge": ridge,
                    **result,
                }
            )
            candidate = result["inner_rmse"], parameter_index, ridge_index
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    selected = profile[best[1] * len(temporal.mariner.RIDGE_GRID) + best[2]]
    parameters, ridge, floor = selected["parameters"], selected["ridge"], selected["floor"]
    matrix, offset = joint.design_and_offset(
        context.early[train], context.late[train], context.alpha, shape, model, parameters
    )
    head = joint.fit_fixed_head(matrix, offset, context.observed[train], ridge, None, floor)
    return joint.JointFit(
        model,
        context.alpha,
        shape,
        parameters,
        ridge,
        None,
        floor,
        best[0],
        head,
        tuple(profile),
        {
            "shapes": [shape],
            "parameters": list(joint.parameter_grid(model)),
            "ridges": list(temporal.mariner.RIDGE_GRID),
            "floor_bounds": list(FLOOR_BOUNDS),
            "floor_contract": "learned common absolute floor; true-shape diagnostic only",
            "nearest_prior": ["NG2-016", "NG2-017"],
            "conditional_solver": "current native NNLS",
        },
    )


def restore(value: dict[str, Any]) -> Fit:
    return joint.JointFit.from_json(value) if "search_space" in value else round2.restore(value)


def cell_folder(context: Context, estimator: str, family: str) -> Path:
    return OUTPUT / "contexts" / context.name / "cells" / f"{estimator}__{family}"


def load_cell(context: Context, estimator: str, family: str, frozen: dict[str, str]) -> Fit:
    folder = cell_folder(context, estimator, family)
    spec = {"context": context.name, "estimator": estimator, "family": family}
    if not completed(folder, spec, frozen):
        raise ValueError(f"Required predecessor is incomplete: {folder}")
    return restore(json.loads((folder / "model.json").read_text()))


def fit_cell(context: Context, estimator: str, family: str, frozen: dict[str, str]) -> tuple[Fit, dict[str, Any]]:
    train = context.train
    arguments = context.early[train], context.late[train], context.observed[train], context.labels
    if estimator in ("native", "ridge_gamma_refined"):
        if context.noise_seed == round2.SEED:
            spec = {"world": context.world, "noise_sd": context.noise_sd}
            verify_saved_source(spec)
            source = saved_source(spec) / "models.json"
            fit = round2.restore(json.loads(source.read_text())[f"{estimator}__{family}"])
            return fit, {"mode": "reused frozen fit", "path": str(source), "sha256": benchmark.sha256(source)}
        if family == "aggregate_mariner":
            if estimator == "native":
                return temporal.fit_aggregate(*arguments, context.alpha, context.pool_tokens), {"mode": "fresh native"}
            aggregate = load_cell(context, "native", family, frozen)
            assert isinstance(aggregate, temporal.TemporalFit)
            return mechanisms.refine_ridge_gamma(*arguments, aggregate), {"mode": "fresh ridge/gamma refinement"}
        aggregate = load_cell(context, estimator, "aggregate_mariner", frozen)
        assert isinstance(aggregate, temporal.TemporalFit)
        return mechanisms.fit_extension(*arguments, aggregate, family), {"mode": "fresh conditional kinetics"}
    null = load_cell(context, estimator, "aggregate_mariner", frozen) if family != "aggregate_mariner" else None
    if null is not None:
        assert isinstance(null, joint.JointFit)
    if estimator == "oracle_true_shape_common_floor":
        return fit_common_floor(context, family), {"mode": "true shape supplied; learned common absolute floor"}
    if estimator == "j1_fixed_shape":
        shapes = (load_cell(context, "native", "aggregate_mariner", frozen).shape,)
    elif estimator in ("oracle_true_shape_true_floor", "oracle_true_shape_shared_gamma"):
        shapes = (context.truth.shape,)
    else:
        shapes = None
    if estimator in ("oracle_true_shape_true_floor", "oracle_all_shape_true_floor"):
        fit = joint.fit_known_floor(
            *arguments, context.alpha, family, floor=context.truth.head.floor, shapes=shapes, matched_null_fit=null
        )
    else:
        fit = joint.fit_joint(*arguments, context.alpha, family, shapes=shapes, matched_null_fit=null)
    return fit, {"mode": "new registered synthetic diagnostic"}


def score_fit(context: Context, fit: Fit) -> tuple[dict[str, Any], pd.DataFrame]:
    query = np.r_[context.asymmetric, context.tied]
    early, late = context.early[query], context.late[query]
    total = early + late
    tied_early, tied_late = context.alpha * total, (1 - context.alpha) * total
    predicted = fit.predict(early, late)
    diagonal = fit.predict(tied_early, tied_late)
    true_diagonal = context.truth.predict(tied_early, tied_late)
    clean, observed = context.clean[query], context.observed[query]
    gain, true_gain = diagonal - predicted, true_diagonal - clean
    a, t = np.arange(len(context.asymmetric)), np.arange(len(context.asymmetric), len(query))
    selected, selected_joint = int(np.argmin(predicted[a])), int(np.argmin(predicted))
    score: dict[str, Any] = {
        "asymmetric_test_n": len(a),
        "actual_tied_diagnostic_n": len(t),
        "clean_asymmetric_rmse": float(np.sqrt(np.mean(np.square(predicted[a] - clean[a])))),
        "observed_asymmetric_rmse": float(np.sqrt(np.mean(np.square(predicted[a] - observed[a])))),
        "clean_actual_tied_rmse": float(np.sqrt(np.mean(np.square(predicted[t] - clean[t])))),
        "observed_actual_tied_rmse": float(np.sqrt(np.mean(np.square(predicted[t] - observed[t])))),
        "clean_counterfactual_tied_rmse": float(np.sqrt(np.mean(np.square(diagonal[a] - true_diagonal[a])))),
        "clean_phase_gain_rmse": float(np.sqrt(np.mean(np.square(gain[a] - true_gain[a])))),
        "max_abs_predicted_phase_gain": float(np.max(abs(gain[a]))),
        "max_abs_false_phase_gain_null": float(np.max(abs(gain[a]))) if context.world == "aggregate_null" else None,
        "clean_regret_at1": float(clean[selected] - np.min(clean[a])),
        "observed_regret_at1": float(observed[selected] - np.min(observed[a])),
        "clean_joint_regret_at1": float(clean[selected_joint] - np.min(clean)),
        "clean_selected_optimism": float(clean[selected] - predicted[selected]),
        "observed_selected_optimism": float(observed[selected] - predicted[selected]),
        "selected_predicted_gain": float(gain[selected]),
        "selected_true_gain": float(true_gain[selected]),
        "selected_row": int(query[selected]),
        "joint_selected_row": int(query[selected_joint]),
        "shape": json.dumps(fit.shape, sort_keys=True),
        "parameters": json.dumps(fit.parameters, sort_keys=True),
        "ridge": fit.ridge,
        "gamma": fit.kappa,
        "final_absolute_floor": fit.head.floor,
    }
    predictions = pd.DataFrame(
        {
            "row_index": query,
            "role": ["test"] * len(a) + ["tied_diagnostic"] * len(t),
            "clean": clean,
            "observed": observed,
            "predicted": predicted,
            "clean_tied_counterfactual": true_diagonal,
            "predicted_tied_counterfactual": diagonal,
            "clean_phase_gain": true_gain,
            "predicted_phase_gain": gain,
        }
    )
    return score, predictions


def inner_predictions(context: Context, fit: Fit) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    train = context.train
    matrix, offset = joint.design_and_offset(
        context.early[train], context.late[train], context.alpha, fit.shape, fit.model, fit.parameters
    )
    known_floor = fit.known_floor if isinstance(fit, joint.JointFit) else None
    predictions = np.empty(len(train))
    floors = np.empty(len(train))
    heads = []
    for fold, (left, right) in enumerate(temporal.inner_splits(context.labels)):
        head = joint.fit_fixed_head(
            matrix[left], offset[left], context.observed[train[left]], fit.ridge, fit.kappa, known_floor
        )
        predictions[right] = temporal.predict_head(head, matrix[right], offset[right])
        floors[right] = head.floor
        heads.append(
            {
                "fold": fold,
                "train_rows": train[left].tolist(),
                "held_rows": train[right].tolist(),
                "intercept": head.intercept,
                "coefficients": head.coefficients.tolist(),
                "floor": head.floor,
            }
        )
    score = float(np.sqrt(np.mean(np.square(predictions - context.observed[train]))))
    expected = fit.inner_rmse if isinstance(fit, joint.JointFit) else fit.conditional_inner_rmse
    if not np.isclose(score, expected, atol=1e-12, rtol=1e-10):
        raise ValueError(f"Selected inner predictions do not replay score: {score} vs {expected}")
    return (
        pd.DataFrame(
            {
                "row_index": train,
                "inner_fold": context.labels,
                "clean": context.clean[train],
                "observed": context.observed[train],
                "predicted": predictions,
                "floor": floors,
            }
        ),
        heads,
    )


def competing_inner_predictions(context: Context, fit: Fit) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replay the best visited null and nonzero states, including losing states."""
    if fit.model == "aggregate_mariner":
        frame, heads = inner_predictions(context, fit)
        return frame.assign(competition_member="best_zero"), {"best_zero": heads, "best_nonzero": "absent from family"}
    frames, evidence = [], {}
    for nonzero in (False, True):
        name = "best_nonzero" if nonzero else "best_zero"
        if isinstance(fit, joint.JointFit):
            eligible = [
                item
                for item in fit.selection_profile
                if (item["parameter_index"] > 0) == nonzero and item["inner_rmse"] is not None
            ]
            selected = min(
                eligible,
                key=lambda item: (
                    item["inner_rmse"],
                    item["parameter_index"],
                    item["shape_index"],
                    item["ridge_index"],
                    item.get("gamma") or 0.0,
                ),
            )
            floor = selected["floor"] if selected["stage"] == "common_absolute_floor" else fit.known_floor
            candidate = dataclasses.replace(
                fit,
                shape=fit.search_space["shapes"][selected["shape_index"]],
                parameters=fit.search_space["parameters"][selected["parameter_index"]],
                ridge=fit.search_space["ridges"][selected["ridge_index"]],
                kappa=selected.get("gamma"),
                known_floor=floor,
                inner_rmse=selected["inner_rmse"],
            )
        else:
            eligible = [item for item in fit.selection_profile if bool(any(item["parameters"].values())) == nonzero]
            selected = min(eligible, key=lambda item: item["conditional_inner_rmse"])
            candidate = dataclasses.replace(
                fit, parameters=selected["parameters"], conditional_inner_rmse=selected["conditional_inner_rmse"]
            )
        frame, heads = inner_predictions(context, candidate)
        frames.append(frame.assign(competition_member=name))
        evidence[name] = {
            "shape": candidate.shape,
            "parameters": candidate.parameters,
            "ridge": candidate.ridge,
            "gamma": candidate.kappa,
            "heads": heads,
        }
    return pd.concat(frames, ignore_index=True), evidence


def run_context(spec: dict[str, Any], estimators: tuple[str, ...] = ESTIMATORS) -> dict[str, Any]:
    frozen = execution_identity()
    context = load_context(spec, frozen)
    with threadpool_limits(limits=1):
        for estimator in estimators:
            for family in FAMILIES:
                cell = {"context": context.name, "estimator": estimator, "family": family}
                folder = cell_folder(context, estimator, family)
                if completed(folder, cell, frozen):
                    print(f"Cached {context.name} {estimator} {family}", flush=True)
                    continue
                print(f"Starting {context.name} {estimator} {family}", flush=True)
                started = time.monotonic()
                execution_identity()
                fit, provenance = fit_cell(context, estimator, family, frozen)
                scores, predictions = score_fit(context, fit)
                inner, heads = inner_predictions(context, fit)
                competing, competition = competing_inner_predictions(context, fit)
                folder.mkdir(parents=True, exist_ok=True)
                model_json = fit.to_json()
                model_json["round3_estimator"] = estimator
                model_json["round3_provenance"] = provenance
                if estimator == "oracle_true_shape_common_floor":
                    model_json["calibration_contract"] = "learned common absolute floor; true-shape oracle diagnostic"
                benchmark.write_json(folder / "model.json", model_json)
                benchmark.write_json(folder / "scores.json", {**spec, **cell, **scores})
                benchmark.write_json(folder / "inner_heads.json", heads)
                benchmark.write_json(folder / "competing_inner_heads.json", competition)
                predictions.to_csv(folder / "predictions.csv", index=False)
                inner.to_csv(folder / "inner_predictions.csv", index=False)
                competing.to_csv(folder / "competing_inner_predictions.csv", index=False)
                finish(
                    folder,
                    cell,
                    (
                        "model.json",
                        "scores.json",
                        "predictions.csv",
                        "inner_predictions.csv",
                        "inner_heads.json",
                        "competing_inner_predictions.csv",
                        "competing_inner_heads.json",
                    ),
                    frozen,
                    elapsed_seconds=time.monotonic() - started,
                    context_receipt_sha256=benchmark.sha256(OUTPUT / "contexts" / context.name / "complete.json"),
                )
                print(
                    f"Finished {context.name} {estimator} {family}: {time.monotonic() - started:.1f}s; "
                    f"clean asym {scores['clean_asymmetric_rmse']:.6f}, tied {scores['clean_actual_tied_rmse']:.6f}",
                    flush=True,
                )
    return {"context": context.name, "cells": len(estimators) * len(FAMILIES)}


def preflight() -> None:
    """Gate full diagnostics on current-helper noiseless representability."""
    frozen = execution_identity()
    folder = OUTPUT / "preflight"
    gate_context = {"stage": "known_shape_known_floor_recovery"}
    if completed(folder, gate_context, frozen):
        require_preflight()
        return
    estimator = "oracle_true_shape_true_floor"
    records, receipts = [], {}
    for spec in registered_contexts():
        if spec["noise_sd"] != 0:
            continue
        run_context(spec, (estimator,))
        context = load_context(spec, frozen)
        for family in FAMILIES:
            cell = cell_folder(context, estimator, family)
            receipts[str(cell / "complete.json")] = benchmark.sha256(cell / "complete.json")
            if context.world == "odds" and family == "aggregate_mariner":
                continue
            predictions = pd.read_csv(cell / "predictions.csv")
            errors = [
                float(np.max(abs(predictions.predicted - predictions.clean))),
                float(np.max(abs(predictions.predicted_tied_counterfactual - predictions.clean_tied_counterfactual))),
                float(np.max(abs(predictions.predicted_phase_gain - predictions.clean_phase_gain))),
            ]
            records.append(
                {"context": context.name, "family": family, "max_errors": errors, "passed": max(errors) < 1e-8}
            )
    folder.mkdir(parents=True, exist_ok=True)
    benchmark.write_json(folder / "recovery.json", {"records": records, "cell_receipts": receipts, "tolerance": 1e-8})
    if len(records) != 3 or not all(record["passed"] for record in records):
        raise ValueError("Current-helper exact oracle preflight failed; scientific diagnostic run is blocked")
    finish(folder, gate_context, ("recovery.json",), frozen)
    print("Preflight passed: all three in-family noiseless oracle checks below 1e-8", flush=True)


def require_preflight() -> None:
    frozen = execution_identity()
    folder = OUTPUT / "preflight"
    if not completed(folder, {"stage": "known_shape_known_floor_recovery"}, frozen):
        raise ValueError("Run preflight before the full synthetic diagnostics")
    report = json.loads((folder / "recovery.json").read_text())
    for path, digest in report["cell_receipts"].items():
        if benchmark.sha256(Path(path)) != digest:
            raise ValueError("Preflight fit receipt changed")


def summarize() -> None:
    frozen = execution_identity()
    rows, inner, receipts = [], [], {}
    for spec in registered_contexts():
        context = load_context(spec, frozen)
        for estimator in ESTIMATORS:
            for family in FAMILIES:
                folder = cell_folder(context, estimator, family)
                cell = {"context": context.name, "estimator": estimator, "family": family}
                if not completed(folder, cell, frozen):
                    raise ValueError(f"Cannot summarize incomplete cell: {folder}")
                rows.append(json.loads((folder / "scores.json").read_text()))
                frame = pd.read_csv(folder / "inner_predictions.csv")
                inner.append(frame.assign(**cell))
                receipts[str(folder / "complete.json")] = benchmark.sha256(folder / "complete.json")
    pd.DataFrame(rows).to_csv(OUTPUT / "scores.csv", index=False)
    pd.concat(inner, ignore_index=True).to_csv(OUTPUT / "selected_inner_predictions.csv", index=False)
    benchmark.write_json(
        OUTPUT / "summary_manifest.json",
        {
            "identity": frozen,
            "cell_receipts": receipts,
            "outputs": {
                name: benchmark.sha256(OUTPUT / name) for name in ("scores.csv", "selected_inner_predictions.csv")
            },
            "claim": "Six synthetic diagnostic contexts only; no qualification or real-outcome fit.",
        },
    )
    print(f"Summarized {len(rows)} estimator cells", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "run", "summarize"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--context", choices=[context_name(spec) for spec in registered_contexts()])
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
        return
    if args.command == "summarize":
        summarize()
        return
    if args.command == "preflight":
        preflight()
        return
    execution_identity()
    require_preflight()
    contexts = [spec for spec in registered_contexts() if args.context is None or context_name(spec) == args.context]
    if args.workers < 1:
        raise ValueError("Worker count must be positive")
    if args.workers == 1:
        for spec in contexts:
            print(json.dumps(run_context(spec), sort_keys=True), flush=True)
        return
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_context, spec) for spec in contexts]
        for future in as_completed(futures):
            print(json.dumps(future.result(), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
