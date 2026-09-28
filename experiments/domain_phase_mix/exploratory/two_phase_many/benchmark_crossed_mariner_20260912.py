# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["fsspec==2026.1.0", "numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Benchmark shared MARINER continuation fits with prefix and action exclusion.

All inputs are frozen local archives. The module has no network operations.
Fits preserve a single intercept, training-only floors, and complete hashes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from fit_two_phase_link_spines_20260907 import load_module, write_json_atomic
from fixed_checkpoint_wspu_models_20260907 import (
    Variant,
    fit_conditional_head,
    head_prediction,
    response_source,
)
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent
INPUT = BASE / "reference_outputs/fixed_checkpoint_branch_wspu_20260907"
OUTPUT = BASE / "reference_outputs/two_phase_mariner_transfer_20260912/crossed"
AUTHORITY = Path("/Users/calvinxu/Projects/Work/Marin/mixture-selection/mixture_selection.py")
ARCHIVE = Path("/Users/calvinxu/Projects/Work/Marin/worktree-archives/20260905-dirty-worktrees")
DSP_SOURCE = (
    ARCHIVE
    / "marin-delphi-y0-y1-surrogate-20260827/files/experiments/domain_phase_mix/exploratory/two_phase_many"
    / "benchmark_delphi_y0_y1_mechanistic_20260827.py"
)
STATES = (
    "shared_bounded_ensemble_kl0p05",
    "shared_bounded_ensemble_kl0p2",
    "shared_bounded_ensemble_kl0p5",
    "cap4_shared_bounded_ensemble_kl0",
    "cap4_shared_bounded_ensemble_kl0p05",
    "cap4_shared_bounded_ensemble_kl0p2",
)
MODELS = {
    "MTP-001": "continuation_log",
    "MTP-002": "cumulative_increment_log",
    "MTP-003": "cumulative_total_log",
    "MTP-004": "cumulative_increment_bpb",
    "MTP-005": "fixed_dsp_increment_bpb",
}
ALPHA = 2400 / 3007
REPRESENTATIVE_SHAPE = {"rate": 0.5, "power": 0.7, "threshold": 3.0}
THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


@dataclass(frozen=True)
class FitJob:
    fold: str
    model: str
    output: str


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    frame = pd.read_csv(INPUT / "data/rows.csv")
    with np.load(INPUT / "data/arrays.npz") as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    assert np.array_equal(frame.row.to_numpy(), np.arange(len(frame)))
    assert np.allclose(frame.target, arrays["target"], rtol=0, atol=1e-14)
    return frame, arrays


@lru_cache(maxsize=4)
def dsp_primitives(path: str, sha256: str) -> ModuleType:
    """Load only two local archived numerical functions, without operational imports."""
    source = Path(path)
    assert digest(source) == sha256
    tree = ast.parse(source.read_text())
    definitions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"benefit", "damage"}
    ]
    assert {node.name for node in definitions} == {"benefit", "damage"}
    module = ModuleType(f"crossed_dsp_{sha256[:12]}")
    module.__dict__["np"] = np
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), module.__dict__)
    return module


def design_matrix(
    prefix_epochs: np.ndarray,
    continuation_epochs: np.ndarray,
    shape: dict[str, float],
    variant: str,
    source: ModuleType,
    dsp_path: Path,
) -> np.ndarray:
    """Build shared branch columns from physical prefix and continuation exposures."""
    continuation = np.atleast_2d(np.asarray(continuation_epochs, float))
    prefix = np.broadcast_to(np.asarray(prefix_epochs, float), continuation.shape)
    if continuation.shape[1] != 39 or np.any(prefix < 0) or np.any(continuation < 0):
        raise ValueError("expected nonnegative N by 39 exposures")
    if variant == "fixed_dsp_increment_bpb":
        dsp = dsp_primitives(str(dsp_path), digest(dsp_path))
        benefit = dsp.benefit(prefix + continuation, 0.3) - dsp.benefit(prefix, 0.3)
        damage = dsp.damage(prefix + continuation, 1.0) - dsp.damage(prefix, 1.0)
        return np.hstack([-benefit, damage])
    if variant == "continuation_log":
        return source.design_matrix(continuation, shape) - source.design_matrix(np.zeros_like(continuation), shape)
    if variant == "cumulative_total_log":
        return source.design_matrix(prefix + continuation, shape)
    if variant not in {"cumulative_increment_log", "cumulative_increment_bpb"}:
        raise ValueError(variant)
    return source.design_matrix(prefix + continuation, shape) - source.design_matrix(prefix, shape)


def predict_fit(
    fit: dict[str, Any], prefix_epochs: np.ndarray, continuation_epochs: np.ndarray, source_path: Path
) -> np.ndarray:
    """Predict from a saved shared fit, without any state-specific terminal anchor."""
    source, sha256 = response_source(source_path)
    assert sha256 == fit["source_sha256"]
    matrix = design_matrix(
        prefix_epochs, continuation_epochs, fit["shape"], fit["variant"], source, Path(fit["dsp_source_path"])
    )
    latent = fit["intercept"] + matrix @ np.asarray(fit["coefficients"])
    if fit["floor"] is None:
        return latent
    return fit["floor"] + np.exp(np.clip(latent, -source.LOG_CLIP, source.LOG_CLIP))


def action_groups(rows: np.ndarray, count: int, frame: pd.DataFrame, legacy: pd.DataFrame) -> dict[str, int]:
    """Reuse the archived outcome-blind geometric partition on unique action coordinates."""
    unique = frame.iloc[rows].drop_duplicates("coordinate_hash").sort_values("coordinate_hash")
    audit = load_module(INPUT / "sources/audit_delphi_phase1_branch_response_20260826.py", "crossed_geometry")
    labels = audit.geometric_folds(legacy.iloc[unique.row.to_numpy(int)], folds=count)
    return dict(zip(unique.coordinate_hash.astype(str), labels.tolist(), strict=True))


def prepare(output: Path) -> dict[str, Any]:
    """Freeze sources and construct target-independent outer and inner exclusions."""
    output.mkdir(parents=True, exist_ok=True)
    sources = output / "sources"
    sources.mkdir(exist_ok=True)
    for source in [AUTHORITY, DSP_SOURCE]:
        destination = sources / source.name
        if destination.exists():
            assert digest(destination) == digest(source), f"source changed: {source}"
        else:
            shutil.copy2(source, destination)
    frame, arrays = load_data()
    legacy = pd.read_csv(INPUT / "data/audit_frame.csv")
    assert legacy.row_id.equals(frame.row_id)
    ordinary = frame.state_id.isin(STATES)
    broad = ordinary & frame.panel.eq("crossed_broad")
    local = ordinary & frame.panel.eq("crossed_local")
    training_mask = broad & (frame.fit_budget | frame.is_tied_control)
    testing_mask = local & (frame.fit_budget | frame.is_tied_control)
    training_all = np.flatnonzero(training_mask)
    testing_all = np.flatnonzero(testing_mask)
    assert len(training_all) == 6 * 51 and len(testing_all) == 6 * 11
    all_rows = np.r_[training_all, testing_all]
    assert np.isfinite(arrays["component_bpb"][all_rows]).all()
    component_error = arrays["component_bpb"][all_rows] @ arrays["component_weights"] - arrays["target"][all_rows]
    assert np.max(np.abs(component_error)) < 2e-7
    # Frozen arrays already include the realized phase lengths and materialized pools.
    assert set(frame.iloc[all_rows].phase_fraction_source) == {"realized_2400_of_3007_updates"}
    assert np.allclose(
        arrays["tied_phase1_epochs"][all_rows],
        arrays["phase0_epochs"][all_rows] * (1 - ALPHA) / ALPHA,
        rtol=1e-8,
        atol=1e-8,
    )
    test_hashes = set(frame.loc[local & frame.fit_budget, "coordinate_hash"])
    original_training = training_all.copy()
    training_all = np.asarray([row for row in training_all if frame.coordinate_hash.iloc[row] not in test_hashes], int)
    contexts = {}
    for held in (*STATES, "seen_prefix"):
        train = training_all if held == "seen_prefix" else training_all[frame.state_id.iloc[training_all].ne(held)]
        test = testing_all if held == "seen_prefix" else testing_all[frame.state_id.iloc[testing_all].eq(held)]
        if held != "seen_prefix":
            forbidden = set(frame.coordinate_hash.iloc[test])
            train = np.asarray([row for row in train if frame.coordinate_hash.iloc[row] not in forbidden], int)
        actions = train[~frame.is_tied_control.iloc[train].to_numpy()]
        groups = action_groups(actions, 3, frame, legacy)
        remaining_states = [state for state in STATES if state != held]
        prefix_group = {state: index % 3 for index, state in enumerate(remaining_states)}
        row_groups = np.asarray([groups.get(str(frame.coordinate_hash.iloc[row]), -1) for row in train])
        row_states = np.asarray([prefix_group[str(frame.state_id.iloc[row])] for row in train])
        calibration = frame.is_tied_control.iloc[train].to_numpy(bool)
        inner = []
        prefix_folds = range(3) if held != "seen_prefix" else [None]
        for prefix_fold in prefix_folds:
            for action_fold in range(3):
                train_prefix = np.ones(len(train), bool) if prefix_fold is None else row_states != prefix_fold
                valid_prefix = np.ones(len(train), bool) if prefix_fold is None else row_states == prefix_fold
                inner_train = np.flatnonzero(train_prefix & ((row_groups != action_fold) | calibration))
                inner_valid = np.flatnonzero(valid_prefix & (row_groups == action_fold) & ~calibration)
                assert len(inner_train) and len(inner_valid) and calibration[inner_train].any()
                assert not set(inner_train).intersection(inner_valid)
                assert not set(frame.coordinate_hash.iloc[train[inner_train]]).intersection(
                    frame.coordinate_hash.iloc[train[inner_valid]]
                )
                if prefix_fold is not None:
                    assert not set(frame.state_id.iloc[train[inner_train]]).intersection(
                        frame.state_id.iloc[train[inner_valid]]
                    )
                inner.append({"train": inner_train.tolist(), "validation": inner_valid.tolist()})
        test_actions = test[~frame.is_tied_control.iloc[test].to_numpy(bool)]
        assert not set(frame.coordinate_hash.iloc[train]).intersection(frame.coordinate_hash.iloc[test_actions])
        if held != "seen_prefix":
            assert held not in set(frame.state_id.iloc[train])
            assert not set(frame.coordinate_hash.iloc[train]).intersection(frame.coordinate_hash.iloc[test])
        contexts[held] = {
            "train": train.tolist(),
            "test": test.tolist(),
            "predict": testing_all.tolist(),
            "calibration": calibration.tolist(),
            "inner_folds": inner,
            "action_groups": groups,
            "prefix_groups": prefix_group,
        }
    identity_paths = [
        Path(__file__),
        output / "PROTOCOL.md",
        sources / AUTHORITY.name,
        sources / DSP_SOURCE.name,
        INPUT / "data/arrays.npz",
        INPUT / "data/rows.csv",
        INPUT / "data/audit_frame.csv",
        INPUT / "data/manifest.json",
        INPUT / "sources/mixture_selection.py",
        INPUT / "sources/audit_delphi_phase1_branch_response_20260826.py",
        BASE / "fixed_checkpoint_wspu_models_20260907.py",
        BASE / "fit_two_phase_link_spines_20260907.py",
    ]
    identity = {str(path.resolve()): digest(path) for path in identity_paths}
    payload = {
        "hashes": identity,
        "contexts": contexts,
        "states": list(STATES),
        "models": MODELS,
        "removed_cross_panel_action_alias_rows": sorted(set(original_training) - set(training_all)),
        "component_aggregation_max_error": float(np.max(np.abs(component_error))),
        "source_sha256": digest(sources / AUTHORITY.name),
        "old_brw_source_sha256": digest(INPUT / "sources/mixture_selection.py"),
        "phase_fraction": ALPHA,
        "training_rows": training_all.tolist(),
        "testing_rows": testing_all.tolist(),
        "nonprimary_state_policy": "exclude observed prefix and replica/hardware-bridge states entirely",
        "continuation_data_seeds": {
            "broad": sorted(frame.loc[broad & frame.fit_budget, "data_seed"].unique().tolist()),
            "local": sorted(frame.loc[local & frame.fit_budget, "data_seed"].unique().tolist()),
        },
    }
    write_json_atomic(output / "prepared.json", payload)
    return payload


def fit_matrix_head(
    matrix: np.ndarray,
    y: np.ndarray,
    calibration: np.ndarray,
    ridge: float,
    variant: str,
    kappa: float,
    source: ModuleType,
):
    anchor = float(np.mean(y[calibration]))
    response_variant = Variant.CUMULATIVE_BPB if variant.endswith("_bpb") else Variant.CUMULATIVE_LOG
    return fit_conditional_head(matrix, y, ridge, response_variant, anchor, 0.0, kappa, source)


def cross_validation(
    matrix: np.ndarray,
    y: np.ndarray,
    calibration: np.ndarray,
    inner: list[dict[str, Any]],
    ridge: float,
    variant: str,
    kappa: float,
    source: ModuleType,
) -> float:
    squared_error = 0.0
    count = 0
    for fold in inner:
        train, valid = np.asarray(fold["train"], int), np.asarray(fold["validation"], int)
        head = fit_matrix_head(matrix[train], y[train], calibration[train], ridge, variant, kappa, source)
        predicted = head_prediction(matrix[valid], head, source)
        if not np.isfinite(predicted).all():
            return math.inf
        squared_error += float(np.sum((y[valid] - predicted) ** 2))
        count += len(valid)
    return math.sqrt(squared_error / count)


def fit_job(job: FitJob) -> dict[str, Any]:
    output = Path(job.output)
    prepared = json.loads((output / "prepared.json").read_text())
    destination = output / "fits" / job.fold / f"{job.model}.json"
    if destination.exists():
        cached = json.loads(destination.read_text())
        assert cached["input_hashes"] == prepared["hashes"]
        assert cached["fold"] == job.fold and cached["model"] == job.model
        return {"fold": job.fold, "model": job.model, "status": "cached", "elapsed": cached["elapsed"]}
    started = time.monotonic()
    _frame, arrays = load_data()
    context = prepared["contexts"][job.fold]
    train = np.asarray(context["train"], int)
    calibration = np.asarray(context["calibration"], bool)
    y = arrays["target"][train]
    source_path = output / "sources/mixture_selection.py"
    dsp_path = output / "sources" / DSP_SOURCE.name
    source, source_hash = response_source(source_path)
    variant = MODELS[job.model]
    shapes = [REPRESENTATIVE_SHAPE] if variant == "fixed_dsp_increment_bpb" else source.SHAPES
    best = (math.inf, -1, -1)
    scores = []
    for shape_index, shape in enumerate(shapes):
        matrix = design_matrix(
            arrays["phase0_epochs"][train], arrays["phase1_epochs"][train], shape, variant, source, dsp_path
        )
        for ridge_index, ridge in enumerate(source.RIDGE_GRID):
            score = cross_validation(
                matrix, y, calibration, context["inner_folds"], ridge, variant, source.KAPPA_PROVISIONAL, source
            )
            candidate = (score, shape_index, ridge_index)
            scores.append([shape_index, ridge_index, score])
            if candidate < best:
                best = candidate
    assert math.isfinite(best[0])
    shape = dict(shapes[best[1]])
    ridge = float(source.RIDGE_GRID[best[2]])
    matrix = design_matrix(
        arrays["phase0_epochs"][train], arrays["phase1_epochs"][train], shape, variant, source, dsp_path
    )
    kappa = source.KAPPA_PROVISIONAL
    search_records = []
    flat = False
    if not variant.endswith("_bpb"):

        def objective(log_kappa: float) -> float:
            value = cross_validation(
                matrix, y, calibration, context["inner_folds"], ridge, variant, math.exp(log_kappa), source
            )
            search_records.append([math.exp(log_kappa), value])
            return value if math.isfinite(value) else source.INFINITE_CV_PENALTY

        low, high = source.KAPPA_BOUNDS
        search = minimize_scalar(
            objective,
            bounds=(math.log(low), math.log(high)),
            method="bounded",
            options={"maxiter": source.KAPPA_SEARCH_EVALUATIONS, "xatol": source.KAPPA_SEARCH_XATOL},
        )
        kappa = float(math.exp(search.x))
        flat = math.log(kappa) >= (1 - source.FLAT_PROFILE_FRACTION) * math.log(high)
        if flat:
            kappa = source.FLAT_PROFILE_KAPPA
    head = fit_matrix_head(matrix, y, calibration, ridge, variant, kappa, source)
    fitted = {
        "model": job.model,
        "variant": variant,
        "shape": shape,
        "ridge": ridge,
        "kappa": None if variant.endswith("_bpb") else kappa,
        "flat_profile": flat,
        "intercept": head.intercept,
        "coefficients": head.coefficients.tolist(),
        "floor": head.floor,
        "source_sha256": source_hash,
        "dsp_source_path": str(dsp_path.resolve()),
        "anchor": float(y[calibration].mean()),
        "noise_sd": 0.0,
        "shape_grid_cv": scores,
        "kappa_search_cv": search_records,
        "inner_cv_rmse": cross_validation(matrix, y, calibration, context["inner_folds"], ridge, variant, kappa, source),
        "training_clamped_deficits": head.clamped_deficits,
    }
    predict_rows = np.asarray(context["predict"], int)
    predictions = predict_fit(
        fitted, arrays["phase0_epochs"][predict_rows], arrays["phase1_epochs"][predict_rows], source_path
    )
    replay = predict_fit(
        json.loads(json.dumps(fitted)),
        arrays["phase0_epochs"][predict_rows],
        arrays["phase1_epochs"][predict_rows],
        source_path,
    )
    assert np.isfinite(predictions).all() and np.array_equal(predictions, replay)
    prediction_matrix = design_matrix(
        arrays["phase0_epochs"][predict_rows], arrays["phase1_epochs"][predict_rows], shape, variant, source, dsp_path
    )
    latent = head.intercept + prediction_matrix @ head.coefficients
    fold_audit = []
    for inner in context["inner_folds"]:
        inner_train = np.asarray(inner["train"], int)
        inner_head = fit_matrix_head(
            matrix[inner_train], y[inner_train], calibration[inner_train], ridge, variant, kappa, source
        )
        fold_audit.append(
            {
                "train_rows": train[inner_train].tolist(),
                "validation_rows": train[np.asarray(inner["validation"], int)].tolist(),
                "floor": inner_head.floor,
                "anchor": float(y[inner_train][calibration[inner_train]].mean()),
                "clamped_deficits": inner_head.clamped_deficits,
            }
        )
    stored = {
        "fold": job.fold,
        "model": job.model,
        "input_hashes": prepared["hashes"],
        "fit": fitted,
        "training_rows": train.tolist(),
        "test_rows": context["test"],
        "prediction_rows": predict_rows.tolist(),
        "predicted": predictions.tolist(),
        "inner_folds": fold_audit,
        "held_outcomes_below_floor": (
            0 if head.floor is None else int(np.sum(arrays["target"][context["test"]] < head.floor))
        ),
        "prediction_guard_count": 0 if head.floor is None else int(np.sum(np.abs(latent) > source.LOG_CLIP)),
        "elapsed": time.monotonic() - started,
    }
    write_json_atomic(destination, stored)
    return {"fold": job.fold, "model": job.model, "status": "fit", "elapsed": stored["elapsed"]}


def singular_summary(matrix: np.ndarray) -> dict[str, Any]:
    centered = matrix - matrix.mean(axis=0)
    scale = np.linalg.norm(centered, axis=0)
    active = scale > 1e-12
    singular = np.linalg.svd(centered[:, active] / scale[active], compute_uv=False)
    positive = singular[singular > max(centered.shape) * np.finfo(float).eps * singular[0]]
    probability = singular / singular.sum()
    return {
        "columns": matrix.shape[1],
        "active_columns": int(active.sum()),
        "numerical_rank": len(positive),
        "effective_rank_singular_entropy": float(np.exp(-np.sum(probability * np.log(np.maximum(probability, 1e-300))))),
        "condition_positive_subspace": float(positive[0] / positive[-1]),
        "singular_values": singular.tolist(),
    }


def structural_checks(output: Path, prepared: dict[str, Any]) -> None:
    """Audit feature identification and noiseless held-policy recovery before outcome fits."""
    _frame, arrays = load_data()
    source, _ = response_source(output / "sources/mixture_selection.py")
    dsp_path = output / "sources" / DSP_SOURCE.name
    rows = np.asarray(prepared["training_rows"], int)
    matrices = {}
    reports = {}
    rng = np.random.default_rng(20260912)
    synthetic = []
    all_rows = np.r_[rows, prepared["testing_rows"]]
    for model, variant in MODELS.items():
        full = design_matrix(
            arrays["phase0_epochs"][all_rows],
            arrays["phase1_epochs"][all_rows],
            REPRESENTATIVE_SHAPE,
            variant,
            source,
            dsp_path,
        )
        matrix = full[: len(rows)]
        matrices[model] = matrix
        reports[model] = singular_summary(matrix)
        theta = rng.uniform(0.001, 0.01, size=78)
        theta /= np.maximum(1, np.max(np.abs(full), axis=0))
        latent = 0.1 + full @ theta
        truth = latent if variant.endswith("_bpb") else 0.8 + np.exp(latent)
        row_position = {row: index for index, row in enumerate(all_rows)}
        for state in STATES:
            context = prepared["contexts"][state]
            train = np.asarray([row_position[row] for row in context["train"]], int)
            test = np.asarray([row_position[row] for row in context["test"]], int)
            intercept, coefficients = source.nonnegative_solve(full[train], latent[train], 0.0)
            predicted_latent = intercept + full[test] @ coefficients
            prediction = predicted_latent if variant.endswith("_bpb") else 0.8 + np.exp(predicted_latent)
            error = prediction - truth[test]
            synthetic.append(
                {
                    "model": model,
                    "held_state": state,
                    "training_latent_rmse": float(
                        np.sqrt(np.mean((intercept + full[train] @ coefficients - latent[train]) ** 2))
                    ),
                    "held_rmse": float(np.sqrt(np.mean(error**2))),
                    "held_max_error": float(np.max(np.abs(error))),
                    "held_action_difference_rmse": float(np.sqrt(np.mean((error[:, None] - error[None, :]) ** 2))),
                    "prediction_recovered_at_1e_6": bool(np.max(np.abs(error)) < 1e-6),
                }
            )
    control = np.column_stack([np.ones(len(rows)), matrices["MTP-001"]])
    residual = matrices["MTP-002"] - control @ np.linalg.lstsq(control, matrices["MTP-002"], rcond=None)[0]
    centered = matrices["MTP-002"] - matrices["MTP-002"].mean(axis=0)
    write_json_atomic(
        output / "structural_checks.json",
        {
            "input_hashes": prepared["hashes"],
            "representative_shape": REPRESENTATIVE_SHAPE,
            "rank": reports,
            "increment_residual_after_continuation": singular_summary(residual),
            "increment_residual_frobenius_fraction": float(np.linalg.norm(residual) / np.linalg.norm(centered)),
            "synthetic": synthetic,
            "interpretation": (
                "Known-shape noiseless recovery separates feature nonidentification from outcome-fit failure. "
                "No new interaction is fitted."
            ),
        },
    )
    pd.DataFrame(synthetic).to_csv(output / "synthetic_recovery.csv", index=False)


def state_metrics(cell: pd.DataFrame) -> dict[str, Any]:
    actual = cell.target.to_numpy(float)
    prediction = cell.prediction.to_numpy(float)
    error = prediction - actual
    pick = int(np.argmin(prediction))
    shortlist = np.argsort(prediction, kind="stable")[:3]
    pairs = np.triu_indices(len(cell), 1)
    pair_error = (error[:, None] - error[None, :])[pairs]
    actual_difference = (actual[:, None] - actual[None, :])[pairs]
    predicted_difference = (prediction[:, None] - prediction[None, :])[pairs]
    return {
        "rows": len(cell),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mean_optimism": float(-error.mean()),
        "spearman": float(spearmanr(actual, prediction).statistic),
        "action_pair_rmse": float(np.sqrt(np.mean(pair_error**2))),
        "action_pair_sign_accuracy": float(np.mean(np.sign(actual_difference) == np.sign(predicted_difference))),
        "top1_regret": float(actual[pick] - actual.min()),
        "top3_regret": float(actual[shortlist].min() - actual.min()),
        "selected_optimism": float(actual[pick] - prediction[pick]),
        "selected_row": int(cell.row.iloc[pick]),
        "selected_action": str(cell.action_id.iloc[pick]),
        "selected_tied": bool(cell.is_tied.iloc[pick]),
        "zero_gain_action_pair_rmse": float(np.sqrt(np.mean(actual_difference**2))),
    }


def evaluate(output: Path) -> None:
    """Score saved fits and audit the selected policies' measured components."""
    frame, arrays = load_data()
    records = []
    metrics = []
    components = []
    for path in sorted((output / "fits").glob("*/*.json")):
        saved = json.loads(path.read_text())
        indices = np.asarray(saved["prediction_rows"], int)
        table = frame.iloc[indices][["row", "row_id", "state_id", "action_id", "target", "coordinate_hash"]].copy()
        table["is_tied"] = frame.is_tied_control.iloc[indices].to_numpy(bool)
        table["prediction"] = saved["predicted"]
        table["fold"] = saved["fold"]
        table["model"] = saved["model"]
        table["primary"] = table.state_id.eq(saved["fold"])
        table["scope"] = "seen_prefix_new_action" if saved["fold"] == "seen_prefix" else "held_prefix_new_action"
        table["held_state"] = table.state_id.eq(saved["fold"])
        records.append(table)
        scored = table if saved["fold"] == "seen_prefix" else table[table.primary]
        for state, cell in scored.groupby("state_id", sort=False):
            tied = cell[cell.is_tied]
            assert len(tied) == 1
            tied_actual = float(tied.target.iloc[0])
            tied_predicted = float(tied.prediction.iloc[0])
            tied_row = int(tied.row.iloc[0])
            for with_tied in (False, True):
                selected = cell if with_tied else cell[~cell.is_tied]
                result = state_metrics(selected)
                pick = selected.loc[selected.row.eq(result["selected_row"])].iloc[0]
                result.update(
                    {
                        "model": saved["model"],
                        "fold": saved["fold"],
                        "state_id": state,
                        "scope": str(table.scope.iloc[0]),
                        "with_tied": with_tied,
                        "observed_gain_vs_tied": tied_actual - float(pick.target),
                        "predicted_gain_vs_tied": tied_predicted - float(pick.prediction),
                        "held_outcomes_below_floor": saved["held_outcomes_below_floor"],
                    }
                )
                metrics.append(result)
                for j, component in enumerate(arrays["component_names"]):
                    components.append(
                        {
                            "model": saved["model"],
                            "fold": saved["fold"],
                            "state_id": state,
                            "with_tied": with_tied,
                            "selected_action": result["selected_action"],
                            "component": str(component),
                            "tied_bpb": float(arrays["component_bpb"][tied_row, j]),
                            "selected_bpb": float(arrays["component_bpb"][result["selected_row"], j]),
                            "difference": float(
                                arrays["component_bpb"][result["selected_row"], j] - arrays["component_bpb"][tied_row, j]
                            ),
                        }
                    )
    pd.concat(records, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(components).to_csv(output / "selected_components.csv", index=False)
    columns = [
        "rmse",
        "mean_optimism",
        "spearman",
        "action_pair_rmse",
        "action_pair_sign_accuracy",
        "top1_regret",
        "top3_regret",
        "selected_optimism",
        "observed_gain_vs_tied",
        "predicted_gain_vs_tied",
    ]
    summary = metric_frame.groupby(["scope", "model", "with_tied"], sort=False)[columns].mean().reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--stage", choices=("prepare", "fit", "evaluate", "all"), default="all")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--folds", nargs="+", choices=[*STATES, "seen_prefix"], default=[*STATES, "seen_prefix"])
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 2:
        raise ValueError("use one or two local workers")
    if any(os.environ.get(name) != "1" for name in THREAD_VARIABLES):
        raise ValueError(f"set all BLAS thread variables to 1: {THREAD_VARIABLES}")
    if args.stage in {"prepare", "all"}:
        prepared = prepare(args.output)
        structural_checks(args.output, prepared)
        print("Prepared six prefix-excluded folds and one seen-prefix diagnostic.", flush=True)
    if args.stage in {"fit", "all"}:
        prepared = json.loads((args.output / "prepared.json").read_text())
        assert all(digest(Path(path)) == sha for path, sha in prepared["hashes"].items())
        jobs = [FitJob(fold, model, str(args.output.resolve())) for fold in args.folds for model in args.models]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending = [pool.submit(fit_job, job) for job in jobs]
            for future in as_completed(pending):
                print(json.dumps(future.result()), flush=True)
    if args.stage in {"evaluate", "all"}:
        evaluate(args.output)


if __name__ == "__main__":
    main()
