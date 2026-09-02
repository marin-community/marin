# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///

"""Compare canonical DSP and OLMix on every retained StarCoder endpoint curve."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix import olmix_loglinear_fit as olmix_loglinear  # noqa: E402
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_dsp_single_phase_ladder_20260824 as dsp_ladder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    starcoder_wsd80_epoch_accounting as epoch_accounting,
)

INVENTORY_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_single_phase_curve_inventory_20260902"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_all_tied_curves_canonical_dsp_20260902"
PRIMARY_TARGET = "eval/paloma/dolma_100_programing_languages-llama3/bpb"
CURVE_INVENTORY_FILE = "curve_inventory.csv"
CURVE_MEMBERSHIPS_FILE = "curve_memberships.csv"
TARGET_OBSERVATIONS_FILE = "target_observations.csv"
INPUT_FILES = (CURVE_INVENTORY_FILE, CURVE_MEMBERSHIPS_FILE, TARGET_OBSERVATIONS_FILE)
EXPECTED_CURVES = 45
INNER_FOLDS = 3
FOLD_SEED = 20_260_902
DENSE_POINTS = 1001
CACHE_SCHEMA_VERSION = 2
PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "scale": 4},
}
FAMILY_LABELS = {
    "fixed_model_token_ladder": "Fixed-model token ladder",
    "matched_nd": "Matched model-size and token-budget ladder",
    "dense_horizon_replay": "Training horizon by StarCoder replay burden",
    "coupled_lr_onset": "Coupled phase-boundary and LR-decay onset",
}
SUPPORT_LABELS = {
    "full": "Full cache",
    "m0125": "0.125x burden",
    "m025": "0.25x burden",
    "m050": "0.5x burden",
    "m100": "1x burden",
    "m200": "2x burden",
    "m400": "4x burden",
}
SUPPORT_ORDER = tuple(SUPPORT_LABELS)
SUPPORT_EPOCH_MULTIPLIERS = {
    "m0125": 0.125,
    "m025": 0.25,
    "m050": 0.5,
    "m100": 1.0,
    "m200": 2.0,
    "m400": 4.0,
}


@dataclass(frozen=True)
class CurveTask:
    """One observed single-phase curve to fit."""

    curve_id: str
    weights: np.ndarray
    response: np.ndarray
    input_hash: str


@dataclass(frozen=True)
class CurveFit:
    """Full-data DSP and OLMix fits evaluated on observed and dense grids."""

    curve_id: str
    input_hash: str
    weights: np.ndarray
    response: np.ndarray
    observed_prediction: np.ndarray
    dense_weights: np.ndarray
    dense_prediction: np.ndarray
    nonlinear_parameters: np.ndarray
    intercept: float
    coefficients: np.ndarray
    olmix_observed_prediction: np.ndarray
    olmix_dense_prediction: np.ndarray
    olmix_log_c: float
    olmix_coefficients: np.ndarray
    olmix_huber_loss: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-dir", type=Path, default=INVENTORY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--restarts", type=int, default=48)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--force-refit", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(curve_id: str, weights: np.ndarray, response: np.ndarray) -> str:
    digest = hashlib.sha256(curve_id.encode("utf-8"))
    digest.update(np.asarray(weights, dtype="<f8").tobytes())
    digest.update(np.asarray(response, dtype="<f8").tobytes())
    return digest.hexdigest()


def stable_seed(curve_id: str) -> int:
    suffix = int.from_bytes(hashlib.sha256(curve_id.encode("utf-8")).digest()[:4], "big")
    return FOLD_SEED + suffix


def interleaved_folds(row_count: int, fold_count: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    labels = np.arange(row_count) % fold_count
    rows = np.arange(row_count)
    return tuple((rows[labels != fold], rows[labels == fold]) for fold in range(fold_count))


def tied_weights(starcoder_share: np.ndarray) -> np.ndarray:
    return np.column_stack([1.0 - starcoder_share, starcoder_share])


def exposures(starcoder_share: np.ndarray) -> np.ndarray:
    """Return the same simulated-epoch coordinates as the four-curve fit."""
    epoch_scales = np.asarray(
        [
            epoch_accounting.SIMULATED_EPOCH_TARGET_BUDGET / epoch_accounting.NEMOTRON_SOURCE_TOKENS,
            epoch_accounting.SIMULATED_EPOCH_TARGET_BUDGET / epoch_accounting.STARCODER_SOURCE_TOKENS,
        ],
        dtype=float,
    )
    return tied_weights(starcoder_share) * epoch_scales[None, :]


def canonical_rung() -> dsp_ladder.Rung:
    return next(rung for rung in dsp_ladder.LADDER if rung.name == "canonical")


def predict_canonical(
    exposure: np.ndarray,
    nonlinear_parameters: np.ndarray,
    intercept: float,
    coefficients: np.ndarray,
) -> np.ndarray:
    design = dsp_ladder.rung_design(exposure, nonlinear_parameters, canonical_rung(), exposure.shape[1])
    return intercept + design @ coefficients


def load_inputs(inventory_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = pd.read_csv(inventory_dir / CURVE_INVENTORY_FILE)
    core = curves.loc[curves["protocol_group"].eq("core_endpoint")].copy()
    if len(core) != EXPECTED_CURVES or not core["primary_target_ready"].all():
        raise ValueError(f"Expected {EXPECTED_CURVES} protocol-ready endpoint curves")

    memberships = pd.read_csv(inventory_dir / CURVE_MEMBERSHIPS_FILE)
    targets = pd.read_csv(inventory_dir / TARGET_OBSERVATIONS_FILE)
    primary = targets.loc[targets["target_id"].eq(PRIMARY_TARGET), ["observation_id", "training_run_id", "bpb"]]
    points = memberships.merge(primary, on=["observation_id", "training_run_id"], validate="many_to_one")
    points = points.loc[points["curve_id"].isin(core["curve_id"])].copy()
    points = points.merge(
        core[["curve_id", "family", "primary_target_points"]],
        on="curve_id",
        validate="many_to_one",
    )
    if points.duplicated(["curve_id", "starcoder_weight"]).any():
        raise ValueError("A core curve has multiple primary observations at one mixture weight")
    counts = points.groupby("curve_id")["starcoder_weight"].nunique()
    expected = core.set_index("curve_id")["primary_target_points"].astype(int)
    if not counts.sort_index().equals(expected.sort_index()):
        raise ValueError("Primary observations do not match the frozen curve inventory")
    if not np.isfinite(points["bpb"]).all():
        raise ValueError("Primary curve data contain nonfinite BPB values")
    return core.sort_values(["family", "curve_id"]).reset_index(drop=True), points


def build_tasks(curves: pd.DataFrame, points: pd.DataFrame) -> list[CurveTask]:
    tasks: list[CurveTask] = []
    for curve_id in curves["curve_id"]:
        block = points.loc[points["curve_id"].eq(curve_id)].sort_values("starcoder_weight")
        weights = block["starcoder_weight"].to_numpy(float)
        response = block["bpb"].to_numpy(float)
        tasks.append(
            CurveTask(
                curve_id=curve_id,
                weights=weights,
                response=response,
                input_hash=array_sha256(curve_id, weights, response),
            )
        )
    return tasks


def fit_curve(task: CurveTask, *, maxiter: int, restarts: int) -> CurveFit:
    exposure = exposures(task.weights)
    nonlinear_parameters, intercept, coefficients = dsp_ladder.fit_rung(
        exposure,
        task.response,
        canonical_rung(),
        interleaved_folds(len(task.response), INNER_FOLDS),
        (),
        seed=stable_seed(task.curve_id),
        maxiter=maxiter,
        restarts=restarts,
    )
    observed_prediction = predict_canonical(exposure, nonlinear_parameters, intercept, coefficients)
    dense_weights = np.linspace(float(task.weights.min()), float(task.weights.max()), DENSE_POINTS)
    dense_prediction = predict_canonical(
        exposures(dense_weights),
        nonlinear_parameters,
        intercept,
        coefficients,
    )
    olmix_fit = olmix_loglinear.fit_olmix_loglinear_model(
        tied_weights(task.weights),
        task.response,
        seed=stable_seed(task.curve_id),
        n_starts=olmix_loglinear.FIT_N_STARTS,
    )
    olmix_observed_prediction = olmix_fit.predict(tied_weights(task.weights))
    olmix_dense_prediction = olmix_fit.predict(tied_weights(dense_weights))
    predictions = (
        observed_prediction,
        dense_prediction,
        olmix_observed_prediction,
        olmix_dense_prediction,
    )
    if not all(np.isfinite(prediction).all() for prediction in predictions):
        raise ValueError(f"{task.curve_id}: a fitted surrogate produced nonfinite predictions")
    return CurveFit(
        curve_id=task.curve_id,
        input_hash=task.input_hash,
        weights=task.weights,
        response=task.response,
        observed_prediction=observed_prediction,
        dense_weights=dense_weights,
        dense_prediction=dense_prediction,
        nonlinear_parameters=nonlinear_parameters,
        intercept=float(intercept),
        coefficients=coefficients,
        olmix_observed_prediction=olmix_observed_prediction,
        olmix_dense_prediction=olmix_dense_prediction,
        olmix_log_c=olmix_fit.log_c,
        olmix_coefficients=np.asarray(olmix_fit.coefficients, dtype=float),
        olmix_huber_loss=olmix_fit.huber_loss,
    )


def fit_cache_path(output_dir: Path, curve_id: str) -> Path:
    return output_dir / "fit_cache" / f"{curve_id}.json"


def cache_metadata(
    *,
    maxiter: int,
    restarts: int,
    optimizer_hash: str,
    olmix_optimizer_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dense_points": DENSE_POINTS,
        "inner_folds": INNER_FOLDS,
        "fold_seed": FOLD_SEED,
        "maxiter": maxiter,
        "restarts": restarts,
        "optimizer_sha256": optimizer_hash,
        "olmix_optimizer_sha256": olmix_optimizer_hash,
        "olmix_huber_delta": olmix_loglinear.DEFAULT_HUBER_DELTA,
        "olmix_starts": olmix_loglinear.FIT_N_STARTS,
    }


def fit_to_payload(result: CurveFit, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        **metadata,
        "curve_id": result.curve_id,
        "input_hash": result.input_hash,
        "weights": result.weights.tolist(),
        "response": result.response.tolist(),
        "observed_prediction": result.observed_prediction.tolist(),
        "dense_weights": result.dense_weights.tolist(),
        "dense_prediction": result.dense_prediction.tolist(),
        "nonlinear_parameters": result.nonlinear_parameters.tolist(),
        "intercept": result.intercept,
        "coefficients": result.coefficients.tolist(),
        "olmix_observed_prediction": result.olmix_observed_prediction.tolist(),
        "olmix_dense_prediction": result.olmix_dense_prediction.tolist(),
        "olmix_log_c": result.olmix_log_c,
        "olmix_coefficients": result.olmix_coefficients.tolist(),
        "olmix_huber_loss": result.olmix_huber_loss,
    }


def payload_to_fit(payload: dict[str, Any]) -> CurveFit:
    return CurveFit(
        curve_id=str(payload["curve_id"]),
        input_hash=str(payload["input_hash"]),
        weights=np.asarray(payload["weights"], dtype=float),
        response=np.asarray(payload["response"], dtype=float),
        observed_prediction=np.asarray(payload["observed_prediction"], dtype=float),
        dense_weights=np.asarray(payload["dense_weights"], dtype=float),
        dense_prediction=np.asarray(payload["dense_prediction"], dtype=float),
        nonlinear_parameters=np.asarray(payload["nonlinear_parameters"], dtype=float),
        intercept=float(payload["intercept"]),
        coefficients=np.asarray(payload["coefficients"], dtype=float),
        olmix_observed_prediction=np.asarray(payload["olmix_observed_prediction"], dtype=float),
        olmix_dense_prediction=np.asarray(payload["olmix_dense_prediction"], dtype=float),
        olmix_log_c=float(payload["olmix_log_c"]),
        olmix_coefficients=np.asarray(payload["olmix_coefficients"], dtype=float),
        olmix_huber_loss=float(payload["olmix_huber_loss"]),
    )


def load_cached_fit(
    path: Path,
    task: CurveTask,
    metadata: dict[str, Any],
) -> CurveFit | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {**metadata, "curve_id": task.curve_id, "input_hash": task.input_hash}
    if any(payload.get(key) != value for key, value in expected.items()):
        return None
    result = payload_to_fit(payload)
    dense_predictions = (result.dense_prediction, result.olmix_dense_prediction)
    if len(result.dense_weights) != DENSE_POINTS or not all(
        np.isfinite(prediction).all() for prediction in dense_predictions
    ):
        return None
    return result


def write_cached_fit(path: Path, result: CurveFit, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(fit_to_payload(result, metadata), separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def fit_all_curves(
    tasks: list[CurveTask],
    *,
    output_dir: Path,
    maxiter: int,
    restarts: int,
    workers: int,
    force_refit: bool,
) -> list[CurveFit]:
    optimizer_hash = file_sha256(Path(dsp_ladder.__file__).resolve())
    olmix_optimizer_hash = file_sha256(Path(olmix_loglinear.__file__).resolve())
    metadata = cache_metadata(
        maxiter=maxiter,
        restarts=restarts,
        optimizer_hash=optimizer_hash,
        olmix_optimizer_hash=olmix_optimizer_hash,
    )
    completed: dict[str, CurveFit] = {}
    pending: list[CurveTask] = []
    for task in tasks:
        cached = None if force_refit else load_cached_fit(fit_cache_path(output_dir, task.curve_id), task, metadata)
        if cached is None:
            pending.append(task)
        else:
            completed[task.curve_id] = cached
    print(f"Reusing {len(completed)} cached fits; fitting {len(pending)} curves with {workers} workers", flush=True)

    fit_one = partial(fit_curve, maxiter=maxiter, restarts=restarts)
    if workers == 1:
        for index, task in enumerate(pending, start=1):
            result = fit_one(task)
            write_cached_fit(fit_cache_path(output_dir, result.curve_id), result, metadata)
            completed[result.curve_id] = result
            print(f"[{index}/{len(pending)}] {result.curve_id}", flush=True)
    elif pending:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as executor:
            futures = {executor.submit(fit_one, task): task.curve_id for task in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                write_cached_fit(fit_cache_path(output_dir, result.curve_id), result, metadata)
                completed[result.curve_id] = result
                print(f"[{index}/{len(pending)}] {result.curve_id}", flush=True)

    missing = sorted({task.curve_id for task in tasks} - completed.keys())
    if missing:
        raise ValueError(f"Missing fitted curves: {missing}")
    return [completed[task.curve_id] for task in tasks]


def compile_outputs(
    curves: pd.DataFrame,
    points: pd.DataFrame,
    results: list[CurveFit],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    curve_lookup = curves.set_index("curve_id")
    point_rows: list[dict[str, Any]] = []
    dense_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    nonlinear_names = ("log_rate_nemotron", "log_rate_starcoder", "threshold_nemotron", "threshold_starcoder")
    coefficient_names = (
        "benefit_amplitude_nemotron",
        "benefit_amplitude_starcoder",
        "harm_amplitude_nemotron",
        "harm_amplitude_starcoder",
    )

    for result in results:
        metadata = curve_lookup.loc[result.curve_id]
        source_points = points.loc[points["curve_id"].eq(result.curve_id)].sort_values("starcoder_weight")
        if not np.allclose(source_points["starcoder_weight"], result.weights, atol=1e-12):
            raise ValueError(f"{result.curve_id}: cached weight grid differs from the inventory")
        if not np.allclose(source_points["bpb"], result.response, atol=1e-12):
            raise ValueError(f"{result.curve_id}: cached response differs from the inventory")

        residual = result.observed_prediction - result.response
        olmix_residual = result.olmix_observed_prediction - result.response
        observed_best = int(np.argmin(result.response))
        fit_grid_best = int(np.argmin(result.observed_prediction))
        olmix_grid_best = int(np.argmin(result.olmix_observed_prediction))
        dense_best = int(np.argmin(result.dense_prediction))
        olmix_dense_best = int(np.argmin(result.olmix_dense_prediction))
        centered = result.response - result.response.mean()
        denominator = float(centered @ centered)
        r_squared = 1.0 - float(residual @ residual) / denominator if denominator > 0 else float("nan")
        olmix_r_squared = 1.0 - float(olmix_residual @ olmix_residual) / denominator if denominator > 0 else float("nan")
        spearman = float(stats.spearmanr(result.observed_prediction, result.response).statistic)
        rmse = float(np.sqrt(np.mean(residual**2)))
        olmix_rmse = float(np.sqrt(np.mean(olmix_residual**2)))

        for (_, source), prediction, error, olmix_prediction, olmix_error in zip(
            source_points.iterrows(),
            result.observed_prediction,
            residual,
            result.olmix_observed_prediction,
            olmix_residual,
            strict=True,
        ):
            point_rows.append(
                {
                    "curve_id": result.curve_id,
                    "family": str(metadata["family"]),
                    "starcoder_weight": float(source["starcoder_weight"]),
                    "observed_bpb": float(source["bpb"]),
                    "full_fit_prediction_bpb": float(prediction),
                    "residual_prediction_minus_observed": float(error),
                    "olmix_full_fit_prediction_bpb": float(olmix_prediction),
                    "olmix_residual_prediction_minus_observed": float(olmix_error),
                    "training_run_id": str(source["training_run_id"]),
                    "observation_id": str(source["observation_id"]),
                }
            )
        dense_rows.extend(
            {
                "curve_id": result.curve_id,
                "family": str(metadata["family"]),
                "starcoder_weight": float(weight),
                "full_fit_prediction_bpb": float(prediction),
                "olmix_full_fit_prediction_bpb": float(olmix_prediction),
            }
            for weight, prediction, olmix_prediction in zip(
                result.dense_weights,
                result.dense_prediction,
                result.olmix_dense_prediction,
                strict=True,
            )
        )
        metric_rows.append(
            {
                "curve_id": result.curve_id,
                "family": str(metadata["family"]),
                "rows": len(result.weights),
                "full_fit_rmse": rmse,
                "full_fit_max_abs_residual": float(np.max(np.abs(residual))),
                "full_fit_r_squared": r_squared,
                "full_fit_spearman": spearman,
                "observed_grid_min_weight": float(result.weights[observed_best]),
                "observed_grid_min_bpb": float(result.response[observed_best]),
                "fit_selected_grid_weight": float(result.weights[fit_grid_best]),
                "fit_selected_grid_actual_bpb": float(result.response[fit_grid_best]),
                "fit_selected_grid_regret": float(result.response[fit_grid_best] - result.response[observed_best]),
                "full_fit_dense_min_weight": float(result.dense_weights[dense_best]),
                "full_fit_dense_min_predicted_bpb": float(result.dense_prediction[dense_best]),
                "dense_min_distance_from_observed_min": float(
                    abs(result.dense_weights[dense_best] - result.weights[observed_best])
                ),
                "olmix_full_fit_rmse": olmix_rmse,
                "olmix_full_fit_max_abs_residual": float(np.max(np.abs(olmix_residual))),
                "olmix_full_fit_r_squared": olmix_r_squared,
                "olmix_fit_selected_grid_weight": float(result.weights[olmix_grid_best]),
                "olmix_fit_selected_grid_actual_bpb": float(result.response[olmix_grid_best]),
                "olmix_fit_selected_grid_regret": float(
                    result.response[olmix_grid_best] - result.response[observed_best]
                ),
                "olmix_full_fit_dense_min_weight": float(result.dense_weights[olmix_dense_best]),
                "olmix_full_fit_dense_min_predicted_bpb": float(result.olmix_dense_prediction[olmix_dense_best]),
                "olmix_huber_loss": result.olmix_huber_loss,
            }
        )
        parameter_rows.append({"curve_id": result.curve_id, "parameter": "intercept", "value": result.intercept})
        parameter_rows.extend(
            {"curve_id": result.curve_id, "parameter": name, "value": float(value)}
            for name, value in zip(nonlinear_names, result.nonlinear_parameters, strict=True)
        )
        parameter_rows.extend(
            {"curve_id": result.curve_id, "parameter": name, "value": float(value)}
            for name, value in zip(coefficient_names, result.coefficients, strict=True)
        )
        parameter_rows.extend(
            [
                {"curve_id": result.curve_id, "parameter": "olmix_log_c", "value": result.olmix_log_c},
                {
                    "curve_id": result.curve_id,
                    "parameter": "olmix_beta_nemotron",
                    "value": float(result.olmix_coefficients[0]),
                },
                {
                    "curve_id": result.curve_id,
                    "parameter": "olmix_beta_starcoder",
                    "value": float(result.olmix_coefficients[1]),
                },
                {
                    "curve_id": result.curve_id,
                    "parameter": "olmix_huber_loss",
                    "value": result.olmix_huber_loss,
                },
            ]
        )

    predictions = pd.DataFrame(point_rows).sort_values(["family", "curve_id", "starcoder_weight"])
    dense = pd.DataFrame(dense_rows).sort_values(["family", "curve_id", "starcoder_weight"])
    metrics = pd.DataFrame(metric_rows).sort_values(["family", "curve_id"]).reset_index(drop=True)
    parameters = pd.DataFrame(parameter_rows).sort_values(["curve_id", "parameter"]).reset_index(drop=True)
    if len(metrics) != EXPECTED_CURVES or metrics["curve_id"].nunique() != EXPECTED_CURVES:
        raise ValueError("Compiled fit metrics are incomplete")
    return predictions, dense, metrics, parameters


def token_label(tokens: float) -> str:
    return f"{tokens / 1e9:.2f}B"


def historical_starcoder_epoch_scale() -> float:
    return epoch_accounting.SIMULATED_EPOCH_TARGET_BUDGET / epoch_accounting.STARCODER_SOURCE_TOKENS


def starcoder_epoch_scale(metadata: pd.Series) -> float:
    """Return StarCoder materialized epochs at p=1 for one physical curve."""
    support_id = str(metadata["support_id"])
    if support_id == "full":
        return float(metadata["planned_materialized_tokens"]) / epoch_accounting.STARCODER_SOURCE_TOKENS
    if support_id in SUPPORT_EPOCH_MULTIPLIERS:
        return historical_starcoder_epoch_scale() * SUPPORT_EPOCH_MULTIPLIERS[support_id]
    if support_id in {"historical_simulated_support", "matched_nd_reference_support"}:
        return historical_starcoder_epoch_scale()
    raise ValueError(f"Unknown StarCoder support for epoch axis: {support_id}")


def epoch_tick_label(value: float, *, maximum: float) -> str:
    if maximum < 0.1:
        return f"{value:.3f} ep"
    if maximum < 2.0:
        return f"{value:.2f} ep"
    if maximum < 20.0:
        return f"{value:.1f} ep"
    return f"{value:.0f} ep"


def cell_id(curve_id: str) -> str:
    return curve_id.split("__")[1]


def scale_mode(curve_id: str) -> str:
    identifier = cell_id(curve_id)
    if "_increase_nd_" in identifier:
        return "N,D"
    if "_increase_n_" in identifier:
        return "N"
    if "_increase_d_" in identifier:
        return "D"
    return "shared"


def short_curve_label(metadata: pd.Series) -> str:
    family = str(metadata["family"])
    curve_id = str(metadata["curve_id"]) if "curve_id" in metadata.index else str(metadata.name)
    if family == "fixed_model_token_ladder":
        return f"{token_label(metadata['planned_materialized_tokens'])} tokens"
    if family == "coupled_lr_onset":
        return f"Onset {metadata['lr_decay_onset_fraction']:.2f}T"
    if family == "dense_horizon_replay":
        return SUPPORT_LABELS[str(metadata["support_id"])]
    rung = cell_id(curve_id).split("_")[0]
    return (
        f"{rung} {scale_mode(curve_id)} | h{int(metadata['hidden_size'])} | "
        f"{token_label(metadata['planned_materialized_tokens'])}"
    )


def dense_section_label(metadata: pd.Series) -> str:
    curve_id = str(metadata.name)
    rung = cell_id(curve_id).split("_")[0]
    return (
        f"{rung} {scale_mode(curve_id)}: h{int(metadata['hidden_size'])}, "
        f"{token_label(metadata['planned_materialized_tokens'])} materialized tokens"
    )


def build_grid_figure(
    curve_ids: list[str],
    *,
    curves: pd.DataFrame,
    predictions: pd.DataFrame,
    dense: pd.DataFrame,
    metrics: pd.DataFrame,
    columns: int,
) -> go.Figure:
    rows = math.ceil(len(curve_ids) / columns)
    curve_lookup = curves.set_index("curve_id")
    metric_lookup = metrics.set_index("curve_id")
    titles = []
    for curve_id in curve_ids:
        metadata = curve_lookup.loc[curve_id]
        metric = metric_lookup.loc[curve_id]
        titles.append(
            f"<b>{short_curve_label(metadata)}</b><br>"
            f"<span style='font-size:9px'>RMSE D/O {metric['full_fit_rmse']:.4f} / "
            f"{metric['olmix_full_fit_rmse']:.4f}<br>"
            f"p* obs/D/O {metric['observed_grid_min_weight']:.3f} / "
            f"{metric['full_fit_dense_min_weight']:.3f} / "
            f"{metric['olmix_full_fit_dense_min_weight']:.3f}</span>"
        )
    figure = make_subplots(
        rows=rows,
        cols=columns,
        subplot_titles=titles,
        horizontal_spacing=0.035 if columns >= 5 else 0.075,
        vertical_spacing=0.23 if rows > 1 else 0.10,
    )

    for index, curve_id in enumerate(curve_ids):
        row = index // columns + 1
        column = index % columns + 1
        point = predictions.loc[predictions["curve_id"].eq(curve_id)].sort_values("starcoder_weight")
        smooth = dense.loc[dense["curve_id"].eq(curve_id)].sort_values("starcoder_weight")
        metric = metric_lookup.loc[curve_id]
        metadata = curve_lookup.loc[curve_id]
        epoch_scale = starcoder_epoch_scale(metadata)
        point_epochs = point["starcoder_weight"].to_numpy() * epoch_scale
        smooth_epochs = smooth["starcoder_weight"].to_numpy() * epoch_scale
        figure.add_trace(
            go.Scatter(
                x=point["starcoder_weight"],
                y=point["observed_bpb"],
                mode="lines+markers",
                line={"color": "#9aa5ad", "width": 1.2},
                marker={"color": "#173042", "size": 6},
                customdata=point_epochs,
                hovertemplate=(
                    "StarCoder p=%{x:.4f}<br>StarCoder materialized epochs=%{customdata:.3f}"
                    "<br>observed=%{y:.6f}<extra>Observed</extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=smooth["starcoder_weight"],
                y=smooth["full_fit_prediction_bpb"],
                mode="lines",
                line={"color": "#d95d39", "width": 2.6},
                customdata=smooth_epochs,
                hovertemplate=(
                    "StarCoder p=%{x:.4f}<br>StarCoder materialized epochs=%{customdata:.3f}"
                    "<br>DSP=%{y:.6f}<extra>Canonical DSP</extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=smooth["starcoder_weight"],
                y=smooth["olmix_full_fit_prediction_bpb"],
                mode="lines",
                line={"color": "#277da1", "width": 2.2, "dash": "dash"},
                customdata=smooth_epochs,
                hovertemplate=(
                    "StarCoder p=%{x:.4f}<br>StarCoder materialized epochs=%{customdata:.3f}"
                    "<br>OLMix=%{y:.6f}<extra>OLMix log-linear</extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=[metric["observed_grid_min_weight"]],
                y=[metric["observed_grid_min_bpb"]],
                mode="markers",
                marker={"color": "#16845b", "size": 12, "symbol": "star"},
                customdata=[metric["observed_grid_min_weight"] * epoch_scale],
                hovertemplate=(
                    "Observed minimum<br>StarCoder p=%{x:.4f}"
                    "<br>StarCoder materialized epochs=%{customdata:.3f}<br>BPB=%{y:.6f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=[metric["full_fit_dense_min_weight"]],
                y=[metric["full_fit_dense_min_predicted_bpb"]],
                mode="markers",
                marker={"color": "#a71930", "size": 10, "symbol": "x", "line": {"width": 2}},
                customdata=[metric["full_fit_dense_min_weight"] * epoch_scale],
                hovertemplate=(
                    "DSP minimum<br>StarCoder p=%{x:.4f}"
                    "<br>StarCoder materialized epochs=%{customdata:.3f}<br>predicted=%{y:.6f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=column,
        )
        combined = np.concatenate(
            [
                point["observed_bpb"].to_numpy(),
                smooth["full_fit_prediction_bpb"].to_numpy(),
                smooth["olmix_full_fit_prediction_bpb"].to_numpy(),
            ]
        )
        lower = float(combined.min())
        upper = float(combined.max())
        padding = max(0.004, 0.07 * (upper - lower))
        figure.update_xaxes(
            range=[float(point["starcoder_weight"].min()) - 0.015, float(point["starcoder_weight"].max()) + 0.015],
            tickvals=[0.0, 0.25, 0.5, 0.75, 1.0],
            tickformat=".2f",
            title="StarCoder p" if row == rows else None,
            row=row,
            col=column,
        )
        figure.update_yaxes(
            range=[lower - padding, upper + padding],
            tickformat=".3f",
            title="BPB" if column == 1 else None,
            row=row,
            col=column,
        )

        weight_min = float(point["starcoder_weight"].min())
        weight_max = float(point["starcoder_weight"].max())
        epoch_tick_weights = np.linspace(weight_min, weight_max, 3)
        base_axis_index = index + 1
        base_xaxis = "x" if base_axis_index == 1 else f"x{base_axis_index}"
        base_yaxis = "y" if base_axis_index == 1 else f"y{base_axis_index}"
        top_axis_index = len(curve_ids) + base_axis_index
        top_xaxis = f"x{top_axis_index}"
        top_axis_key = f"xaxis{top_axis_index}"
        figure.update_layout(
            {
                top_axis_key: {
                    "anchor": base_yaxis,
                    "overlaying": base_xaxis,
                    "matches": base_xaxis,
                    "side": "top",
                    "range": [weight_min - 0.015, weight_max + 0.015],
                    "tickmode": "array",
                    "tickvals": epoch_tick_weights.tolist(),
                    "ticktext": [
                        epoch_tick_label(weight * epoch_scale, maximum=weight_max * epoch_scale)
                        for weight in epoch_tick_weights
                    ],
                    "ticks": "outside",
                    "ticklen": 3,
                    "tickfont": {"size": 9, "color": "#5f7180"},
                    "showgrid": False,
                    "zeroline": False,
                    "fixedrange": False,
                }
            }
        )
        figure.add_trace(
            go.Scatter(
                x=[weight_min, weight_max],
                y=[lower, lower],
                mode="markers",
                marker={"opacity": 0.0},
                hoverinfo="skip",
                showlegend=False,
                xaxis=top_xaxis,
                yaxis=base_yaxis,
            )
        )

    figure.update_layout(
        template="plotly_white",
        height=max(440, 350 * rows + 90),
        margin={"l": 58, "r": 24, "t": 112, "b": 62},
        font={"family": "Avenir Next, Helvetica Neue, sans-serif", "color": "#173042", "size": 11},
        paper_bgcolor="#fbf8f0",
        plot_bgcolor="#fbf8f0",
        hoverlabel={"font_size": 12},
    )
    figure.update_annotations(font={"size": 12, "color": "#173042"}, yshift=30)
    figure.update_xaxes(gridcolor="#e4ded2", zeroline=False)
    figure.update_yaxes(gridcolor="#e4ded2", zeroline=False)
    return figure


def metric_table_html(metrics: pd.DataFrame, curves: pd.DataFrame) -> str:
    table = metrics.merge(
        curves[
            [
                "curve_id",
                "planned_materialized_tokens",
                "hidden_size",
                "support_id",
                "lr_decay_onset_fraction",
            ]
        ],
        on="curve_id",
        validate="one_to_one",
    ).copy()
    table["curve"] = table.apply(short_curve_label, axis=1)
    table["family"] = table["family"].map(FAMILY_LABELS)
    table["DSP RMSE"] = table["full_fit_rmse"].map(lambda value: f"{value:.6f}")
    table["OLMix RMSE"] = table["olmix_full_fit_rmse"].map(lambda value: f"{value:.6f}")
    table["DSP R2"] = table["full_fit_r_squared"].map(lambda value: f"{value:.4f}")
    table["OLMix R2"] = table["olmix_full_fit_r_squared"].map(lambda value: f"{value:.4f}")
    table["observed min p"] = table["observed_grid_min_weight"].map(lambda value: f"{value:.4f}")
    table["DSP min p"] = table["full_fit_dense_min_weight"].map(lambda value: f"{value:.4f}")
    table["OLMix min p"] = table["olmix_full_fit_dense_min_weight"].map(lambda value: f"{value:.4f}")
    table["DSP grid regret"] = table["fit_selected_grid_regret"].map(lambda value: f"{value:+.6f}")
    table["OLMix grid regret"] = table["olmix_fit_selected_grid_regret"].map(lambda value: f"{value:+.6f}")
    table = table.sort_values("full_fit_rmse", ascending=False)
    display = table[
        [
            "family",
            "curve",
            "rows",
            "DSP RMSE",
            "OLMix RMSE",
            "DSP R2",
            "OLMix R2",
            "observed min p",
            "DSP min p",
            "OLMix min p",
            "DSP grid regret",
            "OLMix grid regret",
        ]
    ]
    return display.to_html(index=False, classes="metric-table", table_id="fit-metrics", border=0)


def figure_fragment(figure: go.Figure, *, include_plotlyjs: bool) -> str:
    return pio.to_html(
        figure,
        full_html=False,
        include_plotlyjs="inline" if include_plotlyjs else False,
        config=PLOTLY_CONFIG,
    )


def build_html(
    curves: pd.DataFrame,
    predictions: pd.DataFrame,
    dense: pd.DataFrame,
    metrics: pd.DataFrame,
) -> str:
    curve_lookup = curves.set_index("curve_id")
    fixed_ids = (
        curves.loc[curves["family"].eq("fixed_model_token_ladder")]
        .sort_values("planned_materialized_tokens")["curve_id"]
        .tolist()
    )
    matched_ids = (
        curves.loc[curves["family"].eq("matched_nd")]
        .sort_values(["planned_materialized_tokens", "hidden_size", "curve_id"])["curve_id"]
        .tolist()
    )
    onset_ids = (
        curves.loc[curves["family"].eq("coupled_lr_onset")].sort_values("lr_decay_onset_fraction")["curve_id"].tolist()
    )
    dense_curves = curves.loc[curves["family"].eq("dense_horizon_replay")].copy()
    dense_curves["cell"] = dense_curves["curve_id"].map(cell_id)
    dense_curves["support_rank"] = dense_curves["support_id"].map(
        {value: index for index, value in enumerate(SUPPORT_ORDER)}
    )

    fixed_figure = build_grid_figure(
        fixed_ids,
        curves=curves,
        predictions=predictions,
        dense=dense,
        metrics=metrics,
        columns=2,
    )
    matched_figure = build_grid_figure(
        matched_ids,
        curves=curves,
        predictions=predictions,
        dense=dense,
        metrics=metrics,
        columns=5,
    )
    onset_figure = build_grid_figure(
        onset_ids,
        curves=curves,
        predictions=predictions,
        dense=dense,
        metrics=metrics,
        columns=3,
    )

    fragments: list[str] = []
    fragments.append(
        "<section id='fixed'><h2>Fixed-model token ladder</h2>"
        "<p>One 157M-parameter model trained for four token budgets. Each panel uses its union of regular and "
        "irregular measured tied-mixture coordinates.</p>"
        f"<div class='chart'>{figure_fragment(fixed_figure, include_plotlyjs=True)}</div></section>"
    )
    fragments.append(
        "<section id='matched'><h2>Matched model-size and token-budget ladder</h2>"
        "<p>Ten cells separate increasing model size, increasing training duration, and jointly increasing both. "
        "The measured range is approximately p=0.036 to p=0.90.</p>"
        "<div class='chart-scroll'><div class='chart matched-chart'>"
        f"{figure_fragment(matched_figure, include_plotlyjs=False)}"
        "</div></div></section>"
    )
    dense_fragments: list[str] = []
    for _cell, block in dense_curves.groupby("cell", sort=True):
        ordered = block.sort_values("support_rank")
        curve_ids = ordered["curve_id"].tolist()
        figure = build_grid_figure(
            curve_ids,
            curves=curves,
            predictions=predictions,
            dense=dense,
            metrics=metrics,
            columns=7,
        )
        heading = dense_section_label(curve_lookup.loc[curve_ids[0]])
        dense_fragments.append(
            f"<article class='dense-row'><h3>{heading}</h3><div class='chart-scroll'>"
            f"<div class='chart dense-chart'>{figure_fragment(figure, include_plotlyjs=False)}</div></div></article>"
        )
    fragments.append(
        "<section id='replay'><h2>Training horizon by replay burden</h2>"
        "<p>Four training horizons are crossed with seven physical StarCoder-support settings. Columns progress "
        "from the full cache toward four times the historical replay burden.</p>"
        + "".join(dense_fragments)
        + "</section>"
    )
    fragments.append(
        "<section id='onset'><h2>Coupled LR-decay onset</h2>"
        "<p>Three independently trained 8B-token surfaces move the phase boundary and cosine-decay onset together. "
        "Because p is tied across phases, the data-policy boundary itself is inert.</p>"
        f"<div class='chart'>{figure_fragment(onset_figure, include_plotlyjs=False)}</div></section>"
    )

    median_rmse = float(metrics["full_fit_rmse"].median())
    median_olmix_rmse = float(metrics["olmix_full_fit_rmse"].median())
    worst = metrics.loc[metrics["full_fit_rmse"].idxmax()]
    dsp_wins = int((metrics["full_fit_rmse"] < metrics["olmix_full_fit_rmse"]).sum())
    weight_bounds = predictions.groupby("curve_id")["starcoder_weight"].agg(["min", "max"])
    olmix_minimum = metrics.set_index("curve_id")["olmix_full_fit_dense_min_weight"]
    olmix_edge_minima = int(
        (
            np.isclose(olmix_minimum, weight_bounds["min"], atol=1e-12)
            | np.isclose(olmix_minimum, weight_bounds["max"], atol=1e-12)
        ).sum()
    )
    table = metric_table_html(metrics, curves)
    body = "".join(fragments)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Canonical DSP versus OLMix across 45 StarCoder single-phase curves</title>
<style>
:root {{
  --ink:#173042;
  --muted:#5f7180;
  --paper:#fbf8f0;
  --card:#fffdf8;
  --line:#d9d1c3;
  --accent:#d95d39;
  --good:#16845b;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0;
  color:var(--ink);
  font-family:"Avenir Next","Helvetica Neue",sans-serif;
  background:
    radial-gradient(circle at 90% 0%,#f2e7d2 0,transparent 32rem),
    linear-gradient(180deg,#fbf8f0,#f5f0e7);
}}
main {{ max-width:1720px; margin:0 auto; padding:48px 28px 80px; }}
h1,h2 {{ font-family:Georgia,"Times New Roman",serif; letter-spacing:-0.025em; }}
h1 {{ max-width:1100px; margin:0; font-size:clamp(2.5rem,5vw,5.2rem); line-height:.98; }}
h2 {{ margin:0 0 8px; font-size:2.15rem; }}
h3 {{ margin:28px 0 8px; font-size:1.15rem; }}
p {{ max-width:1000px; color:var(--muted); font-size:1.05rem; line-height:1.58; }}
.lede {{ margin:22px 0 26px; font-size:1.22rem; }}
.nav {{ display:flex; flex-wrap:wrap; gap:9px; margin:24px 0 30px; }}
.nav a {{
  color:var(--ink);
  text-decoration:none;
  border:1px solid #b9c5d1;
  border-radius:999px;
  padding:8px 13px;
  background:rgba(255,255,255,.55);
}}
.cards {{ display:grid; grid-template-columns:repeat(5,minmax(150px,1fr)); gap:12px; margin:24px 0; }}
.card {{
  padding:18px;
  border:1px solid var(--line);
  border-radius:14px;
  background:rgba(255,253,248,.86);
  box-shadow:0 8px 26px rgba(50,40,20,.05);
}}
.card strong {{ display:block; font-family:Georgia,serif; font-size:1.8rem; }}
.card span {{ color:var(--muted); font-size:.88rem; }}
.legend {{
  display:flex;
  flex-wrap:wrap;
  gap:20px;
  align-items:center;
  margin:22px 0 4px;
  padding:13px 16px;
  border-left:4px solid var(--accent);
  background:rgba(255,255,255,.5);
}}
.key {{ display:inline-flex; align-items:center; gap:8px; }}
.swatch {{ width:24px; height:3px; display:inline-block; background:var(--accent); }}
.swatch-olmix {{ width:24px; height:0; display:inline-block; border-top:3px dashed #277da1; }}
.dot {{ width:9px; height:9px; border-radius:50%; display:inline-block; background:var(--ink); }}
.star {{ color:var(--good); font-size:1.15rem; }} .cross {{ color:#a71930; font-size:1.15rem; font-weight:700; }}
.caveat {{
  margin:20px 0 42px;
  padding:18px 20px;
  border:1px solid #d6bda9;
  border-radius:12px;
  background:#fff7ed;
  color:#704a38;
  max-width:1100px;
}}
section {{ margin-top:58px; scroll-margin-top:20px; }}
.chart {{
  border:1px solid var(--line);
  border-radius:14px;
  overflow:hidden;
  background:var(--paper);
  box-shadow:0 9px 30px rgba(47,40,25,.05);
}}
.chart-scroll {{ overflow-x:auto; padding-bottom:4px; }}
.matched-chart {{ min-width:1180px; }} .dense-chart {{ min-width:1500px; }}
.dense-row {{ margin-top:26px; }}
details {{ margin-top:58px; border:1px solid var(--line); border-radius:14px; background:var(--card); padding:18px; }}
summary {{ cursor:pointer; font-family:Georgia,serif; font-size:1.55rem; font-weight:700; }}
.filter {{
  width:min(430px,100%);
  margin:18px 0 12px;
  padding:10px 12px;
  border:1px solid #b8c3cc;
  border-radius:8px;
  font:inherit;
  background:white;
}}
.table-wrap {{ overflow:auto; max-height:680px; }}
.metric-table {{ width:100%; border-collapse:collapse; font-size:.86rem; }}
.metric-table th {{ position:sticky; top:0; background:#173042; color:white; text-align:left; padding:9px; }}
.metric-table td {{ border-bottom:1px solid #e5dfd4; padding:8px 9px; white-space:nowrap; }}
.metric-table tr:nth-child(even) {{ background:#faf6ee; }}
footer {{ margin-top:52px; padding-top:20px; border-top:1px solid var(--line); color:var(--muted); }}
@media (max-width:900px) {{ main {{ padding:30px 14px 60px; }} .cards {{ grid-template-columns:repeat(2,1fr); }} }}
</style>
</head>
<body><main>
<header>
<h1>Canonical DSP versus OLMix across 45 StarCoder single-phase curves</h1>
<p class="lede">
Every retained endpoint curve is shown. Navy points and the thin gray path are measured Programming Languages
BPB; orange is canonical DSP and dashed blue is exact OLMix log-linear, each fit to all points on that curve.
The lower x-axis remains StarCoder fraction <em>p</em>; the synchronized upper axis expresses the same coordinate
as curve-specific StarCoder materialized epochs.
</p>
<nav class="nav">
<a href="#fixed">Token ladder</a>
<a href="#matched">Matched N,D</a>
<a href="#replay">Replay panel</a>
<a href="#onset">LR onset</a>
<a href="#diagnostics">Diagnostics</a>
</nav>
<div class="cards">
<div class="card"><strong>45</strong><span>physical endpoint curves</span></div>
<div class="card"><strong>{median_rmse:.4f}</strong><span>median DSP RMSE</span></div>
<div class="card"><strong>{median_olmix_rmse:.4f}</strong><span>median OLMix RMSE</span></div>
<div class="card"><strong>{dsp_wins}/45</strong><span>lower RMSE for DSP</span></div>
<div class="card"><strong>{olmix_edge_minima}/45</strong><span>OLMix minima at an edge</span></div>
</div>
<div class="legend">
<span class="key"><i class="dot"></i> observed</span>
<span class="key"><i class="swatch"></i> canonical DSP</span>
<span class="key"><i class="swatch-olmix"></i> OLMix log-linear</span>
<span class="key"><b class="star">★</b> observed grid minimum</span>
<span class="key"><b class="cross">x</b> smooth DSP minimum</span>
</div>
<div class="caveat">
<strong>Capacity check, not generalization evidence.</strong> Each full curve selects DSP shape parameters and fits
the linear head using all 15-26 observations. Canonical two-bucket DSP has four nonlinear shape parameters, four
nonnegative amplitudes, and an intercept. OLMix uses the exact positive law
<code>L(p)=c+exp(beta_N(1-p)+beta_S p)</code> with summed Huber loss and 48 starts. Along this one-dimensional edge,
OLMix is necessarily monotone or flat, so it cannot represent an interior U-shaped optimum. Use the separate
benchmark protocol for held-out claims.
</div>
</header>
{body}
<details id="diagnostics" open><summary>Fit diagnostics, worst RMSE first</summary>
<p>
The worst DSP fit is <code>{worst["curve_id"]}</code> at RMSE {worst["full_fit_rmse"]:.6f}.
Filter by family, curve label, or value.
</p>
<input class="filter" id="metric-filter" type="search" placeholder="Filter diagnostics...">
<div class="table-wrap">{table}</div></details>
<footer>
All panels use independent y-axis ranges. Smooth minima are searched only over each curve's measured p-range.
For full-cache curves the upper axis uses the run's materialized token budget divided by the complete StarCoder
pool; finite-support curves use their preregistered replay multiplier. Generated from the deduplicated StarCoder
one-dimensional curve registry.
</footer>
</main>
<script>
const filter = document.getElementById('metric-filter');
filter.addEventListener('input', () => {{
  const query = filter.value.toLowerCase();
  document.querySelectorAll('#fit-metrics tbody tr').forEach(row => {{
    row.style.display = row.textContent.toLowerCase().includes(query) ? '' : 'none';
  }});
}});
</script>
</body></html>"""


def write_report(output_dir: Path, metrics: pd.DataFrame) -> None:
    metrics = metrics.assign(dsp_lower_rmse=metrics["full_fit_rmse"] < metrics["olmix_full_fit_rmse"])
    family = (
        metrics.groupby("family", as_index=False)
        .agg(
            curves=("curve_id", "nunique"),
            median_dsp_rmse=("full_fit_rmse", "median"),
            median_olmix_rmse=("olmix_full_fit_rmse", "median"),
            p90_dsp_rmse=("full_fit_rmse", lambda values: values.quantile(0.9)),
            p90_olmix_rmse=("olmix_full_fit_rmse", lambda values: values.quantile(0.9)),
            dsp_lower_rmse=("dsp_lower_rmse", "sum"),
            dsp_exact_grid_optima=("fit_selected_grid_regret", lambda values: int((values.abs() <= 1e-12).sum())),
            olmix_exact_grid_optima=(
                "olmix_fit_selected_grid_regret",
                lambda values: int((values.abs() <= 1e-12).sum()),
            ),
        )
        .sort_values("family")
    )
    family["family"] = family["family"].map(FAMILY_LABELS)
    for column in ("median_dsp_rmse", "median_olmix_rmse", "p90_dsp_rmse", "p90_olmix_rmse"):
        family[column] = family[column].map(lambda value: f"{value:.6f}")
    worst = metrics.nlargest(10, "full_fit_rmse").copy()
    worst["family"] = worst["family"].map(FAMILY_LABELS)
    worst["DSP RMSE"] = worst["full_fit_rmse"].map(lambda value: f"{value:.6f}")
    worst["OLMix RMSE"] = worst["olmix_full_fit_rmse"].map(lambda value: f"{value:.6f}")
    worst["observed min p"] = worst["observed_grid_min_weight"].map(lambda value: f"{value:.4f}")
    worst["DSP min p"] = worst["full_fit_dense_min_weight"].map(lambda value: f"{value:.4f}")
    worst["OLMix min p"] = worst["olmix_full_fit_dense_min_weight"].map(lambda value: f"{value:.4f}")
    lines = [
        "# Canonical DSP versus OLMix across 45 StarCoder single-phase curves",
        "",
        "This is a descriptive capacity check. Canonical DSP and exact OLMix log-linear are fit independently to "
        "all measured Programming Languages BPB points on each of the 45 endpoint curves in the deduplicated "
        "registry. These are not out-of-fold fits.",
        "",
        "The OLMix law is `L(p) = c + exp(beta_N * (1-p) + beta_S * p)`, fit with summed Huber loss and 48 "
        "starts. It is necessarily monotone or flat on this two-bucket simplex edge, so it cannot express an "
        "interior U-shaped optimum.",
        "",
        family.to_markdown(index=False),
        "",
        "## Worst DSP full-data fits",
        "",
        worst[
            [
                "curve_id",
                "family",
                "rows",
                "DSP RMSE",
                "OLMix RMSE",
                "observed min p",
                "DSP min p",
                "OLMix min p",
            ]
        ].to_markdown(index=False),
        "",
        "The HTML report keeps StarCoder fraction `p` on the lower x-axis and reports the same coordinate as "
        "StarCoder materialized epochs on the upper axis. Full-cache curves use the run's materialized token "
        "budget divided by the full StarCoder pool; finite-support curves use the preregistered replay multiplier.",
        "",
        "Panels use independent y-axis ranges and search both smooth minima only over each curve's measured "
        "mixture range. DSP uses the same canonical rung, three-fold profiled nonlinear objective, and "
        "simulated-epoch coordinates as the validated four-curve artifact.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.maxiter < 1 or args.restarts < 1 or args.workers < 1:
        raise ValueError("maxiter, restarts, and workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves, points = load_inputs(args.inventory_dir)
    tasks = build_tasks(curves, points)
    results = fit_all_curves(
        tasks,
        output_dir=args.output_dir,
        maxiter=args.maxiter,
        restarts=args.restarts,
        workers=args.workers,
        force_refit=args.force_refit,
    )
    predictions, dense, metrics, parameters = compile_outputs(curves, points, results)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    dense.to_csv(args.output_dir / "dense_curves.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    parameters.to_csv(args.output_dir / "full_fit_parameters.csv", index=False)

    input_hashes = {filename: file_sha256(args.inventory_dir / filename) for filename in INPUT_FILES}
    protocol = {
        "schema_version": 2,
        "purpose": (
            "descriptive full-data DSP-versus-OLMix capacity check across every retained StarCoder endpoint curve"
        ),
        "models": {
            "canonical_dsp": {
                "fit_scope": "independent full-data fit per curve",
                "inner_folds": INNER_FOLDS,
                "maxiter": args.maxiter,
                "restarts": args.restarts,
                "optimizer_sha256": file_sha256(Path(dsp_ladder.__file__).resolve()),
            },
            "olmix_loglinear": {
                "fit_scope": "independent full-data fit per curve",
                "law": "c + exp(beta_nemotron * (1 - p) + beta_starcoder * p)",
                "loss": "summed Huber",
                "huber_delta": olmix_loglinear.DEFAULT_HUBER_DELTA,
                "starts": olmix_loglinear.FIT_N_STARTS,
                "optimizer_sha256": file_sha256(Path(olmix_loglinear.__file__).resolve()),
            },
        },
        "target_id": PRIMARY_TARGET,
        "curve_count": len(curves),
        "curve_ids": curves["curve_id"].tolist(),
        "inner_folds": INNER_FOLDS,
        "fold_assignment": "ordered row index modulo fold count",
        "fold_seed": FOLD_SEED,
        "dense_points": DENSE_POINTS,
        "maxiter": args.maxiter,
        "restarts": args.restarts,
        "workers": args.workers,
        "epoch_scales": {
            "nemotron": epoch_accounting.SIMULATED_EPOCH_TARGET_BUDGET / epoch_accounting.NEMOTRON_SOURCE_TOKENS,
            "starcoder": epoch_accounting.SIMULATED_EPOCH_TARGET_BUDGET / epoch_accounting.STARCODER_SOURCE_TOKENS,
        },
        "materialized_epoch_axis": {
            "lower_axis": "StarCoder fraction p",
            "upper_axis": "StarCoder materialized epochs",
            "full_cache_scale": "planned_materialized_tokens / full_starcoder_source_tokens",
            "simulated_support_scale": "simulated_epoch_target_budget / full_starcoder_source_tokens",
            "support_multipliers": SUPPORT_EPOCH_MULTIPLIERS,
            "full_starcoder_source_tokens": epoch_accounting.STARCODER_SOURCE_TOKENS,
            "simulated_epoch_target_budget": epoch_accounting.SIMULATED_EPOCH_TARGET_BUDGET,
        },
        "input_hashes": input_hashes,
        "inventory_manifest_sha256": file_sha256(args.inventory_dir / "manifest.json"),
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "index.html").write_text(
        build_html(curves, predictions, dense, metrics),
        encoding="utf-8",
    )
    write_report(args.output_dir, metrics)
    print(metrics.sort_values("full_fit_rmse", ascending=False).head(10).to_string(index=False), flush=True)
    print(f"Wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
