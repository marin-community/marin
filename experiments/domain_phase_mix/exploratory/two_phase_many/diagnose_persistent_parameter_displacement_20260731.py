# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
#   "wandb",
# ]
# ///
"""Test radial parameter displacement as an observed persistent phase state.

The phase-switch shock identified by SUR-069/070 is transient and its direct
exponential persistence was rejected by SUR-071. This diagnostic instead uses
an independently logged quantity, the total parameter norm, to measure whether
phase 1 creates a durable radial parameter displacement relative to an exact
aggregate-matched tied run.

The transition is selected only from parameter-norm telemetry through step
21000. A policy map is selected only against the resulting telemetry state. A
single response scale is fit on pre-final smooth-target trajectories; step
22000 and the final endpoint are temporal falsification rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import wandb
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "persistent_parameter_displacement_20260731"
RELAXATION_DIR = SCRIPT_DIR / "reference_outputs" / "component_phase_relaxation_20260731"
SWITCH_MAP_DIR = SCRIPT_DIR / "reference_outputs" / "policy_predictable_switch_shock_20260731"
RUN_MANIFEST_PATH = RELAXATION_DIR / "run_manifest.csv"
POLICY_FEATURES_PATH = SWITCH_MAP_DIR / "policy_features.csv"
SWITCH_OOF_PATH = SWITCH_MAP_DIR / "oof_predictions.csv"
TRANSFER_PATH = SWITCH_MAP_DIR / "transfer_predictions.csv"

CANDIDATE_ID = "WSD80-SUR-074"
WANDB_PATH = "marin-community/marin"
PARAMETER_NORM_KEY = "params/norm/total"
HISTORY_CACHE_NAME = "parameter_norm_histories.csv"
UNAVAILABLE_CACHE_NAME = "history_unavailable.csv"
PHASE_BOUNDARY_STEP = 18_310
FINAL_STEP = 22_887
FINAL_TELEMETRY_STEP = 22_880
PRE_BASELINE_WINDOW = (18_110, 18_300)
TRANSITION_FIT_WINDOW = (18_310, 21_000)
TRANSITION_HOLDOUT_STEPS = (22_000, FINAL_TELEMETRY_STEP)
RESPONSE_FIT_STEPS = (19_000, 20_000, 21_000)
RESPONSE_HOLDOUT_STEPS = (22_000, FINAL_STEP)
RATE_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
FEATURE_BLOCKS = {
    "family_shift": ("shift__",),
    "cross_phase": ("shift__", "unfamiliarity__"),
}
BOOTSTRAP_SEED = 20_260_731
BOOTSTRAP_SAMPLES = 5_000
FETCH_ATTEMPTS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preregister", "materialize", "evaluate"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-workers", type=int, default=24)
    return parser.parse_args()


def canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def protocol_payload() -> dict[str, object]:
    return {
        "candidate_id": CANDIDATE_ID,
        "title": "Persistent radial parameter displacement",
        "scope": "development identification diagnostic; not an endpoint surrogate",
        "nearest_prior_routes": ["WSD80-SUR-060", "WSD80-SUR-061", "WSD80-SUR-069", "WSD80-SUR-070", "WSD80-SUR-071"],
        "material_novelty": (
            "An independently logged persistent parameter state replaces the rejected assumption that the "
            "boundary gradient shock itself persists. The state is a paired log total-parameter-norm displacement, "
            "not an endpoint residual, policy distance, or additional RPL coefficient."
        ),
        "data_use": {
            "exposed_before_freeze": [
                "SUR-069/070 shock results and SUR-071 rejection",
                "all historical endpoint outcomes",
                "W&B history-key availability for one one-phase run",
                "the unpaired parameter-norm range for that one run",
            ],
            "not_inspected_before_freeze": [
                "any asymmetric-minus-tied parameter-norm displacement",
                "any relationship between parameter displacement and BPB",
                "any policy prediction of parameter displacement",
            ],
            "interpretation": "Exposed development evidence; any positive result still requires untouched confirmation.",
        },
        "observed_state": {
            "raw_gap": "n_p(s)=log(||theta_2p(s)||_2)-log(||theta_tied(s)||_2)",
            "baseline": "median n_p(s) over steps 18110--18300",
            "displacement": "d_p(s)=n_p(s)-baseline",
            "units": "dimensionless log parameter-norm ratio",
            "tied_invariant": "d=0 for a policy compared with itself",
        },
        "transition": {
            "phase_progress": "s=(step-18310)/(22887-18310)",
            "law": "d_p(s)=q_p*g_k(s), g_k(s)=(1-exp(-k*s))/(1-exp(-k)); g_0(s)=s",
            "interpretation": "q_p is terminal radial displacement; k is a shared dimensionless saturation rate",
            "rate_grid": RATE_GRID,
            "selection": "minimum mean per-pair telemetry RMSE through step 21000",
            "temporal_holdouts": TRANSITION_HOLDOUT_STEPS,
            "outer_stability": "repeat rate selection after withholding each frozen SUR-070 mixture block",
        },
        "policy_map": {
            "features": {
                "family_shift": "SUR-070 predeclared phase-1 family-mass contrasts",
                "cross_phase": "family shift plus SUR-070 counterfactual late unfamiliarity",
            },
            "estimator": "zero-intercept ridge in physical feature units with training-fold RMS scaling",
            "ridge_grid": RIDGE_GRID,
            "folds": "reuse frozen SUR-070 outer mixture blocks; nested ridge selection over remaining blocks",
        },
        "response": {
            "target": "SUR-068 common smooth-target residual",
            "law": "DeltaL_p(s)=a*qhat_p*g_k(s)",
            "parameters": "one signed BPB-per-log-norm response scale a per outer training fold",
            "fit_steps": RESPONSE_FIT_STEPS,
            "temporal_holdouts": RESPONSE_HOLDOUT_STEPS,
            "ablations": ["zero correction", "static terminal state a*qhat_p"],
            "intercept": False,
        },
        "gates": {
            "transition_rate_interior": True,
            "transition_rate_outer_mode_fraction_min": 0.8,
            "transition_fit_improvement_over_linear_min": 0.05,
            "transition_vs_linear_bootstrap_upper_max": 0.0,
            "transition_holdout_no_worse_than_linear": True,
            "state_rank_21000_to_22000_min": 0.60,
            "policy_cross_phase_spearman_min": 0.40,
            "policy_cross_phase_zero_improvement_min": 0.20,
            "policy_cross_phase_vs_shift_bootstrap_upper_max": 0.0,
            "policy_cross_phase_vs_shift_fold_wins_min": 4,
            "response_fit_zero_improvement_min": 0.15,
            "response_dynamic_vs_static_bootstrap_upper_max": 0.0,
            "response_fold_sign_agreement_min": 4,
            "response_step22000_zero_improvement_min": 0.10,
            "response_final_zero_improvement_min": 0.10,
            "response_final_spearman_min": 0.20,
        },
        "forbidden_repairs": [
            "another exponential shock decay or persistent shock offset",
            "endpoint-selected transition rate or state amplitude",
            "per-bucket parameter-norm response coefficients",
            "component-specific rates or a second timescale",
            "intercept, output calibration, or final-endpoint retuning",
            "changing the rate, ridge, feature, or gate grids after materialization",
        ],
        "decision_boundary": (
            "A pass licenses this telemetry-identified transition for a nested temporal-state ablation after the "
            "aggregate spine is selected. A failure closes scalar radial displacement; it does not justify using "
            "the observed state itself as a deployment feature."
        ),
    }


def wrapped_protocol() -> dict[str, object]:
    payload = protocol_payload()
    digest = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    return {"protocol_sha256": digest, "protocol": payload}


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def freeze_protocol(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = wrapped_protocol()
    path = output_dir / "protocol.json"
    if path.exists():
        observed = json.loads(path.read_text())
        if canonical_json(observed) != canonical_json(expected):
            raise RuntimeError(f"Existing protocol differs from current code: {path}")
    else:
        write_json(path, expected)
    print(expected["protocol_sha256"])


def require_frozen_protocol(output_dir: Path) -> None:
    path = output_dir / "protocol.json"
    if not path.exists():
        raise RuntimeError("Run --mode preregister before materialization or evaluation")
    if canonical_json(json.loads(path.read_text())) != canonical_json(wrapped_protocol()):
        raise RuntimeError("Frozen protocol does not match current evaluation code")


def fetch_history(run_id: str) -> pd.DataFrame:
    keys = ["global_step", PARAMETER_NORM_KEY]
    records: list[dict[str, object]] | None = None
    last_error: Exception | None = None
    for attempt in range(FETCH_ATTEMPTS):
        try:
            run = wandb.Api(timeout=120).run(f"{WANDB_PATH}/{run_id}")
            records = list(
                run.scan_history(
                    keys=keys,
                    page_size=1_000,
                    min_step=PRE_BASELINE_WINDOW[0],
                    max_step=FINAL_STEP,
                )
            )
            break
        except Exception as error:
            last_error = error
            if attempt + 1 < FETCH_ATTEMPTS:
                time.sleep(2**attempt)
    if records is None:
        raise RuntimeError(f"W&B history fetch exhausted retries for {run_id}") from last_error
    history = pd.DataFrame.from_records(records)
    missing = [key for key in keys if key not in history]
    if missing:
        raise ValueError(f"Run {run_id} lacks history keys: {missing}")
    history = history.loc[history[PARAMETER_NORM_KEY].notna(), keys].copy()
    if history.empty:
        raise ValueError(f"Run {run_id} has no finite {PARAMETER_NORM_KEY} rows in the requested window")
    history["global_step"] = history["global_step"].astype(int)
    history = history.groupby("global_step", sort=True, as_index=False).last()
    history["wandb_run_id"] = run_id
    return history


def materialize_histories(output_dir: Path, max_workers: int) -> pd.DataFrame:
    cache_path = output_dir / HISTORY_CACHE_NAME
    unavailable_path = output_dir / UNAVAILABLE_CACHE_NAME
    manifest = pd.read_csv(RUN_MANIFEST_PATH)
    expected_ids = set(manifest["wandb_run_id"].astype(str))
    cached = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()
    unavailable = (
        pd.read_csv(unavailable_path) if unavailable_path.exists() else pd.DataFrame(columns=["wandb_run_id", "reason"])
    )
    cached_ids = set(cached.get("wandb_run_id", pd.Series(dtype=str)).astype(str))
    unavailable_ids = set(unavailable["wandb_run_id"].astype(str))
    pending = sorted(expected_ids - cached_ids - unavailable_ids)
    blocks = [cached] if not cached.empty else []
    unavailable_rows = unavailable.to_dict("records")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_history, run_id): run_id for run_id in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            run_id = futures[future]
            try:
                blocks.append(future.result())
            except ValueError as error:
                unavailable_rows.append({"wandb_run_id": run_id, "reason": str(error)})
            except Exception as error:
                raise RuntimeError(f"Failed to fetch parameter history for {run_id}") from error
            if index % 25 == 0:
                pd.concat(blocks, ignore_index=True).to_csv(cache_path, index=False)
    result = pd.concat(blocks, ignore_index=True)
    result = result.drop_duplicates(["wandb_run_id", "global_step"], keep="last")
    result.to_csv(cache_path, index=False)
    unavailable = pd.DataFrame(unavailable_rows, columns=["wandb_run_id", "reason"]).drop_duplicates(
        "wandb_run_id", keep="last"
    )
    unavailable.to_csv(unavailable_path, index=False)
    accounted = set(result["wandb_run_id"].astype(str)) | set(unavailable["wandb_run_id"].astype(str))
    if accounted != expected_ids:
        raise RuntimeError("History materialization does not account for every run")
    return result


def paired_state_histories(histories: pd.DataFrame) -> pd.DataFrame:
    manifest = pd.read_csv(RUN_MANIFEST_PATH)
    joined = manifest.merge(histories, on="wandb_run_id", how="inner", validate="one_to_many")
    one = joined.loc[joined["policy_class"] == "one_phase", ["pair_id", "global_step", PARAMETER_NORM_KEY]].rename(
        columns={PARAMETER_NORM_KEY: "norm_one"}
    )
    two = joined.loc[joined["policy_class"] == "two_phase", ["pair_id", "global_step", PARAMETER_NORM_KEY]].rename(
        columns={PARAMETER_NORM_KEY: "norm_two"}
    )
    paired = one.merge(two, on=["pair_id", "global_step"], how="inner", validate="one_to_one")
    paired = paired.loc[(paired["norm_one"] > 0) & (paired["norm_two"] > 0)].copy()
    paired["log_norm_gap"] = np.log(paired["norm_two"]) - np.log(paired["norm_one"])
    baseline = (
        paired.loc[paired["global_step"].between(*PRE_BASELINE_WINDOW)]
        .groupby("pair_id", sort=False)["log_norm_gap"]
        .median()
        .rename("pre_baseline")
    )
    paired = paired.merge(baseline, on="pair_id", how="inner", validate="many_to_one")
    paired["displacement"] = paired["log_norm_gap"] - paired["pre_baseline"]
    paired["phase_progress"] = (paired["global_step"] - PHASE_BOUNDARY_STEP) / (FINAL_STEP - PHASE_BOUNDARY_STEP)
    return paired.sort_values(["pair_id", "global_step"]).reset_index(drop=True)


def transition_shape(progress: np.ndarray, rate: float) -> np.ndarray:
    progress = np.asarray(progress, dtype=float)
    if rate == 0:
        return progress
    return -np.expm1(-rate * progress) / -np.expm1(-rate)


def fit_pair_amplitudes(frame: pd.DataFrame, rate: float) -> pd.DataFrame:
    rows = []
    for pair_id, block in frame.groupby("pair_id", sort=True):
        shape = transition_shape(block["phase_progress"].to_numpy(), rate)
        observed = block["displacement"].to_numpy(dtype=float)
        denominator = float(shape @ shape)
        if denominator <= 1e-14:
            continue
        amplitude = float(shape @ observed / denominator)
        prediction = amplitude * shape
        rows.append(
            {
                "pair_id": pair_id,
                "terminal_displacement": amplitude,
                "telemetry_rmse": rmse(observed, prediction),
                "telemetry_rows": len(block),
            }
        )
    return pd.DataFrame(rows)


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean(np.square(predicted - observed))))


def safe_spearman(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if np.std(observed) < 1e-14 or np.std(predicted) < 1e-14:
        return 0.0
    return float(spearmanr(observed, predicted).statistic)


def bootstrap_rmse_difference(
    frame: pd.DataFrame,
    prediction_a: str,
    prediction_b: str,
    seed: int,
) -> tuple[float, float, float]:
    grouped = {key: block.index.to_numpy() for key, block in frame.groupby("pair_id", sort=True)}
    keys = np.asarray(list(grouped), dtype=object)
    rng = np.random.default_rng(seed)
    differences = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    for index in range(BOOTSTRAP_SAMPLES):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        rows = np.concatenate([grouped[key] for key in sampled])
        block = frame.loc[rows]
        differences[index] = rmse(block["observed"], block[prediction_a]) - rmse(block["observed"], block[prediction_b])
    return (
        float(np.mean(differences)),
        float(np.quantile(differences, 0.025)),
        float(np.quantile(differences, 0.975)),
    )


def bootstrap_pair_mean_difference(
    frame: pd.DataFrame,
    value_a: str,
    value_b: str,
    seed: int,
) -> tuple[float, float, float]:
    if frame["pair_id"].duplicated().any():
        raise ValueError("Pair-mean bootstrap requires one row per pair")
    rng = np.random.default_rng(seed)
    differences = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    values_a = frame[value_a].to_numpy(dtype=float)
    values_b = frame[value_b].to_numpy(dtype=float)
    for index in range(BOOTSTRAP_SAMPLES):
        sampled = rng.integers(0, len(frame), size=len(frame))
        differences[index] = float(np.mean(values_a[sampled] - values_b[sampled]))
    return (
        float(np.mean(differences)),
        float(np.quantile(differences, 0.025)),
        float(np.quantile(differences, 0.975)),
    )


def outer_fold_map() -> pd.DataFrame:
    predictions = pd.read_csv(SWITCH_OOF_PATH)
    folds = predictions.loc[predictions["target"] == "gradient_log_jump", ["pair_id", "outer_fold"]]
    return folds.drop_duplicates("pair_id", keep="last")


def select_transition(state_fit: pd.DataFrame, folds: pd.DataFrame) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    pair_folds = folds.set_index("pair_id")["outer_fold"]
    metrics = []
    for rate in RATE_GRID:
        amplitudes = fit_pair_amplitudes(state_fit, rate)
        metrics.append({"rate": rate, "mean_pair_rmse": float(amplitudes["telemetry_rmse"].mean())})
    metrics_frame = pd.DataFrame(metrics)
    selected = float(metrics_frame.sort_values(["mean_pair_rmse", "rate"]).iloc[0]["rate"])
    outer_rows = []
    for outer_fold in sorted(folds["outer_fold"].unique()):
        train_ids = set(pair_folds.index[pair_folds != outer_fold])
        train = state_fit.loc[state_fit["pair_id"].isin(train_ids)]
        candidates = []
        for rate in RATE_GRID:
            candidates.append((fit_pair_amplitudes(train, rate)["telemetry_rmse"].mean(), rate))
        outer_rows.append({"outer_fold": int(outer_fold), "selected_rate": float(min(candidates)[1])})
    return selected, metrics_frame, pd.DataFrame(outer_rows)


def feature_columns(features: pd.DataFrame, block: str) -> list[str]:
    prefixes = FEATURE_BLOCKS[block]
    return [column for column in features.columns if column != "pair_id" and column.startswith(prefixes)]


def fit_ridge(features: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    scale = np.sqrt(np.mean(np.square(features), axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    scaled = features / scale
    coefficients = np.linalg.solve(scaled.T @ scaled + alpha * np.eye(scaled.shape[1]), scaled.T @ target)
    return coefficients / scale


def select_ridge(
    features: np.ndarray,
    target: np.ndarray,
    fold_labels: np.ndarray,
    train_mask: np.ndarray,
) -> float:
    candidates = []
    for alpha in RIDGE_GRID:
        prediction = np.full(len(target), np.nan)
        for inner_fold in np.unique(fold_labels[train_mask]):
            inner_test = train_mask & (fold_labels == inner_fold)
            inner_train = train_mask & ~inner_test
            coefficients = fit_ridge(features[inner_train], target[inner_train], alpha)
            prediction[inner_test] = features[inner_test] @ coefficients
        candidates.append((rmse(target[train_mask], prediction[train_mask]), alpha))
    return float(min(candidates)[1])


def policy_oof_predictions(amplitudes: pd.DataFrame, folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(POLICY_FEATURES_PATH)
    data = amplitudes.merge(features, on="pair_id", how="inner", validate="one_to_one")
    data = data.merge(folds, on="pair_id", how="inner", validate="one_to_one")
    parameter_rows = []
    for block in FEATURE_BLOCKS:
        columns = feature_columns(data, block)
        matrix = data[columns].to_numpy(dtype=float)
        target = data["terminal_displacement"].to_numpy(dtype=float)
        labels = data["outer_fold"].to_numpy(dtype=int)
        prediction = np.full(len(data), np.nan)
        for outer_fold in sorted(np.unique(labels)):
            test = labels == outer_fold
            train = ~test
            alpha = select_ridge(matrix, target, labels, train)
            coefficients = fit_ridge(matrix[train], target[train], alpha)
            prediction[test] = matrix[test] @ coefficients
            for feature, coefficient in zip(columns, coefficients, strict=True):
                parameter_rows.append(
                    {
                        "block": block,
                        "outer_fold": int(outer_fold),
                        "ridge": alpha,
                        "feature": feature,
                        "coefficient": float(coefficient),
                    }
                )
        data[f"predicted__{block}"] = prediction
    return data, pd.DataFrame(parameter_rows)


def response_predictions(
    policy_predictions: pd.DataFrame,
    selected_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    transfer = pd.read_csv(TRANSFER_PATH)
    columns = ["pair_id", "global_step", "common_residual"]
    data = transfer[columns].merge(
        policy_predictions[["pair_id", "outer_fold", "predicted__cross_phase"]],
        on="pair_id",
        how="inner",
        validate="many_to_one",
    )
    data["phase_progress"] = (data["global_step"] - PHASE_BOUNDARY_STEP) / (FINAL_STEP - PHASE_BOUNDARY_STEP)
    data["state_dynamic"] = data["predicted__cross_phase"] * transition_shape(data["phase_progress"], selected_rate)
    data["state_static"] = data["predicted__cross_phase"]
    data["observed"] = data["common_residual"]
    data["predicted__zero"] = 0.0
    data["predicted__dynamic"] = np.nan
    data["predicted__static"] = np.nan
    fit_steps = data["global_step"].isin(RESPONSE_FIT_STEPS)
    parameter_rows = []
    for outer_fold in sorted(data["outer_fold"].unique()):
        test = data["outer_fold"] == outer_fold
        train = ~test & fit_steps
        for name in ("dynamic", "static"):
            state = data[f"state_{name}"].to_numpy(dtype=float)
            observed = data["observed"].to_numpy(dtype=float)
            denominator = float(state[train] @ state[train])
            amplitude = 0.0 if denominator <= 1e-14 else float(state[train] @ observed[train] / denominator)
            data.loc[test, f"predicted__{name}"] = amplitude * state[test]
            parameter_rows.append({"outer_fold": int(outer_fold), "model": name, "response_scale": amplitude})
    return data, pd.DataFrame(parameter_rows)


def metric_row(frame: pd.DataFrame, prediction: str, scope: str) -> dict[str, object]:
    observed = frame["observed"].to_numpy(dtype=float)
    predicted = frame[prediction].to_numpy(dtype=float)
    zero_rmse = rmse(observed, np.zeros_like(observed))
    prediction_rmse = rmse(observed, predicted)
    return {
        "scope": scope,
        "model": prediction.removeprefix("predicted__"),
        "rows": len(frame),
        "pairs": frame["pair_id"].nunique(),
        "rmse": prediction_rmse,
        "zero_improvement": 1.0 - prediction_rmse / zero_rmse,
        "spearman": safe_spearman(observed, predicted),
        "bias": float(np.mean(predicted - observed)),
        "amplitude_ratio": float(np.std(predicted) / np.std(observed)) if np.std(observed) > 0 else 0.0,
    }


def nearest_step_state(state: pd.DataFrame, step: int) -> pd.DataFrame:
    block = state.copy()
    block["distance"] = np.abs(block["global_step"] - step)
    return block.sort_values(["pair_id", "distance"]).drop_duplicates("pair_id", keep="first")


def evaluate(output_dir: Path) -> None:
    histories = pd.read_csv(output_dir / HISTORY_CACHE_NAME)
    state = paired_state_histories(histories)
    folds = outer_fold_map()
    fit_state = state.loc[state["global_step"].between(*TRANSITION_FIT_WINDOW) & (state["phase_progress"] >= 0)].copy()
    selected_rate, transition_metrics, outer_rate_amplitudes = select_transition(fit_state, folds)
    selected_amplitudes = fit_pair_amplitudes(fit_state, selected_rate)
    linear_amplitudes = fit_pair_amplitudes(fit_state, 0.0).rename(
        columns={"terminal_displacement": "linear_terminal_displacement", "telemetry_rmse": "linear_telemetry_rmse"}
    )
    selected_amplitudes = selected_amplitudes.merge(
        linear_amplitudes[["pair_id", "linear_terminal_displacement", "linear_telemetry_rmse"]],
        on="pair_id",
        how="inner",
        validate="one_to_one",
    )

    telemetry_comparison = selected_amplitudes.rename(columns={"telemetry_rmse": "candidate_rmse"}).copy()
    telemetry_bootstrap = bootstrap_pair_mean_difference(
        telemetry_comparison,
        "candidate_rmse",
        "linear_telemetry_rmse",
        BOOTSTRAP_SEED,
    )

    holdout_rows = []
    for step in TRANSITION_HOLDOUT_STEPS:
        observed = nearest_step_state(state.loc[state["global_step"] >= PHASE_BOUNDARY_STEP], step)
        comparison = observed.merge(selected_amplitudes, on="pair_id", how="inner", validate="one_to_one")
        progress = (step - PHASE_BOUNDARY_STEP) / (FINAL_STEP - PHASE_BOUNDARY_STEP)
        comparison["candidate_prediction"] = (
            comparison["terminal_displacement"] * transition_shape(np.asarray([progress]), selected_rate)[0]
        )
        comparison["linear_prediction"] = comparison["linear_terminal_displacement"] * progress
        holdout_rows.append(
            {
                "global_step": step,
                "pairs": len(comparison),
                "candidate_rmse": rmse(comparison["displacement"], comparison["candidate_prediction"]),
                "linear_rmse": rmse(comparison["displacement"], comparison["linear_prediction"]),
            }
        )
    holdout_metrics = pd.DataFrame(holdout_rows)

    state_21000 = nearest_step_state(state, 21_000)[["pair_id", "displacement"]].rename(
        columns={"displacement": "displacement_21000"}
    )
    state_22000 = nearest_step_state(state, 22_000)[["pair_id", "displacement"]].rename(
        columns={"displacement": "displacement_22000"}
    )
    rank_frame = state_21000.merge(state_22000, on="pair_id", how="inner", validate="one_to_one")
    state_rank = safe_spearman(rank_frame["displacement_21000"], rank_frame["displacement_22000"])

    policy_predictions, policy_parameters = policy_oof_predictions(selected_amplitudes, folds)
    policy_metrics = []
    for block in FEATURE_BLOCKS:
        prediction = f"predicted__{block}"
        policy_metrics.append(
            {
                "block": block,
                "rows": len(policy_predictions),
                "rmse": rmse(policy_predictions["terminal_displacement"], policy_predictions[prediction]),
                "zero_improvement": (
                    1.0
                    - rmse(policy_predictions["terminal_displacement"], policy_predictions[prediction])
                    / rmse(policy_predictions["terminal_displacement"], np.zeros(len(policy_predictions)))
                ),
                "spearman": safe_spearman(policy_predictions["terminal_displacement"], policy_predictions[prediction]),
            }
        )
    policy_metrics_frame = pd.DataFrame(policy_metrics)
    policy_comparison = policy_predictions.rename(columns={"terminal_displacement": "observed"})
    policy_bootstrap = bootstrap_rmse_difference(
        policy_comparison,
        "predicted__cross_phase",
        "predicted__family_shift",
        BOOTSTRAP_SEED + 1,
    )
    fold_wins = 0
    for outer_fold in sorted(policy_predictions["outer_fold"].unique()):
        block = policy_predictions.loc[policy_predictions["outer_fold"] == outer_fold]
        cross = rmse(block["terminal_displacement"], block["predicted__cross_phase"])
        shift = rmse(block["terminal_displacement"], block["predicted__family_shift"])
        fold_wins += int(cross < shift)

    response, response_parameters = response_predictions(policy_predictions, selected_rate)
    response_metrics = []
    scopes = {"fit": RESPONSE_FIT_STEPS, "step22000": (22_000,), "final": (FINAL_STEP,)}
    for scope, steps in scopes.items():
        block = response.loc[response["global_step"].isin(steps)]
        for prediction in ("predicted__zero", "predicted__static", "predicted__dynamic"):
            response_metrics.append(metric_row(block, prediction, scope))
    response_metrics_frame = pd.DataFrame(response_metrics)
    fit_response = response.loc[response["global_step"].isin(RESPONSE_FIT_STEPS)]
    response_bootstrap = bootstrap_rmse_difference(
        fit_response,
        "predicted__dynamic",
        "predicted__static",
        BOOTSTRAP_SEED + 2,
    )
    dynamic_signs = response_parameters.loc[response_parameters["model"] == "dynamic", "response_scale"]
    sign_agreement = int(max((dynamic_signs > 0).sum(), (dynamic_signs < 0).sum()))

    outer_rates = outer_rate_amplitudes[["outer_fold", "selected_rate"]].drop_duplicates()
    mode_fraction = float(outer_rates["selected_rate"].value_counts(normalize=True).max())
    transition_improvement = 1.0 - float(selected_amplitudes["telemetry_rmse"].mean()) / float(
        selected_amplitudes["linear_telemetry_rmse"].mean()
    )
    selected_interior = selected_rate not in (RATE_GRID[0], RATE_GRID[-1])
    holdout_no_worse = bool((holdout_metrics["candidate_rmse"] <= holdout_metrics["linear_rmse"]).all())
    policy_cross = policy_metrics_frame.set_index("block").loc["cross_phase"]
    response_index = response_metrics_frame.set_index(["scope", "model"])
    response_fit_improvement = float(response_index.loc[("fit", "dynamic"), "zero_improvement"])
    response_step22000_improvement = float(response_index.loc[("step22000", "dynamic"), "zero_improvement"])
    response_final_improvement = float(response_index.loc[("final", "dynamic"), "zero_improvement"])
    response_final_spearman = float(response_index.loc[("final", "dynamic"), "spearman"])
    gate_rows = [
        ("transition_rate_interior", selected_interior, selected_rate),
        ("transition_rate_outer_mode_fraction", mode_fraction >= 0.8, mode_fraction),
        ("transition_fit_improvement_over_linear", transition_improvement >= 0.05, transition_improvement),
        ("transition_vs_linear_bootstrap_upper", telemetry_bootstrap[2] <= 0.0, telemetry_bootstrap[2]),
        ("transition_holdout_no_worse_than_linear", holdout_no_worse, holdout_no_worse),
        ("state_rank_21000_to_22000", state_rank >= 0.60, state_rank),
        ("policy_cross_phase_spearman", policy_cross["spearman"] >= 0.40, policy_cross["spearman"]),
        (
            "policy_cross_phase_zero_improvement",
            policy_cross["zero_improvement"] >= 0.20,
            policy_cross["zero_improvement"],
        ),
        ("policy_cross_phase_vs_shift_bootstrap_upper", policy_bootstrap[2] <= 0.0, policy_bootstrap[2]),
        ("policy_cross_phase_vs_shift_fold_wins", fold_wins >= 4, fold_wins),
        (
            "response_fit_zero_improvement",
            response_fit_improvement >= 0.15,
            response_fit_improvement,
        ),
        ("response_dynamic_vs_static_bootstrap_upper", response_bootstrap[2] <= 0.0, response_bootstrap[2]),
        ("response_fold_sign_agreement", sign_agreement >= 4, sign_agreement),
        (
            "response_step22000_zero_improvement",
            response_step22000_improvement >= 0.10,
            response_step22000_improvement,
        ),
        (
            "response_final_zero_improvement",
            response_final_improvement >= 0.10,
            response_final_improvement,
        ),
        (
            "response_final_spearman",
            response_final_spearman >= 0.20,
            response_final_spearman,
        ),
    ]
    gates = pd.DataFrame(gate_rows, columns=["gate", "passed", "value"])
    passed = bool(gates["passed"].all())

    output_dir.mkdir(parents=True, exist_ok=True)
    state.to_csv(output_dir / "paired_parameter_state.csv", index=False)
    transition_metrics.to_csv(output_dir / "transition_metrics.csv", index=False)
    outer_rates.to_csv(output_dir / "outer_selected_rates.csv", index=False)
    selected_amplitudes.to_csv(output_dir / "terminal_state.csv", index=False)
    holdout_metrics.to_csv(output_dir / "transition_holdout_metrics.csv", index=False)
    policy_predictions.to_csv(output_dir / "policy_predictions.csv", index=False)
    policy_parameters.to_csv(output_dir / "policy_parameters.csv", index=False)
    policy_metrics_frame.to_csv(output_dir / "policy_metrics.csv", index=False)
    response.to_csv(output_dir / "response_predictions.csv", index=False)
    response_parameters.to_csv(output_dir / "response_parameters.csv", index=False)
    response_metrics_frame.to_csv(output_dir / "response_metrics.csv", index=False)
    gates.to_csv(output_dir / "acceptance_gate.csv", index=False)

    decision = {
        "candidate_id": CANDIDATE_ID,
        "passed": passed,
        "selected_rate": selected_rate,
        "transition_fit_improvement_over_linear": transition_improvement,
        "transition_bootstrap_candidate_minus_linear": telemetry_bootstrap,
        "state_rank_21000_to_22000": state_rank,
        "policy_bootstrap_cross_phase_minus_shift": policy_bootstrap,
        "policy_cross_phase_vs_shift_fold_wins": fold_wins,
        "response_bootstrap_dynamic_minus_static": response_bootstrap,
        "response_fold_sign_agreement": sign_agreement,
    }
    write_json(output_dir / "decision.json", decision)
    write_report(output_dir, decision, gates, policy_metrics_frame, response_metrics_frame, holdout_metrics)
    write_plot(output_dir, state, selected_amplitudes, policy_predictions, response, selected_rate)
    print(json.dumps(decision, indent=2, sort_keys=True))


def write_report(
    output_dir: Path,
    decision: dict[str, object],
    gates: pd.DataFrame,
    policy_metrics: pd.DataFrame,
    response_metrics: pd.DataFrame,
    holdout_metrics: pd.DataFrame,
) -> None:
    status = "passes" if decision["passed"] else "fails"
    report = f"""# Persistent radial parameter displacement

`{CANDIDATE_ID}` {status} its frozen development gate.

The diagnostic measures the asymmetric-minus-tied change in log total parameter norm relative to its pre-switch
baseline. Its transition and policy map are identified entirely from optimizer telemetry; final BPB is used only as
a temporal falsification outcome.

## Decision

- Selected saturation rate: `{decision['selected_rate']}`.
- Transition improvement over the linear displacement ablation:
  `{decision['transition_fit_improvement_over_linear']:.1%}`.
- Step-21000 to step-22000 state-rank persistence: `{decision['state_rank_21000_to_22000']:.3f}`.
- Overall gate: `{'PASS' if decision['passed'] else 'FAIL'}`.

## Gates

{gates.to_markdown(index=False)}

## Policy map

{policy_metrics.to_markdown(index=False)}

## Smooth-target response

{response_metrics.to_markdown(index=False)}

## Telemetry holdouts

{holdout_metrics.to_markdown(index=False)}

This is exposed development evidence. A pass licenses only a nested temporal-state test after the aggregate spine is
selected; it is not confirmation and does not license using observed optimizer telemetry at deployment.
"""
    (output_dir / "report.md").write_text(report)


def write_plot(
    output_dir: Path,
    state: pd.DataFrame,
    amplitudes: pd.DataFrame,
    policy: pd.DataFrame,
    response: pd.DataFrame,
    selected_rate: float,
) -> None:
    figure = make_subplots(rows=1, cols=3, subplot_titles=("Observed state", "Policy map", "Final response"))
    progress_grid = np.linspace(0, 1, 100)
    quantiles = amplitudes["terminal_displacement"].quantile([0.1, 0.5, 0.9]).to_numpy()
    colors = ("#1a9850", "#fee08b", "#d73027")
    for quantile, color, label in zip(quantiles, colors, ("q10", "median", "q90"), strict=True):
        figure.add_trace(
            go.Scatter(
                x=PHASE_BOUNDARY_STEP + progress_grid * (FINAL_STEP - PHASE_BOUNDARY_STEP),
                y=quantile * transition_shape(progress_grid, selected_rate),
                mode="lines",
                name=f"transition {label}",
                line={"color": color},
            ),
            row=1,
            col=1,
        )
    sampled_pairs = set(
        amplitudes.sort_values("terminal_displacement").iloc[np.linspace(0, len(amplitudes) - 1, 15).astype(int)][
            "pair_id"
        ]
    )
    for pair_id, block in state.loc[
        state["pair_id"].isin(sampled_pairs) & (state["global_step"] >= PHASE_BOUNDARY_STEP)
    ].groupby("pair_id"):
        figure.add_trace(
            go.Scatter(
                x=block["global_step"],
                y=block["displacement"],
                mode="lines",
                line={"width": 1},
                opacity=0.25,
                name=str(pair_id),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Scatter(
            x=policy["terminal_displacement"],
            y=policy["predicted__cross_phase"],
            mode="markers",
            marker={"color": policy["terminal_displacement"], "colorscale": "RdYlGn_r", "size": 6},
            name="OOF policy state",
        ),
        row=1,
        col=2,
    )
    final = response.loc[response["global_step"] == FINAL_STEP]
    figure.add_trace(
        go.Scatter(
            x=final["observed"],
            y=final["predicted__dynamic"],
            mode="markers",
            marker={"color": final["observed"], "colorscale": "RdYlGn_r", "size": 6},
            name="Final holdout",
        ),
        row=1,
        col=3,
    )
    figure.update_xaxes(title_text="Global step", row=1, col=1)
    figure.update_yaxes(title_text="Log-norm displacement", row=1, col=1)
    figure.update_xaxes(title_text="Observed terminal state", row=1, col=2)
    figure.update_yaxes(title_text="OOF predicted state", row=1, col=2)
    figure.update_xaxes(title_text="Observed common residual", row=1, col=3)
    figure.update_yaxes(title_text="Predicted common residual", row=1, col=3)
    figure.update_layout(
        title="Persistent radial parameter displacement", template="plotly_white", height=620, width=1500
    )
    figure.write_html(output_dir / "persistent_parameter_displacement.html", include_plotlyjs=True)


def main() -> None:
    args = parse_args()
    if args.mode == "preregister":
        freeze_protocol(args.output_dir)
        return
    require_frozen_protocol(args.output_dir)
    if args.mode == "materialize":
        histories = materialize_histories(args.output_dir, args.max_workers)
        print(f"materialized {len(histories)} telemetry rows across {histories['wandb_run_id'].nunique()} runs")
        return
    evaluate(args.output_dir)


if __name__ == "__main__":
    main()
