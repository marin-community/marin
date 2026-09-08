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
"""Test architecture-relative parameter redistribution as a phase state.

SUR-074 established a persistent change in total parameter norm but rejected
that scalar radius as a performance state. This diagnostic removes the global
radius and learns one signed module-redistribution direction exclusively from
parameter-norm telemetry. Final BPB never selects the direction, transition,
policy features, or regularization.
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
from diagnose_persistent_parameter_displacement_20260731 import (  # pyrefly: ignore[missing-import]
    BOOTSTRAP_SEED,
    FINAL_STEP,
    FINAL_TELEMETRY_STEP,
    PHASE_BOUNDARY_STEP,
    POLICY_FEATURES_PATH,
    PRE_BASELINE_WINDOW,
    RESPONSE_FIT_STEPS,
    RESPONSE_HOLDOUT_STEPS,
    RIDGE_GRID,
    RUN_MANIFEST_PATH,
    SWITCH_OOF_PATH,
    TRANSFER_PATH,
    TRANSITION_FIT_WINDOW,
    TRANSITION_HOLDOUT_STEPS,
    WANDB_PATH,
    bootstrap_pair_mean_difference,
    bootstrap_rmse_difference,
    canonical_json,
    fit_ridge,
    rmse,
    safe_spearman,
    transition_shape,
    write_json,
)
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "architecture_relative_parameter_state_20260731"
CANDIDATE_ID = "WSD80-SUR-075"
TOTAL_NORM_KEY = "params/norm/total"
PARAMETER_PREFIX = "params/norm/"
EXPECTED_PARAMETER_TENSORS = 110
FETCH_ATTEMPTS = 3
HISTORY_CACHE_NAME = "module_norm_histories.csv"
UNAVAILABLE_CACHE_NAME = "history_unavailable.csv"
TELEMETRY_STEPS = (19_000, 20_000, 21_000, 22_000, FINAL_TELEMETRY_STEP)
MODULE_GROUPS = (
    "embeddings",
    "input_layernorm",
    "post_attention_layernorm",
    "attention_q",
    "attention_k",
    "attention_v",
    "attention_o",
    "mlp_gate",
    "mlp_up",
    "mlp_down",
    "final_norm",
)
FEATURE_BLOCKS = {
    "shift": ("shift__",),
    "shift_repetition": ("shift__", "late_repetition__"),
    "shift_unfamiliarity": ("shift__", "unfamiliarity__"),
    "full": ("shift__", "late_repetition__", "unfamiliarity__"),
}
RATE_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
PCA_BOOTSTRAPS = 2_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preregister", "materialize", "evaluate"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-workers", type=int, default=24)
    return parser.parse_args()


def protocol_payload() -> dict[str, object]:
    return {
        "candidate_id": CANDIDATE_ID,
        "title": "Architecture-relative parameter redistribution",
        "scope": "development identification diagnostic; not an endpoint surrogate",
        "nearest_prior_routes": [
            "WSD80-SUR-060",
            "WSD80-SUR-061",
            "WSD80-SUR-062",
            "WSD80-SUR-069",
            "WSD80-SUR-070",
            "WSD80-SUR-071",
            "WSD80-SUR-074",
        ],
        "material_novelty": (
            "SUR-074 retained only the global parameter radius. This diagnostic removes that radius and measures "
            "signed redistribution among eleven predeclared architecture modules. One uncentered principal direction "
            "is learned from telemetry alone, so there are no per-layer BPB coefficients or endpoint-selected "
            "state axes."
        ),
        "data_use": {
            "exposed_before_freeze": [
                "all SUR-074 scalar-radius results",
                "W&B key names and key counts for 20 representative runs",
                "the fact that 110 per-tensor parameter norms are logged",
            ],
            "not_inspected_before_freeze": [
                "any asymmetric-minus-tied module-relative displacement",
                "any telemetry principal direction or explained energy",
                "any relationship between module-relative displacement and BPB",
            ],
            "interpretation": "Exposed development evidence; a pass cannot confirm a final surrogate.",
        },
        "observed_state": {
            "module_groups": MODULE_GROUPS,
            "group_norm": "||theta_g||=(sum_{j in g} ||theta_j||^2)^1/2",
            "relative_gap": (
                "u_pg(s)=log(||theta_2p,g||/||theta_2p,total||)-" "log(||theta_tied,g||/||theta_tied,total||)"
            ),
            "baseline": "d_pg(s)=u_pg(s)-median_pre_switch[u_pg]",
            "direction": "first uncentered right singular vector of d_p(21000), telemetry only",
            "score": "q_p(s)=v^T d_p(s)",
            "units": "dimensionless log relative-norm ratio",
            "radial_invariant": "multiplying every parameter norm by one common factor leaves d unchanged",
            "tied_invariant": "d=0 for a policy compared with itself",
        },
        "transition": {
            "phase_progress": "s=(step-18310)/(22887-18310)",
            "law": "q_p(s)=a_p*g_k(s), g_k(s)=(1-exp(-k*s))/(1-exp(-k)); g_0(s)=s",
            "rate_grid": RATE_GRID,
            "selection": "minimum mean per-pair telemetry RMSE through step 21000",
            "temporal_holdouts": TRANSITION_HOLDOUT_STEPS,
        },
        "policy_map": {
            "candidate_blocks": FEATURE_BLOCKS,
            "selection": "nested block and ridge selection using telemetry scores only",
            "ridge_grid": RIDGE_GRID,
            "folds": "frozen SUR-070 mixture blocks",
            "intercept": False,
        },
        "response": {
            "target": "SUR-068 common smooth-target residual",
            "law": "DeltaL_p(s)=c*qhat_p*g_k(s)",
            "parameters": "one signed BPB-per-state response scale per outer training fold",
            "fit_steps": RESPONSE_FIT_STEPS,
            "temporal_holdouts": RESPONSE_HOLDOUT_STEPS,
            "ablations": ["zero correction", "static terminal state", "SUR-074 scalar radial state"],
            "intercept": False,
        },
        "gates": {
            "direction_explained_energy_min": 0.25,
            "direction_outer_cosine_min": 0.80,
            "direction_bootstrap_cosine_low_min": 0.80,
            "transition_rate_interior": True,
            "transition_outer_mode_fraction_min": 0.80,
            "transition_fit_improvement_over_linear_min": 0.05,
            "transition_vs_linear_bootstrap_upper_max": 0.0,
            "transition_holdout_no_worse_than_linear": True,
            "state_rank_21000_to_22000_min": 0.60,
            "policy_spearman_min": 0.40,
            "policy_zero_improvement_min": 0.20,
            "response_fit_zero_improvement_min": 0.15,
            "response_dynamic_vs_static_bootstrap_upper_max": 0.0,
            "response_fold_sign_agreement_min": 4,
            "response_step22000_zero_improvement_min": 0.10,
            "response_final_zero_improvement_min": 0.10,
            "response_final_spearman_min": 0.20,
        },
        "forbidden_repairs": [
            "per-layer or per-module BPB response coefficients",
            "more than one telemetry direction after seeing endpoint outcomes",
            "endpoint-selected PCA centering, scaling, component count, or sign",
            "another transition timescale or persistent offset",
            "endpoint calibration, intercept, or final-endpoint retuning",
            "changing feature, ridge, rate, or gate grids after materialization",
        ],
        "decision_boundary": (
            "A pass licenses one architecture-relative state for a nested surrogate ablation after the aggregate "
            "spine is selected. A failure closes existing norm telemetry as a useful temporal state and leaves a "
            "switch-time intervention as the admissible next identification route."
        ),
    }


def wrapped_protocol() -> dict[str, object]:
    payload = protocol_payload()
    digest = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    return {"protocol_sha256": digest, "protocol": payload}


def freeze_protocol(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = wrapped_protocol()
    path = output_dir / "protocol.json"
    if path.exists():
        if canonical_json(json.loads(path.read_text())) != canonical_json(expected):
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


def module_group(key: str) -> str:
    suffix = key.removeprefix(PARAMETER_PREFIX)
    if suffix == "embeddings.token_embeddings.weight":
        return "embeddings"
    if suffix == "transformer.norm.weight":
        return "final_norm"
    patterns = {
        ".input_layernorm.weight": "input_layernorm",
        ".post_attention_layernorm.weight": "post_attention_layernorm",
        ".self_attn.q_proj.weight": "attention_q",
        ".self_attn.k_proj.weight": "attention_k",
        ".self_attn.v_proj.weight": "attention_v",
        ".self_attn.o_proj.weight": "attention_o",
        ".mlp.gate_proj.weight": "mlp_gate",
        ".mlp.up_proj.weight": "mlp_up",
        ".mlp.down_proj.weight": "mlp_down",
    }
    matches = [group for pattern, group in patterns.items() if pattern in suffix]
    if len(matches) != 1:
        raise ValueError(f"Cannot assign parameter norm key to exactly one group: {key}")
    return matches[0]


def fetch_history(run_id: str) -> pd.DataFrame:
    records: list[dict[str, object]] | None = None
    last_error: Exception | None = None
    for attempt in range(FETCH_ATTEMPTS):
        try:
            run = wandb.Api(timeout=120).run(f"{WANDB_PATH}/{run_id}")
            history_keys = sorted(
                key
                for key in run._attrs["historyKeys"]["keys"]
                if key.startswith(PARAMETER_PREFIX) and key != TOTAL_NORM_KEY
            )
            if len(history_keys) != EXPECTED_PARAMETER_TENSORS:
                raise ValueError(
                    f"Run {run_id} has {len(history_keys)} parameter tensors; expected {EXPECTED_PARAMETER_TENSORS}"
                )
            grouped: dict[str, list[str]] = {group: [] for group in MODULE_GROUPS}
            for key in history_keys:
                grouped[module_group(key)].append(key)
            if any(not keys for keys in grouped.values()):
                raise ValueError(f"Run {run_id} has an empty module group")
            requested = ["global_step", TOTAL_NORM_KEY, *history_keys]
            records = list(
                run.scan_history(
                    keys=requested,
                    page_size=1_000,
                    min_step=PRE_BASELINE_WINDOW[0],
                    max_step=FINAL_STEP,
                )
            )
            history = pd.DataFrame.from_records(records)
            if history.empty:
                raise ValueError(f"Run {run_id} has no parameter-norm rows in the requested window")
            keep = history["global_step"].between(*PRE_BASELINE_WINDOW) | history["global_step"].isin(TELEMETRY_STEPS)
            history = history.loc[keep].copy()
            if history.empty or TOTAL_NORM_KEY not in history:
                raise ValueError(f"Run {run_id} has no finite total parameter norms in the requested windows")
            result = history[["global_step", TOTAL_NORM_KEY]].copy()
            for group, keys in grouped.items():
                values = history[keys].to_numpy(dtype=float)
                result[f"module_norm__{group}"] = np.sqrt(np.nansum(np.square(values), axis=1))
            required = [TOTAL_NORM_KEY, *(f"module_norm__{group}" for group in MODULE_GROUPS)]
            required_values = result[required].to_numpy(dtype=float)
            valid = np.isfinite(required_values).all(axis=1) & (required_values > 0).all(axis=1)
            result = result.loc[pd.Series(np.asarray(valid, dtype=bool), index=result.index)]
            if result.empty:
                raise ValueError(f"Run {run_id} has no complete positive module-norm rows")
            result["global_step"] = result["global_step"].astype(int)
            result = result.groupby("global_step", sort=True, as_index=False).last()
            result["wandb_run_id"] = run_id
            return result
        except ValueError:
            raise
        except Exception as error:
            last_error = error
            if attempt + 1 < FETCH_ATTEMPTS:
                time.sleep(2**attempt)
    raise RuntimeError(f"W&B history fetch exhausted retries for {run_id}") from last_error


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
                raise RuntimeError(f"Failed to fetch module history for {run_id}") from error
            if index % 25 == 0 and blocks:
                pd.concat(blocks, ignore_index=True).to_csv(cache_path, index=False)
    if not blocks:
        raise RuntimeError("No module histories were materialized")
    result = pd.concat(blocks, ignore_index=True).drop_duplicates(["wandb_run_id", "global_step"], keep="last")
    result.to_csv(cache_path, index=False)
    unavailable = pd.DataFrame(unavailable_rows, columns=["wandb_run_id", "reason"]).drop_duplicates(
        "wandb_run_id", keep="last"
    )
    unavailable.to_csv(unavailable_path, index=False)
    accounted = set(result["wandb_run_id"].astype(str)) | set(unavailable["wandb_run_id"].astype(str))
    if accounted != expected_ids:
        raise RuntimeError("History materialization does not account for every run")
    return result


def paired_module_histories(histories: pd.DataFrame) -> pd.DataFrame:
    manifest = pd.read_csv(RUN_MANIFEST_PATH)
    joined = manifest.merge(histories, on="wandb_run_id", how="inner", validate="one_to_many")
    value_columns = [TOTAL_NORM_KEY, *(f"module_norm__{group}" for group in MODULE_GROUPS)]
    one = joined.loc[joined["policy_class"] == "one_phase", ["pair_id", "global_step", *value_columns]].copy()
    two = joined.loc[joined["policy_class"] == "two_phase", ["pair_id", "global_step", *value_columns]].copy()
    one = one.rename(columns={column: f"one__{column}" for column in value_columns})
    two = two.rename(columns={column: f"two__{column}" for column in value_columns})
    paired = one.merge(two, on=["pair_id", "global_step"], how="inner", validate="one_to_one")
    displacement_columns = []
    for group in MODULE_GROUPS:
        one_relative = paired[f"one__module_norm__{group}"] / paired[f"one__{TOTAL_NORM_KEY}"]
        two_relative = paired[f"two__module_norm__{group}"] / paired[f"two__{TOTAL_NORM_KEY}"]
        gap = np.log(two_relative) - np.log(one_relative)
        column = f"displacement__{group}"
        paired[f"gap__{group}"] = gap
        baseline = (
            paired.loc[paired["global_step"].between(*PRE_BASELINE_WINDOW)]
            .groupby("pair_id", sort=False)[f"gap__{group}"]
            .median()
            .rename(f"baseline__{group}")
        )
        paired = paired.merge(baseline, on="pair_id", how="inner", validate="many_to_one")
        paired[column] = paired[f"gap__{group}"] - paired[f"baseline__{group}"]
        displacement_columns.append(column)
    paired["phase_progress"] = (paired["global_step"] - PHASE_BOUNDARY_STEP) / (FINAL_STEP - PHASE_BOUNDARY_STEP)
    return paired[["pair_id", "global_step", "phase_progress", *displacement_columns]].sort_values(
        ["pair_id", "global_step"]
    )


def orient_direction(direction: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    if reference is not None:
        return direction if float(direction @ reference) >= 0 else -direction
    index = int(np.argmax(np.abs(direction)))
    return direction if direction[index] >= 0 else -direction


def principal_direction(matrix: np.ndarray, reference: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    _, singular_values, right = np.linalg.svd(np.asarray(matrix, dtype=float), full_matrices=False)
    direction = orient_direction(right[0], reference)
    explained = float(np.square(singular_values[0]) / np.square(singular_values).sum())
    return direction, explained


def nearest_state(state: pd.DataFrame, step: int) -> pd.DataFrame:
    frame = state.copy()
    frame["distance"] = np.abs(frame["global_step"] - step)
    return frame.sort_values(["pair_id", "distance"]).drop_duplicates("pair_id", keep="first")


def displacement_columns() -> list[str]:
    return [f"displacement__{group}" for group in MODULE_GROUPS]


def project_state(state: pd.DataFrame, direction: np.ndarray) -> pd.DataFrame:
    projected = state[["pair_id", "global_step", "phase_progress"]].copy()
    projected["score"] = state[displacement_columns()].to_numpy(dtype=float) @ direction
    return projected


def fit_pair_amplitudes(frame: pd.DataFrame, rate: float) -> pd.DataFrame:
    rows = []
    for pair_id, block in frame.groupby("pair_id", sort=True):
        shape = transition_shape(block["phase_progress"].to_numpy(), rate)
        observed = block["score"].to_numpy(dtype=float)
        denominator = float(shape @ shape)
        if denominator <= 1e-14:
            continue
        amplitude = float(shape @ observed / denominator)
        rows.append(
            {
                "pair_id": pair_id,
                "terminal_score": amplitude,
                "telemetry_rmse": rmse(observed, amplitude * shape),
            }
        )
    return pd.DataFrame(rows)


def select_rate(frame: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    rows = []
    for rate in RATE_GRID:
        amplitudes = fit_pair_amplitudes(frame, rate)
        rows.append({"rate": rate, "mean_pair_rmse": float(amplitudes["telemetry_rmse"].mean())})
    metrics = pd.DataFrame(rows)
    selected = float(metrics.sort_values(["mean_pair_rmse", "rate"]).iloc[0]["rate"])
    return selected, metrics


def outer_fold_map() -> pd.DataFrame:
    predictions = pd.read_csv(SWITCH_OOF_PATH)
    folds = predictions.loc[predictions["target"] == "gradient_log_jump", ["pair_id", "outer_fold"]]
    return folds.drop_duplicates("pair_id", keep="last")


def feature_columns(features: pd.DataFrame, block: str) -> list[str]:
    prefixes = FEATURE_BLOCKS[block]
    return [str(column) for column in features if column != "pair_id" and str(column).startswith(prefixes)]


def select_policy_model(
    features: pd.DataFrame,
    target: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
) -> tuple[str, float]:
    candidates = []
    for block in FEATURE_BLOCKS:
        columns = feature_columns(features, block)
        matrix = features[columns].to_numpy(dtype=float)
        for alpha in RIDGE_GRID:
            prediction = np.full(len(target), np.nan)
            for inner_fold in np.unique(labels[train]):
                inner_test = train & (labels == inner_fold)
                inner_train = train & ~inner_test
                coefficients = fit_ridge(matrix[inner_train], target[inner_train], alpha)
                prediction[inner_test] = matrix[inner_test] @ coefficients
            candidates.append((rmse(target[train], prediction[train]), block, alpha))
    _, block, alpha = min(candidates)
    return str(block), float(alpha)


def fold_state_and_policy_predictions(
    state: pd.DataFrame,
    folds: pd.DataFrame,
    full_direction: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    terminal = nearest_state(state, 21_000)
    features = pd.read_csv(POLICY_FEATURES_PATH).merge(folds, on="pair_id", how="inner", validate="one_to_one")
    labels = features["outer_fold"].to_numpy(dtype=int)
    rows = []
    parameter_rows = []
    direction_rows = []
    for outer_fold in sorted(np.unique(labels)):
        train_ids = set(features.loc[features["outer_fold"] != outer_fold, "pair_id"])
        train_terminal = terminal.loc[terminal["pair_id"].isin(train_ids)]
        direction, explained = principal_direction(train_terminal[displacement_columns()].to_numpy(), full_direction)
        cosine = float(direction @ full_direction)
        projected = project_state(state, direction)
        fit_projected = projected.loc[
            projected["pair_id"].isin(train_ids)
            & projected["global_step"].between(*TRANSITION_FIT_WINDOW)
            & (projected["phase_progress"] >= 0)
        ]
        selected_rate, _ = select_rate(fit_projected)
        amplitudes = fit_pair_amplitudes(
            projected.loc[projected["global_step"].between(*TRANSITION_FIT_WINDOW) & (projected["phase_progress"] >= 0)],
            selected_rate,
        )
        data = features.merge(
            amplitudes[["pair_id", "terminal_score"]], on="pair_id", how="inner", validate="one_to_one"
        )
        target = data["terminal_score"].to_numpy(dtype=float)
        data_labels = data["outer_fold"].to_numpy(dtype=int)
        train = data_labels != outer_fold
        test = ~train
        block, alpha = select_policy_model(data, target, data_labels, train)
        columns = feature_columns(data, block)
        matrix = data[columns].to_numpy(dtype=float)
        coefficients = fit_ridge(matrix[train], target[train], alpha)
        prediction = matrix[test] @ coefficients
        for index, predicted in zip(data.index[test], prediction, strict=True):
            rows.append(
                {
                    "pair_id": data.loc[index, "pair_id"],
                    "outer_fold": int(outer_fold),
                    "observed_terminal_score": float(data.loc[index, "terminal_score"]),
                    "predicted_terminal_score": float(predicted),
                    "selected_rate": selected_rate,
                    "selected_block": block,
                    "selected_ridge": alpha,
                }
            )
        for column, coefficient in zip(columns, coefficients, strict=True):
            parameter_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "selected_rate": selected_rate,
                    "selected_block": block,
                    "selected_ridge": alpha,
                    "feature": column,
                    "coefficient": float(coefficient),
                }
            )
        direction_rows.append(
            {
                "outer_fold": int(outer_fold),
                "explained_energy": explained,
                "cosine_to_full": cosine,
                "selected_rate": selected_rate,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(parameter_rows), pd.DataFrame(direction_rows)


def response_predictions(policy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    transfer = pd.read_csv(TRANSFER_PATH)
    data = transfer[["pair_id", "global_step", "common_residual"]].merge(
        policy[["pair_id", "outer_fold", "predicted_terminal_score", "selected_rate"]],
        on="pair_id",
        how="inner",
        validate="many_to_one",
    )
    data["phase_progress"] = (data["global_step"] - PHASE_BOUNDARY_STEP) / (FINAL_STEP - PHASE_BOUNDARY_STEP)
    data["state_dynamic"] = [
        score * transition_shape(np.asarray([progress]), rate)[0]
        for score, progress, rate in zip(
            data["predicted_terminal_score"], data["phase_progress"], data["selected_rate"], strict=True
        )
    ]
    data["state_static"] = data["predicted_terminal_score"]
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


def bootstrap_direction_stability(matrix: np.ndarray, full_direction: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED + 10)
    cosines = np.empty(PCA_BOOTSTRAPS)
    for index in range(PCA_BOOTSTRAPS):
        sampled = rng.integers(0, len(matrix), size=len(matrix))
        direction, _ = principal_direction(matrix[sampled], full_direction)
        cosines[index] = float(direction @ full_direction)
    return float(np.median(cosines)), float(np.quantile(cosines, 0.025)), float(np.quantile(cosines, 0.975))


def evaluate(output_dir: Path) -> None:
    histories = pd.read_csv(output_dir / HISTORY_CACHE_NAME)
    state = paired_module_histories(histories)
    folds = outer_fold_map()
    terminal = nearest_state(state, 21_000)
    terminal_matrix = terminal[displacement_columns()].to_numpy(dtype=float)
    full_direction, explained_energy = principal_direction(terminal_matrix)
    bootstrap_cosine = bootstrap_direction_stability(terminal_matrix, full_direction)

    projected = project_state(state, full_direction)
    fit_projected = projected.loc[
        projected["global_step"].between(*TRANSITION_FIT_WINDOW) & (projected["phase_progress"] >= 0)
    ]
    selected_rate, transition_metrics = select_rate(fit_projected)
    selected_amplitudes = fit_pair_amplitudes(fit_projected, selected_rate)
    linear_amplitudes = fit_pair_amplitudes(fit_projected, 0.0).rename(
        columns={"terminal_score": "linear_terminal_score", "telemetry_rmse": "linear_telemetry_rmse"}
    )
    selected_amplitudes = selected_amplitudes.merge(
        linear_amplitudes[["pair_id", "linear_terminal_score", "linear_telemetry_rmse"]],
        on="pair_id",
        how="inner",
        validate="one_to_one",
    )
    telemetry_comparison = selected_amplitudes.rename(columns={"telemetry_rmse": "candidate_rmse"})
    telemetry_bootstrap = bootstrap_pair_mean_difference(
        telemetry_comparison, "candidate_rmse", "linear_telemetry_rmse", BOOTSTRAP_SEED + 11
    )

    holdout_rows = []
    for step in TRANSITION_HOLDOUT_STEPS:
        observed = nearest_state(projected, step)
        comparison = observed.merge(selected_amplitudes, on="pair_id", how="inner", validate="one_to_one")
        progress = (step - PHASE_BOUNDARY_STEP) / (FINAL_STEP - PHASE_BOUNDARY_STEP)
        comparison["candidate_prediction"] = (
            comparison["terminal_score"] * transition_shape(np.asarray([progress]), selected_rate)[0]
        )
        comparison["linear_prediction"] = comparison["linear_terminal_score"] * progress
        holdout_rows.append(
            {
                "global_step": step,
                "pairs": len(comparison),
                "candidate_rmse": rmse(comparison["score"], comparison["candidate_prediction"]),
                "linear_rmse": rmse(comparison["score"], comparison["linear_prediction"]),
            }
        )
    holdout_metrics = pd.DataFrame(holdout_rows)

    state_21000 = nearest_state(projected, 21_000)[["pair_id", "score"]].rename(columns={"score": "score_21000"})
    state_22000 = nearest_state(projected, 22_000)[["pair_id", "score"]].rename(columns={"score": "score_22000"})
    rank_frame = state_21000.merge(state_22000, on="pair_id", how="inner", validate="one_to_one")
    state_rank = safe_spearman(rank_frame["score_21000"], rank_frame["score_22000"])

    policy, policy_parameters, fold_directions = fold_state_and_policy_predictions(state, folds, full_direction)
    policy_zero_rmse = rmse(policy["observed_terminal_score"], np.zeros(len(policy)))
    policy_rmse = rmse(policy["observed_terminal_score"], policy["predicted_terminal_score"])
    policy_metrics = {
        "rows": len(policy),
        "rmse": policy_rmse,
        "zero_improvement": 1.0 - policy_rmse / policy_zero_rmse,
        "spearman": safe_spearman(policy["observed_terminal_score"], policy["predicted_terminal_score"]),
    }

    response, response_parameters = response_predictions(policy)
    response_metrics = []
    fit_scope = response.loc[response["global_step"].isin(RESPONSE_FIT_STEPS)]
    for prediction in ("predicted__zero", "predicted__static", "predicted__dynamic"):
        response_metrics.append(metric_row(fit_scope, prediction, "fit"))
    for step in RESPONSE_HOLDOUT_STEPS:
        scope = response.loc[response["global_step"] == step]
        for prediction in ("predicted__zero", "predicted__static", "predicted__dynamic"):
            response_metrics.append(metric_row(scope, prediction, f"step{step}" if step != FINAL_STEP else "final"))
    response_metrics_frame = pd.DataFrame(response_metrics)
    response_fit = response.loc[response["global_step"].isin(RESPONSE_FIT_STEPS)]
    dynamic_static_bootstrap = bootstrap_rmse_difference(
        response_fit,
        "predicted__dynamic",
        "predicted__static",
        BOOTSTRAP_SEED + 12,
    )

    loading_rows = [
        {"module_group": group, "loading": float(value)}
        for group, value in zip(MODULE_GROUPS, full_direction, strict=True)
    ]
    loadings = pd.DataFrame(loading_rows)
    selected_rate_outer_mode = float(fold_directions["selected_rate"].mode().iloc[0])
    outer_mode_fraction = float(np.mean(fold_directions["selected_rate"] == selected_rate_outer_mode))
    fit_dynamic = response_metrics_frame.query("scope == 'fit' and model == 'dynamic'").iloc[0]
    step_dynamic = response_metrics_frame.query("scope == 'step22000' and model == 'dynamic'").iloc[0]
    final_dynamic = response_metrics_frame.query("scope == 'final' and model == 'dynamic'").iloc[0]
    response_signs = response_parameters.loc[response_parameters["model"] == "dynamic", "response_scale"]
    sign_agreement = int(max((response_signs > 0).sum(), (response_signs < 0).sum()))
    transition_improvement = 1.0 - float(telemetry_comparison["candidate_rmse"].mean()) / float(
        telemetry_comparison["linear_telemetry_rmse"].mean()
    )
    gates = {
        "direction_explained_energy": explained_energy >= 0.25,
        "direction_outer_cosine": float(fold_directions["cosine_to_full"].min()) >= 0.80,
        "direction_bootstrap_cosine_low": bootstrap_cosine[1] >= 0.80,
        "transition_rate_interior": selected_rate not in (min(RATE_GRID), max(RATE_GRID)),
        "transition_outer_mode_fraction": outer_mode_fraction >= 0.80,
        "transition_fit_improvement_over_linear": transition_improvement >= 0.05,
        "transition_vs_linear_bootstrap_upper": telemetry_bootstrap[2] <= 0.0,
        "transition_holdout_no_worse_than_linear": bool(
            (holdout_metrics["candidate_rmse"] <= holdout_metrics["linear_rmse"]).all()
        ),
        "state_rank_21000_to_22000": state_rank >= 0.60,
        "policy_spearman": policy_metrics["spearman"] >= 0.40,
        "policy_zero_improvement": policy_metrics["zero_improvement"] >= 0.20,
        "response_fit_zero_improvement": float(fit_dynamic["zero_improvement"]) >= 0.15,
        "response_dynamic_vs_static_bootstrap_upper": dynamic_static_bootstrap[2] <= 0.0,
        "response_fold_sign_agreement": sign_agreement >= 4,
        "response_step22000_zero_improvement": float(step_dynamic["zero_improvement"]) >= 0.10,
        "response_final_zero_improvement": float(final_dynamic["zero_improvement"]) >= 0.10,
        "response_final_spearman": float(final_dynamic["spearman"]) >= 0.20,
    }
    gate_values = {
        "direction_explained_energy": explained_energy,
        "direction_outer_cosine": float(fold_directions["cosine_to_full"].min()),
        "direction_bootstrap_cosine_low": bootstrap_cosine[1],
        "transition_rate_interior": selected_rate,
        "transition_outer_mode_fraction": outer_mode_fraction,
        "transition_fit_improvement_over_linear": transition_improvement,
        "transition_vs_linear_bootstrap_upper": telemetry_bootstrap[2],
        "transition_holdout_no_worse_than_linear": float(gates["transition_holdout_no_worse_than_linear"]),
        "state_rank_21000_to_22000": state_rank,
        "policy_spearman": policy_metrics["spearman"],
        "policy_zero_improvement": policy_metrics["zero_improvement"],
        "response_fit_zero_improvement": float(fit_dynamic["zero_improvement"]),
        "response_dynamic_vs_static_bootstrap_upper": dynamic_static_bootstrap[2],
        "response_fold_sign_agreement": sign_agreement,
        "response_step22000_zero_improvement": float(step_dynamic["zero_improvement"]),
        "response_final_zero_improvement": float(final_dynamic["zero_improvement"]),
        "response_final_spearman": float(final_dynamic["spearman"]),
    }
    gate_frame = pd.DataFrame(
        [{"gate": name, "passed": passed, "value": gate_values[name]} for name, passed in gates.items()]
    )
    passed = bool(all(gates.values()))

    state.to_csv(output_dir / "paired_module_state.csv", index=False)
    projected.to_csv(output_dir / "projected_state.csv", index=False)
    loadings.to_csv(output_dir / "direction_loadings.csv", index=False)
    transition_metrics.to_csv(output_dir / "transition_metrics.csv", index=False)
    holdout_metrics.to_csv(output_dir / "transition_holdout_metrics.csv", index=False)
    fold_directions.to_csv(output_dir / "fold_directions.csv", index=False)
    policy.to_csv(output_dir / "policy_predictions.csv", index=False)
    policy_parameters.to_csv(output_dir / "policy_parameters.csv", index=False)
    response.to_csv(output_dir / "response_predictions.csv", index=False)
    response_parameters.to_csv(output_dir / "response_parameters.csv", index=False)
    response_metrics_frame.to_csv(output_dir / "response_metrics.csv", index=False)
    gate_frame.to_csv(output_dir / "acceptance_gate.csv", index=False)
    decision = {
        "candidate_id": CANDIDATE_ID,
        "passed": passed,
        "selected_rate": selected_rate,
        "explained_energy": explained_energy,
        "bootstrap_direction_cosine": bootstrap_cosine,
        "state_rank_21000_to_22000": state_rank,
        "policy_metrics": policy_metrics,
        "telemetry_bootstrap_candidate_minus_linear": telemetry_bootstrap,
        "response_bootstrap_dynamic_minus_static": dynamic_static_bootstrap,
        "gates": gates,
    }
    write_json(output_dir / "decision.json", decision)
    write_report(output_dir, decision, gate_frame, loadings, holdout_metrics, response_metrics_frame)
    write_plot(output_dir, loadings, projected, policy, response)
    print(json.dumps(decision, indent=2, sort_keys=True))


def write_report(
    output_dir: Path,
    decision: dict[str, object],
    gates: pd.DataFrame,
    loadings: pd.DataFrame,
    holdouts: pd.DataFrame,
    response_metrics: pd.DataFrame,
) -> None:
    verdict = "passes" if decision["passed"] else "fails"
    report = f"""# Architecture-relative parameter redistribution

`{CANDIDATE_ID}` {verdict} its frozen development gate.

The diagnostic removes total parameter radius and learns one signed architecture-module redistribution direction
from telemetry alone. Final BPB is used only as a temporal falsification outcome.

## Decision

- Selected saturation rate: `{decision['selected_rate']}`.
- Direction explained energy: `{decision['explained_energy']:.3f}`.
- Step-21000 to step-22000 score-rank persistence: `{decision['state_rank_21000_to_22000']:.3f}`.
- Overall gate: `{'PASS' if decision['passed'] else 'FAIL'}`.

## Gates

{gates.to_markdown(index=False)}

## Direction loadings

{loadings.sort_values('loading').to_markdown(index=False)}

## Telemetry holdouts

{holdouts.to_markdown(index=False)}

## Smooth-target response

{response_metrics.to_markdown(index=False)}

This is exposed development evidence. A pass licenses only a nested temporal-state test after the aggregate spine is
selected; it is not confirmation and does not license using observed optimizer telemetry at deployment.
"""
    (output_dir / "report.md").write_text(report)


def write_plot(
    output_dir: Path,
    loadings: pd.DataFrame,
    projected: pd.DataFrame,
    policy: pd.DataFrame,
    response: pd.DataFrame,
) -> None:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Telemetry-only direction", "State trajectories", "Policy prediction", "Final response"),
    )
    ordered = loadings.sort_values("loading")
    figure.add_trace(
        go.Bar(x=ordered["loading"], y=ordered["module_group"], orientation="h", marker_color="#274c77"),
        row=1,
        col=1,
    )
    sampled_pairs = projected["pair_id"].drop_duplicates().iloc[:: max(1, projected["pair_id"].nunique() // 40)]
    for _, block in projected.loc[projected["pair_id"].isin(sampled_pairs)].groupby("pair_id", sort=False):
        figure.add_trace(
            go.Scatter(x=block["phase_progress"], y=block["score"], mode="lines", line={"width": 1}, opacity=0.35),
            row=1,
            col=2,
        )
    figure.add_trace(
        go.Scatter(
            x=policy["observed_terminal_score"],
            y=policy["predicted_terminal_score"],
            mode="markers",
            marker={
                "color": policy["observed_terminal_score"],
                "colorscale": "RdYlGn_r",
                "size": 7,
            },
        ),
        row=2,
        col=1,
    )
    final = response.loc[response["global_step"] == FINAL_STEP]
    figure.add_trace(
        go.Scatter(
            x=final["observed"],
            y=final["predicted__dynamic"],
            mode="markers",
            marker={"color": final["observed"], "colorscale": "RdYlGn_r", "size": 7},
        ),
        row=2,
        col=2,
    )
    figure.update_layout(
        title="Architecture-relative parameter state",
        template="plotly_white",
        height=900,
        width=1250,
        showlegend=False,
    )
    figure.write_html(
        output_dir / "architecture_relative_parameter_state.html",
        include_plotlyjs=True,
        config={"toImageButtonOptions": {"format": "png", "scale": 4}},
    )


def main() -> None:
    args = parse_args()
    if args.mode == "preregister":
        freeze_protocol(args.output_dir)
        return
    require_frozen_protocol(args.output_dir)
    if args.mode == "materialize":
        histories = materialize_histories(args.output_dir, args.max_workers)
        print(f"materialized {histories['wandb_run_id'].nunique()} runs and {len(histories)} rows")
        return
    evaluate(args.output_dir)


if __name__ == "__main__":
    main()
