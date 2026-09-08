# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build three-phase dense policy datasets for offline-control v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.offline_rl.contracts import DEFAULT_OBJECTIVE_METRIC

logger = logging.getLogger(__name__)

TOTAL_STEPS = 5722
PHASE_BOUNDARIES = (1888, 3824)
WINDOW_STEPS = 320
WINDOW_BINS = 32
WINDOW_BIN_WIDTH = WINDOW_STEPS // WINDOW_BINS
PRETRAIN_STRIDE = 20
ACTION_GRID = np.linspace(0.0, 1.0, 21, dtype=np.float32)
SEQUENCE_CHANNELS = (
    "train/loss",
    "optim/learning_rate",
    "optim/adam_lr",
    "grad/norm/total",
    "params/norm/total",
    "eval/loss",
    DEFAULT_OBJECTIVE_METRIC,
    "eval_event_mask",
    "relative_time_to_boundary",
)
SUMMARY_FEATURES = (
    "decision_index",
    "num_phases_total",
    "remaining_decisions",
    "budget_frac_consumed",
    "budget_frac_remaining",
    "global_step",
    "last_train_loss",
    "last_eval_loss",
    "last_obj_bpb",
    "train_eval_gap",
    "steps_since_last_eval_frac",
    "optim/learning_rate",
    "optim/adam_lr",
    "avg_lr_to_next_boundary",
    "avg_lr_remaining",
    "grad/norm/total",
    "params/norm/total",
    "grad_to_param_norm",
    "prev_action_starcoder",
    "cumulative_starcoder_exposure",
    "delta_prev_action",
    "phase_train_loss_mean",
    "phase_train_loss_delta",
    "phase_eval_loss_delta",
    "phase_obj_bpb_delta",
    "phase_grad_norm_mean",
    "phase_grad_norm_std",
    "phase_params_norm_mean",
    "phase_params_norm_delta",
    "seq_train_loss_last",
    "seq_train_loss_mean",
    "seq_train_loss_std",
    "seq_train_loss_slope",
    "seq_learning_rate_last",
    "seq_learning_rate_mean",
    "seq_learning_rate_std",
    "seq_learning_rate_slope",
    "seq_adam_lr_last",
    "seq_adam_lr_mean",
    "seq_adam_lr_std",
    "seq_adam_lr_slope",
    "seq_grad_norm_total_last",
    "seq_grad_norm_total_mean",
    "seq_grad_norm_total_std",
    "seq_grad_norm_total_slope",
    "seq_params_norm_total_last",
    "seq_params_norm_total_mean",
    "seq_params_norm_total_std",
    "seq_params_norm_total_slope",
    "seq_eval_loss_last",
    "seq_eval_loss_mean",
    "seq_eval_loss_std",
    "seq_eval_loss_slope",
    "seq_obj_bpb_last",
    "seq_obj_bpb_mean",
    "seq_obj_bpb_std",
    "seq_obj_bpb_slope",
)


@dataclass(frozen=True)
class BuildThreePhaseDenseDatasetConfig:
    """Config for building the three-phase dense policy dataset."""

    input_dir: str
    output_dir: str
    objective_metric: str = DEFAULT_OBJECTIVE_METRIC
    feature_coverage_threshold: float = 0.95
    n_cv_folds: int = 5
    window_steps: int = WINDOW_STEPS
    window_bins: int = WINDOW_BINS
    pretrain_stride: int = PRETRAIN_STRIDE
    action_grid_bins: int = len(ACTION_GRID)
    action_grid_low: float = 0.0
    action_grid_high: float = 1.0


@dataclass(frozen=True)
class RunHistoryCache:
    """Per-run dense history arrays used to avoid repeated pandas filtering in v3 builders."""

    total_steps: int
    raw_by_metric: dict[str, np.ndarray]
    carryforward_by_metric: dict[str, np.ndarray]
    valid_steps_by_metric: dict[str, np.ndarray]
    valid_values_by_metric: dict[str, np.ndarray]
    eval_event_mask: np.ndarray


def _deterministic_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def assign_grouped_folds(runs_df: pd.DataFrame, n_folds: int) -> dict[str, int]:
    assignments: dict[str, int] = {}
    run_ids = sorted(runs_df["wandb_run_id"].astype(str).unique(), key=lambda item: (_deterministic_hash(item), item))
    for index, run_id in enumerate(run_ids):
        assignments[run_id] = index % n_folds
    return assignments


def discretize_action(action: float, action_grid: np.ndarray) -> tuple[int, float]:
    index = int(np.argmin(np.abs(action_grid - float(action))))
    return index, float(action_grid[index])


def build_action_grid(config: BuildThreePhaseDenseDatasetConfig) -> np.ndarray:
    """Build the discrete StarCoder action grid for v3."""
    if config.action_grid_bins < 2:
        raise ValueError("action_grid_bins must be at least 2")
    return np.linspace(
        config.action_grid_low,
        config.action_grid_high,
        config.action_grid_bins,
        dtype=np.float32,
    )


def _prepare_run_history_cache(history: pd.DataFrame, *, total_steps: int, objective_metric: str) -> RunHistoryCache:
    metrics = (
        "train/loss",
        "optim/learning_rate",
        "optim/adam_lr",
        "grad/norm/total",
        "params/norm/total",
        "eval/loss",
        objective_metric,
    )
    length = max(int(total_steps), int(history["step"].max()) + 1 if not history.empty else int(total_steps))
    steps = history["step"].to_numpy(dtype=np.int32, copy=False)
    raw_by_metric: dict[str, np.ndarray] = {}
    carryforward_by_metric: dict[str, np.ndarray] = {}
    valid_steps_by_metric: dict[str, np.ndarray] = {}
    valid_values_by_metric: dict[str, np.ndarray] = {}

    for metric_name in metrics:
        raw = np.full(length, np.nan, dtype=np.float32)
        if metric_name in history.columns:
            values = history[metric_name].to_numpy(dtype=np.float32, copy=False)
            valid = ~np.isnan(values)
            if valid.any():
                metric_steps = steps[valid]
                metric_values = values[valid]
                raw[metric_steps] = metric_values
                valid_steps_by_metric[metric_name] = metric_steps
                valid_values_by_metric[metric_name] = metric_values
            else:
                valid_steps_by_metric[metric_name] = np.zeros(0, dtype=np.int32)
                valid_values_by_metric[metric_name] = np.zeros(0, dtype=np.float32)
        else:
            valid_steps_by_metric[metric_name] = np.zeros(0, dtype=np.int32)
            valid_values_by_metric[metric_name] = np.zeros(0, dtype=np.float32)
        raw_by_metric[metric_name] = raw
        carryforward_by_metric[metric_name] = pd.Series(raw, copy=False).ffill().to_numpy(dtype=np.float32, copy=False)

    eval_mask = np.zeros(length, dtype=np.float32)
    if "eval/loss" in history.columns:
        eval_values = history["eval/loss"].to_numpy(dtype=np.float32, copy=False)
        eval_mask[steps[~np.isnan(eval_values)]] = 1.0
    if objective_metric in history.columns:
        objective_values = history[objective_metric].to_numpy(dtype=np.float32, copy=False)
        eval_mask[steps[~np.isnan(objective_values)]] = 1.0

    return RunHistoryCache(
        total_steps=length,
        raw_by_metric=raw_by_metric,
        carryforward_by_metric=carryforward_by_metric,
        valid_steps_by_metric=valid_steps_by_metric,
        valid_values_by_metric=valid_values_by_metric,
        eval_event_mask=eval_mask,
    )


def _phase_lengths(total_steps: int, phase_boundaries: tuple[int, ...]) -> list[int]:
    starts = [0, *phase_boundaries]
    ends = [*phase_boundaries, total_steps]
    return [int(end - start) for start, end in zip(starts, ends, strict=True)]


def _decision_steps(phase_boundaries: tuple[int, ...]) -> list[int]:
    return [0, *phase_boundaries]


def _last_metric_at_or_before(
    history: pd.DataFrame, metric_col: str, decision_step: int
) -> tuple[float | None, int | None]:
    if metric_col not in history.columns:
        return None, None
    metric = history.loc[(history["step"] <= decision_step) & history[metric_col].notna(), ["step", metric_col]]
    if metric.empty:
        return None, None
    row = metric.sort_values("step").iloc[-1]
    return float(row[metric_col]), int(row["step"])


def _first_metric_after(history: pd.DataFrame, metric_col: str, step: int) -> float | None:
    if metric_col not in history.columns:
        return None
    metric = history.loc[(history["step"] > step) & history[metric_col].notna(), metric_col]
    if metric.empty:
        return None
    return float(metric.iloc[0])


def _mean_metric_between(history: pd.DataFrame, metric_col: str, start_step: int, end_step: int) -> float | None:
    if metric_col not in history.columns:
        return None
    metric = history.loc[
        (history["step"] > start_step) & (history["step"] <= end_step) & history[metric_col].notna(), metric_col
    ]
    if metric.empty:
        return None
    return float(metric.mean())


def _window_bins(window_steps: int, window_bins: int, decision_step: int) -> list[tuple[int, int]]:
    bin_width = max(1, window_steps // window_bins)
    start = max(0, decision_step - window_steps)
    return [(start + i * bin_width, start + (i + 1) * bin_width) for i in range(window_bins)]


def _carryforward_metric(history: pd.DataFrame, metric_col: str, step: int) -> float | None:
    value, _ = _last_metric_at_or_before(history, metric_col, step)
    return value


def _first_metric_after_from_cache(cache: RunHistoryCache, metric_col: str, step: int) -> float | None:
    valid_steps = cache.valid_steps_by_metric.get(metric_col)
    valid_values = cache.valid_values_by_metric.get(metric_col)
    if valid_steps is None or valid_values is None or len(valid_steps) == 0:
        return None
    index = int(np.searchsorted(valid_steps, step, side="right"))
    if index >= len(valid_values):
        return None
    return float(valid_values[index])


def _carryforward_metric_from_cache(cache: RunHistoryCache, metric_col: str, step: int) -> float | None:
    values = cache.carryforward_by_metric.get(metric_col)
    if values is None or len(values) == 0:
        return None
    bounded_step = min(max(int(step), 0), len(values) - 1)
    value = values[bounded_step]
    return None if np.isnan(value) else float(value)


def build_sequence_window(
    history: pd.DataFrame,
    *,
    decision_step: int,
    objective_metric: str,
    window_steps: int,
    window_bins: int,
    history_cache: RunHistoryCache | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    channels = len(SEQUENCE_CHANNELS)
    values = np.zeros((window_bins, channels), dtype=np.float32)
    mask = np.zeros(window_bins, dtype=np.float32)
    if decision_step <= 0:
        return values, mask

    bins = _window_bins(window_steps, window_bins, decision_step)
    for index, (bin_start, bin_end) in enumerate(bins):
        mask[index] = 1.0
        if history_cache is None:
            in_bin = history[(history["step"] >= bin_start) & (history["step"] < bin_end)]
            values[index, 0] = (
                float(in_bin["train/loss"].mean())
                if "train/loss" in in_bin and in_bin["train/loss"].notna().any()
                else 0.0
            )
            values[index, 1] = (
                float(in_bin["optim/learning_rate"].mean())
                if "optim/learning_rate" in in_bin and in_bin["optim/learning_rate"].notna().any()
                else 0.0
            )
            values[index, 2] = (
                float(in_bin["optim/adam_lr"].mean())
                if "optim/adam_lr" in in_bin and in_bin["optim/adam_lr"].notna().any()
                else 0.0
            )
            values[index, 3] = (
                float(in_bin["grad/norm/total"].mean())
                if "grad/norm/total" in in_bin and in_bin["grad/norm/total"].notna().any()
                else 0.0
            )
            values[index, 4] = (
                float(in_bin["params/norm/total"].mean())
                if "params/norm/total" in in_bin and in_bin["params/norm/total"].notna().any()
                else 0.0
            )
            eval_loss = _carryforward_metric(history, "eval/loss", bin_end - 1)
            obj = _carryforward_metric(history, objective_metric, bin_end - 1)
            has_eval = (("eval/loss" in in_bin.columns) and in_bin["eval/loss"].notna().any()) or (
                (objective_metric in in_bin.columns) and in_bin[objective_metric].notna().any()
            )
        else:
            train_slice = history_cache.raw_by_metric["train/loss"][bin_start:bin_end]
            lr_slice = history_cache.raw_by_metric["optim/learning_rate"][bin_start:bin_end]
            adam_lr_slice = history_cache.raw_by_metric["optim/adam_lr"][bin_start:bin_end]
            grad_slice = history_cache.raw_by_metric["grad/norm/total"][bin_start:bin_end]
            param_slice = history_cache.raw_by_metric["params/norm/total"][bin_start:bin_end]
            values[index, 0] = float(np.nanmean(train_slice)) if np.isfinite(train_slice).any() else 0.0
            values[index, 1] = float(np.nanmean(lr_slice)) if np.isfinite(lr_slice).any() else 0.0
            values[index, 2] = float(np.nanmean(adam_lr_slice)) if np.isfinite(adam_lr_slice).any() else 0.0
            values[index, 3] = float(np.nanmean(grad_slice)) if np.isfinite(grad_slice).any() else 0.0
            values[index, 4] = float(np.nanmean(param_slice)) if np.isfinite(param_slice).any() else 0.0
            eval_loss = _carryforward_metric_from_cache(history_cache, "eval/loss", bin_end - 1)
            obj = _carryforward_metric_from_cache(history_cache, objective_metric, bin_end - 1)
            has_eval = bool(history_cache.eval_event_mask[bin_start:bin_end].any())
        values[index, 5] = 0.0 if eval_loss is None else float(eval_loss)
        values[index, 6] = 0.0 if obj is None else float(obj)
        values[index, 7] = 1.0 if has_eval else 0.0
        values[index, 8] = float((bin_end - decision_step) / float(window_steps))
    return values, mask


def _slope(values: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 0
    if valid.sum() < 2:
        return 0.0
    y = values[valid].astype(np.float64)
    x = np.arange(len(values), dtype=np.float64)[valid]
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom <= 0.0:
        return 0.0
    return float(((x * (y - y.mean())).sum()) / denom)


def _sequence_summary(window: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    prefix = {
        0: "seq_train_loss",
        1: "seq_learning_rate",
        2: "seq_adam_lr",
        3: "seq_grad_norm_total",
        4: "seq_params_norm_total",
        5: "seq_eval_loss",
        6: "seq_obj_bpb",
    }
    output: dict[str, float] = {}
    valid = mask > 0
    for channel_index, name in prefix.items():
        channel = window[:, channel_index]
        if valid.any():
            current = channel[valid]
            output[f"{name}_last"] = float(current[-1])
            output[f"{name}_mean"] = float(current.mean())
            output[f"{name}_std"] = float(current.std(ddof=0))
        else:
            output[f"{name}_last"] = 0.0
            output[f"{name}_mean"] = 0.0
            output[f"{name}_std"] = 0.0
        output[f"{name}_slope"] = _slope(channel, mask)
    return output


def _compute_exposure(actions: list[float], phase_lengths: list[int], decision_index: int) -> tuple[float, float, float]:
    if decision_index <= 0:
        return 0.5, 0.0, 0.0
    consumed = sum(phase_lengths[:decision_index])
    weighted = sum(actions[idx] * phase_lengths[idx] for idx in range(decision_index))
    prev_action = actions[decision_index - 1]
    prev_prev = actions[decision_index - 2] if decision_index > 1 else prev_action
    cumulative = float(weighted / consumed) if consumed > 0 else 0.0
    delta_prev = float(prev_action - prev_prev) if decision_index > 1 else 0.0
    return float(prev_action), cumulative, delta_prev


def _phase_metric_delta(history: pd.DataFrame, metric_col: str, start_step: int, end_step: int) -> float:
    if metric_col not in history.columns:
        return 0.0
    metric = history.loc[
        (history["step"] >= start_step) & (history["step"] <= end_step) & history[metric_col].notna(),
        ["step", metric_col],
    ].sort_values("step")
    if len(metric) < 2:
        return 0.0
    return float(metric.iloc[-1][metric_col] - metric.iloc[0][metric_col])


def _phase_metric_mean(history: pd.DataFrame, metric_col: str, start_step: int, end_step: int) -> float:
    if metric_col not in history.columns:
        return 0.0
    metric = history.loc[
        (history["step"] >= start_step) & (history["step"] <= end_step) & history[metric_col].notna(), metric_col
    ]
    if metric.empty:
        return 0.0
    return float(metric.mean())


def _phase_metric_std(history: pd.DataFrame, metric_col: str, start_step: int, end_step: int) -> float:
    if metric_col not in history.columns:
        return 0.0
    metric = history.loc[
        (history["step"] >= start_step) & (history["step"] <= end_step) & history[metric_col].notna(), metric_col
    ]
    if metric.empty:
        return 0.0
    return float(metric.std(ddof=0))


def _feature_row(
    *,
    run_row: pd.Series,
    history: pd.DataFrame,
    objective_metric: str,
    decision_index: int,
    window_steps: int,
    window_bins: int,
    history_cache: RunHistoryCache | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray] | None:
    total_steps = int(run_row["total_steps"])
    num_phases_total = int(run_row.get("num_phases_total", 3))
    phase_boundaries = tuple(json.loads(run_row["phase_boundaries_json"]))
    decision_steps = _decision_steps(phase_boundaries)
    phase_lengths = _phase_lengths(total_steps, phase_boundaries)
    decision_step = decision_steps[decision_index]
    last_decision_index = num_phases_total - 1
    next_step = total_steps if decision_index == last_decision_index else decision_steps[decision_index + 1]
    phase_start = 0 if decision_index == 0 else decision_steps[decision_index - 1]

    actions = [
        float(run_row[f"phase_{idx}_starcoder"])
        for idx in range(num_phases_total)
        if f"phase_{idx}_starcoder" in run_row and not pd.isna(run_row[f"phase_{idx}_starcoder"])
    ]
    if len(actions) != num_phases_total:
        return None

    prev_action, cumulative_exposure, delta_prev_action = _compute_exposure(actions, phase_lengths, decision_index)
    last_train_loss, _ = _last_metric_at_or_before(history, "train/loss", decision_step)
    last_eval_loss, eval_step = _last_metric_at_or_before(history, "eval/loss", decision_step)
    last_obj_bpb, obj_step = _last_metric_at_or_before(history, objective_metric, decision_step)
    last_lr, _ = _last_metric_at_or_before(history, "optim/learning_rate", decision_step)
    last_adam_lr, _ = _last_metric_at_or_before(history, "optim/adam_lr", decision_step)
    last_grad_norm, _ = _last_metric_at_or_before(history, "grad/norm/total", decision_step)
    last_param_norm, _ = _last_metric_at_or_before(history, "params/norm/total", decision_step)

    if any(step is not None for step in (eval_step, obj_step)):
        last_eval_like_step = max(step for step in (eval_step, obj_step) if step is not None)
        steps_since_last_eval_frac = (decision_step - last_eval_like_step) / float(total_steps)
    else:
        steps_since_last_eval_frac = 1.0

    window, mask = build_sequence_window(
        history,
        decision_step=decision_step,
        objective_metric=objective_metric,
        window_steps=window_steps,
        window_bins=window_bins,
        history_cache=history_cache,
    )
    state: dict[str, float] = {
        "num_phases_total": float(num_phases_total),
        "decision_index": float(decision_index),
        "remaining_decisions": float(last_decision_index - decision_index),
        "budget_frac_consumed": float(decision_step) / float(total_steps),
        "budget_frac_remaining": 1.0 - float(decision_step) / float(total_steps),
        "global_step": float(decision_step),
        "prev_action_starcoder": float(prev_action),
        "cumulative_starcoder_exposure": float(cumulative_exposure),
        "delta_prev_action": float(delta_prev_action),
        "last_train_loss": 0.0 if last_train_loss is None else float(last_train_loss),
        "last_eval_loss": 0.0 if last_eval_loss is None else float(last_eval_loss),
        "last_obj_bpb": 0.0 if last_obj_bpb is None else float(last_obj_bpb),
        "train_eval_gap": (
            0.0 if last_train_loss is None or last_eval_loss is None else float(last_eval_loss - last_train_loss)
        ),
        "steps_since_last_eval_frac": float(steps_since_last_eval_frac),
        "optim/learning_rate": 0.0 if last_lr is None else float(last_lr),
        "optim/adam_lr": 0.0 if last_adam_lr is None else float(last_adam_lr),
        "avg_lr_to_next_boundary": _mean_metric_between(history, "optim/learning_rate", decision_step, next_step) or 0.0,
        "avg_lr_remaining": _mean_metric_between(history, "optim/learning_rate", decision_step, total_steps) or 0.0,
        "grad/norm/total": 0.0 if last_grad_norm is None else float(last_grad_norm),
        "params/norm/total": 0.0 if last_param_norm is None else float(last_param_norm),
        "grad_to_param_norm": (
            0.0
            if last_grad_norm is None or last_param_norm in (None, 0.0)
            else float(last_grad_norm / max(last_param_norm, 1e-6))
        ),
        "phase_train_loss_mean": _phase_metric_mean(history, "train/loss", phase_start, decision_step),
        "phase_train_loss_delta": _phase_metric_delta(history, "train/loss", phase_start, decision_step),
        "phase_eval_loss_delta": _phase_metric_delta(history, "eval/loss", phase_start, decision_step),
        "phase_obj_bpb_delta": _phase_metric_delta(history, objective_metric, phase_start, decision_step),
        "phase_grad_norm_mean": _phase_metric_mean(history, "grad/norm/total", phase_start, decision_step),
        "phase_grad_norm_std": _phase_metric_std(history, "grad/norm/total", phase_start, decision_step),
        "phase_params_norm_mean": _phase_metric_mean(history, "params/norm/total", phase_start, decision_step),
        "phase_params_norm_delta": _phase_metric_delta(history, "params/norm/total", phase_start, decision_step),
    }
    state.update(_sequence_summary(window, mask))
    return state, window, mask


def build_decision_state_features(
    *,
    run_row: pd.Series,
    history: pd.DataFrame,
    objective_metric: str,
    decision_index: int,
    window_steps: int = WINDOW_STEPS,
    window_bins: int = WINDOW_BINS,
    history_cache: RunHistoryCache | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray] | None:
    """Build the summary-state features and dense window for one boundary decision."""
    return _feature_row(
        run_row=run_row,
        history=history,
        objective_metric=objective_metric,
        decision_index=decision_index,
        window_steps=window_steps,
        window_bins=window_bins,
        history_cache=history_cache,
    )


def _decision_row(
    *,
    run_row: pd.Series,
    state: dict[str, float],
    decision_index: int,
    next_obj_bpb: float,
    final_objective: float,
    action_starcoder: float,
    action_grid: np.ndarray,
    fold_assignments: dict[str, int],
) -> dict[str, Any]:
    """Attach action/reward metadata to a boundary state row."""
    total_steps = int(run_row["total_steps"])
    num_phases_total = int(run_row.get("num_phases_total", 3))
    phase_boundaries = tuple(json.loads(run_row["phase_boundaries_json"]))
    decision_steps = _decision_steps(phase_boundaries)
    last_decision_index = num_phases_total - 1
    next_step = total_steps if decision_index == last_decision_index else decision_steps[decision_index + 1]
    last_obj_bpb = float(state.get("last_obj_bpb", 0.0))
    action_idx, action_grid_value = discretize_action(action_starcoder, action_grid)
    reward_dense_raw = float(
        (last_obj_bpb - next_obj_bpb) if decision_index > 0 or last_obj_bpb != 0.0 else -next_obj_bpb
    )
    row: dict[str, Any] = {
        "row_id": f"{run_row['wandb_run_id']}::decision::{decision_index}",
        "wandb_run_id": str(run_row["wandb_run_id"]),
        "run_family": str(run_row.get("run_family", "three_phase_starcoder")),
        "source_experiment": run_row.get("source_experiment"),
        "local_run_id": run_row.get("local_run_id"),
        "run_name": run_row.get("run_name"),
        "t": int(decision_index),
        "decision_step": int(decision_steps[decision_index]),
        "next_step": int(next_step),
        "total_steps": int(total_steps),
        "action_starcoder": float(action_starcoder),
        "action_idx": int(action_idx),
        "action_grid_value": float(action_grid_value),
        "done": bool(decision_index == last_decision_index),
        "final_objective": float(final_objective),
        "next_obj_bpb": float(next_obj_bpb),
        "reward_dense_raw": float(reward_dense_raw),
        "reward_terminal_raw": float(-final_objective if decision_index == 2 else 0.0),
        "phase_boundaries_json": run_row.get("phase_boundaries_json"),
        "cv_fold": int(fold_assignments[str(run_row["wandb_run_id"])]),
        "episode_id": str(run_row["wandb_run_id"]),
        "step_in_episode": int(decision_index),
    }
    row.update(state)
    return row


def _build_pretrain_windows(
    run_row: pd.Series,
    history: pd.DataFrame,
    *,
    objective_metric: str,
    window_steps: int,
    window_bins: int,
    stride: int,
    history_cache: RunHistoryCache | None = None,
) -> list[dict[str, Any]]:
    phase_boundaries = tuple(json.loads(run_row["phase_boundaries_json"]))
    num_phases_total = int(run_row.get("num_phases_total", len(phase_boundaries) + 1))
    starts = [0, *phase_boundaries]
    ends = [*phase_boundaries, int(run_row["total_steps"])]
    actions = [float(run_row[f"phase_{idx}_starcoder"]) for idx in range(num_phases_total)]
    rows: list[dict[str, Any]] = []
    for phase_index, (start_step, end_step) in enumerate(zip(starts, ends, strict=True)):
        cursor = start_step + window_steps
        while cursor < end_step:
            window, mask = build_sequence_window(
                history,
                decision_step=cursor,
                objective_metric=objective_metric,
                window_steps=window_steps,
                window_bins=window_bins,
                history_cache=history_cache,
            )
            rows.append(
                {
                    "row_id": f"{run_row['wandb_run_id']}::pretrain::{phase_index}::{cursor}",
                    "wandb_run_id": str(run_row["wandb_run_id"]),
                    "phase_index": int(phase_index),
                    "cursor_step": int(cursor),
                    "action_starcoder": float(actions[phase_index]),
                    "next_train_loss": (
                        (
                            _first_metric_after_from_cache(history_cache, "train/loss", cursor)
                            if history_cache is not None
                            else _first_metric_after(history, "train/loss", cursor)
                        )
                        or 0.0
                    ),
                    "next_eval_bpb": (
                        (
                            _first_metric_after_from_cache(history_cache, objective_metric, cursor)
                            if history_cache is not None
                            else _first_metric_after(history, objective_metric, cursor)
                        )
                        or float(run_row[objective_metric])
                    ),
                    "window": window,
                    "mask": mask,
                }
            )
            cursor += stride
    return rows


def _impute_selected_features(
    frame: pd.DataFrame, selected_features: tuple[str, ...]
) -> tuple[pd.DataFrame, dict[str, float]]:
    result = frame.copy()
    defaults: dict[str, float] = {}
    for feature_name in selected_features:
        if feature_name not in result.columns:
            result[feature_name] = np.nan
        series = result[feature_name].replace([np.inf, -np.inf], np.nan)
        default = float(series.median()) if series.notna().any() else 0.0
        result[feature_name] = series.fillna(default).astype(float)
        defaults[feature_name] = default
    return result, defaults


def build_three_phase_dense_policy_dataset(
    config: BuildThreePhaseDenseDatasetConfig,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_df = pd.read_parquet(input_dir / "runs.parquet")
    history_wide_df = pd.read_parquet(input_dir / "history_dense_wide.parquet")
    three_phase_runs = runs_df[runs_df["run_family"] == "three_phase_starcoder"].copy().reset_index(drop=True)
    fold_assignments = assign_grouped_folds(three_phase_runs, config.n_cv_folds)
    action_grid = build_action_grid(config)
    history_by_run = {
        str(run_id): frame.sort_values("step").reset_index(drop=True)
        for run_id, frame in history_wide_df.groupby("wandb_run_id", sort=False)
    }

    decision_rows: list[dict[str, Any]] = []
    sequence_by_row_id: dict[str, np.ndarray] = {}
    mask_by_row_id: dict[str, np.ndarray] = {}
    pretrain_meta: list[dict[str, Any]] = []
    pretrain_windows: list[np.ndarray] = []
    pretrain_masks: list[np.ndarray] = []

    for _, run_row in three_phase_runs.iterrows():
        run_id = str(run_row["wandb_run_id"])
        history = history_by_run.get(run_id)
        if history is None or history.empty:
            continue
        history_cache = _prepare_run_history_cache(
            history,
            total_steps=int(run_row["total_steps"]),
            objective_metric=config.objective_metric,
        )
        final_objective = (
            float(run_row[config.objective_metric])
            if config.objective_metric in run_row and not pd.isna(run_row[config.objective_metric])
            else None
        )
        if final_objective is None:
            metric = (
                history[config.objective_metric].dropna()
                if config.objective_metric in history.columns
                else pd.Series(dtype=float)
            )
            if metric.empty:
                continue
            final_objective = float(metric.iloc[-1])
        for decision_index in range(3):
            built = build_decision_state_features(
                run_row=run_row,
                history=history,
                objective_metric=config.objective_metric,
                decision_index=decision_index,
                window_steps=config.window_steps,
                window_bins=config.window_bins,
                history_cache=history_cache,
            )
            if built is None:
                continue
            state, window, mask = built
            next_decision_step = total_steps = int(run_row["total_steps"])
            phase_boundaries = tuple(json.loads(run_row["phase_boundaries_json"]))
            decision_steps = _decision_steps(phase_boundaries)
            next_decision_step = total_steps if decision_index == 2 else decision_steps[decision_index + 1]
            next_obj_bpb, _ = _last_metric_at_or_before(history, config.objective_metric, next_decision_step)
            if next_obj_bpb is None:
                next_obj_bpb = final_objective
            action_starcoder = float(run_row[f"phase_{decision_index}_starcoder"])
            row = _decision_row(
                run_row=run_row,
                state=state,
                decision_index=decision_index,
                next_obj_bpb=float(next_obj_bpb),
                final_objective=float(final_objective),
                action_starcoder=action_starcoder,
                action_grid=action_grid,
                fold_assignments=fold_assignments,
            )
            decision_rows.append(row)
            sequence_by_row_id[row["row_id"]] = window
            mask_by_row_id[row["row_id"]] = mask

        for item in _build_pretrain_windows(
            run_row,
            history,
            objective_metric=config.objective_metric,
            window_steps=config.window_steps,
            window_bins=config.window_bins,
            stride=config.pretrain_stride,
            history_cache=history_cache,
        ):
            pretrain_meta.append({key: value for key, value in item.items() if key not in {"window", "mask"}})
            pretrain_windows.append(item["window"])
            pretrain_masks.append(item["mask"])

    decisions = pd.DataFrame(decision_rows)
    if decisions.empty:
        raise ValueError("No decision rows were produced.")
    pre_impute = decisions.copy()
    coverage_rows = []
    selected_feature_keys: list[str] = []
    for feature_name in SUMMARY_FEATURES:
        coverage = (
            float(pre_impute[feature_name].replace([np.inf, -np.inf], np.nan).notna().mean())
            if feature_name in pre_impute.columns
            else 0.0
        )
        selected = coverage >= config.feature_coverage_threshold
        coverage_rows.append(
            {
                "feature_name": feature_name,
                "coverage": coverage,
                "selected": selected,
            }
        )
        if selected:
            selected_feature_keys.append(feature_name)
    if not selected_feature_keys:
        raise ValueError("No summary features met the coverage threshold.")
    decisions, feature_defaults = _impute_selected_features(decisions, tuple(selected_feature_keys))
    decisions = decisions.sort_values(["wandb_run_id", "step_in_episode"]).reset_index(drop=True)
    decisions["return_to_go_dense_raw"] = decisions.groupby("episode_id", sort=False)["reward_dense_raw"].transform(
        lambda values: np.flip(np.cumsum(np.flip(values.to_numpy(dtype=np.float32)))).astype(np.float32)
    )

    decisions.to_parquet(output_dir / "decisions.parquet", index=False)

    ordered_row_ids = decisions["row_id"].tolist()
    sequence_arrays = {
        "row_ids": np.asarray(ordered_row_ids, dtype=str),
        "sequences": np.stack([sequence_by_row_id[row_id] for row_id in ordered_row_ids], axis=0).astype(np.float32),
        "masks": np.stack([mask_by_row_id[row_id] for row_id in ordered_row_ids], axis=0).astype(np.float32),
    }
    np.savez_compressed(output_dir / "decision_sequences.npz", **sequence_arrays)

    pretrain_payload = {
        "row_ids": np.asarray([row["row_id"] for row in pretrain_meta], dtype=str),
        "wandb_run_id": np.asarray([row["wandb_run_id"] for row in pretrain_meta], dtype=str),
        "phase_index": np.asarray([row["phase_index"] for row in pretrain_meta], dtype=np.int64),
        "cursor_step": np.asarray([row["cursor_step"] for row in pretrain_meta], dtype=np.int64),
        "action_starcoder": np.asarray([row["action_starcoder"] for row in pretrain_meta], dtype=np.float32),
        "next_train_loss": np.asarray([row["next_train_loss"] for row in pretrain_meta], dtype=np.float32),
        "next_eval_bpb": np.asarray([row["next_eval_bpb"] for row in pretrain_meta], dtype=np.float32),
        "sequences": (
            np.stack(pretrain_windows, axis=0).astype(np.float32)
            if pretrain_windows
            else np.zeros((0, config.window_bins, len(SEQUENCE_CHANNELS)), dtype=np.float32)
        ),
        "masks": (
            np.stack(pretrain_masks, axis=0).astype(np.float32)
            if pretrain_masks
            else np.zeros((0, config.window_bins), dtype=np.float32)
        ),
    }
    np.savez_compressed(output_dir / "pretrain_sequences.npz", **pretrain_payload)

    feature_manifest = {
        "objective_metric": config.objective_metric,
        "feature_coverage_threshold": config.feature_coverage_threshold,
        "selected_feature_keys": selected_feature_keys,
        "candidate_feature_keys": list(SUMMARY_FEATURES),
        "feature_defaults": feature_defaults,
        "rows": coverage_rows,
    }
    with (output_dir / "feature_manifest.json").open("w") as f:
        json.dump(feature_manifest, f, indent=2, sort_keys=True)

    sequence_manifest = {
        "window_steps": config.window_steps,
        "window_bins": config.window_bins,
        "bin_width": config.window_steps // config.window_bins,
        "channels": list(SEQUENCE_CHANNELS),
        "sequence_shape": [len(decisions), config.window_bins, len(SEQUENCE_CHANNELS)],
        "pretrain_count": len(pretrain_meta),
        "pretrain_stride": config.pretrain_stride,
    }
    with (output_dir / "sequence_manifest.json").open("w") as f:
        json.dump(sequence_manifest, f, indent=2, sort_keys=True)

    manifest = {
        "objective_metric": config.objective_metric,
        "n_runs": int(decisions["wandb_run_id"].nunique()),
        "n_decision_rows": len(decisions),
        "action_grid": [float(value) for value in action_grid.tolist()],
        "selected_feature_keys": selected_feature_keys,
        "n_cv_folds": int(config.n_cv_folds),
        "window_steps": int(config.window_steps),
        "window_bins": int(config.window_bins),
    }
    with (output_dir / "dataset_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return decisions, sequence_arrays, manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build three-phase dense policy datasets.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="experiments/domain_phase_mix/offline_rl/artifacts/three_phase_starcoder_dense_v3",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/domain_phase_mix/offline_rl/artifacts/three_phase_starcoder_dense_v3",
    )
    parser.add_argument("--objective-metric", type=str, default=DEFAULT_OBJECTIVE_METRIC)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    build_three_phase_dense_policy_dataset(
        BuildThreePhaseDenseDatasetConfig(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            objective_metric=args.objective_metric,
        )
    )


if __name__ == "__main__":
    main()
