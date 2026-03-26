# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build offline-RL transitions for 3-phase StarCoder."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.offline_rl.contracts import (
    DEFAULT_STATE_KEYS,
    RLFeatureConfig,
    default_feature_config,
)

PHASE_COLUMN_TEMPLATE = "phase_{phase_idx}_starcoder"


@dataclass(frozen=True)
class BuildTransitionsConfig:
    """Config for converting run histories into transition datasets."""

    input_dir: str
    output_dir: str
    feature_config: RLFeatureConfig
    split_seed: int = 42
    train_fraction: float = 0.8


def _deterministic_hash(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()[:8]
    return int(digest, 16)


def assign_split(run_id: str, seed: int, train_fraction: float) -> str:
    """Assign one run id into train/val split deterministically."""
    score = _deterministic_hash(run_id, seed) / float(16**8)
    return "train" if score < train_fraction else "val"


def _feature_defaults(frame: pd.DataFrame, keys: tuple[str, ...]) -> dict[str, float]:
    defaults: dict[str, float] = {}
    for key in keys:
        if key not in frame.columns:
            defaults[key] = 0.0
            continue
        series = frame[key].replace([np.inf, -np.inf], np.nan)
        value = float(series.median()) if series.notna().any() else 0.0
        defaults[key] = value
    defaults["phase_index"] = 0.0
    defaults["tokens_frac"] = 0.0
    defaults["steps_since_last_eval_frac"] = 1.0
    defaults["prev_action_starcoder"] = 0.5
    return defaults


def _last_metric_at_or_before(
    history: pd.DataFrame,
    metric_col: str,
    decision_step: int,
) -> tuple[float | None, int | None]:
    if metric_col not in history.columns:
        return None, None
    metric = history.loc[(history["step"] <= decision_step) & history[metric_col].notna(), ["step", metric_col]]
    if metric.empty:
        return None, None
    row = metric.sort_values("step").iloc[-1]
    return float(row[metric_col]), int(row["step"])


def extract_decision_state(
    history: pd.DataFrame,
    decision_step: int,
    phase_index: int,
    prev_action_starcoder: float,
    total_steps: int,
    objective_metric: str,
    defaults: dict[str, float],
) -> dict[str, float]:
    """Extract decision-time state features from a run history."""
    train_loss, _ = _last_metric_at_or_before(history, "train/loss", decision_step)
    eval_loss, _ = _last_metric_at_or_before(history, "eval/loss", decision_step)
    last_obj_bpb, last_eval_step = _last_metric_at_or_before(history, objective_metric, decision_step)

    tokens_frac = decision_step / float(total_steps)
    if last_eval_step is None:
        steps_since_eval_frac = 1.0
    else:
        steps_since_eval_frac = (decision_step - last_eval_step) / float(total_steps)

    state = {
        "phase_index": float(phase_index),
        "last_train_loss": train_loss if train_loss is not None else defaults["last_train_loss"],
        "last_eval_loss": eval_loss if eval_loss is not None else defaults["last_eval_loss"],
        "last_obj_bpb": last_obj_bpb if last_obj_bpb is not None else defaults["last_obj_bpb"],
        "tokens_frac": tokens_frac,
        "steps_since_last_eval_frac": steps_since_eval_frac,
        "prev_action_starcoder": prev_action_starcoder,
    }
    return state


def _impute_state_columns(frame: pd.DataFrame, feature_keys: tuple[str, ...]) -> pd.DataFrame:
    result = frame.copy()
    for key in feature_keys:
        if key not in result.columns:
            result[key] = 0.0
        series = result[key].replace([np.inf, -np.inf], np.nan)
        default = float(series.median()) if series.notna().any() else 0.0
        result[key] = series.fillna(default).astype(float)
    return result


def add_reward_standardization(
    frame: pd.DataFrame,
    reward_col: str,
    split_col: str = "split",
) -> tuple[pd.DataFrame, float, float]:
    """Add standardized reward column based on train-split stats."""
    result = frame.copy()
    train_rewards = result.loc[result[split_col] == "train", reward_col]
    reward_mean = float(train_rewards.mean()) if train_rewards.notna().any() else 0.0
    reward_std = float(train_rewards.std(ddof=0)) if train_rewards.notna().any() else 0.0
    if reward_std <= 0.0:
        reward_std = 1.0
    standardized_col = reward_col.replace("_raw", "_std")
    result[standardized_col] = (result[reward_col] - reward_mean) / reward_std
    return result, reward_mean, reward_std


def augment_suffix_fragments(base_transitions: pd.DataFrame) -> pd.DataFrame:
    """Create suffix-fragment episodes starting at t=0,1,2 for each run."""
    rows: list[pd.DataFrame] = []
    for run_id, run_df in base_transitions.groupby("wandb_run_id"):
        run_df = run_df.sort_values("t").reset_index(drop=True)
        for fragment_start in range(len(run_df)):
            fragment = run_df.iloc[fragment_start:].copy()
            fragment["fragment_start"] = fragment_start
            fragment["step_in_episode"] = np.arange(len(fragment), dtype=int)
            fragment["episode_id"] = f"{run_id}:frag{fragment_start}"
            rows.append(fragment)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _extract_final_objective(run_row: pd.Series, history: pd.DataFrame, objective_metric: str) -> float | None:
    value = run_row.get(objective_metric)
    if isinstance(value, int | float):
        return float(value)
    if objective_metric not in history.columns:
        return None
    metric = history[objective_metric].dropna()
    if metric.empty:
        return None
    return float(metric.iloc[-1])


def build_transition_tables(
    runs_df: pd.DataFrame,
    history_wide_df: pd.DataFrame,
    feature_config: RLFeatureConfig,
    split_seed: int,
    train_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    """Build base transition rows plus fragment-augmented episodes."""
    if runs_df.empty:
        raise ValueError("runs dataframe is empty")

    defaults = _feature_defaults(history_wide_df, DEFAULT_STATE_KEYS)
    boundary_0, boundary_1 = feature_config.phase_end_steps
    decision_steps = (0, boundary_0, boundary_1)

    base_rows: list[dict] = []

    for _, run_row in runs_df.iterrows():
        wandb_run_id = str(run_row["wandb_run_id"])
        run_history = history_wide_df[history_wide_df["wandb_run_id"] == wandb_run_id].copy()
        run_history = run_history.sort_values("step")
        split = assign_split(wandb_run_id, split_seed, train_fraction)

        actions = []
        for phase_idx in range(3):
            col = PHASE_COLUMN_TEMPLATE.format(phase_idx=phase_idx)
            if col not in run_row:
                raise ValueError(f"Missing action column '{col}' in run table")
            actions.append(float(run_row[col]))

        final_objective = _extract_final_objective(run_row, run_history, feature_config.objective_metric)
        if final_objective is None:
            continue

        state_by_t: list[dict[str, float]] = []
        for t, decision_step in enumerate(decision_steps):
            prev_action = actions[t - 1] if t > 0 else defaults["prev_action_starcoder"]
            state = extract_decision_state(
                history=run_history,
                decision_step=decision_step,
                phase_index=t,
                prev_action_starcoder=float(prev_action),
                total_steps=feature_config.total_steps,
                objective_metric=feature_config.objective_metric,
                defaults=defaults,
            )
            state_by_t.append(state)

        objectives = [state["last_obj_bpb"] for state in state_by_t]
        objectives.append(final_objective)

        for t in range(3):
            reward_raw = -final_objective if t == 2 else 0.0
            if t < 2:
                reward_delta_raw = -(objectives[t + 1] - objectives[t])
            else:
                reward_delta_raw = -final_objective

            row = {
                "wandb_run_id": wandb_run_id,
                "source_experiment": run_row.get("source_experiment"),
                "local_run_id": run_row.get("local_run_id"),
                "split": split,
                "t": t,
                "decision_step": decision_steps[t],
                "done": bool(t == 2),
                "action_starcoder": actions[t],
                "final_objective": final_objective,
                "reward_raw": reward_raw,
                "reward_delta_raw": reward_delta_raw,
                "next_obj_bpb": objectives[t + 1],
            }
            row.update(state_by_t[t])
            base_rows.append(row)

    base = pd.DataFrame(base_rows)
    if base.empty:
        raise ValueError("No transitions produced; verify objective and phase columns")

    base = _impute_state_columns(base, DEFAULT_STATE_KEYS)
    base, reward_mean, reward_std = add_reward_standardization(base, "reward_raw")
    base, reward_delta_mean, reward_delta_std = add_reward_standardization(base, "reward_delta_raw")
    episodes = augment_suffix_fragments(base)

    manifest: dict[str, float | int | str] = {
        "n_runs": int(base["wandb_run_id"].nunique()),
        "n_base_transitions": len(base),
        "n_episode_rows": len(episodes),
        "n_augmented_episodes": int(episodes["episode_id"].nunique()),
        "objective_metric": feature_config.objective_metric,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "reward_delta_mean": reward_delta_mean,
        "reward_delta_std": reward_delta_std,
    }
    return base, episodes, manifest


def run_build(config: BuildTransitionsConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    """Run transition building end-to-end."""
    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_df = pd.read_parquet(input_dir / "runs.parquet")
    history_wide_df = pd.read_parquet(input_dir / "history_wide.parquet")

    base, episodes, manifest = build_transition_tables(
        runs_df=runs_df,
        history_wide_df=history_wide_df,
        feature_config=config.feature_config,
        split_seed=config.split_seed,
        train_fraction=config.train_fraction,
    )

    base.to_parquet(output_dir / "transitions.parquet", index=False)
    episodes.to_parquet(output_dir / "episodes.parquet", index=False)
    with (output_dir / "dataset_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return base, episodes, manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline-RL transitions for 3-phase StarCoder.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="experiments/domain_phase_mix/offline_rl/artifacts/three_phase_starcoder",
        help="Directory containing runs.parquet and history_wide.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/domain_phase_mix/offline_rl/artifacts/three_phase_starcoder",
        help="Directory for transitions outputs.",
    )
    parser.add_argument(
        "--objective-metric",
        type=str,
        default=default_feature_config().objective_metric,
        help="Objective metric key.",
    )
    parser.add_argument("--split-seed", type=int, default=42, help="Train/val split seed.")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Train split fraction.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_build(
        BuildTransitionsConfig(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            feature_config=default_feature_config(args.objective_metric),
            split_seed=args.split_seed,
            train_fraction=args.train_fraction,
        )
    )


if __name__ == "__main__":
    main()
