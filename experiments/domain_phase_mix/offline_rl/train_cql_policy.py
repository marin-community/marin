# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train an offline CQL policy on 3-phase StarCoder transitions."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.offline_rl.contracts import DEFAULT_STATE_KEYS, DEFAULT_OBJECTIVE_METRIC
from experiments.domain_phase_mix.offline_rl.policy_artifact import PolicyArtifactV1, save_policy_artifact

logger = logging.getLogger(__name__)


class CQLBackend(Protocol):
    """Minimal backend interface used by the trainer."""

    def fit_steps(self, n_steps: int) -> None:
        """Train for *n_steps* offline updates."""

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predict actions for observations."""

    def predict_value(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Predict Q-values for observation/action pairs."""

    def save(self, path: str) -> None:
        """Persist the backend model."""


@dataclass(frozen=True)
class TrainCQLConfig:
    """Configuration for offline CQL training."""

    input_dir: str
    output_dir: str
    objective_metric: str = DEFAULT_OBJECTIVE_METRIC
    epochs: int = 10
    steps_per_epoch: int = 500
    gamma: float = 0.99
    device: str = "auto"
    action_low: float = 0.05
    action_high: float = 0.95


@dataclass(frozen=True)
class EpochMetrics:
    """Scored metrics for one epoch checkpoint."""

    epoch: int
    td_error: float
    conservative_gap: float
    score: float
    model_path: str


def resolve_device(device: str) -> str:
    """Resolve training device from a user-facing selector."""
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _load_dataset(input_dir: str) -> tuple[pd.DataFrame, dict]:
    root = Path(input_dir)
    episodes = pd.read_parquet(root / "episodes.parquet")
    with (root / "dataset_manifest.json").open() as f:
        manifest = json.load(f)
    return episodes, manifest


def _to_numpy(df: pd.DataFrame, state_keys: tuple[str, ...], reward_col: str) -> dict[str, np.ndarray]:
    ordered = df.sort_values(["episode_id", "step_in_episode"]).reset_index(drop=True)
    obs = ordered.loc[:, list(state_keys)].to_numpy(dtype=np.float32)
    act = ordered.loc[:, ["action_starcoder"]].to_numpy(dtype=np.float32)
    rew = ordered.loc[:, reward_col].to_numpy(dtype=np.float32)
    done = ordered.loc[:, "done"].astype(bool).to_numpy()

    next_obs = obs.copy()
    if len(ordered) > 0:
        for idx in range(len(ordered) - 1):
            same_episode = ordered.at[idx, "episode_id"] == ordered.at[idx + 1, "episode_id"]
            if same_episode:
                next_obs[idx] = obs[idx + 1]
        next_obs[-1] = obs[-1]

    return {
        "obs": obs,
        "actions": act,
        "rewards": rew,
        "dones": done,
        "next_obs": next_obs,
    }


def _build_d3rlpy_backend(
    train_batch: dict[str, np.ndarray],
    resolved_device: str,
) -> CQLBackend:
    try:
        from d3rlpy.algos import CQLConfig
        from d3rlpy.dataset import MDPDataset
    except ImportError as exc:
        raise RuntimeError("d3rlpy is required for CQL training. Install with `uv add d3rlpy`.") from exc

    dataset = MDPDataset(
        observations=train_batch["obs"],
        actions=train_batch["actions"],
        rewards=train_batch["rewards"],
        terminals=train_batch["dones"],
    )

    algo = CQLConfig().create(device=resolved_device)

    class _Backend:
        def __init__(self):
            self._algo = algo
            self._dataset = dataset

        def fit_steps(self, n_steps: int) -> None:
            self._algo.fit(self._dataset, n_steps=n_steps, n_steps_per_epoch=n_steps, show_progress=False)

        def predict(self, observations: np.ndarray) -> np.ndarray:
            return np.asarray(self._algo.predict(observations), dtype=np.float32)

        def predict_value(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
            return np.asarray(self._algo.predict_value(observations, actions), dtype=np.float32)

        def save(self, path: str) -> None:
            # Prefer saving full learnable object when available.
            if hasattr(self._algo, "save"):
                self._algo.save(path)
            else:
                self._algo.save_model(path)

    return _Backend()


def _compute_epoch_metrics(
    backend: CQLBackend,
    batch: dict[str, np.ndarray],
    gamma: float,
    conservative_floor: float,
) -> tuple[float, float, float]:
    q_behavior = backend.predict_value(batch["obs"], batch["actions"]).reshape(-1)
    policy_actions = backend.predict(batch["obs"]).astype(np.float32).reshape(-1, 1)
    q_policy = backend.predict_value(batch["obs"], policy_actions).reshape(-1)
    q_next = backend.predict_value(batch["next_obs"], backend.predict(batch["next_obs"]).astype(np.float32)).reshape(-1)

    targets = batch["rewards"] + gamma * (1.0 - batch["dones"].astype(np.float32)) * q_next
    td_error = float(np.mean((q_behavior - targets) ** 2))
    conservative_gap = float(np.mean(np.maximum(q_policy - q_behavior, 0.0)))
    trend_penalty = max(0.0, conservative_gap - conservative_floor)
    score = td_error + 0.1 * trend_penalty
    return td_error, conservative_gap, score


def _run_single_variant(
    *,
    label: str,
    reward_col: str,
    episodes: pd.DataFrame,
    state_keys: tuple[str, ...],
    config: TrainCQLConfig,
    output_dir: Path,
    backend_factory,
) -> tuple[list[EpochMetrics], Path]:
    train_rows = episodes[episodes["split"] == "train"].copy()
    val_rows = episodes[episodes["split"] == "val"].copy()
    train_batch = _to_numpy(train_rows, state_keys, reward_col)
    val_batch = _to_numpy(val_rows, state_keys, reward_col)

    backend = backend_factory(train_batch, resolve_device(config.device))
    variant_dir = output_dir / label
    variant_dir.mkdir(parents=True, exist_ok=True)

    best_floor = float("inf")
    metrics: list[EpochMetrics] = []

    for epoch in range(1, config.epochs + 1):
        backend.fit_steps(config.steps_per_epoch)
        conservative_floor = best_floor if np.isfinite(best_floor) else 0.0
        td_error, conservative_gap, score = _compute_epoch_metrics(
            backend=backend,
            batch=val_batch,
            gamma=config.gamma,
            conservative_floor=conservative_floor,
        )
        best_floor = min(best_floor, conservative_gap)

        model_path = variant_dir / f"epoch_{epoch:03d}.d3"
        backend.save(str(model_path))
        metrics.append(
            EpochMetrics(
                epoch=epoch,
                td_error=td_error,
                conservative_gap=conservative_gap,
                score=score,
                model_path=str(model_path),
            )
        )

    best = min(metrics, key=lambda item: item.score)
    best_path = variant_dir / "model_best.d3"
    final_path = variant_dir / "model_final.d3"
    shutil.copyfile(best.model_path, best_path)
    shutil.copyfile(metrics[-1].model_path, final_path)
    return metrics, best_path


def _write_metrics_report(
    *,
    output_dir: Path,
    primary_metrics: list[EpochMetrics],
    delta_metrics: list[EpochMetrics],
) -> None:
    rows = []
    for label, series in (("terminal", primary_metrics), ("delta", delta_metrics)):
        for item in series:
            rows.append(
                {
                    "reward_variant": label,
                    "epoch": item.epoch,
                    "td_error": item.td_error,
                    "conservative_gap": item.conservative_gap,
                    "score": item.score,
                    "model_path": item.model_path,
                }
            )
    report_df = pd.DataFrame(rows)
    report_df.to_csv(output_dir / "offline_training_report.csv", index=False)

    summary = {
        "best_terminal_epoch": int(min(primary_metrics, key=lambda item: item.score).epoch),
        "best_delta_epoch": int(min(delta_metrics, key=lambda item: item.score).epoch),
    }
    with (output_dir / "offline_training_report.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def train_policy(
    config: TrainCQLConfig,
    backend_factory=_build_d3rlpy_backend,
) -> PolicyArtifactV1:
    """Train terminal and delta reward variants; return primary policy artifact."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes, manifest = _load_dataset(config.input_dir)
    state_keys = tuple(key for key in DEFAULT_STATE_KEYS if key in episodes.columns)
    if len(state_keys) != len(DEFAULT_STATE_KEYS):
        raise ValueError("Episode dataset is missing required state columns")

    primary_metrics, primary_best_path = _run_single_variant(
        label="terminal_reward",
        reward_col="reward_std",
        episodes=episodes,
        state_keys=state_keys,
        config=config,
        output_dir=output_dir,
        backend_factory=backend_factory,
    )
    delta_metrics, _ = _run_single_variant(
        label="delta_reward",
        reward_col="reward_delta_std",
        episodes=episodes,
        state_keys=state_keys,
        config=config,
        output_dir=output_dir,
        backend_factory=backend_factory,
    )
    _write_metrics_report(output_dir=output_dir, primary_metrics=primary_metrics, delta_metrics=delta_metrics)

    train_rows = episodes[episodes["split"] == "train"]
    state_mean = [float(train_rows[key].mean()) for key in state_keys]
    state_std = []
    for key in state_keys:
        value = float(train_rows[key].std(ddof=0))
        state_std.append(value if value > 0 else 1.0)
    artifact = PolicyArtifactV1(
        kind="d3rlpy_cql_continuous_v1",
        objective_metric=config.objective_metric,
        state_keys=state_keys,
        action_low=config.action_low,
        action_high=config.action_high,
        state_mean=state_mean,
        state_std=state_std,
        reward_mean=float(manifest.get("reward_mean", 0.0)),
        reward_std=float(manifest.get("reward_std", 1.0)),
        model_path=str(primary_best_path.resolve()),
    )
    save_policy_artifact(output_dir / "policy_artifact.json", artifact)
    return artifact


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train offline CQL baseline for 3-phase StarCoder.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="experiments/domain_phase_mix/offline_rl/artifacts/three_phase_starcoder",
        help="Directory containing episodes.parquet and dataset_manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/domain_phase_mix/offline_rl/artifacts/three_phase_starcoder/cql_policy",
        help="Directory to write trained policy artifacts.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--steps-per-epoch", type=int, default=500, help="Offline updates per epoch.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: auto, mps, cpu, cuda:0, etc.",
    )
    parser.add_argument("--action-low", type=float, default=0.05, help="Minimum allowed action.")
    parser.add_argument("--action-high", type=float, default=0.95, help="Maximum allowed action.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    train_policy(
        TrainCQLConfig(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            gamma=args.gamma,
            device=args.device,
            action_low=args.action_low,
            action_high=args.action_high,
        )
    )


if __name__ == "__main__":
    main()
