# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Policy artifact format and helpers for offline-RL baselines."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class PolicyArtifactV1:
    """Serialized metadata for a trained d3rlpy CQL policy."""

    kind: Literal["d3rlpy_cql_continuous_v1"]
    objective_metric: str
    state_keys: tuple[str, ...]
    action_low: float
    action_high: float
    state_mean: list[float]
    state_std: list[float]
    reward_mean: float
    reward_std: float
    model_path: str


def save_policy_artifact(path: str | Path, artifact: PolicyArtifactV1) -> None:
    """Persist a policy artifact to JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as f:
        json.dump(dataclasses.asdict(artifact), f, indent=2, sort_keys=True)


def load_policy_artifact(path: str | Path) -> PolicyArtifactV1:
    """Load a policy artifact from JSON."""
    source = Path(path)
    with source.open() as f:
        payload = json.load(f)
    return PolicyArtifactV1(
        kind=payload["kind"],
        objective_metric=payload["objective_metric"],
        state_keys=tuple(payload["state_keys"]),
        action_low=float(payload["action_low"]),
        action_high=float(payload["action_high"]),
        state_mean=[float(x) for x in payload["state_mean"]],
        state_std=[float(x) for x in payload["state_std"]],
        reward_mean=float(payload["reward_mean"]),
        reward_std=float(payload["reward_std"]),
        model_path=str(payload["model_path"]),
    )


def normalize_state(state: dict[str, float], artifact: PolicyArtifactV1) -> np.ndarray:
    """Normalize a state dict to the policy's expected feature vector."""
    raw = np.asarray([float(state[key]) for key in artifact.state_keys], dtype=np.float32)
    mean = np.asarray(artifact.state_mean, dtype=np.float32)
    std = np.asarray(artifact.state_std, dtype=np.float32)
    safe_std = np.where(std <= 0.0, 1.0, std)
    return (raw - mean) / safe_std


def clip_action(action: float, artifact: PolicyArtifactV1) -> float:
    """Clip an action to policy bounds."""
    return float(np.clip(action, artifact.action_low, artifact.action_high))
