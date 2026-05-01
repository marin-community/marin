# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Policy artifact formats and helpers for offline-RL baselines."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from experiments.domain_phase_mix.offline_rl.contracts import PolicyKindV2


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


@dataclass(frozen=True)
class PolicyArtifactV2:
    """Serialized metadata for pooled offline-control v2 policies."""

    kind: PolicyKindV2
    objective_metric: str
    state_keys: tuple[str, ...]
    action_low: float
    action_high: float
    action_values: list[float]
    state_mean: list[float]
    state_std: list[float]
    reward_mean: float
    reward_std: float
    model_path: str
    aux_paths: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, float | int | str | bool] = field(default_factory=dict)


AnyPolicyArtifact = PolicyArtifactV1 | PolicyArtifactV2


def save_policy_artifact(path: str | Path, artifact: AnyPolicyArtifact) -> None:
    """Persist a policy artifact to JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as f:
        json.dump(dataclasses.asdict(artifact), f, indent=2, sort_keys=True)


save_policy_artifact_v2 = save_policy_artifact


def load_policy_artifact(path: str | Path) -> AnyPolicyArtifact:
    """Load a v1 or v2 policy artifact from JSON."""
    source = Path(path)
    with source.open() as f:
        payload = json.load(f)

    kind = str(payload["kind"])
    if kind == "d3rlpy_cql_continuous_v1":
        return PolicyArtifactV1(
            kind="d3rlpy_cql_continuous_v1",
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

    return PolicyArtifactV2(
        kind=kind,  # type: ignore[arg-type]
        objective_metric=payload["objective_metric"],
        state_keys=tuple(payload["state_keys"]),
        action_low=float(payload["action_low"]),
        action_high=float(payload["action_high"]),
        action_values=[float(x) for x in payload.get("action_values", [])],
        state_mean=[float(x) for x in payload["state_mean"]],
        state_std=[float(x) for x in payload["state_std"]],
        reward_mean=float(payload["reward_mean"]),
        reward_std=float(payload["reward_std"]),
        model_path=str(payload["model_path"]),
        aux_paths={str(k): str(v) for k, v in payload.get("aux_paths", {}).items()},
        metadata=payload.get("metadata", {}),
    )


load_policy_artifact_v2 = load_policy_artifact


def normalize_state(state: dict[str, float], artifact: PolicyArtifactV1 | PolicyArtifactV2) -> np.ndarray:
    """Normalize a state dict to the policy's expected feature vector."""
    raw = np.asarray([float(state[key]) for key in artifact.state_keys], dtype=np.float32)
    mean = np.asarray(artifact.state_mean, dtype=np.float32)
    std = np.asarray(artifact.state_std, dtype=np.float32)
    safe_std = np.where(std <= 0.0, 1.0, std)
    return (raw - mean) / safe_std


def clip_action(action: float, artifact: PolicyArtifactV1 | PolicyArtifactV2) -> float:
    """Clip an action to policy bounds."""
    return float(np.clip(action, artifact.action_low, artifact.action_high))


def discretize_action(action: float, artifact: PolicyArtifactV2) -> tuple[int, float]:
    """Map a continuous action onto the nearest v2 discrete action value."""
    if not artifact.action_values:
        raise ValueError("Policy artifact does not define a discrete action grid.")
    values = np.asarray(artifact.action_values, dtype=np.float32)
    index = int(np.argmin(np.abs(values - action)))
    return index, float(values[index])
