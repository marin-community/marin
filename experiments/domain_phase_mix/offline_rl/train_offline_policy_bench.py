# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train and compare pooled offline-control baselines for StarCoder mixture policies."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from joblib import dump
from sklearn.ensemble import HistGradientBoostingRegressor

from experiments.domain_phase_mix.offline_rl.build_pooled_transition_dataset import discretize_action
from experiments.domain_phase_mix.offline_rl.contracts import DEFAULT_OBJECTIVE_METRIC, PolicyKindV2
from experiments.domain_phase_mix.offline_rl.ope import (
    ActionPolicy,
    BehaviorPolicyModel,
    RewardModelBundle,
    direct_method_value,
    doubly_robust_estimate,
    fit_behavior_policy,
    fit_fqe_model,
    fit_reward_models,
    initial_state_value,
    policy_diagnostics,
    weighted_importance_sampling,
)
from experiments.domain_phase_mix.offline_rl.policy_artifact import PolicyArtifactV2, save_policy_artifact

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OfflinePolicyBenchConfig:
    """Configuration for the pooled v2 offline-control benchmark."""

    input_dir: str
    output_dir: str
    objective_metric: str = DEFAULT_OBJECTIVE_METRIC
    random_state: int = 42
    gamma: float = 1.0
    support_threshold: float = 0.02
    bound_rate_threshold: float = 0.25
    reward_bonus_weight: float = 0.15
    final_model_weight: float = 1.0
    fqe_iterations: int = 8
    d3rlpy_steps: int = 800
    d3rlpy_steps_per_epoch: int = 200
    discrete_iql_epochs: int = 160
    discrete_iql_batch_size: int = 64
    discrete_iql_hidden_size: int = 64
    include_continuous_cql: bool = True
    device: str = "auto"


@dataclass(frozen=True)
class StateStats:
    """Normalization statistics for policy training."""

    mean: np.ndarray
    std: np.ndarray

    def normalize(self, frame: pd.DataFrame, feature_keys: tuple[str, ...]) -> np.ndarray:
        raw = frame.loc[:, list(feature_keys)].to_numpy(dtype=np.float32)
        safe_std = np.where(self.std <= 0.0, 1.0, self.std)
        return (raw - self.mean) / safe_std


class FixedSchedulePolicy:
    """Static baseline policy using one fixed StarCoder schedule."""

    def __init__(self, action_grid: np.ndarray, schedule_by_decision: dict[int, float]):
        self.name = "fixed_best_schedule"
        self._value_by_decision: dict[int, float] = {}
        self._index_by_decision: dict[int, int] = {}
        for decision_index, value in schedule_by_decision.items():
            action_idx, action_value = discretize_action(float(value), action_grid)
            self._index_by_decision[int(decision_index)] = action_idx
            self._value_by_decision[int(decision_index)] = action_value

    def predict_action_indices(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            [self._index_by_decision[int(decision_index)] for decision_index in frame["decision_index"].tolist()],
            dtype=np.int64,
        )


class OutcomePlannerPolicy:
    """Support-constrained finite-horizon planner trained with backward induction."""

    def __init__(
        self,
        *,
        name: str,
        feature_keys: tuple[str, ...],
        action_grid: np.ndarray,
        q_models: dict[int, HistGradientBoostingRegressor],
        behavior_policy: BehaviorPolicyModel,
        support_threshold: float,
    ):
        self.name = name
        self.feature_keys = feature_keys
        self.action_grid = action_grid
        self.q_models = q_models
        self.behavior_policy = behavior_policy
        self.support_threshold = support_threshold

    def _stage_q_values(self, frame: pd.DataFrame, action_values: np.ndarray) -> np.ndarray:
        decision_index = int(frame["decision_index"].iloc[0])
        model = self.q_models[decision_index]
        inputs = np.concatenate(
            [
                frame.loc[:, list(self.feature_keys)].to_numpy(dtype=np.float32),
                action_values.reshape(-1, 1).astype(np.float32),
            ],
            axis=1,
        )
        return model.predict(inputs).astype(np.float32)

    def best_supported_actions(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        outputs = np.zeros(len(frame), dtype=np.int64)
        values = np.zeros(len(frame), dtype=np.float32)
        for _, stage_frame in frame.groupby("decision_index", sort=False):
            stage_positions = stage_frame.index.to_numpy(dtype=int)
            if len(stage_positions) == 0:
                continue
            repeated_inputs = np.repeat(
                stage_frame.loc[:, list(self.feature_keys)].to_numpy(dtype=np.float32),
                len(self.action_grid),
                axis=0,
            )
            repeated = pd.DataFrame(
                repeated_inputs,
                columns=list(self.feature_keys),
            )
            action_candidates = np.tile(self.action_grid, len(stage_frame)).astype(np.float32)
            support_candidates = np.tile(np.arange(len(self.action_grid), dtype=np.int64), len(stage_frame))
            support_matrix = self.behavior_policy.predict_proba(repeated)
            support = support_matrix[np.arange(len(repeated)), support_candidates].reshape(
                len(stage_frame),
                len(self.action_grid),
            )
            q_values = self._stage_q_values(repeated, action_candidates).reshape(len(stage_frame), len(self.action_grid))

            for local_index, global_index in enumerate(stage_positions):
                supported = np.where(support[local_index] >= self.support_threshold)[0]
                candidate_indices = (
                    supported
                    if len(supported) > 0
                    else np.where(support[local_index] == np.max(support[local_index]))[0]
                )
                best_local = int(candidate_indices[np.argmax(q_values[local_index, candidate_indices])])
                outputs[global_index] = best_local
                values[global_index] = q_values[local_index, best_local]
        return outputs, values

    def predict_action_indices(self, frame: pd.DataFrame) -> np.ndarray:
        outputs, _ = self.best_supported_actions(frame.reset_index(drop=True))
        return outputs


class D3RLPyDiscretePolicy:
    """Wrapper around a discrete d3rlpy policy."""

    def __init__(
        self,
        *,
        name: str,
        algo,
        feature_keys: tuple[str, ...],
        action_grid: np.ndarray,
        state_stats: StateStats,
    ):
        self.name = name
        self.algo = algo
        self.feature_keys = feature_keys
        self.action_grid = action_grid
        self.state_stats = state_stats

    def predict_action_indices(self, frame: pd.DataFrame) -> np.ndarray:
        states = self.state_stats.normalize(frame, self.feature_keys)
        return np.asarray(self.algo.predict(states), dtype=np.int64).reshape(-1)

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.d3"
        if hasattr(self.algo, "save"):
            self.algo.save(str(model_path))
        else:
            self.algo.save_model(str(model_path))
        return model_path


class D3RLPyContinuousCQLPolicy:
    """Wrapper around a continuous d3rlpy CQL policy with discrete projection."""

    def __init__(
        self,
        *,
        algo,
        feature_keys: tuple[str, ...],
        action_grid: np.ndarray,
        state_stats: StateStats,
    ):
        self.name = "continuous_cql_ablation"
        self.algo = algo
        self.feature_keys = feature_keys
        self.action_grid = action_grid
        self.state_stats = state_stats

    def predict_action_indices(self, frame: pd.DataFrame) -> np.ndarray:
        states = self.state_stats.normalize(frame, self.feature_keys)
        raw = np.asarray(self.algo.predict(states), dtype=np.float32).reshape(len(frame), -1)[:, 0]
        return np.asarray([discretize_action(float(value), self.action_grid)[0] for value in raw], dtype=np.int64)

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.d3"
        if hasattr(self.algo, "save"):
            self.algo.save(str(model_path))
        else:
            self.algo.save_model(str(model_path))
        return model_path


class _IQLMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class TorchDiscreteIQLPolicy:
    """Minimal discrete IQL implementation for short-horizon pooled control."""

    def __init__(
        self,
        *,
        feature_keys: tuple[str, ...],
        action_grid: np.ndarray,
        state_stats: StateStats,
        q_net: _IQLMLP,
        v_net: _IQLMLP,
        policy_net: _IQLMLP,
        device: torch.device,
    ):
        self.name = "discrete_iql"
        self.feature_keys = feature_keys
        self.action_grid = action_grid
        self.state_stats = state_stats
        self.q_net = q_net
        self.v_net = v_net
        self.policy_net = policy_net
        self.device = device

    def predict_action_indices(self, frame: pd.DataFrame) -> np.ndarray:
        states = torch.tensor(
            self.state_stats.normalize(frame, self.feature_keys),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            logits = self.policy_net.forward(states)
            return torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.pt"
        torch.save(
            {
                "q_state_dict": self.q_net.state_dict(),
                "v_state_dict": self.v_net.state_dict(),
                "policy_state_dict": self.policy_net.state_dict(),
                "input_dim": int(self.q_net.net[0].in_features),
                "action_dim": int(self.q_net.net[-1].out_features),
                "hidden_size": int(self.q_net.net[0].out_features),
            },
            model_path,
        )
        return model_path


def resolve_device(device: str) -> str:
    """Resolve training device from a user-facing selector."""
    if device != "auto":
        return device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def compute_state_stats(frame: pd.DataFrame, feature_keys: tuple[str, ...]) -> StateStats:
    """Compute train-fold normalization statistics for policy inputs."""
    mean = frame.loc[:, list(feature_keys)].mean(axis=0).to_numpy(dtype=np.float32)
    std = frame.loc[:, list(feature_keys)].std(axis=0, ddof=0).to_numpy(dtype=np.float32)
    std = np.where(std <= 0.0, 1.0, std)
    return StateStats(mean=mean, std=std)


def _reward_standardize(train_rewards: pd.Series, rewards: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(train_rewards.mean()) if train_rewards.notna().any() else 0.0
    std = float(train_rewards.std(ddof=0)) if train_rewards.notna().any() else 1.0
    if std <= 0.0:
        std = 1.0
    return ((rewards - mean) / std).astype(np.float32), mean, std


def _state_action_inputs(frame: pd.DataFrame, feature_keys: tuple[str, ...], action_values: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            frame.loc[:, list(feature_keys)].to_numpy(dtype=np.float32),
            action_values.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    )


def _select_best_supported_actions(
    *,
    frame: pd.DataFrame,
    feature_keys: tuple[str, ...],
    action_grid: np.ndarray,
    q_models: dict[int, HistGradientBoostingRegressor],
    behavior_policy: BehaviorPolicyModel,
    support_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    planner = OutcomePlannerPolicy(
        name="outcome_planner",
        feature_keys=feature_keys,
        action_grid=action_grid,
        q_models=q_models,
        behavior_policy=behavior_policy,
        support_threshold=support_threshold,
    )
    return planner.best_supported_actions(frame.reset_index(drop=True))


def train_outcome_planner(
    frame: pd.DataFrame,
    feature_keys: tuple[str, ...],
    action_grid: np.ndarray,
    behavior_policy: BehaviorPolicyModel,
    config: OfflinePolicyBenchConfig,
) -> OutcomePlannerPolicy:
    """Fit a finite-horizon Q planner using backward induction on logged transitions."""
    ordered = frame.sort_values(["episode_id", "step_in_episode"]).reset_index(drop=True)
    next_features = ordered.groupby("episode_id", sort=False)[list(feature_keys)].shift(-1)
    q_models: dict[int, HistGradientBoostingRegressor] = {}

    for decision_index in sorted(ordered["decision_index"].unique(), reverse=True):
        stage_frame = ordered[ordered["decision_index"] == decision_index].copy()
        targets = stage_frame["reward_dense_raw"].to_numpy(dtype=np.float32)
        nonterminal_index = stage_frame.index[~stage_frame["done"].to_numpy(dtype=bool)]
        if len(nonterminal_index) > 0:
            next_rows = next_features.loc[nonterminal_index, list(feature_keys)].copy()
            _, continuation = _select_best_supported_actions(
                frame=next_rows,
                feature_keys=feature_keys,
                action_grid=action_grid,
                q_models=q_models,
                behavior_policy=behavior_policy,
                support_threshold=config.support_threshold,
            )
            targets[~stage_frame["done"].to_numpy(dtype=bool)] += config.gamma * continuation

        model = HistGradientBoostingRegressor(
            random_state=config.random_state + int(decision_index),
            max_depth=4,
            max_iter=64,
        )
        model.fit(
            _state_action_inputs(
                stage_frame,
                feature_keys,
                stage_frame["action_starcoder"].to_numpy(dtype=np.float32),
            ),
            targets,
        )
        q_models[int(decision_index)] = model

    return OutcomePlannerPolicy(
        name="outcome_planner",
        feature_keys=feature_keys,
        action_grid=action_grid,
        q_models=q_models,
        behavior_policy=behavior_policy,
        support_threshold=config.support_threshold,
    )


def _decision_state_defaults(frame: pd.DataFrame, feature_keys: tuple[str, ...]) -> dict[str, dict[str, float]]:
    defaults: dict[str, dict[str, float]] = {}
    for decision_index, stage_frame in frame.groupby("decision_index", sort=True):
        defaults[str(int(decision_index))] = {
            feature_name: float(stage_frame[feature_name].median()) for feature_name in feature_keys
        }
    return defaults


def _family_decision_state_defaults(
    frame: pd.DataFrame,
    feature_keys: tuple[str, ...],
) -> dict[str, dict[str, dict[str, float]]]:
    defaults: dict[str, dict[str, dict[str, float]]] = {}
    for run_family, family_frame in frame.groupby("run_family", sort=True):
        defaults[str(run_family)] = _decision_state_defaults(family_frame, feature_keys)
    return defaults


def _build_discrete_dataset(
    frame: pd.DataFrame,
    feature_keys: tuple[str, ...],
    state_stats: StateStats,
    reward_mean: float,
    reward_std: float,
    action_count: int,
):
    from d3rlpy.dataset import MDPDataset

    ordered = frame.sort_values(["episode_id", "step_in_episode"]).reset_index(drop=True)
    observations = state_stats.normalize(ordered, feature_keys)
    actions = ordered["action_idx"].to_numpy(dtype=np.int64)
    rewards = ((ordered["reward_dense_raw"].to_numpy(dtype=np.float32) - reward_mean) / reward_std).astype(np.float32)
    terminals = ordered["done"].astype(np.float32).to_numpy()
    return MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        action_size=action_count,
    )


def _build_continuous_dataset(
    frame: pd.DataFrame,
    feature_keys: tuple[str, ...],
    state_stats: StateStats,
    reward_mean: float,
    reward_std: float,
):
    from d3rlpy.dataset import MDPDataset

    ordered = frame.sort_values(["episode_id", "step_in_episode"]).reset_index(drop=True)
    observations = state_stats.normalize(ordered, feature_keys)
    actions = ordered.loc[:, ["action_starcoder"]].to_numpy(dtype=np.float32)
    rewards = ((ordered["reward_dense_raw"].to_numpy(dtype=np.float32) - reward_mean) / reward_std).astype(np.float32)
    terminals = ordered["done"].astype(np.float32).to_numpy()
    return MDPDataset(observations=observations, actions=actions, rewards=rewards, terminals=terminals)


def train_discrete_cql_policy(
    frame: pd.DataFrame,
    feature_keys: tuple[str, ...],
    state_stats: StateStats,
    reward_mean: float,
    reward_std: float,
    action_grid: np.ndarray,
    config: OfflinePolicyBenchConfig,
    output_dir: Path,
) -> D3RLPyDiscretePolicy:
    from d3rlpy.algos import DiscreteCQLConfig
    from d3rlpy.logging import FileAdapterFactory

    dataset = _build_discrete_dataset(
        frame,
        feature_keys,
        state_stats,
        reward_mean,
        reward_std,
        action_count=len(action_grid),
    )
    algo = DiscreteCQLConfig(gamma=config.gamma).create(device=resolve_device(config.device))
    algo.fit(
        dataset,
        n_steps=config.d3rlpy_steps,
        n_steps_per_epoch=config.d3rlpy_steps_per_epoch,
        show_progress=False,
        with_timestamp=False,
        experiment_name="discrete_cql",
        logger_adapter=FileAdapterFactory(root_dir=str(output_dir / "logs")),
    )
    return D3RLPyDiscretePolicy(
        name="discrete_cql",
        algo=algo,
        feature_keys=feature_keys,
        action_grid=action_grid,
        state_stats=state_stats,
    )


def train_discrete_bc_policy(
    frame: pd.DataFrame,
    feature_keys: tuple[str, ...],
    state_stats: StateStats,
    reward_mean: float,
    reward_std: float,
    action_grid: np.ndarray,
    config: OfflinePolicyBenchConfig,
    output_dir: Path,
) -> D3RLPyDiscretePolicy:
    from d3rlpy.algos import DiscreteBCConfig
    from d3rlpy.logging import FileAdapterFactory

    dataset = _build_discrete_dataset(
        frame,
        feature_keys,
        state_stats,
        reward_mean,
        reward_std,
        action_count=len(action_grid),
    )
    algo = DiscreteBCConfig().create(device=resolve_device(config.device))
    algo.fit(
        dataset,
        n_steps=config.d3rlpy_steps,
        n_steps_per_epoch=config.d3rlpy_steps_per_epoch,
        show_progress=False,
        with_timestamp=False,
        experiment_name="discrete_bc",
        logger_adapter=FileAdapterFactory(root_dir=str(output_dir / "logs")),
    )
    return D3RLPyDiscretePolicy(
        name="discrete_bc",
        algo=algo,
        feature_keys=feature_keys,
        action_grid=action_grid,
        state_stats=state_stats,
    )


def train_continuous_cql_policy(
    frame: pd.DataFrame,
    feature_keys: tuple[str, ...],
    state_stats: StateStats,
    reward_mean: float,
    reward_std: float,
    action_grid: np.ndarray,
    config: OfflinePolicyBenchConfig,
    output_dir: Path,
) -> D3RLPyContinuousCQLPolicy:
    from d3rlpy.algos import CQLConfig
    from d3rlpy.logging import FileAdapterFactory

    dataset = _build_continuous_dataset(frame, feature_keys, state_stats, reward_mean, reward_std)
    algo = CQLConfig(gamma=config.gamma).create(device=resolve_device(config.device))
    algo.fit(
        dataset,
        n_steps=config.d3rlpy_steps,
        n_steps_per_epoch=config.d3rlpy_steps_per_epoch,
        show_progress=False,
        with_timestamp=False,
        experiment_name="continuous_cql_ablation",
        logger_adapter=FileAdapterFactory(root_dir=str(output_dir / "logs")),
    )
    return D3RLPyContinuousCQLPolicy(
        algo=algo,
        feature_keys=feature_keys,
        action_grid=action_grid,
        state_stats=state_stats,
    )


def _expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return weight * diff.pow(2)


def train_discrete_iql_policy(
    frame: pd.DataFrame,
    feature_keys: tuple[str, ...],
    state_stats: StateStats,
    reward_mean: float,
    reward_std: float,
    action_grid: np.ndarray,
    config: OfflinePolicyBenchConfig,
) -> TorchDiscreteIQLPolicy:
    ordered = frame.sort_values(["episode_id", "step_in_episode"]).reset_index(drop=True)
    observations = state_stats.normalize(ordered, feature_keys)
    rewards = ((ordered["reward_dense_raw"].to_numpy(dtype=np.float32) - reward_mean) / reward_std).astype(np.float32)
    actions = ordered["action_idx"].to_numpy(dtype=np.int64)
    dones = ordered["done"].astype(bool).to_numpy()

    next_rows = ordered.groupby("episode_id", sort=False).shift(-1)
    next_observations = observations.copy()
    has_next = ~dones
    next_observations[has_next] = next_rows.loc[has_next, list(feature_keys)].to_numpy(dtype=np.float32)
    next_observations[has_next] = (next_observations[has_next] - state_stats.mean) / state_stats.std

    input_dim = observations.shape[1]
    action_dim = int(ordered["action_idx"].max()) + 1
    device = torch.device(resolve_device(config.device))

    q_net = _IQLMLP(input_dim, action_dim, config.discrete_iql_hidden_size).to(device)
    v_net = _IQLMLP(input_dim, 1, config.discrete_iql_hidden_size).to(device)
    policy_net = _IQLMLP(input_dim, action_dim, config.discrete_iql_hidden_size).to(device)

    q_opt = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    v_opt = torch.optim.Adam(v_net.parameters(), lr=1e-3)
    p_opt = torch.optim.Adam(policy_net.parameters(), lr=1e-3)

    obs_tensor = torch.tensor(observations, dtype=torch.float32, device=device)
    next_obs_tensor = torch.tensor(next_observations, dtype=torch.float32, device=device)
    act_tensor = torch.tensor(actions, dtype=torch.int64, device=device)
    rew_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    done_tensor = torch.tensor(dones.astype(np.float32), dtype=torch.float32, device=device)

    batch_size = min(config.discrete_iql_batch_size, len(ordered))
    rng = np.random.default_rng(config.random_state)

    for _ in range(config.discrete_iql_epochs):
        indices = rng.permutation(len(ordered))
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            obs = obs_tensor[batch_idx]
            next_obs = next_obs_tensor[batch_idx]
            act = act_tensor[batch_idx]
            rew = rew_tensor[batch_idx]
            done = done_tensor[batch_idx]

            with torch.no_grad():
                target = rew + config.gamma * (1.0 - done) * v_net.forward(next_obs).squeeze(-1)
            q_values = q_net.forward(obs)
            q_taken = q_values.gather(1, act.unsqueeze(1)).squeeze(1)
            q_loss = F.mse_loss(q_taken, target)
            q_opt.zero_grad()
            q_loss.backward()
            q_opt.step()

            with torch.no_grad():
                q_detached = q_net.forward(obs).gather(1, act.unsqueeze(1)).squeeze(1)
            v = v_net.forward(obs).squeeze(-1)
            v_loss = _expectile_loss(q_detached - v, tau=0.7).mean()
            v_opt.zero_grad()
            v_loss.backward()
            v_opt.step()

            with torch.no_grad():
                advantage = q_net.forward(obs).gather(1, act.unsqueeze(1)).squeeze(1) - v_net.forward(obs).squeeze(-1)
                weights = torch.clamp(torch.exp(advantage / 0.3), max=100.0)
            logits = policy_net.forward(obs)
            bc_loss = F.cross_entropy(logits, act, reduction="none")
            p_loss = torch.mean(weights * bc_loss)
            p_opt.zero_grad()
            p_loss.backward()
            p_opt.step()

    return TorchDiscreteIQLPolicy(
        feature_keys=feature_keys,
        action_grid=action_grid,
        state_stats=state_stats,
        q_net=q_net,
        v_net=v_net,
        policy_net=policy_net,
        device=device,
    )


def _best_fixed_schedule(train_three_phase: pd.DataFrame) -> dict[int, float]:
    final_rows = train_three_phase[train_three_phase["decision_index"] == 0]
    best_run_id = str(final_rows.sort_values("final_objective").iloc[0]["wandb_run_id"])
    run_frame = train_three_phase[train_three_phase["wandb_run_id"] == best_run_id].sort_values("decision_index")
    return {int(row.decision_index): float(row.action_starcoder) for row in run_frame.itertuples()}


def _policy_artifact(
    *,
    kind: PolicyKindV2,
    model_path: Path,
    feature_keys: tuple[str, ...],
    action_grid: np.ndarray,
    state_stats: StateStats,
    reward_mean: float,
    reward_std: float,
    objective_metric: str,
    metadata: dict[str, Any],
    aux_paths: dict[str, str] | None = None,
) -> PolicyArtifactV2:
    return PolicyArtifactV2(
        kind=kind,
        objective_metric=objective_metric,
        state_keys=feature_keys,
        action_low=float(action_grid.min()),
        action_high=float(action_grid.max()),
        action_values=[float(value) for value in action_grid.tolist()],
        state_mean=[float(value) for value in state_stats.mean.tolist()],
        state_std=[float(value) for value in state_stats.std.tolist()],
        reward_mean=reward_mean,
        reward_std=reward_std,
        model_path=str(model_path.resolve()),
        aux_paths=aux_paths or {},
        metadata=metadata,
    )


def _sensitivity_rows(
    policy: ActionPolicy,
    reference_frame: pd.DataFrame,
    feature_keys: tuple[str, ...],
    action_grid: np.ndarray,
    method_name: str,
) -> list[dict[str, Any]]:
    if reference_frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    medians = reference_frame.loc[:, list(feature_keys)].median(axis=0)
    decision_template = int(reference_frame["decision_index"].mode().iloc[0])
    base = {feature: float(medians[feature]) for feature in feature_keys}
    for feature_name in feature_keys:
        sweep = np.linspace(-2.0, 2.0, 17)
        for value in sweep:
            row = dict(base)
            row[feature_name] = float(medians[feature_name] + value)
            row["decision_index"] = decision_template
            frame = pd.DataFrame([row])
            action_idx = int(policy.predict_action_indices(frame)[0])
            rows.append(
                {
                    "method": method_name,
                    "decision_index": decision_template,
                    "feature_name": feature_name,
                    "feature_delta": float(value),
                    "action_idx": action_idx,
                    "action_value": float(action_grid[action_idx]),
                }
            )
    return rows


def _evaluate_policy(
    *,
    policy: ActionPolicy,
    train_frame: pd.DataFrame,
    val_three_phase: pd.DataFrame,
    feature_keys: tuple[str, ...],
    behavior_policy: BehaviorPolicyModel,
    reward_models: RewardModelBundle,
    action_grid: np.ndarray,
    config: OfflinePolicyBenchConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    fqe = fit_fqe_model(
        train_frame=train_frame,
        feature_keys=feature_keys,
        policy=policy,
        action_grid=action_grid,
        gamma=config.gamma,
        random_state=config.random_state,
        iterations=config.fqe_iterations,
    )
    direct_value = direct_method_value(val_three_phase, policy, reward_models)
    fqe_value = initial_state_value(val_three_phase, policy, fqe, action_grid)
    dr_value = doubly_robust_estimate(
        val_three_phase,
        policy,
        behavior_policy,
        fqe,
        action_grid,
        gamma=config.gamma,
    )
    wis_value = weighted_importance_sampling(val_three_phase, policy, behavior_policy, gamma=config.gamma)
    diagnostics_df, summary = policy_diagnostics(
        frame=val_three_phase,
        policy=policy,
        behavior_policy=behavior_policy,
        action_grid=tuple(float(value) for value in action_grid.tolist()),
        support_threshold=config.support_threshold,
    )
    metrics = {
        "method": policy.name,
        "direct_value": direct_value,
        "fqe_value": fqe_value,
        "dr_value": dr_value,
        "wis_value": wis_value,
        **summary,
    }
    return metrics, diagnostics_df.assign(method=policy.name)


def run_offline_policy_bench(config: OfflinePolicyBenchConfig) -> dict[str, Any]:
    """Run grouped CV offline policy selection and save full-data artifacts."""
    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    decisions = pd.read_parquet(input_dir / "decisions.parquet")
    with (input_dir / "feature_manifest.json").open() as f:
        feature_manifest = json.load(f)
    with (input_dir / "dataset_manifest.json").open() as f:
        dataset_manifest = json.load(f)

    feature_keys = tuple(feature_manifest["selected_feature_keys"])
    action_grid = np.asarray(dataset_manifest["action_grid"], dtype=np.float32)
    n_folds = int(dataset_manifest["n_cv_folds"])

    fold_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[pd.DataFrame] = []
    sensitivity_rows: list[dict[str, Any]] = []

    for fold in range(n_folds):
        train_frame = decisions[decisions["cv_fold"] != fold].copy().reset_index(drop=True)
        val_frame = decisions[decisions["cv_fold"] == fold].copy().reset_index(drop=True)
        val_three_phase = val_frame[val_frame["run_family"] == "three_phase_starcoder"].copy().reset_index(drop=True)
        if train_frame.empty or val_three_phase.empty:
            continue

        state_stats = compute_state_stats(train_frame, feature_keys)
        train_reward_std, reward_mean, reward_std = _reward_standardize(
            train_frame["reward_dense_raw"],
            train_frame["reward_dense_raw"].to_numpy(dtype=np.float32),
        )
        del train_reward_std

        behavior_policy = fit_behavior_policy(
            train_frame,
            feature_keys=feature_keys,
            action_count=len(action_grid),
            random_state=config.random_state,
        )
        reward_models = fit_reward_models(
            train_frame,
            feature_keys=feature_keys,
            random_state=config.random_state,
            final_model_weight=config.final_model_weight,
            reward_bonus_weight=config.reward_bonus_weight,
        )
        best_fixed_policy = FixedSchedulePolicy(
            action_grid,
            _best_fixed_schedule(train_frame[train_frame["run_family"] == "three_phase_starcoder"]),
        )
        outcome_planner = train_outcome_planner(
            train_frame,
            feature_keys,
            action_grid,
            behavior_policy,
            config,
        )

        fold_dir = output_dir / f"fold_{fold}"
        policies: list[ActionPolicy] = [
            best_fixed_policy,
            outcome_planner,
            train_discrete_iql_policy(
                train_frame,
                feature_keys,
                state_stats,
                reward_mean,
                reward_std,
                action_grid,
                config,
            ),
            train_discrete_cql_policy(
                train_frame,
                feature_keys,
                state_stats,
                reward_mean,
                reward_std,
                action_grid,
                config,
                fold_dir / "discrete_cql",
            ),
            train_discrete_bc_policy(
                train_frame,
                feature_keys,
                state_stats,
                reward_mean,
                reward_std,
                action_grid,
                config,
                fold_dir / "discrete_bc",
            ),
        ]
        if config.include_continuous_cql:
            policies.append(
                train_continuous_cql_policy(
                    train_frame,
                    feature_keys,
                    state_stats,
                    reward_mean,
                    reward_std,
                    action_grid,
                    config,
                    fold_dir / "continuous_cql",
                )
            )

        for policy in policies:
            metrics, diagnostics = _evaluate_policy(
                policy=policy,
                train_frame=train_frame,
                val_three_phase=val_three_phase,
                feature_keys=feature_keys,
                behavior_policy=behavior_policy,
                reward_models=reward_models,
                action_grid=action_grid,
                config=config,
            )
            metrics["fold"] = fold
            fold_rows.append(metrics)
            diagnostics_rows.append(diagnostics.assign(fold=fold))

            reference = val_three_phase[val_three_phase["decision_index"] == 0]
            sensitivity_rows.extend(_sensitivity_rows(policy, reference, feature_keys, action_grid, policy.name))

    scores_df = pd.DataFrame(fold_rows)
    if scores_df.empty:
        raise ValueError("No fold scores were produced.")
    diagnostics_df = pd.concat(diagnostics_rows, ignore_index=True) if diagnostics_rows else pd.DataFrame()
    sensitivity_df = pd.DataFrame(sensitivity_rows)

    scores_df.to_csv(output_dir / "fold_scores.csv", index=False)
    diagnostics_df.to_csv(output_dir / "policy_diagnostics.csv", index=False)
    sensitivity_df.to_csv(output_dir / "sensitivity_curves.csv", index=False)

    summary_df = (
        scores_df.groupby("method", as_index=False)
        .agg(
            direct_value_mean=("direct_value", "mean"),
            fqe_value_mean=("fqe_value", "mean"),
            dr_value_mean=("dr_value", "mean"),
            wis_value_mean=("wis_value", "mean"),
            unsupported_rate_mean=("unsupported_rate", "mean"),
            boundary_rate_mean=("boundary_rate", "mean"),
            mean_support_mean=("mean_support", "mean"),
        )
        .sort_values("dr_value_mean", ascending=False)
        .reset_index(drop=True)
    )
    summary_df.to_csv(output_dir / "policy_summary.csv", index=False)

    baseline_bc = scores_df[scores_df["method"] == "discrete_bc"].set_index("fold")
    baseline_fixed = scores_df[scores_df["method"] == "fixed_best_schedule"].set_index("fold")

    gate_rows: list[dict[str, Any]] = []
    candidate_methods = [
        method for method in summary_df["method"].tolist() if method not in {"fixed_best_schedule", "discrete_bc"}
    ]
    passing_candidates: list[str] = []
    for method in candidate_methods:
        method_scores = scores_df[scores_df["method"] == method].set_index("fold")
        common_folds = sorted(set(method_scores.index) & set(baseline_bc.index) & set(baseline_fixed.index))
        beat_folds = 0
        for fold in common_folds:
            row = method_scores.loc[fold]
            bc_row = baseline_bc.loc[fold]
            fixed_row = baseline_fixed.loc[fold]
            if (
                row["fqe_value"] > bc_row["fqe_value"]
                and row["fqe_value"] > fixed_row["fqe_value"]
                and row["dr_value"] > bc_row["dr_value"]
                and row["dr_value"] > fixed_row["dr_value"]
            ):
                beat_folds += 1
        unsupported_rate = float(method_scores["unsupported_rate"].mean())
        boundary_rate = float(method_scores["boundary_rate"].mean())
        passed = beat_folds >= 3 and unsupported_rate <= 0.15 and boundary_rate <= config.bound_rate_threshold
        gate_rows.append(
            {
                "method": method,
                "beat_baselines_folds": beat_folds,
                "unsupported_rate_mean": unsupported_rate,
                "boundary_rate_mean": boundary_rate,
                "passed": passed,
            }
        )
        if passed:
            passing_candidates.append(method)

    gate_df = pd.DataFrame(gate_rows)
    gate_df.to_csv(output_dir / "rollout_gate_report.csv", index=False)

    best_fqe = summary_df.sort_values("fqe_value_mean", ascending=False).iloc[0]["method"]
    best_dr = summary_df.sort_values("dr_value_mean", ascending=False).iloc[0]["method"]
    selected_method = None
    if best_fqe == best_dr and best_fqe in set(passing_candidates):
        selected_method = str(best_fqe)

    full_state_stats = compute_state_stats(decisions, feature_keys)
    _, full_reward_mean, full_reward_std = _reward_standardize(
        decisions["reward_dense_raw"], decisions["reward_dense_raw"].to_numpy(dtype=np.float32)
    )
    full_behavior = fit_behavior_policy(
        decisions,
        feature_keys=feature_keys,
        action_count=len(action_grid),
        random_state=config.random_state,
    )
    full_outcome_planner = train_outcome_planner(
        decisions,
        feature_keys,
        action_grid,
        full_behavior,
        config,
    )

    full_models_dir = output_dir / "full_models"
    full_models_dir.mkdir(parents=True, exist_ok=True)
    final_artifacts: dict[str, str] = {}
    decision_state_defaults_path = output_dir / "artifacts" / "decision_state_defaults.json"
    family_decision_state_defaults_path = output_dir / "artifacts" / "decision_state_defaults_by_family.json"
    decision_state_defaults_path.parent.mkdir(parents=True, exist_ok=True)
    with decision_state_defaults_path.open("w") as f:
        json.dump(_decision_state_defaults(decisions, feature_keys), f, indent=2, sort_keys=True)
    with family_decision_state_defaults_path.open("w") as f:
        json.dump(_family_decision_state_defaults(decisions, feature_keys), f, indent=2, sort_keys=True)

    outcome_bundle_path = full_models_dir / "outcome_planner" / "planner.joblib"
    outcome_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "feature_keys": feature_keys,
            "action_grid": action_grid.tolist(),
            "support_threshold": config.support_threshold,
            "q_models": full_outcome_planner.q_models,
            "behavior_policy": full_behavior,
        },
        outcome_bundle_path,
    )
    final_artifacts["outcome_planner"] = str(outcome_bundle_path)

    discrete_iql = train_discrete_iql_policy(
        decisions,
        feature_keys,
        full_state_stats,
        full_reward_mean,
        full_reward_std,
        action_grid,
        config,
    )
    final_artifacts["discrete_iql"] = str(discrete_iql.save(full_models_dir / "discrete_iql"))

    discrete_cql = train_discrete_cql_policy(
        decisions,
        feature_keys,
        full_state_stats,
        full_reward_mean,
        full_reward_std,
        action_grid,
        config,
        full_models_dir / "discrete_cql",
    )
    final_artifacts["discrete_cql"] = str(discrete_cql.save(full_models_dir / "discrete_cql"))

    discrete_bc = train_discrete_bc_policy(
        decisions,
        feature_keys,
        full_state_stats,
        full_reward_mean,
        full_reward_std,
        action_grid,
        config,
        full_models_dir / "discrete_bc",
    )
    final_artifacts["discrete_bc"] = str(discrete_bc.save(full_models_dir / "discrete_bc"))

    if config.include_continuous_cql:
        continuous_cql = train_continuous_cql_policy(
            decisions,
            feature_keys,
            full_state_stats,
            full_reward_mean,
            full_reward_std,
            action_grid,
            config,
            full_models_dir / "continuous_cql",
        )
        final_artifacts["continuous_cql_ablation"] = str(continuous_cql.save(full_models_dir / "continuous_cql"))

    artifact_map = {
        "outcome_planner": _policy_artifact(
            kind="sklearn_outcome_planner_v2",
            model_path=Path(final_artifacts["outcome_planner"]),
            feature_keys=feature_keys,
            action_grid=action_grid,
            state_stats=full_state_stats,
            reward_mean=full_reward_mean,
            reward_std=full_reward_std,
            objective_metric=config.objective_metric,
            metadata={"support_threshold": config.support_threshold},
            aux_paths={
                "decision_state_defaults": str(decision_state_defaults_path.resolve()),
                "decision_state_defaults_by_family": str(family_decision_state_defaults_path.resolve()),
            },
        ),
        "discrete_iql": _policy_artifact(
            kind="torch_discrete_iql_v2",
            model_path=Path(final_artifacts["discrete_iql"]),
            feature_keys=feature_keys,
            action_grid=action_grid,
            state_stats=full_state_stats,
            reward_mean=full_reward_mean,
            reward_std=full_reward_std,
            objective_metric=config.objective_metric,
            metadata={},
            aux_paths={
                "decision_state_defaults": str(decision_state_defaults_path.resolve()),
                "decision_state_defaults_by_family": str(family_decision_state_defaults_path.resolve()),
            },
        ),
        "discrete_cql": _policy_artifact(
            kind="d3rlpy_discrete_cql_v2",
            model_path=Path(final_artifacts["discrete_cql"]),
            feature_keys=feature_keys,
            action_grid=action_grid,
            state_stats=full_state_stats,
            reward_mean=full_reward_mean,
            reward_std=full_reward_std,
            objective_metric=config.objective_metric,
            metadata={},
            aux_paths={
                "decision_state_defaults": str(decision_state_defaults_path.resolve()),
                "decision_state_defaults_by_family": str(family_decision_state_defaults_path.resolve()),
            },
        ),
        "discrete_bc": _policy_artifact(
            kind="d3rlpy_discrete_bc_v2",
            model_path=Path(final_artifacts["discrete_bc"]),
            feature_keys=feature_keys,
            action_grid=action_grid,
            state_stats=full_state_stats,
            reward_mean=full_reward_mean,
            reward_std=full_reward_std,
            objective_metric=config.objective_metric,
            metadata={},
            aux_paths={
                "decision_state_defaults": str(decision_state_defaults_path.resolve()),
                "decision_state_defaults_by_family": str(family_decision_state_defaults_path.resolve()),
            },
        ),
    }
    if config.include_continuous_cql and "continuous_cql_ablation" in final_artifacts:
        artifact_map["continuous_cql_ablation"] = _policy_artifact(
            kind="d3rlpy_cql_continuous_v2",
            model_path=Path(final_artifacts["continuous_cql_ablation"]),
            feature_keys=feature_keys,
            action_grid=action_grid,
            state_stats=full_state_stats,
            reward_mean=full_reward_mean,
            reward_std=full_reward_std,
            objective_metric=config.objective_metric,
            metadata={},
            aux_paths={
                "decision_state_defaults": str(decision_state_defaults_path.resolve()),
                "decision_state_defaults_by_family": str(family_decision_state_defaults_path.resolve()),
            },
        )

    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for method, artifact in artifact_map.items():
        save_policy_artifact(artifacts_dir / f"{method}.json", artifact)

    if selected_method is not None:
        save_policy_artifact(output_dir / "selected_policy_artifact.json", artifact_map[selected_method])

    summary = {
        "selected_method": selected_method,
        "best_fqe_method": str(best_fqe),
        "best_dr_method": str(best_dr),
        "passing_candidates": passing_candidates,
        "feature_keys": list(feature_keys),
        "final_artifacts": final_artifacts,
    }
    with (output_dir / "offline_policy_report.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare pooled offline-control StarCoder policies.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="experiments/domain_phase_mix/offline_rl/artifacts/starcoder_pooled_v2",
        help="Directory containing decisions.parquet and feature_manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/domain_phase_mix/offline_rl/artifacts/starcoder_pooled_v2/policy_bench",
        help="Directory to write benchmark outputs.",
    )
    parser.add_argument("--objective-metric", type=str, default=DEFAULT_OBJECTIVE_METRIC)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-continuous-cql", action="store_true", help="Disable the continuous CQL ablation.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    run_offline_policy_bench(
        OfflinePolicyBenchConfig(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            objective_metric=args.objective_metric,
            include_continuous_cql=not args.no_continuous_cql,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
