# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline policy evaluation utilities for the pooled StarCoder v2 benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


class ActionPolicy(Protocol):
    """Minimal policy interface used by the offline benchmark."""

    name: str

    def predict_action_indices(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict one discrete action index for each state row."""


@dataclass(frozen=True)
class DecisionBatch:
    """Aligned decision rows plus optional dense sequence features."""

    frame: pd.DataFrame
    sequences: np.ndarray | None = None
    masks: np.ndarray | None = None

    def __post_init__(self) -> None:
        normalized = self.frame.reset_index(drop=True)
        object.__setattr__(self, "frame", normalized)
        if self.sequences is not None and len(self.sequences) != len(normalized):
            raise ValueError("sequences must align with frame rows")
        if self.masks is not None and len(self.masks) != len(normalized):
            raise ValueError("masks must align with frame rows")

    def take(self, positions: np.ndarray | list[int]) -> DecisionBatch:
        index = np.asarray(positions, dtype=np.int64)
        return DecisionBatch(
            frame=self.frame.iloc[index].reset_index(drop=True),
            sequences=None if self.sequences is None else self.sequences[index],
            masks=None if self.masks is None else self.masks[index],
        )


class BatchActionPolicy(Protocol):
    """Policy interface for sequence-aware decision batches."""

    name: str

    def predict_action_indices_batch(self, batch: DecisionBatch) -> np.ndarray:
        """Predict one discrete action index for each row in a decision batch."""


@dataclass(frozen=True)
class BehaviorPolicyModel:
    """Behavior policy classifier wrapper with fixed action dimensionality."""

    estimator: HistGradientBoostingClassifier | DummyClassifier
    feature_keys: tuple[str, ...]
    action_count: int

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        inputs = frame.loc[:, list(self.feature_keys)].to_numpy(dtype=np.float32)
        raw = self.estimator.predict_proba(inputs)
        if isinstance(raw, list):
            raw = raw[0]
        result = np.zeros((len(frame), self.action_count), dtype=np.float32)
        classes = np.asarray(getattr(self.estimator, "classes_", np.arange(raw.shape[1])), dtype=int)
        result[:, classes] = np.asarray(raw, dtype=np.float32)
        row_sums = result.sum(axis=1, keepdims=True)
        safe_row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)
        return result / safe_row_sums


@dataclass(frozen=True)
class RewardModelBundle:
    """Stage-conditioned reward/final-objective models for direct estimation."""

    feature_keys: tuple[str, ...]
    action_grid: tuple[float, ...]
    dense_reward_models: dict[int, HistGradientBoostingRegressor]
    final_objective_models: dict[int, HistGradientBoostingRegressor]
    final_model_weight: float
    reward_bonus_weight: float

    def _inputs(self, frame: pd.DataFrame, action_values: np.ndarray) -> np.ndarray:
        base = frame.loc[:, list(self.feature_keys)].to_numpy(dtype=np.float32)
        return np.concatenate([base, action_values.reshape(-1, 1).astype(np.float32)], axis=1)

    def predict_dense_reward(self, frame: pd.DataFrame, action_values: np.ndarray) -> np.ndarray:
        outputs = np.zeros(len(frame), dtype=np.float32)
        for decision_index, stage_frame in frame.groupby("decision_index", sort=False):
            model = self.dense_reward_models[int(decision_index)]
            indexer = stage_frame.index.to_numpy()
            outputs[indexer] = model.predict(self._inputs(stage_frame, action_values[indexer])).astype(np.float32)
        return outputs

    def predict_final_objective(self, frame: pd.DataFrame, action_values: np.ndarray) -> np.ndarray:
        outputs = np.zeros(len(frame), dtype=np.float32)
        for decision_index, stage_frame in frame.groupby("decision_index", sort=False):
            model = self.final_objective_models.get(int(decision_index))
            indexer = stage_frame.index.to_numpy()
            if model is None:
                outputs[indexer] = math.nan
                continue
            outputs[indexer] = model.predict(self._inputs(stage_frame, action_values[indexer])).astype(np.float32)
        return outputs

    def blended_stage_scores(self, frame: pd.DataFrame, action_values: np.ndarray) -> np.ndarray:
        final_pred = self.predict_final_objective(frame, action_values)
        dense_pred = self.predict_dense_reward(frame, action_values)
        score = np.where(np.isnan(final_pred), 0.0, self.final_model_weight * final_pred)
        score = score - self.reward_bonus_weight * dense_pred
        score = np.where(np.isnan(final_pred), -self.reward_bonus_weight * dense_pred, score)
        return score.astype(np.float32)


@dataclass(frozen=True)
class FittedQEvaluator:
    """Fitted-Q evaluation model for one deterministic policy."""

    model: HistGradientBoostingRegressor
    feature_keys: tuple[str, ...]

    def predict_q(self, frame: pd.DataFrame, action_values: np.ndarray) -> np.ndarray:
        inputs = np.concatenate(
            [
                frame.loc[:, list(self.feature_keys)].to_numpy(dtype=np.float32),
                action_values.reshape(-1, 1).astype(np.float32),
            ],
            axis=1,
        )
        return self.model.predict(inputs).astype(np.float32)


def predict_action_indices_for_batch(
    policy: ActionPolicy | BatchActionPolicy,
    batch: DecisionBatch,
) -> np.ndarray:
    """Predict actions for either summary-only or sequence-aware policies."""
    if hasattr(policy, "predict_action_indices_batch"):
        return np.asarray(policy.predict_action_indices_batch(batch), dtype=np.int64).reshape(-1)
    return np.asarray(policy.predict_action_indices(batch.frame), dtype=np.int64).reshape(-1)


def fit_behavior_policy(
    frame: pd.DataFrame,
    feature_keys: tuple[str, ...],
    action_count: int,
    random_state: int,
) -> BehaviorPolicyModel:
    """Fit a behavior policy over discrete action bins."""
    inputs = frame.loc[:, list(feature_keys)].to_numpy(dtype=np.float32)
    targets = frame["action_idx"].to_numpy(dtype=np.int64)
    if len(np.unique(targets)) < 2:
        estimator: HistGradientBoostingClassifier | DummyClassifier = DummyClassifier(strategy="prior")
    else:
        estimator = HistGradientBoostingClassifier(random_state=random_state, max_depth=4, max_iter=64)
    estimator.fit(inputs, targets)
    return BehaviorPolicyModel(estimator=estimator, feature_keys=feature_keys, action_count=action_count)


def fit_reward_models(
    frame: pd.DataFrame,
    feature_keys: tuple[str, ...],
    random_state: int,
    final_model_weight: float,
    reward_bonus_weight: float,
) -> RewardModelBundle:
    """Fit stage-conditioned direct models for dense reward and final objective."""
    dense_models: dict[int, HistGradientBoostingRegressor] = {}
    final_models: dict[int, HistGradientBoostingRegressor] = {}
    for decision_index, stage_frame in frame.groupby("decision_index"):
        inputs = np.concatenate(
            [
                stage_frame.loc[:, list(feature_keys)].to_numpy(dtype=np.float32),
                stage_frame.loc[:, ["action_starcoder"]].to_numpy(dtype=np.float32),
            ],
            axis=1,
        )
        dense_model = HistGradientBoostingRegressor(random_state=random_state, max_depth=4, max_iter=64)
        dense_model.fit(inputs, stage_frame["reward_dense_raw"].to_numpy(dtype=np.float32))
        dense_models[int(decision_index)] = dense_model

        stage_three_phase = stage_frame[stage_frame["run_family"] == "three_phase_starcoder"]
        if len(stage_three_phase) > 8:
            final_inputs = np.concatenate(
                [
                    stage_three_phase.loc[:, list(feature_keys)].to_numpy(dtype=np.float32),
                    stage_three_phase.loc[:, ["action_starcoder"]].to_numpy(dtype=np.float32),
                ],
                axis=1,
            )
            final_model = HistGradientBoostingRegressor(random_state=random_state, max_depth=4, max_iter=64)
            final_model.fit(final_inputs, stage_three_phase["final_objective"].to_numpy(dtype=np.float32))
            final_models[int(decision_index)] = final_model

    return RewardModelBundle(
        feature_keys=feature_keys,
        action_grid=tuple(sorted(frame["action_grid_value"].unique().tolist())),
        dense_reward_models=dense_models,
        final_objective_models=final_models,
        final_model_weight=final_model_weight,
        reward_bonus_weight=reward_bonus_weight,
    )


def fit_fqe_model(
    train_frame: pd.DataFrame,
    feature_keys: tuple[str, ...],
    policy: ActionPolicy,
    action_grid: np.ndarray,
    gamma: float,
    random_state: int,
    iterations: int = 8,
) -> FittedQEvaluator:
    """Fit a generic fitted-Q evaluator for a deterministic discrete policy."""
    working = train_frame.sort_values(["episode_id", "step_in_episode"]).reset_index(drop=True).copy()
    model = HistGradientBoostingRegressor(random_state=random_state, max_depth=4, max_iter=64)
    targets = working["reward_dense_raw"].to_numpy(dtype=np.float32)

    for _ in range(iterations):
        inputs = np.concatenate(
            [
                working.loc[:, list(feature_keys)].to_numpy(dtype=np.float32),
                working.loc[:, ["action_starcoder"]].to_numpy(dtype=np.float32),
            ],
            axis=1,
        )
        model.fit(inputs, targets)

        bootstrap = np.zeros(len(working), dtype=np.float32)
        next_frame = working.groupby("episode_id", sort=False).shift(-1)
        has_next = ~working["done"].to_numpy(dtype=bool)
        if has_next.any():
            next_rows = working.loc[has_next, :].copy()
            next_rows.loc[:, list(feature_keys)] = next_frame.loc[has_next, list(feature_keys)].to_numpy(
                dtype=np.float32
            )
            next_actions_idx = policy.predict_action_indices(next_rows)
            value_map = {int(idx): float(value) for idx, value in enumerate(action_grid.tolist())}
            next_action_values_array = np.asarray([value_map[int(idx)] for idx in next_actions_idx], dtype=np.float32)
            bootstrap[has_next] = model.predict(
                np.concatenate(
                    [
                        next_rows.loc[:, list(feature_keys)].to_numpy(dtype=np.float32),
                        next_action_values_array.reshape(-1, 1),
                    ],
                    axis=1,
                )
            ).astype(np.float32)
        targets = working["reward_dense_raw"].to_numpy(dtype=np.float32) + gamma * bootstrap

    return FittedQEvaluator(model=model, feature_keys=feature_keys)


def initial_state_value(
    frame: pd.DataFrame,
    policy: ActionPolicy,
    fqe: FittedQEvaluator,
    action_grid: np.ndarray,
) -> float:
    """Estimate policy value from held-out initial states using fitted Q."""
    initial = frame[frame["decision_index"] == 0].copy().reset_index(drop=True)
    if initial.empty:
        return math.nan
    action_idx = policy.predict_action_indices(initial)
    value_map = {int(idx): float(value) for idx, value in enumerate(action_grid.tolist())}
    action_values = np.asarray([value_map[int(idx)] for idx in action_idx], dtype=np.float32)
    return float(np.mean(fqe.predict_q(initial, action_values)))


def doubly_robust_estimate(
    frame: pd.DataFrame,
    policy: ActionPolicy,
    behavior_policy: BehaviorPolicyModel,
    fqe: FittedQEvaluator,
    action_grid: np.ndarray,
    gamma: float,
) -> float:
    """Compute recursive doubly robust estimate over held-out episodes."""
    if frame.empty:
        return math.nan

    value_map = {int(idx): float(value) for idx, value in enumerate(action_grid.tolist())}
    episode_returns: list[float] = []
    for _, episode in frame.groupby("episode_id", sort=False):
        episode = episode.sort_values("step_in_episode").reset_index(drop=True)
        probs = behavior_policy.predict_proba(episode)
        target_actions = policy.predict_action_indices(episode)
        next_dr = 0.0
        for position in reversed(range(len(episode))):
            row = episode.iloc[[position]].copy()
            target_idx = int(target_actions[position])
            target_value = np.asarray([value_map[target_idx]], dtype=np.float32)
            v_hat = float(fqe.predict_q(row, target_value)[0])

            logged_idx = int(episode.iloc[position]["action_idx"])
            logged_value = np.asarray([float(episode.iloc[position]["action_starcoder"])], dtype=np.float32)
            q_logged = float(fqe.predict_q(row, logged_value)[0])
            behavior_prob = float(max(probs[position, logged_idx], 1e-6))
            indicator = 1.0 if logged_idx == target_idx else 0.0
            rho = indicator / behavior_prob
            reward = float(episode.iloc[position]["reward_dense_raw"])
            dr = v_hat + rho * (reward + gamma * next_dr - q_logged)
            next_dr = dr
        episode_returns.append(next_dr)
    return float(np.mean(episode_returns))


def weighted_importance_sampling(
    frame: pd.DataFrame,
    policy: ActionPolicy,
    behavior_policy: BehaviorPolicyModel,
    gamma: float,
) -> float:
    """Compute self-normalized per-decision importance sampling as a sanity check."""
    numerators: list[float] = []
    denominators: list[float] = []
    for _, episode in frame.groupby("episode_id", sort=False):
        episode = episode.sort_values("step_in_episode").reset_index(drop=True)
        probs = behavior_policy.predict_proba(episode)
        target_actions = policy.predict_action_indices(episode)
        weight = 1.0
        total = 0.0
        for position in range(len(episode)):
            logged_idx = int(episode.iloc[position]["action_idx"])
            target_idx = int(target_actions[position])
            if logged_idx != target_idx:
                weight = 0.0
                break
            weight /= float(max(probs[position, logged_idx], 1e-6))
            total += (gamma**position) * float(episode.iloc[position]["reward_dense_raw"])
        numerators.append(weight * total)
        denominators.append(weight)
    denom = float(np.sum(denominators))
    if denom <= 0.0:
        return math.nan
    return float(np.sum(numerators) / denom)


def direct_method_value(frame: pd.DataFrame, policy: ActionPolicy, reward_models: RewardModelBundle) -> float:
    """Estimate value with the direct stage-0 final-objective model on held-out initial states."""
    initial = frame[frame["decision_index"] == 0].copy().reset_index(drop=True)
    if initial.empty or 0 not in reward_models.final_objective_models:
        return math.nan
    target_idx = policy.predict_action_indices(initial)
    action_values = np.asarray([reward_models.action_grid[int(idx)] for idx in target_idx], dtype=np.float32)
    predicted_final = reward_models.predict_final_objective(initial, action_values)
    return float(np.mean(-predicted_final))


def direct_method_value_batch(
    batch: DecisionBatch,
    policy: ActionPolicy | BatchActionPolicy,
    reward_models: RewardModelBundle,
) -> float:
    """Batch-aware direct estimate using held-out initial states."""
    initial_positions = np.flatnonzero(batch.frame["decision_index"].to_numpy(dtype=np.int64) == 0)
    if len(initial_positions) == 0 or 0 not in reward_models.final_objective_models:
        return math.nan
    initial_batch = batch.take(initial_positions)
    target_idx = predict_action_indices_for_batch(policy, initial_batch)
    action_values = np.asarray([reward_models.action_grid[int(idx)] for idx in target_idx], dtype=np.float32)
    predicted_final = reward_models.predict_final_objective(initial_batch.frame, action_values)
    return float(np.mean(-predicted_final))


def policy_diagnostics(
    frame: pd.DataFrame,
    policy: ActionPolicy,
    behavior_policy: BehaviorPolicyModel,
    action_grid: tuple[float, ...],
    support_threshold: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compute per-stage action/support diagnostics on held-out states."""
    predicted_idx = policy.predict_action_indices(frame)
    behavior_probs = behavior_policy.predict_proba(frame)
    chosen_support = np.asarray([behavior_probs[i, int(idx)] for i, idx in enumerate(predicted_idx)], dtype=np.float32)
    predicted_values = np.asarray([action_grid[int(idx)] for idx in predicted_idx], dtype=np.float32)

    rows: list[dict[str, float | int]] = []
    for decision_index, stage in frame.groupby("decision_index", sort=True):
        stage_indexer = stage.index.to_numpy(dtype=int)
        stage_actions = predicted_idx[stage_indexer]
        stage_values = predicted_values[stage_indexer]
        stage_support = chosen_support[stage_indexer]
        counts = np.bincount(stage_actions, minlength=len(action_grid)).astype(np.float32)
        probs = counts / max(float(counts.sum()), 1.0)
        entropy = float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum())
        rows.append(
            {
                "decision_index": int(decision_index),
                "count": len(stage),
                "action_entropy": entropy,
                "mean_support": float(stage_support.mean()),
                "unsupported_rate": float((stage_support < support_threshold).mean()),
                "boundary_rate": float(((stage_actions == 0) | (stage_actions == len(action_grid) - 1)).mean()),
                "mean_action_value": float(stage_values.mean()),
            }
        )
    summary = {
        "unsupported_rate": float((chosen_support < support_threshold).mean()),
        "boundary_rate": float(((predicted_idx == 0) | (predicted_idx == len(action_grid) - 1)).mean()),
        "mean_support": float(chosen_support.mean()),
    }
    return pd.DataFrame(rows), summary


def fit_fqe_model_batch(
    train_batch: DecisionBatch,
    feature_keys: tuple[str, ...],
    policy: ActionPolicy | BatchActionPolicy,
    action_grid: np.ndarray,
    gamma: float,
    random_state: int,
    iterations: int = 8,
) -> FittedQEvaluator:
    """Fit FQE while allowing sequence-aware policies on next states."""
    order = train_batch.frame.sort_values(["episode_id", "step_in_episode"]).index.to_numpy(dtype=np.int64)
    working = train_batch.frame.iloc[order].reset_index(drop=True).copy()
    sequences = None if train_batch.sequences is None else train_batch.sequences[order]
    masks = None if train_batch.masks is None else train_batch.masks[order]

    model = HistGradientBoostingRegressor(random_state=random_state, max_depth=4, max_iter=64)
    targets = working["reward_dense_raw"].to_numpy(dtype=np.float32)

    for _ in range(iterations):
        inputs = np.concatenate(
            [
                working.loc[:, list(feature_keys)].to_numpy(dtype=np.float32),
                working.loc[:, ["action_starcoder"]].to_numpy(dtype=np.float32),
            ],
            axis=1,
        )
        model.fit(inputs, targets)

        bootstrap = np.zeros(len(working), dtype=np.float32)
        next_frame = working.groupby("episode_id", sort=False)[list(feature_keys)].shift(-1)
        has_next = ~working["done"].to_numpy(dtype=bool)
        if has_next.any():
            next_rows = working.loc[has_next, :].copy()
            next_rows.loc[:, list(feature_keys)] = next_frame.loc[has_next, list(feature_keys)].to_numpy(
                dtype=np.float32
            )
            next_batch = DecisionBatch(
                frame=next_rows.reset_index(drop=True),
                sequences=None if sequences is None else sequences[has_next],
                masks=None if masks is None else masks[has_next],
            )
            next_actions_idx = predict_action_indices_for_batch(policy, next_batch)
            next_action_values = np.asarray([float(action_grid[int(idx)]) for idx in next_actions_idx], dtype=np.float32)
            bootstrap[has_next] = model.predict(
                np.concatenate(
                    [
                        next_rows.loc[:, list(feature_keys)].to_numpy(dtype=np.float32),
                        next_action_values.reshape(-1, 1),
                    ],
                    axis=1,
                )
            ).astype(np.float32)
        targets = working["reward_dense_raw"].to_numpy(dtype=np.float32) + gamma * bootstrap

    return FittedQEvaluator(model=model, feature_keys=feature_keys)


def initial_state_value_batch(
    batch: DecisionBatch,
    policy: ActionPolicy | BatchActionPolicy,
    fqe: FittedQEvaluator,
    action_grid: np.ndarray,
) -> float:
    """Estimate held-out initial-state value for sequence-aware policies."""
    initial_positions = np.flatnonzero(batch.frame["decision_index"].to_numpy(dtype=np.int64) == 0)
    if len(initial_positions) == 0:
        return math.nan
    initial_batch = batch.take(initial_positions)
    action_idx = predict_action_indices_for_batch(policy, initial_batch)
    action_values = np.asarray([float(action_grid[int(idx)]) for idx in action_idx], dtype=np.float32)
    return float(np.mean(fqe.predict_q(initial_batch.frame, action_values)))


def doubly_robust_estimate_batch(
    batch: DecisionBatch,
    policy: ActionPolicy | BatchActionPolicy,
    behavior_policy: BehaviorPolicyModel,
    fqe: FittedQEvaluator,
    action_grid: np.ndarray,
    gamma: float,
) -> float:
    """Compute recursive doubly robust estimates for sequence-aware policies."""
    if batch.frame.empty:
        return math.nan

    value_map = {int(idx): float(value) for idx, value in enumerate(action_grid.tolist())}
    episode_returns: list[float] = []
    for _, positions in batch.frame.groupby("episode_id", sort=False).indices.items():
        episode_batch = batch.take(np.asarray(sorted(positions), dtype=np.int64))
        episode = episode_batch.frame
        probs = behavior_policy.predict_proba(episode)
        target_actions = predict_action_indices_for_batch(policy, episode_batch)
        next_dr = 0.0
        for position in reversed(range(len(episode))):
            row_batch = episode_batch.take([position])
            row = row_batch.frame
            target_idx = int(target_actions[position])
            target_value = np.asarray([value_map[target_idx]], dtype=np.float32)
            v_hat = float(fqe.predict_q(row, target_value)[0])
            logged_idx = int(episode.iloc[position]["action_idx"])
            logged_value = np.asarray([float(episode.iloc[position]["action_starcoder"])], dtype=np.float32)
            q_logged = float(fqe.predict_q(row, logged_value)[0])
            behavior_prob = float(max(probs[position, logged_idx], 1e-6))
            indicator = 1.0 if logged_idx == target_idx else 0.0
            rho = indicator / behavior_prob
            reward = float(episode.iloc[position]["reward_dense_raw"])
            dr = v_hat + rho * (reward + gamma * next_dr - q_logged)
            next_dr = dr
        episode_returns.append(next_dr)
    return float(np.mean(episode_returns))


def weighted_importance_sampling_batch(
    batch: DecisionBatch,
    policy: ActionPolicy | BatchActionPolicy,
    behavior_policy: BehaviorPolicyModel,
    gamma: float,
) -> float:
    """Compute self-normalized WIS for sequence-aware policies."""
    numerators: list[float] = []
    denominators: list[float] = []
    for _, positions in batch.frame.groupby("episode_id", sort=False).indices.items():
        episode_batch = batch.take(np.asarray(sorted(positions), dtype=np.int64))
        episode = episode_batch.frame
        probs = behavior_policy.predict_proba(episode)
        target_actions = predict_action_indices_for_batch(policy, episode_batch)
        weight = 1.0
        total = 0.0
        for position in range(len(episode)):
            logged_idx = int(episode.iloc[position]["action_idx"])
            target_idx = int(target_actions[position])
            if logged_idx != target_idx:
                weight = 0.0
                break
            weight /= float(max(probs[position, logged_idx], 1e-6))
            total += (gamma**position) * float(episode.iloc[position]["reward_dense_raw"])
        numerators.append(weight * total)
        denominators.append(weight)
    denom = float(np.sum(denominators))
    if denom <= 0.0:
        return math.nan
    return float(np.sum(numerators) / denom)


def policy_diagnostics_batch(
    batch: DecisionBatch,
    policy: ActionPolicy | BatchActionPolicy,
    behavior_policy: BehaviorPolicyModel,
    action_grid: tuple[float, ...],
    support_threshold: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compute diagnostics for sequence-aware policies."""
    frame = batch.frame.reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame(), {
            "unsupported_rate": math.nan,
            "boundary_rate": math.nan,
            "mean_support": math.nan,
            "action_entropy": math.nan,
        }
    predicted_idx = predict_action_indices_for_batch(policy, batch)
    behavior_probs = behavior_policy.predict_proba(frame)
    chosen_support = np.asarray([behavior_probs[i, int(idx)] for i, idx in enumerate(predicted_idx)], dtype=np.float32)
    predicted_values = np.asarray([action_grid[int(idx)] for idx in predicted_idx], dtype=np.float32)

    rows: list[dict[str, float | int]] = []
    for decision_index, stage in frame.groupby("decision_index", sort=True):
        stage_indexer = stage.index.to_numpy(dtype=int)
        stage_actions = predicted_idx[stage_indexer]
        stage_values = predicted_values[stage_indexer]
        stage_support = chosen_support[stage_indexer]
        counts = np.bincount(stage_actions, minlength=len(action_grid)).astype(np.float32)
        probs = counts / max(float(counts.sum()), 1.0)
        entropy = float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum())
        rows.append(
            {
                "decision_index": int(decision_index),
                "count": len(stage),
                "action_entropy": entropy,
                "mean_support": float(stage_support.mean()),
                "unsupported_rate": float((stage_support < support_threshold).mean()),
                "boundary_rate": float(((stage_actions == 0) | (stage_actions == len(action_grid) - 1)).mean()),
                "mean_action_value": float(stage_values.mean()),
            }
        )
    counts = np.bincount(predicted_idx, minlength=len(action_grid)).astype(np.float32)
    probs = counts / max(float(counts.sum()), 1.0)
    summary = {
        "unsupported_rate": float((chosen_support < support_threshold).mean()),
        "boundary_rate": float(((predicted_idx == 0) | (predicted_idx == len(action_grid) - 1)).mean()),
        "mean_support": float(chosen_support.mean()),
        "action_entropy": float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum()),
    }
    return pd.DataFrame(rows), summary
