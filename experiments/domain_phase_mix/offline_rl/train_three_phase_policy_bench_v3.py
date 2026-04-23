# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train and compare three-phase offline-control v3 policies."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from joblib import dump
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from experiments.domain_phase_mix.offline_rl.build_three_phase_dense_policy_dataset import (
    SEQUENCE_CHANNELS,
)
from experiments.domain_phase_mix.offline_rl.contracts import DEFAULT_OBJECTIVE_METRIC, PolicyKindV2
from experiments.domain_phase_mix.offline_rl.ope import (
    ActionPolicy,
    BatchActionPolicy,
    BehaviorPolicyModel,
    DecisionBatch,
    RewardModelBundle,
    direct_method_value_batch,
    doubly_robust_estimate_batch,
    fit_behavior_policy,
    fit_fqe_model_batch,
    fit_reward_models,
    initial_state_value_batch,
    policy_diagnostics_batch,
    predict_action_indices_for_batch,
    weighted_importance_sampling_batch,
)
from experiments.domain_phase_mix.offline_rl.policy_artifact import PolicyArtifactV2, save_policy_artifact

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThreePhasePolicyBenchV3Config:
    """Configuration for the three-phase offline-control v3 benchmark."""

    input_dir: str
    output_dir: str
    objective_metric: str = DEFAULT_OBJECTIVE_METRIC
    random_state: int = 42
    gamma: float = 1.0
    support_lambda: float = 0.05
    support_floor: float = 1e-4
    diagnostic_support_threshold: float = 0.02
    boundary_rate_threshold: float = 0.25
    unsupported_rate_threshold: float = 0.15
    fqe_iterations: int = 8
    q_max_iter: int = 96
    discrete_bc_max_iter: int = 96
    pretrain_epochs: int = 40
    finetune_epochs: int = 120
    target_update_period: int = 10
    batch_size: int = 64
    hidden_size: int = 64
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_ffn_dim: int = 128
    dropout: float = 0.1
    learning_rate: float = 1e-3
    device: str = "auto"


@dataclass(frozen=True)
class StateStats:
    """Normalization statistics for summary-state features."""

    mean: np.ndarray
    std: np.ndarray

    def normalize(self, frame: pd.DataFrame, feature_keys: tuple[str, ...]) -> np.ndarray:
        raw = frame.loc[:, list(feature_keys)].to_numpy(dtype=np.float32)
        safe_std = np.where(self.std <= 0.0, 1.0, self.std)
        return (raw - self.mean) / safe_std


class FixedSchedulePolicy(ActionPolicy):
    """Static baseline using the best historical schedule in the train fold."""

    def __init__(self, action_grid: np.ndarray, schedule_by_decision: dict[int, float]):
        self.name = "fixed_best_schedule"
        self._index_by_decision: dict[int, int] = {}
        for decision_index, value in schedule_by_decision.items():
            idx = int(np.argmin(np.abs(action_grid - float(value))))
            self._index_by_decision[int(decision_index)] = idx

    def predict_action_indices(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            [self._index_by_decision[int(decision_index)] for decision_index in frame["decision_index"].tolist()],
            dtype=np.int64,
        )


class SklearnDiscreteBCPolicy(ActionPolicy):
    """Summary-only discrete BC baseline fit with gradient-boosted classification."""

    def __init__(
        self,
        *,
        estimator: HistGradientBoostingClassifier | DummyClassifier,
        feature_keys: tuple[str, ...],
    ):
        self.name = "discrete_bc"
        self.estimator = estimator
        self.feature_keys = feature_keys

    def predict_action_indices(self, frame: pd.DataFrame) -> np.ndarray:
        inputs = frame.loc[:, list(self.feature_keys)].to_numpy(dtype=np.float32)
        return np.asarray(self.estimator.predict(inputs), dtype=np.int64)


class DynamicQPlannerV3(BatchActionPolicy):
    """Finite-horizon dynamic-Q planner with soft support regularization."""

    def __init__(
        self,
        *,
        feature_keys: tuple[str, ...],
        action_grid: np.ndarray,
        q_models: dict[int, HistGradientBoostingRegressor],
        behavior_policy: BehaviorPolicyModel,
        support_lambda: float,
        support_floor: float,
    ):
        self.name = "dynamic_q_planner_v3"
        self.feature_keys = feature_keys
        self.action_grid = np.asarray(action_grid, dtype=np.float32)
        self.q_models = q_models
        self.behavior_policy = behavior_policy
        self.support_lambda = support_lambda
        self.support_floor = support_floor

    def predict_action_indices(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_action_indices_batch(DecisionBatch(frame=frame))

    def predict_action_indices_batch(self, batch: DecisionBatch) -> np.ndarray:
        outputs, _ = self.best_actions(batch)
        return outputs

    def best_actions(self, batch: DecisionBatch) -> tuple[np.ndarray, np.ndarray]:
        frame = batch.frame.reset_index(drop=True)
        outputs = np.zeros(len(frame), dtype=np.int64)
        values = np.zeros(len(frame), dtype=np.float32)
        for decision_index, stage_frame in frame.groupby("decision_index", sort=False):
            stage_positions = stage_frame.index.to_numpy(dtype=np.int64)
            if len(stage_positions) == 0:
                continue
            repeated_inputs = np.repeat(
                stage_frame.loc[:, list(self.feature_keys)].to_numpy(dtype=np.float32),
                len(self.action_grid),
                axis=0,
            )
            repeated = pd.DataFrame(repeated_inputs, columns=list(self.feature_keys))
            candidate_indices = np.tile(np.arange(len(self.action_grid), dtype=np.int64), len(stage_frame))
            candidate_actions = np.tile(self.action_grid, len(stage_frame)).astype(np.float32)
            support_matrix = self.behavior_policy.predict_proba(repeated)
            support = support_matrix[np.arange(len(repeated)), candidate_indices].reshape(
                len(stage_frame), len(self.action_grid)
            )
            model = self.q_models[int(decision_index)]
            q_values = (
                model.predict(
                    np.concatenate(
                        [
                            repeated.loc[:, list(self.feature_keys)].to_numpy(dtype=np.float32),
                            candidate_actions.reshape(-1, 1),
                        ],
                        axis=1,
                    )
                )
                .astype(np.float32)
                .reshape(len(stage_frame), len(self.action_grid))
            )
            score = q_values + self.support_lambda * np.log(np.maximum(support, self.support_floor))
            score = np.where(support >= self.support_floor, score, -np.inf)
            for local_index, global_index in enumerate(stage_positions):
                if not np.isfinite(score[local_index]).any():
                    best_local = int(np.argmax(support[local_index]))
                else:
                    best_local = int(np.argmax(score[local_index]))
                outputs[global_index] = best_local
                values[global_index] = q_values[local_index, best_local]
        return outputs, values


class _SequenceQNetwork(torch.nn.Module):
    def __init__(
        self,
        *,
        summary_dim: int,
        sequence_dim: int,
        action_dim: int,
        hidden_size: int,
    ):
        super().__init__()
        self.summary_tower = torch.nn.Sequential(
            torch.nn.Linear(summary_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
        )
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, hidden_size),
            torch.nn.ReLU(),
        )
        self.q_head = torch.nn.Linear(hidden_size, action_dim)
        self.bc_head = torch.nn.Linear(hidden_size, action_dim)
        self.next_bpb_head = torch.nn.Linear(hidden_size, 1)
        self.final_bpb_head = torch.nn.Linear(hidden_size, 1)
        self.pretrain_train_head = torch.nn.Linear(hidden_size, 1)
        self.pretrain_eval_head = torch.nn.Linear(hidden_size, 1)
        self.sequence_dim = sequence_dim
        self.hidden_size = hidden_size

    def encode_sequence(self, sequences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def fused_hidden(self, summary: torch.Tensor, sequences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        seq_hidden = self.encode_sequence(sequences, masks)
        summary_hidden = self.summary_tower(summary)
        return self.fusion(torch.cat([summary_hidden, seq_hidden], dim=1))

    def forward(
        self, summary: torch.Tensor, sequences: torch.Tensor, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.fused_hidden(summary, sequences, masks)
        return (
            self.q_head(hidden),
            self.bc_head(hidden),
            self.next_bpb_head(hidden).squeeze(-1),
            self.final_bpb_head(hidden).squeeze(-1),
        )

    def forward_pretrain(self, sequences: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode_sequence(sequences, masks)
        return (
            self.pretrain_train_head(hidden).squeeze(-1),
            self.pretrain_eval_head(hidden).squeeze(-1),
        )


class GRUQNetwork(_SequenceQNetwork):
    def __init__(self, *, summary_dim: int, sequence_dim: int, action_dim: int, hidden_size: int):
        super().__init__(
            summary_dim=summary_dim,
            sequence_dim=sequence_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
        )
        self.input_proj = torch.nn.Linear(sequence_dim, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, batch_first=True)

    def encode_sequence(self, sequences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        projected = self.input_proj(sequences)
        outputs, _ = self.gru(projected)
        lengths = masks.sum(dim=1).long().clamp(min=1)
        gather_index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, outputs.size(-1))
        gathered = outputs.gather(1, gather_index).squeeze(1)
        zero_mask = masks.sum(dim=1) <= 0
        if zero_mask.any():
            gathered = gathered.clone()
            gathered[zero_mask] = 0.0
        return gathered


class TransformerQNetwork(_SequenceQNetwork):
    def __init__(
        self,
        *,
        summary_dim: int,
        sequence_dim: int,
        action_dim: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        max_length: int,
    ):
        super().__init__(
            summary_dim=summary_dim,
            sequence_dim=sequence_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
        )
        self.input_proj = torch.nn.Linear(sequence_dim, hidden_size)
        self.pos_embedding = torch.nn.Parameter(torch.zeros(1, max_length, hidden_size))
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        # MPS does not support the nested-tensor mask fast path used by the default encoder.
        # Disable it explicitly so the same model can train on Apple Silicon without CPU fallback.
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

    def encode_sequence(self, sequences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if sequences.shape[0] == 0:
            return torch.zeros((0, self.hidden_size), dtype=sequences.dtype, device=sequences.device)
        safe_masks = masks.clone()
        zero_mask = safe_masks.sum(dim=1) <= 0
        if zero_mask.any():
            safe_masks = safe_masks.clone()
            safe_masks[zero_mask, 0] = 1.0
        projected = self.input_proj(sequences) + self.pos_embedding[:, : sequences.shape[1], :]
        key_padding_mask = safe_masks <= 0
        encoded = self.encoder(projected, src_key_padding_mask=key_padding_mask)
        valid = safe_masks.unsqueeze(-1)
        denom = valid.sum(dim=1).clamp(min=1.0)
        pooled = (encoded * valid).sum(dim=1) / denom
        if zero_mask.any():
            pooled = pooled.clone()
            pooled[zero_mask] = 0.0
        return pooled


class TorchSequenceQPolicy(BatchActionPolicy):
    """Sequence-aware Q policy with soft support regularization."""

    def __init__(
        self,
        *,
        name: Literal["gru_q_v3", "transformer_q_v3"],
        model: _SequenceQNetwork,
        feature_keys: tuple[str, ...],
        action_grid: np.ndarray,
        state_stats: StateStats,
        behavior_policy: BehaviorPolicyModel,
        support_lambda: float,
        support_floor: float,
        device: torch.device,
    ):
        self.name = name
        self.model = model
        self.feature_keys = feature_keys
        self.action_grid = np.asarray(action_grid, dtype=np.float32)
        self.state_stats = state_stats
        self.behavior_policy = behavior_policy
        self.support_lambda = support_lambda
        self.support_floor = support_floor
        self.device = device
        self.model.eval()

    def _normalized_summary(self, frame: pd.DataFrame) -> torch.Tensor:
        values = self.state_stats.normalize(frame, self.feature_keys)
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def predict_action_indices(self, frame: pd.DataFrame) -> np.ndarray:
        zeros = np.zeros((len(frame), 32, len(SEQUENCE_CHANNELS)), dtype=np.float32)
        masks = np.zeros((len(frame), 32), dtype=np.float32)
        return self.predict_action_indices_batch(DecisionBatch(frame=frame, sequences=zeros, masks=masks))

    def predict_action_indices_batch(self, batch: DecisionBatch) -> np.ndarray:
        if batch.sequences is None or batch.masks is None:
            raise ValueError(f"{self.name} requires aligned sequence inputs")
        with torch.no_grad():
            summary = self._normalized_summary(batch.frame)
            sequences = torch.tensor(batch.sequences, dtype=torch.float32, device=self.device)
            masks = torch.tensor(batch.masks, dtype=torch.float32, device=self.device)
            q_values, _, _, _ = self.model(summary, sequences, masks)
            support = self.behavior_policy.predict_proba(batch.frame)
            support_tensor = torch.tensor(support, dtype=torch.float32, device=self.device)
            score = q_values + self.support_lambda * torch.log(torch.clamp(support_tensor, min=self.support_floor))
            score = torch.where(
                support_tensor >= self.support_floor,
                score,
                torch.full_like(score, float("-inf")),
            )
            if (~torch.isfinite(score)).all(dim=1).any():
                best_support = torch.argmax(support_tensor, dim=1)
                best_score = torch.argmax(torch.where(torch.isfinite(score), score, torch.full_like(score, -1e9)), dim=1)
                invalid = (~torch.isfinite(score)).all(dim=1)
                best_score[invalid] = best_support[invalid]
                return best_score.cpu().numpy().astype(np.int64)
            return torch.argmax(score, dim=1).cpu().numpy().astype(np.int64)

    def save(self, output_dir: Path, *, metadata: dict[str, Any]) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.pt"
        torch.save(
            {
                "name": self.name,
                "state_dict": self.model.state_dict(),
                "feature_keys": list(self.feature_keys),
                "action_grid": self.action_grid.tolist(),
                "state_mean": self.state_stats.mean.tolist(),
                "state_std": self.state_stats.std.tolist(),
                **metadata,
            },
            model_path,
        )
        return model_path


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def compute_state_stats(frame: pd.DataFrame, feature_keys: tuple[str, ...]) -> StateStats:
    mean = frame.loc[:, list(feature_keys)].mean(axis=0).to_numpy(dtype=np.float32)
    std = frame.loc[:, list(feature_keys)].std(axis=0, ddof=0).to_numpy(dtype=np.float32)
    std = np.where(std <= 0.0, 1.0, std)
    return StateStats(mean=mean, std=std)


def _state_action_inputs(frame: pd.DataFrame, feature_keys: tuple[str, ...], action_values: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            frame.loc[:, list(feature_keys)].to_numpy(dtype=np.float32),
            action_values.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    )


def _decision_state_defaults(frame: pd.DataFrame, feature_keys: tuple[str, ...]) -> dict[str, dict[str, float]]:
    defaults: dict[str, dict[str, float]] = {}
    for decision_index, stage_frame in frame.groupby("decision_index", sort=True):
        defaults[str(int(decision_index))] = {
            feature_name: float(stage_frame[feature_name].median()) for feature_name in feature_keys
        }
    return defaults


def _best_fixed_schedule(train_frame: pd.DataFrame) -> dict[int, float]:
    initial = train_frame[train_frame["decision_index"] == 0].copy()
    best_run_id = str(initial.sort_values("final_objective").iloc[0]["wandb_run_id"])
    run_frame = train_frame[train_frame["wandb_run_id"] == best_run_id].sort_values("decision_index")
    return {int(row.decision_index): float(row.action_starcoder) for row in run_frame.itertuples()}


def _fit_discrete_bc_policy(
    train_frame: pd.DataFrame,
    feature_keys: tuple[str, ...],
    random_state: int,
) -> SklearnDiscreteBCPolicy:
    inputs = train_frame.loc[:, list(feature_keys)].to_numpy(dtype=np.float32)
    targets = train_frame["action_idx"].to_numpy(dtype=np.int64)
    if len(np.unique(targets)) < 2:
        estimator: HistGradientBoostingClassifier | DummyClassifier = DummyClassifier(strategy="prior")
    else:
        estimator = HistGradientBoostingClassifier(random_state=random_state, max_depth=4, max_iter=96)
    estimator.fit(inputs, targets)
    return SklearnDiscreteBCPolicy(estimator=estimator, feature_keys=feature_keys)


def _canonical_initial_batch(
    train_batch: DecisionBatch,
    feature_keys: tuple[str, ...],
) -> DecisionBatch:
    initial = train_batch.frame[train_batch.frame["decision_index"] == 0].copy().reset_index(drop=True)
    medians = {feature: float(initial[feature].median()) for feature in feature_keys}
    frame = pd.DataFrame([medians])
    sequences = np.zeros((1, train_batch.sequences.shape[1], train_batch.sequences.shape[2]), dtype=np.float32)
    masks = np.zeros((1, train_batch.masks.shape[1]), dtype=np.float32)
    return DecisionBatch(frame=frame, sequences=sequences, masks=masks)


def _phase0_top_decile_limit(train_frame: pd.DataFrame, action_grid: np.ndarray) -> float:
    initial = train_frame[train_frame["decision_index"] == 0].copy()
    cutoff = max(1, math.ceil(0.1 * len(initial)))
    top = initial.sort_values("final_objective").head(cutoff)
    q75 = float(top["action_starcoder"].quantile(0.75))
    bin_width = float(action_grid[1] - action_grid[0]) if len(action_grid) > 1 else 0.0
    return q75 + bin_width


def _ordered_batch(batch: DecisionBatch) -> tuple[DecisionBatch, np.ndarray]:
    order = batch.frame.sort_values(["episode_id", "step_in_episode"]).index.to_numpy(dtype=np.int64)
    return batch.take(order), order


def _sequence_targets(
    batch: DecisionBatch, feature_keys: tuple[str, ...], state_stats: StateStats
) -> dict[str, np.ndarray]:
    ordered_batch, _ = _ordered_batch(batch)
    frame = ordered_batch.frame
    summary = state_stats.normalize(frame, feature_keys)
    sequences = np.asarray(ordered_batch.sequences, dtype=np.float32)
    masks = np.asarray(ordered_batch.masks, dtype=np.float32)
    actions = frame["action_idx"].to_numpy(dtype=np.int64)
    rewards = frame["reward_dense_raw"].to_numpy(dtype=np.float32)
    next_bpb = frame["next_obj_bpb"].to_numpy(dtype=np.float32)
    final_bpb = frame["final_objective"].to_numpy(dtype=np.float32)
    dones = frame["done"].astype(np.float32).to_numpy(dtype=np.float32)

    next_summary = np.zeros_like(summary, dtype=np.float32)
    next_sequences = np.zeros_like(sequences, dtype=np.float32)
    next_masks = np.zeros_like(masks, dtype=np.float32)
    for _, episode in frame.groupby("episode_id", sort=False):
        positions = episode.index.to_numpy(dtype=np.int64)
        if len(positions) <= 1:
            continue
        next_positions = positions[1:]
        current_positions = positions[:-1]
        next_summary[current_positions] = summary[next_positions]
        next_sequences[current_positions] = sequences[next_positions]
        next_masks[current_positions] = masks[next_positions]

    return {
        "summary": summary,
        "sequences": sequences,
        "masks": masks,
        "actions": actions,
        "rewards": rewards,
        "next_bpb": next_bpb,
        "final_bpb": final_bpb,
        "dones": dones,
        "next_summary": next_summary,
        "next_sequences": next_sequences,
        "next_masks": next_masks,
    }


def _train_pretrain_heads(
    model: _SequenceQNetwork,
    *,
    sequences: np.ndarray,
    masks: np.ndarray,
    next_train_loss: np.ndarray,
    next_eval_bpb: np.ndarray,
    device: torch.device,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    random_state: int,
) -> None:
    if len(sequences) == 0:
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(random_state)
    seq_tensor = torch.tensor(sequences, dtype=torch.float32, device=device)
    mask_tensor = torch.tensor(masks, dtype=torch.float32, device=device)
    next_train_tensor = torch.tensor(next_train_loss, dtype=torch.float32, device=device)
    next_eval_tensor = torch.tensor(next_eval_bpb, dtype=torch.float32, device=device)
    model.train()
    for _ in range(epochs):
        indices = rng.permutation(len(sequences))
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            pred_train, pred_eval = model.forward_pretrain(seq_tensor[batch_idx], mask_tensor[batch_idx])
            loss = F.mse_loss(pred_train, next_train_tensor[batch_idx]) + 0.5 * F.mse_loss(
                pred_eval, next_eval_tensor[batch_idx]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()


def _train_sequence_policy(
    *,
    name: Literal["gru_q_v3", "transformer_q_v3"],
    train_batch: DecisionBatch,
    pretrain_payload: dict[str, np.ndarray],
    feature_keys: tuple[str, ...],
    state_stats: StateStats,
    action_grid: np.ndarray,
    behavior_policy: BehaviorPolicyModel,
    config: ThreePhasePolicyBenchV3Config,
) -> TorchSequenceQPolicy:
    device = resolve_device(config.device)
    if train_batch.sequences is None or train_batch.masks is None:
        raise ValueError("sequence policies require aligned decision sequences")
    torch.manual_seed(config.random_state)

    if name == "gru_q_v3":
        model: _SequenceQNetwork = GRUQNetwork(
            summary_dim=len(feature_keys),
            sequence_dim=len(SEQUENCE_CHANNELS),
            action_dim=len(action_grid),
            hidden_size=config.hidden_size,
        )
    else:
        model = TransformerQNetwork(
            summary_dim=len(feature_keys),
            sequence_dim=len(SEQUENCE_CHANNELS),
            action_dim=len(action_grid),
            hidden_size=config.hidden_size,
            num_layers=config.transformer_layers,
            num_heads=config.transformer_heads,
            ffn_dim=config.transformer_ffn_dim,
            dropout=config.dropout,
            max_length=train_batch.sequences.shape[1],
        )
    model.to(device)

    _train_pretrain_heads(
        model,
        sequences=np.asarray(pretrain_payload["sequences"], dtype=np.float32),
        masks=np.asarray(pretrain_payload["masks"], dtype=np.float32),
        next_train_loss=np.asarray(pretrain_payload["next_train_loss"], dtype=np.float32),
        next_eval_bpb=np.asarray(pretrain_payload["next_eval_bpb"], dtype=np.float32),
        device=device,
        learning_rate=config.learning_rate,
        epochs=config.pretrain_epochs,
        batch_size=config.batch_size,
        random_state=config.random_state,
    )

    tensors = _sequence_targets(train_batch, feature_keys, state_stats)
    summary = torch.tensor(tensors["summary"], dtype=torch.float32, device=device)
    sequences = torch.tensor(tensors["sequences"], dtype=torch.float32, device=device)
    masks = torch.tensor(tensors["masks"], dtype=torch.float32, device=device)
    actions = torch.tensor(tensors["actions"], dtype=torch.int64, device=device)
    rewards = torch.tensor(tensors["rewards"], dtype=torch.float32, device=device)
    next_bpb = torch.tensor(tensors["next_bpb"], dtype=torch.float32, device=device)
    final_bpb = torch.tensor(tensors["final_bpb"], dtype=torch.float32, device=device)
    dones = torch.tensor(tensors["dones"], dtype=torch.float32, device=device)
    next_summary = torch.tensor(tensors["next_summary"], dtype=torch.float32, device=device)
    next_sequences = torch.tensor(tensors["next_sequences"], dtype=torch.float32, device=device)
    next_masks = torch.tensor(tensors["next_masks"], dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    target_model = copy.deepcopy(model)
    rng = np.random.default_rng(config.random_state)
    model.train()
    for epoch in range(config.finetune_epochs):
        with torch.no_grad():
            target_model.eval()
            next_q, _, _, _ = target_model(next_summary, next_sequences, next_masks)
            next_values = next_q.max(dim=1).values
            td_target = rewards + config.gamma * (1.0 - dones) * next_values
        indices = rng.permutation(len(summary))
        for start in range(0, len(indices), config.batch_size):
            batch_idx = indices[start : start + config.batch_size]
            q_values, bc_logits, pred_next_bpb, pred_final_bpb = model(
                summary[batch_idx],
                sequences[batch_idx],
                masks[batch_idx],
            )
            q_taken = q_values.gather(1, actions[batch_idx].unsqueeze(1)).squeeze(1)
            loss = (
                F.mse_loss(q_taken, td_target[batch_idx])
                + 0.25 * F.cross_entropy(bc_logits, actions[batch_idx])
                + 0.1 * F.mse_loss(pred_next_bpb, next_bpb[batch_idx])
                + 0.1 * F.mse_loss(pred_final_bpb, final_bpb[batch_idx])
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % config.target_update_period == 0:
            target_model.load_state_dict(model.state_dict())
    model.eval()
    return TorchSequenceQPolicy(
        name=name,
        model=model,
        feature_keys=feature_keys,
        action_grid=action_grid,
        state_stats=state_stats,
        behavior_policy=behavior_policy,
        support_lambda=config.support_lambda,
        support_floor=config.support_floor,
        device=device,
    )


def _select_best_actions(
    *,
    frame: pd.DataFrame,
    feature_keys: tuple[str, ...],
    action_grid: np.ndarray,
    q_models: dict[int, HistGradientBoostingRegressor],
    behavior_policy: BehaviorPolicyModel,
    support_lambda: float,
    support_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    planner = DynamicQPlannerV3(
        feature_keys=feature_keys,
        action_grid=action_grid,
        q_models=q_models,
        behavior_policy=behavior_policy,
        support_lambda=support_lambda,
        support_floor=support_floor,
    )
    return planner.best_actions(DecisionBatch(frame=frame))


def _train_dynamic_q_planner(
    train_frame: pd.DataFrame,
    feature_keys: tuple[str, ...],
    action_grid: np.ndarray,
    behavior_policy: BehaviorPolicyModel,
    config: ThreePhasePolicyBenchV3Config,
) -> DynamicQPlannerV3:
    ordered = train_frame.sort_values(["episode_id", "step_in_episode"]).reset_index(drop=True)
    next_features = ordered.groupby("episode_id", sort=False)[list(feature_keys)].shift(-1)
    q_models: dict[int, HistGradientBoostingRegressor] = {}
    for decision_index in sorted(ordered["decision_index"].unique(), reverse=True):
        stage_frame = ordered[ordered["decision_index"] == decision_index].copy()
        targets = stage_frame["reward_dense_raw"].to_numpy(dtype=np.float32)
        nonterminal_mask = ~stage_frame["done"].to_numpy(dtype=bool)
        if nonterminal_mask.any():
            next_rows = stage_frame.loc[nonterminal_mask, list(feature_keys)].copy()
            next_rows.loc[:, list(feature_keys)] = next_features.loc[
                stage_frame.index[nonterminal_mask], list(feature_keys)
            ].to_numpy(dtype=np.float32)
            _, continuation = _select_best_actions(
                frame=next_rows,
                feature_keys=feature_keys,
                action_grid=action_grid,
                q_models=q_models,
                behavior_policy=behavior_policy,
                support_lambda=config.support_lambda,
                support_floor=config.support_floor,
            )
            targets[nonterminal_mask] += config.gamma * continuation
        model = HistGradientBoostingRegressor(
            random_state=config.random_state + int(decision_index),
            max_depth=4,
            max_iter=config.q_max_iter,
        )
        model.fit(
            _state_action_inputs(stage_frame, feature_keys, stage_frame["action_starcoder"].to_numpy(dtype=np.float32)),
            targets,
        )
        q_models[int(decision_index)] = model
    return DynamicQPlannerV3(
        feature_keys=feature_keys,
        action_grid=action_grid,
        q_models=q_models,
        behavior_policy=behavior_policy,
        support_lambda=config.support_lambda,
        support_floor=config.support_floor,
    )


def _policy_artifact(
    *,
    kind: PolicyKindV2,
    model_path: Path,
    feature_keys: tuple[str, ...],
    action_grid: np.ndarray,
    state_stats: StateStats,
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
        reward_mean=0.0,
        reward_std=1.0,
        model_path=str(model_path.resolve()),
        aux_paths=aux_paths or {},
        metadata=metadata,
    )


def _load_decision_batch(
    input_dir: Path,
) -> tuple[pd.DataFrame, DecisionBatch, dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    decisions = pd.read_parquet(input_dir / "decisions.parquet")
    with np.load(input_dir / "decision_sequences.npz", allow_pickle=False) as sequence_npz:
        row_ids = sequence_npz["row_ids"].astype(str)
        sequences = sequence_npz["sequences"].astype(np.float32)
        masks = sequence_npz["masks"].astype(np.float32)
    if decisions["row_id"].tolist() != row_ids.tolist():
        raise ValueError("decision_sequences.npz row_ids do not align with decisions.parquet")
    with np.load(input_dir / "pretrain_sequences.npz", allow_pickle=False) as pretrain_npz:
        pretrain_payload = {key: pretrain_npz[key] for key in pretrain_npz.files}
    with (input_dir / "feature_manifest.json").open() as f:
        feature_manifest = json.load(f)
    with (input_dir / "dataset_manifest.json").open() as f:
        dataset_manifest = json.load(f)
    batch = DecisionBatch(frame=decisions, sequences=sequences, masks=masks)
    return decisions, batch, pretrain_payload, feature_manifest, dataset_manifest


def _batch_from_mask(batch: DecisionBatch, mask: np.ndarray) -> DecisionBatch:
    positions = np.flatnonzero(mask.astype(bool))
    return batch.take(positions)


def _evaluate_policy(
    *,
    policy: ActionPolicy | BatchActionPolicy,
    train_batch: DecisionBatch,
    val_batch: DecisionBatch,
    feature_keys: tuple[str, ...],
    behavior_policy: BehaviorPolicyModel,
    reward_models: RewardModelBundle,
    action_grid: np.ndarray,
    config: ThreePhasePolicyBenchV3Config,
) -> tuple[dict[str, Any], pd.DataFrame]:
    fqe = fit_fqe_model_batch(
        train_batch=train_batch,
        feature_keys=feature_keys,
        policy=policy,
        action_grid=action_grid,
        gamma=config.gamma,
        random_state=config.random_state,
        iterations=config.fqe_iterations,
    )
    direct_value = direct_method_value_batch(val_batch, policy, reward_models)
    fqe_value = initial_state_value_batch(val_batch, policy, fqe, action_grid)
    dr_value = doubly_robust_estimate_batch(
        val_batch,
        policy,
        behavior_policy,
        fqe,
        action_grid,
        gamma=config.gamma,
    )
    wis_value = weighted_importance_sampling_batch(val_batch, policy, behavior_policy, gamma=config.gamma)
    diagnostics_df, summary = policy_diagnostics_batch(
        batch=val_batch,
        policy=policy,
        behavior_policy=behavior_policy,
        action_grid=tuple(float(value) for value in action_grid.tolist()),
        support_threshold=config.diagnostic_support_threshold,
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


def run_three_phase_policy_bench_v3(config: ThreePhasePolicyBenchV3Config) -> dict[str, Any]:
    """Run grouped CV for dynamic-Q and sequence policies on dense three-phase data."""
    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    decisions, full_batch, pretrain_payload, feature_manifest, dataset_manifest = _load_decision_batch(input_dir)
    feature_keys = tuple(feature_manifest["selected_feature_keys"])
    action_grid = np.asarray(dataset_manifest["action_grid"], dtype=np.float32)
    n_folds = int(dataset_manifest["n_cv_folds"])

    fold_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[pd.DataFrame] = []
    canonical_rows: list[dict[str, Any]] = []

    for fold in range(n_folds):
        train_mask = decisions["cv_fold"].to_numpy(dtype=np.int64) != fold
        val_mask = decisions["cv_fold"].to_numpy(dtype=np.int64) == fold
        train_frame = decisions.loc[train_mask].copy().reset_index(drop=True)
        val_frame = decisions.loc[val_mask].copy().reset_index(drop=True)
        if train_frame.empty or val_frame.empty:
            continue
        train_batch = _batch_from_mask(full_batch, train_mask)
        val_batch = _batch_from_mask(full_batch, val_mask)

        state_stats = compute_state_stats(train_frame, feature_keys)
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
            final_model_weight=1.0,
            reward_bonus_weight=0.0,
        )

        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        policies: list[ActionPolicy | BatchActionPolicy] = [
            FixedSchedulePolicy(action_grid, _best_fixed_schedule(train_frame)),
            _fit_discrete_bc_policy(train_frame, feature_keys, config.random_state),
            _train_dynamic_q_planner(train_frame, feature_keys, action_grid, behavior_policy, config),
            _train_sequence_policy(
                name="gru_q_v3",
                train_batch=train_batch,
                pretrain_payload=pretrain_payload,
                feature_keys=feature_keys,
                state_stats=state_stats,
                action_grid=action_grid,
                behavior_policy=behavior_policy,
                config=config,
            ),
            _train_sequence_policy(
                name="transformer_q_v3",
                train_batch=train_batch,
                pretrain_payload=pretrain_payload,
                feature_keys=feature_keys,
                state_stats=state_stats,
                action_grid=action_grid,
                behavior_policy=behavior_policy,
                config=config,
            ),
        ]

        canonical_batch = _canonical_initial_batch(train_batch, feature_keys)
        phase0_limit = _phase0_top_decile_limit(train_frame, action_grid)
        for policy in policies:
            metrics, diagnostics = _evaluate_policy(
                policy=policy,
                train_batch=train_batch,
                val_batch=val_batch,
                feature_keys=feature_keys,
                behavior_policy=behavior_policy,
                reward_models=reward_models,
                action_grid=action_grid,
                config=config,
            )
            action_idx = int(predict_action_indices_for_batch(policy, canonical_batch)[0])
            canonical_action = float(action_grid[action_idx])
            metrics.update(
                {
                    "fold": fold,
                    "canonical_phase0_action": canonical_action,
                    "canonical_phase0_passed": canonical_action <= phase0_limit,
                }
            )
            fold_rows.append(metrics)
            diagnostics_rows.append(diagnostics.assign(fold=fold))
            canonical_rows.append(
                {
                    "fold": fold,
                    "method": policy.name,
                    "canonical_phase0_action": canonical_action,
                    "phase0_limit": phase0_limit,
                    "passed": canonical_action <= phase0_limit,
                }
            )

    scores_df = pd.DataFrame(fold_rows)
    if scores_df.empty:
        raise ValueError("No fold scores were produced.")
    diagnostics_df = pd.concat(diagnostics_rows, ignore_index=True) if diagnostics_rows else pd.DataFrame()
    canonical_df = pd.DataFrame(canonical_rows)

    scores_df.to_csv(output_dir / "fold_scores.csv", index=False)
    diagnostics_df.to_csv(output_dir / "policy_diagnostics.csv", index=False)
    canonical_df.to_csv(output_dir / "canonical_phase0_checks.csv", index=False)

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
            canonical_phase0_action_mean=("canonical_phase0_action", "mean"),
            canonical_phase0_pass_rate=("canonical_phase0_passed", "mean"),
        )
        .sort_values("dr_value_mean", ascending=False)
        .reset_index(drop=True)
    )
    summary_df.to_csv(output_dir / "policy_summary.csv", index=False)

    baseline_bc = scores_df[scores_df["method"] == "discrete_bc"].set_index("fold")
    baseline_fixed = scores_df[scores_df["method"] == "fixed_best_schedule"].set_index("fold")
    candidate_methods = [
        method for method in summary_df["method"].tolist() if method not in {"fixed_best_schedule", "discrete_bc"}
    ]
    gate_rows: list[dict[str, Any]] = []
    passing_methods: list[str] = []
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
        canonical_pass_rate = float(method_scores["canonical_phase0_passed"].mean())
        passed = (
            beat_folds >= 3
            and unsupported_rate <= config.unsupported_rate_threshold
            and boundary_rate <= config.boundary_rate_threshold
            and canonical_pass_rate >= 1.0
        )
        gate_rows.append(
            {
                "method": method,
                "beat_baselines_folds": beat_folds,
                "unsupported_rate_mean": unsupported_rate,
                "boundary_rate_mean": boundary_rate,
                "canonical_phase0_pass_rate": canonical_pass_rate,
                "passed": passed,
            }
        )
        if passed:
            passing_methods.append(method)
    gate_df = pd.DataFrame(gate_rows)
    gate_df.to_csv(output_dir / "rollout_gate_report.csv", index=False)

    full_state_stats = compute_state_stats(decisions, feature_keys)
    full_behavior = fit_behavior_policy(
        decisions,
        feature_keys=feature_keys,
        action_count=len(action_grid),
        random_state=config.random_state,
    )
    full_dynamic_q = _train_dynamic_q_planner(decisions, feature_keys, action_grid, full_behavior, config)
    full_discrete_bc = _fit_discrete_bc_policy(decisions, feature_keys, config.random_state)
    full_gru = _train_sequence_policy(
        name="gru_q_v3",
        train_batch=full_batch,
        pretrain_payload=pretrain_payload,
        feature_keys=feature_keys,
        state_stats=full_state_stats,
        action_grid=action_grid,
        behavior_policy=full_behavior,
        config=config,
    )
    full_transformer = _train_sequence_policy(
        name="transformer_q_v3",
        train_batch=full_batch,
        pretrain_payload=pretrain_payload,
        feature_keys=feature_keys,
        state_stats=full_state_stats,
        action_grid=action_grid,
        behavior_policy=full_behavior,
        config=config,
    )

    artifacts_dir = output_dir / "artifacts"
    full_models_dir = output_dir / "full_models"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    full_models_dir.mkdir(parents=True, exist_ok=True)

    decision_defaults_path = artifacts_dir / "decision_state_defaults.json"
    with decision_defaults_path.open("w") as f:
        json.dump(_decision_state_defaults(decisions, feature_keys), f, indent=2, sort_keys=True)
    feature_manifest_copy = artifacts_dir / "feature_manifest.json"
    dataset_manifest_copy = artifacts_dir / "dataset_manifest.json"
    feature_manifest_copy.write_text(json.dumps(feature_manifest, indent=2, sort_keys=True))
    dataset_manifest_copy.write_text(json.dumps(dataset_manifest, indent=2, sort_keys=True))

    dynamic_q_bundle_path = full_models_dir / "dynamic_q_planner_v3" / "planner.joblib"
    dynamic_q_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "feature_keys": feature_keys,
            "action_grid": action_grid.tolist(),
            "q_models": full_dynamic_q.q_models,
            "behavior_policy": full_behavior,
            "support_lambda": config.support_lambda,
            "support_floor": config.support_floor,
        },
        dynamic_q_bundle_path,
    )

    bc_model_dir = full_models_dir / "discrete_bc"
    bc_model_dir.mkdir(parents=True, exist_ok=True)
    bc_bundle_path = bc_model_dir / "model.joblib"
    dump({"feature_keys": feature_keys, "estimator": full_discrete_bc.estimator}, bc_bundle_path)

    support_bundle_path = artifacts_dir / "support_bundle.joblib"
    dump(
        {
            "behavior_policy": full_behavior,
            "support_lambda": config.support_lambda,
            "support_floor": config.support_floor,
            "sequence_channels": list(SEQUENCE_CHANNELS),
        },
        support_bundle_path,
    )

    gru_model_path = full_gru.save(
        full_models_dir / "gru_q_v3",
        metadata={
            "hidden_size": config.hidden_size,
            "sequence_type": "gru",
            "sequence_channels": list(SEQUENCE_CHANNELS),
        },
    )
    transformer_model_path = full_transformer.save(
        full_models_dir / "transformer_q_v3",
        metadata={
            "hidden_size": config.hidden_size,
            "transformer_layers": config.transformer_layers,
            "transformer_heads": config.transformer_heads,
            "transformer_ffn_dim": config.transformer_ffn_dim,
            "dropout": config.dropout,
            "sequence_type": "transformer",
            "sequence_channels": list(SEQUENCE_CHANNELS),
            "max_length": int(full_batch.sequences.shape[1]),
        },
    )

    artifact_map = {
        "dynamic_q_planner_v3": _policy_artifact(
            kind="sklearn_dynamic_q_planner_v3",
            model_path=dynamic_q_bundle_path,
            feature_keys=feature_keys,
            action_grid=action_grid,
            state_stats=full_state_stats,
            objective_metric=config.objective_metric,
            metadata={
                "support_lambda": config.support_lambda,
                "support_floor": config.support_floor,
                "sequence_model": False,
            },
            aux_paths={
                "decision_state_defaults": str(decision_defaults_path.resolve()),
                "feature_manifest": str(feature_manifest_copy.resolve()),
                "dataset_manifest": str(dataset_manifest_copy.resolve()),
            },
        ),
        "discrete_bc": _policy_artifact(
            kind="d3rlpy_discrete_bc_v2",
            model_path=bc_bundle_path,
            feature_keys=feature_keys,
            action_grid=action_grid,
            state_stats=full_state_stats,
            objective_metric=config.objective_metric,
            metadata={"sequence_model": False},
            aux_paths={
                "decision_state_defaults": str(decision_defaults_path.resolve()),
                "feature_manifest": str(feature_manifest_copy.resolve()),
                "dataset_manifest": str(dataset_manifest_copy.resolve()),
            },
        ),
        "gru_q_v3": _policy_artifact(
            kind="torch_gru_q_v3",
            model_path=gru_model_path,
            feature_keys=feature_keys,
            action_grid=action_grid,
            state_stats=full_state_stats,
            objective_metric=config.objective_metric,
            metadata={
                "support_lambda": config.support_lambda,
                "support_floor": config.support_floor,
                "sequence_model": True,
            },
            aux_paths={
                "support_bundle": str(support_bundle_path.resolve()),
                "decision_state_defaults": str(decision_defaults_path.resolve()),
                "feature_manifest": str(feature_manifest_copy.resolve()),
                "dataset_manifest": str(dataset_manifest_copy.resolve()),
            },
        ),
        "transformer_q_v3": _policy_artifact(
            kind="torch_transformer_q_v3",
            model_path=transformer_model_path,
            feature_keys=feature_keys,
            action_grid=action_grid,
            state_stats=full_state_stats,
            objective_metric=config.objective_metric,
            metadata={
                "support_lambda": config.support_lambda,
                "support_floor": config.support_floor,
                "sequence_model": True,
            },
            aux_paths={
                "support_bundle": str(support_bundle_path.resolve()),
                "decision_state_defaults": str(decision_defaults_path.resolve()),
                "feature_manifest": str(feature_manifest_copy.resolve()),
                "dataset_manifest": str(dataset_manifest_copy.resolve()),
            },
        ),
    }
    for method, artifact in artifact_map.items():
        save_policy_artifact(artifacts_dir / f"{method}.json", artifact)

    passing_artifacts = {
        method: str((artifacts_dir / f"{method}.json").resolve()) for method in passing_methods if method in artifact_map
    }
    with (output_dir / "passing_policy_artifacts.json").open("w") as f:
        json.dump(passing_artifacts, f, indent=2, sort_keys=True)

    summary = {
        "passing_methods": passing_methods,
        "feature_keys": list(feature_keys),
        "artifacts": {method: str((artifacts_dir / f"{method}.json").resolve()) for method in artifact_map},
    }
    with (output_dir / "offline_policy_report.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train three-phase offline-control v3 policies.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="experiments/domain_phase_mix/offline_rl/artifacts/three_phase_starcoder_dense_v3",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/domain_phase_mix/offline_rl/artifacts/three_phase_starcoder_dense_v3/policy_bench_v3",
    )
    parser.add_argument("--objective-metric", type=str, default=DEFAULT_OBJECTIVE_METRIC)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    run_three_phase_policy_bench_v3(
        ThreePhasePolicyBenchV3Config(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            objective_metric=args.objective_metric,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
