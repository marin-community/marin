# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.offline_rl.build_three_phase_dense_policy_dataset import (
    BuildThreePhaseDenseDatasetConfig,
    SEQUENCE_CHANNELS,
    build_three_phase_dense_policy_dataset,
)
from experiments.domain_phase_mix.offline_rl.collect_three_phase_starcoder_dataset import build_wide_history
from experiments.domain_phase_mix.offline_rl.collect_three_phase_starcoder_dense_dataset import (
    _merge_group_histories,
    collect_dense_history_from_run,
)
from experiments.domain_phase_mix.offline_rl.evaluate_policy_three_phase_starcoder import (
    _discover_exact_checkpoint,
)
from experiments.domain_phase_mix.offline_rl.ope import (
    DecisionBatch,
    fit_behavior_policy,
    predict_action_indices_for_batch,
)
from experiments.domain_phase_mix.offline_rl.train_three_phase_policy_bench_v3 import (
    TransformerQNetwork,
    ThreePhasePolicyBenchV3Config,
    _canonical_initial_batch,
    _fit_discrete_bc_policy,
    _phase0_top_decile_limit,
    _train_dynamic_q_planner,
    _train_sequence_policy,
    compute_state_stats,
)


class _FakeDenseRun:
    id = "run-a"
    display_name = "pinlin_calvin_xu/data_mixture/three_phase_starcoder_1/run_00000"

    def __init__(self):
        self._train_rows = [
            {"_step": 0, "train/loss": 4.0, "optim/learning_rate": 0.0, "optim/adam_lr": 0.0},
            {"_step": 1, "train/loss": 3.9, "optim/learning_rate": 0.1, "optim/adam_lr": 0.05},
            {"_step": 2, "train/loss": 3.8, "optim/learning_rate": 0.2, "optim/adam_lr": 0.1},
        ]
        self._norm_rows = [
            {"_step": 0, "grad/norm/total": 1.0, "params/norm/total": 10.0},
            {"_step": 2, "grad/norm/total": 1.2, "params/norm/total": 10.2},
        ]
        self._eval_rows = [
            {
                "_step": 0,
                "eval/loss": 4.5,
                "eval/paloma/dolma_100_programing_languages/bpb": 1.6,
            },
            {
                "_step": 2,
                "eval/loss": 4.2,
                "eval/paloma/dolma_100_programing_languages/bpb": 1.4,
            },
        ]

    def scan_history(self, keys):
        key_set = set(keys)
        if {"train/loss", "optim/learning_rate", "optim/adam_lr"}.issubset(key_set):
            return list(self._train_rows)
        if {"grad/norm/total", "params/norm/total"}.issubset(key_set):
            return list(self._norm_rows)
        if {
            "eval/loss",
            "eval/paloma/dolma_100_programing_languages/bpb",
        }.issubset(key_set):
            return list(self._eval_rows)
        return []


def test_collect_dense_history_from_run_merges_sparse_and_dense_groups():
    history = collect_dense_history_from_run(_FakeDenseRun())
    assert history["step"].tolist() == [0, 1, 2]
    assert history.loc[history["step"] == 1, "train/loss"].iloc[0] == 3.9
    assert np.isnan(history.loc[history["step"] == 1, "eval/loss"]).all()
    assert history.loc[history["step"] == 2, "grad/norm/total"].iloc[0] == 1.2
    assert history.loc[history["step"] == 2, "eval/paloma/dolma_100_programing_languages/bpb"].iloc[0] == 1.4


def test_build_wide_history_does_not_create_cartesian_product():
    history_long = pd.DataFrame(
        [
            {
                "wandb_run_id": "run-a",
                "source_experiment": "exp-a",
                "local_run_id": 1,
                "run_name": "exp-a/run-a",
                "step": 0,
                "total_tokens": None,
                "metric_key": "train/loss",
                "metric_value": 4.0,
                "_scan_index": 0,
            },
            {
                "wandb_run_id": "run-a",
                "source_experiment": "exp-a",
                "local_run_id": 1,
                "run_name": "exp-a/run-a",
                "step": 1,
                "total_tokens": None,
                "metric_key": "train/loss",
                "metric_value": 3.9,
                "_scan_index": 1,
            },
            {
                "wandb_run_id": "run-b",
                "source_experiment": "exp-b",
                "local_run_id": 2,
                "run_name": "exp-b/run-b",
                "step": 0,
                "total_tokens": None,
                "metric_key": "train/loss",
                "metric_value": 4.2,
                "_scan_index": 2,
            },
            {
                "wandb_run_id": "run-b",
                "source_experiment": "exp-b",
                "local_run_id": 2,
                "run_name": "exp-b/run-b",
                "step": 1,
                "total_tokens": None,
                "metric_key": "train/loss",
                "metric_value": 4.1,
                "_scan_index": 3,
            },
        ]
    )

    history = build_wide_history(history_long)
    assert len(history) == 4
    assert history["wandb_run_id"].tolist() == ["run-a", "run-a", "run-b", "run-b"]
    assert history["step"].tolist() == [0, 1, 0, 1]


def test_merge_group_histories_keeps_single_total_tokens_column():
    train = pd.DataFrame(
        [
            {
                "wandb_run_id": "run-a",
                "source_experiment": "exp-a",
                "local_run_id": 1,
                "run_name": "exp-a/run-a",
                "step": 0,
                "total_tokens": None,
                "train/loss": 4.0,
            }
        ]
    )
    eval_df = pd.DataFrame(
        [
            {
                "wandb_run_id": "run-a",
                "source_experiment": "exp-a",
                "local_run_id": 1,
                "run_name": "exp-a/run-a",
                "step": 0,
                "total_tokens": None,
                "eval/loss": 4.5,
            }
        ]
    )

    merged = _merge_group_histories({"train": train, "eval": eval_df})
    assert len(merged) == 1
    assert "total_tokens_x" not in merged.columns
    assert "total_tokens_y" not in merged.columns
    assert merged.loc[0, "train/loss"] == 4.0
    assert merged.loc[0, "eval/loss"] == 4.5


def _small_dense_inputs(tmp_path: Path) -> Path:
    input_dir = tmp_path / "dense_input"
    input_dir.mkdir(parents=True)
    runs = pd.DataFrame(
        [
            {
                "wandb_run_id": "run-a",
                "run_name": "three/run-a",
                "source_experiment": "exp-a",
                "run_family": "three_phase_starcoder",
                "local_run_id": 1,
                "num_phases_total": 3,
                "total_steps": 80,
                "phase_boundaries_json": json.dumps([30, 60]),
                "eval/paloma/dolma_100_programing_languages/bpb": 0.9,
                "phase_0_starcoder": 0.0,
                "phase_1_starcoder": 0.4,
                "phase_2_starcoder": 0.2,
            },
            {
                "wandb_run_id": "run-b",
                "run_name": "three/run-b",
                "source_experiment": "exp-b",
                "run_family": "three_phase_starcoder",
                "local_run_id": 2,
                "num_phases_total": 3,
                "total_steps": 80,
                "phase_boundaries_json": json.dumps([30, 60]),
                "eval/paloma/dolma_100_programing_languages/bpb": 1.1,
                "phase_0_starcoder": 1.0,
                "phase_1_starcoder": 0.8,
                "phase_2_starcoder": 0.6,
            },
        ]
    )
    history_rows = []
    for run_id, final_bpb, scale in (("run-a", 0.9, 1.0), ("run-b", 1.1, 1.2)):
        for step in range(0, 81, 10):
            history_rows.append(
                {
                    "wandb_run_id": run_id,
                    "source_experiment": "exp-a" if run_id == "run-a" else "exp-b",
                    "local_run_id": 1 if run_id == "run-a" else 2,
                    "run_name": f"three/{run_id}",
                    "step": step,
                    "train/loss": 4.0 * scale - 0.02 * step,
                    "optim/learning_rate": max(0.0, 0.02 - 0.0002 * step),
                    "optim/adam_lr": max(0.0, 0.008 - 0.00008 * step),
                    "grad/norm/total": 1.0 * scale + 0.01 * step,
                    "params/norm/total": 10.0 * scale + 0.02 * step,
                    "eval/loss": 4.5 * scale - 0.025 * step,
                    "eval/paloma/dolma_100_programing_languages/bpb": final_bpb + max(0, 80 - step) * 0.01,
                }
            )
    pd.DataFrame(history_rows).to_parquet(input_dir / "history_dense_wide.parquet", index=False)
    runs.to_parquet(input_dir / "runs.parquet", index=False)
    return input_dir


def test_build_three_phase_dense_dataset_outputs_decisions_sequences_and_pretrain(tmp_path: Path):
    input_dir = _small_dense_inputs(tmp_path)
    output_dir = tmp_path / "dense_output"
    decisions, sequence_arrays, manifest = build_three_phase_dense_policy_dataset(
        BuildThreePhaseDenseDatasetConfig(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            window_steps=20,
            window_bins=4,
            pretrain_stride=5,
        )
    )
    assert len(decisions) == 6
    assert manifest["action_grid"][0] == 0.0
    assert manifest["action_grid"][-1] == 1.0
    assert sequence_arrays["sequences"].shape == (6, 4, len(SEQUENCE_CHANNELS))
    assert sequence_arrays["masks"][0].tolist() == [0.0, 0.0, 0.0, 0.0]
    with np.load(output_dir / "pretrain_sequences.npz", allow_pickle=False) as payload:
        assert len(payload["row_ids"]) > len(decisions)


def _synthetic_policy_batch() -> tuple[pd.DataFrame, DecisionBatch, dict[str, np.ndarray], tuple[str, ...]]:
    feature_keys = (
        "decision_index",
        "remaining_decisions",
        "budget_frac_consumed",
        "last_obj_bpb",
        "prev_action_starcoder",
    )
    action_grid = np.linspace(0.0, 1.0, 21, dtype=np.float32)
    rows = []
    sequences = []
    masks = []
    for run_idx in range(10):
        good = run_idx < 8
        actions = [0.0, 0.3, 0.25] if good else [0.8, 0.7, 0.6]
        objectives = [1.35, 1.02, 0.85] if good else [1.55, 1.28, 1.12]
        rewards = [0.33, 0.17, 0.0] if good else [0.12, 0.08, 0.0]
        for decision_index in range(3):
            last_obj = objectives[max(0, decision_index - 1)] if decision_index > 0 else (1.45 if good else 1.85)
            row = {
                "row_id": f"run-{run_idx}::{decision_index}",
                "wandb_run_id": f"run-{run_idx}",
                "run_family": "three_phase_starcoder",
                "source_experiment": "synthetic",
                "local_run_id": run_idx,
                "run_name": f"run-{run_idx}",
                "decision_index": float(decision_index),
                "remaining_decisions": float(2 - decision_index),
                "budget_frac_consumed": float(decision_index) / 3.0,
                "last_obj_bpb": float(last_obj),
                "prev_action_starcoder": float(actions[decision_index - 1] if decision_index > 0 else 0.5),
                "action_starcoder": float(actions[decision_index]),
                "action_idx": int(np.argmin(np.abs(action_grid - actions[decision_index]))),
                "reward_dense_raw": float(rewards[decision_index]),
                "next_obj_bpb": float(objectives[decision_index]),
                "final_objective": float(objectives[-1]),
                "done": bool(decision_index == 2),
                "episode_id": f"run-{run_idx}",
                "step_in_episode": decision_index,
            }
            rows.append(row)
            seq = np.zeros((32, len(SEQUENCE_CHANNELS)), dtype=np.float32)
            seq[:, 0] = np.linspace(4.2 if good else 4.8, 3.4 if good else 4.2, 32)
            seq[:, 5] = np.linspace(4.5 if good else 5.0, 3.8 if good else 4.5, 32)
            seq[:, 6] = np.linspace(1.6, objectives[decision_index], 32)
            seq[:, 7] = 0.0
            seq[:, 8] = np.linspace(-1.0, -0.05, 32)
            mask = np.ones(32, dtype=np.float32)
            if decision_index == 0:
                seq[:] = 0.0
                mask[:] = 0.0
            sequences.append(seq)
            masks.append(mask)
    frame = pd.DataFrame(rows)
    batch = DecisionBatch(frame=frame, sequences=np.stack(sequences), masks=np.stack(masks))
    pretrain_payload = {
        "row_ids": np.asarray([], dtype=str),
        "wandb_run_id": np.asarray([], dtype=str),
        "phase_index": np.asarray([], dtype=np.int64),
        "cursor_step": np.asarray([], dtype=np.int64),
        "action_starcoder": np.asarray([], dtype=np.float32),
        "next_train_loss": np.asarray([], dtype=np.float32),
        "next_eval_bpb": np.asarray([], dtype=np.float32),
        "sequences": np.zeros((0, 32, len(SEQUENCE_CHANNELS)), dtype=np.float32),
        "masks": np.zeros((0, 32), dtype=np.float32),
    }
    return frame, batch, pretrain_payload, feature_keys


def test_v3_policies_prefer_low_initial_starcoder_on_synthetic_data():
    frame, batch, pretrain_payload, feature_keys = _synthetic_policy_batch()
    config = ThreePhasePolicyBenchV3Config(
        input_dir="unused",
        output_dir="unused",
        pretrain_epochs=0,
        finetune_epochs=120,
        batch_size=16,
        learning_rate=3e-3,
        device="cpu",
    )
    action_grid = np.linspace(0.0, 1.0, 21, dtype=np.float32)
    state_stats = compute_state_stats(frame, feature_keys)
    behavior_policy = fit_behavior_policy(frame, feature_keys, action_count=len(action_grid), random_state=0)

    dynamic_q = _train_dynamic_q_planner(frame, feature_keys, action_grid, behavior_policy, config)
    gru = _train_sequence_policy(
        name="gru_q_v3",
        train_batch=batch,
        pretrain_payload=pretrain_payload,
        feature_keys=feature_keys,
        state_stats=state_stats,
        action_grid=action_grid,
        behavior_policy=behavior_policy,
        config=config,
    )
    transformer = _train_sequence_policy(
        name="transformer_q_v3",
        train_batch=batch,
        pretrain_payload=pretrain_payload,
        feature_keys=feature_keys,
        state_stats=state_stats,
        action_grid=action_grid,
        behavior_policy=behavior_policy,
        config=config,
    )

    canonical = _canonical_initial_batch(batch, feature_keys)
    good_initial_positions = np.flatnonzero(
        (frame["decision_index"].to_numpy(dtype=np.float32) == 0.0)
        & (frame["final_objective"].to_numpy(dtype=np.float32) < 0.9)
    )
    good_initial = batch.take(good_initial_positions[:1])
    assert action_grid[predict_action_indices_for_batch(dynamic_q, canonical)[0]] <= 0.35
    assert action_grid[predict_action_indices_for_batch(dynamic_q, good_initial)[0]] <= 0.35
    assert action_grid[predict_action_indices_for_batch(gru, good_initial)[0]] <= 0.35
    assert action_grid[predict_action_indices_for_batch(transformer, good_initial)[0]] <= 0.35


def test_canonical_phase0_limit_rejects_high_action_schedule():
    frame, _, _, _ = _synthetic_policy_batch()
    limit = _phase0_top_decile_limit(frame, np.linspace(0.0, 1.0, 21, dtype=np.float32))
    high_policy = _fit_discrete_bc_policy(frame.assign(action_idx=20), ("decision_index",), random_state=0)
    del high_policy
    assert limit < 0.5
    assert 0.8 > limit


def test_discover_exact_checkpoint_returns_boundary_path(tmp_path: Path):
    output_dir = tmp_path / "rollout"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "step-1518").mkdir()
    (checkpoint_dir / "step-1887").mkdir()
    resolved = _discover_exact_checkpoint(str(output_dir), 1887)
    assert resolved.endswith("step-1887")


def test_transformer_q_network_disables_nested_tensor_fast_path():
    network = TransformerQNetwork(
        summary_dim=8,
        sequence_dim=len(SEQUENCE_CHANNELS),
        action_dim=21,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        dropout=0.1,
        max_length=32,
    )
    assert network.encoder.enable_nested_tensor is False
