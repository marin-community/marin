# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.offline_rl.build_transitions import (
    augment_suffix_fragments,
    build_transition_tables,
)
from experiments.domain_phase_mix.offline_rl.collect_three_phase_starcoder_dataset import (
    build_wide_history,
    dedupe_history_rows,
)
from experiments.domain_phase_mix.offline_rl.contracts import default_feature_config
from experiments.domain_phase_mix.offline_rl.evaluate_policy_three_phase_starcoder import (
    EvaluateConfig,
    _should_force_local_dry_run,
    build_phase_train_plan,
)
from experiments.domain_phase_mix.offline_rl.policy_artifact import (
    PolicyArtifactV1,
    clip_action,
    load_policy_artifact,
    save_policy_artifact,
)
from experiments.domain_phase_mix.offline_rl.train_cql_policy import TrainCQLConfig, train_policy


def test_dedupe_history_rows_keeps_latest_entry():
    long_df = pd.DataFrame(
        [
            {
                "wandb_run_id": "run-a",
                "source_experiment": "x",
                "local_run_id": 1,
                "run_name": "x/run_00001",
                "step": 100,
                "total_tokens": 1.0,
                "metric_key": "eval/loss",
                "metric_value": 2.0,
                "_scan_index": 0,
            },
            {
                "wandb_run_id": "run-a",
                "source_experiment": "x",
                "local_run_id": 1,
                "run_name": "x/run_00001",
                "step": 100,
                "total_tokens": 1.0,
                "metric_key": "eval/loss",
                "metric_value": 1.5,
                "_scan_index": 1,
            },
        ]
    )
    deduped = dedupe_history_rows(long_df)
    assert len(deduped) == 1
    assert float(deduped.iloc[0]["metric_value"]) == 1.5


def _synthetic_run_tables():
    runs_df = pd.DataFrame(
        [
            {
                "wandb_run_id": "run-a",
                "source_experiment": "exp",
                "local_run_id": 1,
                "phase_0_starcoder": 0.1,
                "phase_1_starcoder": 0.2,
                "phase_2_starcoder": 0.3,
                "eval/paloma/dolma_100_programing_languages/bpb": 0.8,
            },
            {
                "wandb_run_id": "run-b",
                "source_experiment": "exp",
                "local_run_id": 2,
                "phase_0_starcoder": 0.4,
                "phase_1_starcoder": 0.5,
                "phase_2_starcoder": 0.6,
                "eval/paloma/dolma_100_programing_languages/bpb": 0.7,
            },
        ]
    )
    long_rows = []
    for run_id in ("run-a", "run-b"):
        for step, train_loss, eval_loss, obj, tokens in (
            (0, 3.0, np.nan, np.nan, 0.0),
            (1888, 2.0, 2.2, 1.0, 1.0),
            (3833, 1.5, 1.7, 0.9, 2.0),
            (5722, 1.2, 1.4, 0.8 if run_id == "run-a" else 0.7, 3.0),
        ):
            for key, value in (
                ("train/loss", train_loss),
                ("eval/loss", eval_loss),
                ("eval/paloma/dolma_100_programing_languages/bpb", obj),
            ):
                if pd.isna(value):
                    continue
                long_rows.append(
                    {
                        "wandb_run_id": run_id,
                        "source_experiment": "exp",
                        "local_run_id": 1 if run_id == "run-a" else 2,
                        "run_name": f"exp/{run_id}",
                        "step": step,
                        "total_tokens": tokens,
                        "metric_key": key,
                        "metric_value": value,
                        "_scan_index": len(long_rows),
                    }
                )
    wide_df = build_wide_history(dedupe_history_rows(pd.DataFrame(long_rows)))
    return runs_df, wide_df


def test_build_transition_dataset_integrity_and_rewards():
    runs_df, wide_df = _synthetic_run_tables()
    base, episodes, manifest = build_transition_tables(
        runs_df=runs_df,
        history_wide_df=wide_df,
        feature_config=default_feature_config(),
        split_seed=7,
        train_fraction=0.8,
    )
    assert len(base) == 6
    assert len(base.groupby("wandb_run_id")) == 2
    assert len(episodes) == 12  # (3 + 2 + 1) * 2 runs
    assert manifest["n_base_transitions"] == 6
    assert all(col in base.columns for col in ("reward_std", "reward_delta_std"))

    non_terminal = base[base["t"] < 2]
    terminal = base[base["t"] == 2]
    assert (non_terminal["reward_raw"] == 0.0).all()
    assert (terminal["reward_raw"] < 0.0).all()

    required_features = [
        "phase_index",
        "last_train_loss",
        "last_eval_loss",
        "last_obj_bpb",
        "tokens_frac",
        "steps_since_last_eval_frac",
        "prev_action_starcoder",
    ]
    assert np.isfinite(base[required_features].to_numpy()).all()


def test_fragment_augmentation_suffix_lengths():
    frame = pd.DataFrame(
        [
            {"wandb_run_id": "run-a", "t": 0, "done": False},
            {"wandb_run_id": "run-a", "t": 1, "done": False},
            {"wandb_run_id": "run-a", "t": 2, "done": True},
        ]
    )
    augmented = augment_suffix_fragments(frame)
    counts = augmented.groupby("episode_id")["t"].count().sort_values().tolist()
    assert counts == [1, 2, 3]


def test_policy_artifact_roundtrip_and_clip(tmp_path):
    artifact = PolicyArtifactV1(
        kind="d3rlpy_cql_continuous_v1",
        objective_metric="eval/loss",
        state_keys=("a", "b"),
        action_low=0.1,
        action_high=0.9,
        state_mean=[0.0, 0.0],
        state_std=[1.0, 1.0],
        reward_mean=0.0,
        reward_std=1.0,
        model_path="dummy.d3",
    )
    path = tmp_path / "policy.json"
    save_policy_artifact(path, artifact)
    loaded = load_policy_artifact(path)
    assert loaded == artifact
    assert clip_action(5.0, loaded) == 0.9
    assert clip_action(-5.0, loaded) == 0.1


def test_cql_smoke_with_fake_backend(tmp_path):
    episodes = pd.DataFrame(
        [
            {
                "wandb_run_id": "run-a",
                "episode_id": "run-a:frag0",
                "step_in_episode": 0,
                "split": "train",
                "done": False,
                "action_starcoder": 0.2,
                "reward_std": 0.0,
                "reward_delta_std": 0.1,
                "phase_index": 0.0,
                "last_train_loss": 1.0,
                "last_eval_loss": 1.1,
                "last_obj_bpb": 1.2,
                "tokens_frac": 0.0,
                "steps_since_last_eval_frac": 1.0,
                "prev_action_starcoder": 0.5,
            },
            {
                "wandb_run_id": "run-a",
                "episode_id": "run-a:frag0",
                "step_in_episode": 1,
                "split": "train",
                "done": True,
                "action_starcoder": 0.3,
                "reward_std": -1.0,
                "reward_delta_std": -0.1,
                "phase_index": 1.0,
                "last_train_loss": 0.9,
                "last_eval_loss": 1.0,
                "last_obj_bpb": 1.1,
                "tokens_frac": 0.5,
                "steps_since_last_eval_frac": 0.1,
                "prev_action_starcoder": 0.2,
            },
            {
                "wandb_run_id": "run-b",
                "episode_id": "run-b:frag0",
                "step_in_episode": 0,
                "split": "val",
                "done": False,
                "action_starcoder": 0.4,
                "reward_std": 0.0,
                "reward_delta_std": 0.1,
                "phase_index": 0.0,
                "last_train_loss": 1.1,
                "last_eval_loss": 1.2,
                "last_obj_bpb": 1.3,
                "tokens_frac": 0.0,
                "steps_since_last_eval_frac": 1.0,
                "prev_action_starcoder": 0.5,
            },
            {
                "wandb_run_id": "run-b",
                "episode_id": "run-b:frag0",
                "step_in_episode": 1,
                "split": "val",
                "done": True,
                "action_starcoder": 0.5,
                "reward_std": -1.0,
                "reward_delta_std": -0.2,
                "phase_index": 1.0,
                "last_train_loss": 1.0,
                "last_eval_loss": 1.1,
                "last_obj_bpb": 1.2,
                "tokens_frac": 0.5,
                "steps_since_last_eval_frac": 0.1,
                "prev_action_starcoder": 0.4,
            },
        ]
    )
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    episodes.to_parquet(input_dir / "episodes.parquet", index=False)
    with (input_dir / "dataset_manifest.json").open("w") as f:
        json.dump({"reward_mean": -0.8, "reward_std": 0.3}, f)

    class _FakeBackend:
        def __init__(self, train_batch, device):
            self._bias = 0.0

        def fit_steps(self, n_steps: int) -> None:
            self._bias += float(n_steps) * 0.0001

        def predict(self, observations: np.ndarray) -> np.ndarray:
            out = observations[:, :1] * 0.2 + 0.3 + self._bias
            return out.astype(np.float32)

        def predict_value(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
            return (actions.reshape(-1) + observations[:, 0] * 0.01 + self._bias).astype(np.float32)

        def save(self, path: str) -> None:
            Path(path).write_text("fake-model")

    artifact = train_policy(
        TrainCQLConfig(
            input_dir=str(input_dir),
            output_dir=str(tmp_path / "output"),
            epochs=2,
            steps_per_epoch=5,
            device="cpu",
        ),
        backend_factory=lambda train_batch, device: _FakeBackend(train_batch, device),
    )
    assert artifact.kind == "d3rlpy_cql_continuous_v1"
    assert Path(artifact.model_path).exists()
    assert (tmp_path / "output" / "policy_artifact.json").exists()


def test_online_phase_planning_handoffs():
    p0 = build_phase_train_plan(0, checkpoint_path=None)
    p1 = build_phase_train_plan(1, checkpoint_path="gs://x/checkpoint-1")
    p2 = build_phase_train_plan(2, checkpoint_path="gs://x/checkpoint-2")
    assert p0.initialize_from_checkpoint_path is None
    assert p1.initialize_from_checkpoint_path is not None
    assert p2.initialize_from_checkpoint_path is not None
    assert p1.reset_data_loader_on_init is False
    assert p2.reset_data_loader_on_init is False


def test_local_fallback_preflight(monkeypatch):
    module_path = "experiments.domain_phase_mix.offline_rl.evaluate_policy_three_phase_starcoder.get_vm_region"
    monkeypatch.setattr(module_path, lambda: (_ for _ in ()).throw(ValueError("no metadata")))
    cfg = EvaluateConfig(
        policy_artifact_path="x/policy_artifact.json",
        output_dir="tmp",
        n_replicates=1,
        allow_local_fallback=True,
    )
    assert _should_force_local_dry_run(cfg) is True

    cfg_no_fallback = EvaluateConfig(
        policy_artifact_path="x/policy_artifact.json",
        output_dir="tmp",
        n_replicates=1,
        allow_local_fallback=False,
    )
    assert _should_force_local_dry_run(cfg_no_fallback) is False
