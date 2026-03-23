# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.domain_phase_mix.offline_rl import evaluate_policy_two_phase_starcoder as two_phase_eval
from experiments.domain_phase_mix.offline_rl import evaluate_policy_three_phase_starcoder as eval_runner
from experiments.domain_phase_mix.offline_rl.build_pooled_transition_dataset import (
    BuildPooledTransitionConfig,
    build_action_grid,
    build_pooled_transition_dataset,
    discretize_action,
)
from experiments.domain_phase_mix.offline_rl.build_transitions import (
    augment_suffix_fragments,
    build_transition_tables,
)
from experiments.domain_phase_mix.offline_rl.collect_pooled_starcoder_dataset import (
    _resolve_phase_weights,
    _scan_history_batch,
)
from experiments.domain_phase_mix.offline_rl.collect_three_phase_starcoder_dataset import (
    build_wide_history,
    dedupe_history_rows,
)
from experiments.domain_phase_mix.offline_rl.contracts import (
    PooledDatasetConfig,
    default_feature_config,
)
from experiments.domain_phase_mix.offline_rl.ope import (
    direct_method_value,
    doubly_robust_estimate,
    fit_behavior_policy,
    fit_fqe_model,
    fit_reward_models,
    initial_state_value,
    policy_diagnostics,
)
from experiments.domain_phase_mix.offline_rl.policy_artifact import (
    PolicyArtifactV1,
    PolicyArtifactV2,
    clip_action,
    load_policy_artifact,
    save_policy_artifact,
)
from experiments.domain_phase_mix.offline_rl.train_cql_policy import TrainCQLConfig, train_policy
from experiments.domain_phase_mix.offline_rl.train_offline_policy_bench import (
    OfflinePolicyBenchConfig,
    run_offline_policy_bench,
)


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


def test_build_wide_history_preserves_rows_without_local_run_id():
    long_df = pd.DataFrame(
        [
            {
                "wandb_run_id": "run-a",
                "source_experiment": "x",
                "local_run_id": None,
                "run_name": "x/run-a",
                "step": 100,
                "total_tokens": None,
                "metric_key": "train/loss",
                "metric_value": 2.0,
                "_scan_index": 0,
            },
            {
                "wandb_run_id": "run-a",
                "source_experiment": "x",
                "local_run_id": None,
                "run_name": "x/run-a",
                "step": 100,
                "total_tokens": None,
                "metric_key": "eval/loss",
                "metric_value": 2.5,
                "_scan_index": 1,
            },
        ]
    )
    wide = build_wide_history(dedupe_history_rows(long_df))
    assert len(wide) == 1
    assert wide.iloc[0]["run_name"] == "x/run-a"
    assert float(wide.iloc[0]["train/loss"]) == 2.0
    assert float(wide.iloc[0]["eval/loss"]) == 2.5


def test_scan_history_batch_skips_after_retry_exhaustion():
    class _AlwaysFailRun:
        id = "run-fail"

        def scan_history(self, keys):
            raise RuntimeError(f"boom: {keys}")

    rows = _scan_history_batch(_AlwaysFailRun(), ("grad/norm/total",), attempts=2, backoff_seconds=0.0)
    assert rows == []


def test_resolve_phase_weights_handles_legacy_base_offset():
    phase_weights = {"phase_0": {"starcoder": 0.2}, "phase_1": {"starcoder": 0.3}}
    resolved = _resolve_phase_weights(
        map_from_configs={("exp", 42): phase_weights},
        map_from_csv={},
        source_experiment="exp",
        local_run_id=90042,
        wandb_run_id="run-id",
    )
    assert resolved == phase_weights


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


def test_policy_model_path_falls_back_to_artifact_local_full_models(tmp_path):
    artifact_dir = tmp_path / "bundle"
    model_dir = artifact_dir / "full_models" / "outcome_planner"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "planner.joblib"
    model_path.write_bytes(b"planner")
    artifact_path = artifact_dir / "selected_policy_artifact.json"
    save_policy_artifact(
        artifact_path,
        PolicyArtifactV2(
            kind="sklearn_outcome_planner_v2",
            objective_metric="eval/loss",
            state_keys=("decision_index",),
            action_low=0.05,
            action_high=0.95,
            action_values=[0.05, 0.5, 0.95],
            state_mean=[0.0],
            state_std=[1.0],
            reward_mean=0.0,
            reward_std=1.0,
            model_path="/tmp/moved/planner.joblib",
            metadata={"support_threshold": 0.02},
        ),
    )

    resolved = eval_runner._resolve_policy_model_path(str(artifact_path), "/tmp/moved/planner.joblib")
    assert resolved == model_path.resolve()


def test_outcome_planner_predicts_supported_highest_q_action(tmp_path, monkeypatch):
    class _FakeBehaviorPolicy:
        def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
            return np.asarray(
                [
                    [0.01, 0.49, 0.50],
                    [0.01, 0.49, 0.50],
                    [0.01, 0.49, 0.50],
                ],
                dtype=np.float32,
            )

    class _FakeQModel:
        def predict(self, inputs: np.ndarray) -> np.ndarray:
            del inputs
            return np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

    artifact_dir = tmp_path / "planner_eval"
    artifact_dir.mkdir(parents=True)
    artifact_path = artifact_dir / "selected_policy_artifact.json"
    save_policy_artifact(
        artifact_path,
        PolicyArtifactV2(
            kind="sklearn_outcome_planner_v2",
            objective_metric="eval/loss",
            state_keys=("decision_index", "last_obj_bpb"),
            action_low=0.05,
            action_high=0.95,
            action_values=[0.1, 0.2, 0.3],
            state_mean=[0.0, 1.0],
            state_std=[1.0, 1.0],
            reward_mean=0.0,
            reward_std=1.0,
            model_path="/tmp/elsewhere/planner.joblib",
            metadata={"support_threshold": 0.02},
        ),
    )
    fallback_model = artifact_dir / "full_models" / "outcome_planner" / "planner.joblib"
    fallback_model.parent.mkdir(parents=True)
    fallback_model.write_bytes(b"planner")

    monkeypatch.setattr(
        eval_runner.joblib,
        "load",
        lambda _: {
            "feature_keys": ("decision_index", "last_obj_bpb"),
            "action_grid": [0.1, 0.2, 0.3],
            "q_models": {1: _FakeQModel()},
            "behavior_policy": _FakeBehaviorPolicy(),
            "support_threshold": 0.02,
        },
    )
    eval_runner._load_policy_artifact_cached.cache_clear()
    eval_runner._load_outcome_planner_bundle.cache_clear()

    action = eval_runner._policy_predict_action(
        str(artifact_path),
        {"decision_index": 1.0, "last_obj_bpb": 0.9},
    )
    assert math.isclose(action, 0.3, rel_tol=0.0, abs_tol=1e-6)


def test_outcome_planner_legacy_reward_bundle_uses_min_score(tmp_path, monkeypatch):
    class _FakeBehaviorPolicy:
        def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
            return np.asarray(
                [
                    [0.4, 0.3, 0.3],
                    [0.4, 0.3, 0.3],
                    [0.4, 0.3, 0.3],
                ],
                dtype=np.float32,
            )

    class _FakeRewardBundle:
        def blended_stage_scores(self, frame: pd.DataFrame, action_values: np.ndarray) -> np.ndarray:
            del frame, action_values
            return np.asarray([0.8, 0.1, 0.4], dtype=np.float32)

    artifact_dir = tmp_path / "planner_eval_legacy"
    artifact_dir.mkdir(parents=True)
    artifact_path = artifact_dir / "selected_policy_artifact.json"
    save_policy_artifact(
        artifact_path,
        PolicyArtifactV2(
            kind="sklearn_outcome_planner_v2",
            objective_metric="eval/loss",
            state_keys=("decision_index", "last_obj_bpb"),
            action_low=0.05,
            action_high=0.95,
            action_values=[0.1, 0.2, 0.3],
            state_mean=[0.0, 1.0],
            state_std=[1.0, 1.0],
            reward_mean=0.0,
            reward_std=1.0,
            model_path="/tmp/elsewhere/planner.joblib",
            metadata={"support_threshold": 0.02},
        ),
    )
    fallback_model = artifact_dir / "full_models" / "outcome_planner" / "planner.joblib"
    fallback_model.parent.mkdir(parents=True)
    fallback_model.write_bytes(b"planner")

    monkeypatch.setattr(
        eval_runner.joblib,
        "load",
        lambda _: {
            "feature_keys": ("decision_index", "last_obj_bpb"),
            "action_grid": [0.1, 0.2, 0.3],
            "reward_models": _FakeRewardBundle(),
            "behavior_policy": _FakeBehaviorPolicy(),
            "support_threshold": 0.02,
        },
    )
    eval_runner._load_policy_artifact_cached.cache_clear()
    eval_runner._load_outcome_planner_bundle.cache_clear()

    action = eval_runner._policy_predict_action(
        str(artifact_path),
        {"decision_index": 1.0, "last_obj_bpb": 0.9},
    )
    assert math.isclose(action, 0.2, rel_tol=0.0, abs_tol=1e-6)


def test_history_from_completed_run_uses_batched_scan_history(monkeypatch):
    calls: list[tuple[int, int, float]] = []

    def _fake_collect(run, metric_keys, history_batch_size, retry_attempts, backoff_seconds):
        del run
        calls.append((history_batch_size, retry_attempts, backoff_seconds))
        assert metric_keys == ("train/loss", "eval/loss")
        return [
            {
                "wandb_run_id": "run-a",
                "source_experiment": "exp",
                "local_run_id": 1,
                "run_name": "exp/run-a",
                "step": 0,
                "total_tokens": 0.0,
                "metric_key": "train/loss",
                "metric_value": 3.0,
                "_scan_index": 0,
            },
            {
                "wandb_run_id": "run-a",
                "source_experiment": "exp",
                "local_run_id": 1,
                "run_name": "exp/run-a",
                "step": 0,
                "total_tokens": 0.0,
                "metric_key": "eval/loss",
                "metric_value": 3.5,
                "_scan_index": 1,
            },
        ]

    monkeypatch.setattr(eval_runner, "collect_history_long_rows_batched", _fake_collect)
    history = eval_runner._history_from_completed_run(object(), ("train/loss", "eval/loss"))
    assert calls == [(eval_runner.ONLINE_HISTORY_BATCH_SIZE, 3, 2.0)]
    assert history["train/loss"].tolist() == [3.0]
    assert history["eval/loss"].tolist() == [3.5]


def test_initial_policy_state_uses_decision_zero_defaults(tmp_path):
    artifact_dir = tmp_path / "planner_eval"
    artifact_dir.mkdir(parents=True)
    defaults_path = artifact_dir / "decision_state_defaults.json"
    defaults_path.write_text(
        json.dumps(
            {
                "0": {
                    "decision_index": 0.0,
                    "num_phases_total": 3.0,
                    "remaining_decisions": 2.0,
                    "budget_frac_consumed": 0.0,
                    "budget_frac_remaining": 1.0,
                    "last_obj_bpb": 1.23,
                    "prev_action_starcoder": 0.5,
                }
            }
        )
    )
    artifact_path = artifact_dir / "selected_policy_artifact.json"
    save_policy_artifact(
        artifact_path,
        PolicyArtifactV2(
            kind="d3rlpy_discrete_bc_v2",
            objective_metric="eval/loss",
            state_keys=(
                "decision_index",
                "num_phases_total",
                "remaining_decisions",
                "budget_frac_consumed",
                "budget_frac_remaining",
                "last_obj_bpb",
                "prev_action_starcoder",
            ),
            action_low=0.05,
            action_high=0.95,
            action_values=[0.05, 0.5, 0.95],
            state_mean=[9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            state_std=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            reward_mean=0.0,
            reward_std=1.0,
            model_path="model.d3",
            aux_paths={"decision_state_defaults": str(defaults_path)},
        ),
    )
    eval_runner._load_policy_artifact_cached.cache_clear()
    eval_runner._load_decision_state_defaults.cache_clear()

    state = eval_runner._initial_policy_state(str(artifact_path))
    assert state["decision_index"] == 0.0
    assert state["last_obj_bpb"] == 1.23
    assert state["remaining_decisions"] == 2.0


def test_initial_policy_state_prefers_family_specific_defaults(tmp_path):
    artifact_dir = tmp_path / "family_defaults_eval"
    artifact_dir.mkdir(parents=True)
    defaults_path = artifact_dir / "decision_state_defaults.json"
    defaults_path.write_text(json.dumps({"0": {"last_obj_bpb": 9.0}}))
    family_defaults_path = artifact_dir / "decision_state_defaults_by_family.json"
    family_defaults_path.write_text(
        json.dumps(
            {
                "two_phase_starcoder": {
                    "0": {
                        "decision_index": 0.0,
                        "num_phases_total": 2.0,
                        "remaining_decisions": 1.0,
                        "budget_frac_consumed": 0.0,
                        "budget_frac_remaining": 1.0,
                        "last_obj_bpb": 1.5,
                        "prev_action_starcoder": 0.5,
                    }
                }
            }
        )
    )
    artifact_path = artifact_dir / "selected_policy_artifact.json"
    save_policy_artifact(
        artifact_path,
        PolicyArtifactV2(
            kind="d3rlpy_discrete_bc_v2",
            objective_metric="eval/loss",
            state_keys=(
                "decision_index",
                "num_phases_total",
                "remaining_decisions",
                "budget_frac_consumed",
                "budget_frac_remaining",
                "last_obj_bpb",
                "prev_action_starcoder",
            ),
            action_low=0.05,
            action_high=0.95,
            action_values=[0.05, 0.5, 0.95],
            state_mean=[9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            state_std=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            reward_mean=0.0,
            reward_std=1.0,
            model_path="model.d3",
            aux_paths={
                "decision_state_defaults": str(defaults_path),
                "decision_state_defaults_by_family": str(family_defaults_path),
            },
        ),
    )
    eval_runner._load_policy_artifact_cached.cache_clear()
    eval_runner._load_decision_state_defaults.cache_clear()
    eval_runner._load_family_decision_state_defaults.cache_clear()

    state = eval_runner._initial_policy_state(
        str(artifact_path),
        num_phases_total=2,
        total_steps=3814,
        phase_end_steps=(1904,),
        run_family="two_phase_starcoder",
    )
    assert state["decision_index"] == 0.0
    assert state["last_obj_bpb"] == 1.5
    assert state["remaining_decisions"] == 1.0


def test_two_phase_hypothetical_state_uses_family_defaults_and_prior_action(tmp_path):
    artifact_dir = tmp_path / "two_phase_eval"
    artifact_dir.mkdir(parents=True)
    family_defaults_path = artifact_dir / "decision_state_defaults_by_family.json"
    family_defaults_path.write_text(
        json.dumps(
            {
                "two_phase_starcoder": {
                    "1": {
                        "decision_index": 1.0,
                        "num_phases_total": 2.0,
                        "remaining_decisions": 0.0,
                        "budget_frac_consumed": 1904.0 / 3814.0,
                        "budget_frac_remaining": 1.0 - (1904.0 / 3814.0),
                        "last_obj_bpb": 0.9,
                        "steps_since_last_eval_frac": 0.2,
                        "global_step": 1904.0,
                        "prev_action_starcoder": 0.05,
                        "cumulative_starcoder_exposure": 0.05,
                        "delta_prev_action": 0.0,
                    }
                }
            }
        )
    )
    artifact_path = artifact_dir / "selected_policy_artifact.json"
    save_policy_artifact(
        artifact_path,
        PolicyArtifactV2(
            kind="d3rlpy_discrete_bc_v2",
            objective_metric="eval/loss",
            state_keys=(
                "decision_index",
                "num_phases_total",
                "remaining_decisions",
                "budget_frac_consumed",
                "budget_frac_remaining",
                "last_obj_bpb",
                "steps_since_last_eval_frac",
                "global_step",
                "prev_action_starcoder",
                "cumulative_starcoder_exposure",
                "delta_prev_action",
            ),
            action_low=0.05,
            action_high=0.95,
            action_values=[0.05, 0.5, 0.95],
            state_mean=[0.0] * 11,
            state_std=[1.0] * 11,
            reward_mean=0.0,
            reward_std=1.0,
            model_path="model.d3",
            aux_paths={"decision_state_defaults_by_family": str(family_defaults_path)},
        ),
    )
    eval_runner._load_policy_artifact_cached.cache_clear()
    eval_runner._load_family_decision_state_defaults.cache_clear()

    state = two_phase_eval._hypothetical_state_for_decision(
        artifact_path=str(artifact_path),
        config=two_phase_eval.EvaluateTwoPhaseConfig(
            policy_artifact_path=str(artifact_path),
            output_dir=str(tmp_path / "out"),
        ),
        decision_index=1,
        prior_actions=[0.23],
    )
    assert math.isclose(state["prev_action_starcoder"], 0.23, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(state["cumulative_starcoder_exposure"], 0.23, rel_tol=0.0, abs_tol=1e-6)
    assert state["remaining_decisions"] == 0.0
    assert state["last_obj_bpb"] == 0.9
    assert state["steps_since_last_eval_frac"] == 0.2


def test_two_phase_inspect_policy_writes_actions(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "two_phase_inspect"
    artifact_dir.mkdir(parents=True)
    family_defaults_path = artifact_dir / "decision_state_defaults_by_family.json"
    family_defaults_path.write_text(
        json.dumps(
            {
                "two_phase_starcoder": {
                    "0": {
                        "decision_index": 0.0,
                        "num_phases_total": 2.0,
                        "remaining_decisions": 1.0,
                        "budget_frac_consumed": 0.0,
                        "budget_frac_remaining": 1.0,
                        "last_obj_bpb": 5.0,
                        "global_step": 0.0,
                        "prev_action_starcoder": 0.5,
                        "cumulative_starcoder_exposure": 0.0,
                        "delta_prev_action": 0.0,
                    },
                    "1": {
                        "decision_index": 1.0,
                        "num_phases_total": 2.0,
                        "remaining_decisions": 0.0,
                        "budget_frac_consumed": 1904.0 / 3814.0,
                        "budget_frac_remaining": 1.0 - (1904.0 / 3814.0),
                        "last_obj_bpb": 1.0,
                        "global_step": 1904.0,
                        "prev_action_starcoder": 0.05,
                        "cumulative_starcoder_exposure": 0.05,
                        "delta_prev_action": 0.0,
                    },
                }
            }
        )
    )
    artifact_path = artifact_dir / "selected_policy_artifact.json"
    save_policy_artifact(
        artifact_path,
        PolicyArtifactV2(
            kind="d3rlpy_discrete_bc_v2",
            objective_metric="eval/loss",
            state_keys=(
                "decision_index",
                "num_phases_total",
                "remaining_decisions",
                "budget_frac_consumed",
                "budget_frac_remaining",
                "last_obj_bpb",
                "global_step",
                "prev_action_starcoder",
                "cumulative_starcoder_exposure",
                "delta_prev_action",
            ),
            action_low=0.05,
            action_high=0.95,
            action_values=[0.05, 0.5, 0.95],
            state_mean=[0.0] * 10,
            state_std=[1.0] * 10,
            reward_mean=0.0,
            reward_std=1.0,
            model_path="model.d3",
            aux_paths={"decision_state_defaults_by_family": str(family_defaults_path)},
        ),
    )
    eval_runner._load_policy_artifact_cached.cache_clear()
    eval_runner._load_family_decision_state_defaults.cache_clear()

    def _fake_predict(_artifact_path, state, device="cpu"):
        return 0.05 if state["remaining_decisions"] == 1.0 else 0.23

    monkeypatch.setattr(eval_runner, "_policy_predict_action", _fake_predict)
    config = two_phase_eval.EvaluateTwoPhaseConfig(
        policy_artifact_path=str(artifact_path),
        output_dir=str(tmp_path / "inspect_out"),
        inspect_only=True,
    )
    result = two_phase_eval.inspect_policy(config)

    assert result.iloc[0]["phase_0_starcoder"] == 0.05
    assert result.iloc[0]["phase_1_starcoder"] == 0.23
    assert (tmp_path / "inspect_out" / "policy_inspection_summary.json").exists()


def test_torch_discrete_iql_policy_prediction(tmp_path):
    artifact_dir = tmp_path / "iql_eval"
    artifact_dir.mkdir(parents=True)
    model_path = artifact_dir / "model.pt"
    policy_net = eval_runner._EvalIQLMLP(input_dim=2, output_dim=3, hidden_size=4)
    with torch.no_grad():
        for parameter in policy_net.parameters():
            parameter.zero_()
        last_linear = policy_net.net[-1]
        last_linear.bias.copy_(torch.tensor([0.0, 2.0, 1.0], dtype=torch.float32))
    torch.save(
        {
            "policy_state_dict": policy_net.state_dict(),
            "q_state_dict": policy_net.state_dict(),
            "v_state_dict": policy_net.state_dict(),
            "input_dim": 2,
            "action_dim": 3,
            "hidden_size": 4,
        },
        model_path,
    )
    artifact_path = artifact_dir / "selected_policy_artifact.json"
    save_policy_artifact(
        artifact_path,
        PolicyArtifactV2(
            kind="torch_discrete_iql_v2",
            objective_metric="eval/loss",
            state_keys=("decision_index", "last_obj_bpb"),
            action_low=0.05,
            action_high=0.95,
            action_values=[0.05, 0.5, 0.95],
            state_mean=[0.0, 0.0],
            state_std=[1.0, 1.0],
            reward_mean=0.0,
            reward_std=1.0,
            model_path=str(model_path),
        ),
    )
    eval_runner._load_policy_artifact_cached.cache_clear()
    eval_runner._load_torch_discrete_iql_policy.cache_clear()

    action = eval_runner._policy_predict_action(
        str(artifact_path),
        {"decision_index": 0.0, "last_obj_bpb": 1.0},
    )
    assert math.isclose(action, 0.5, rel_tol=0.0, abs_tol=1e-6)


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
    from experiments.domain_phase_mix.offline_rl.evaluate_policy_three_phase_starcoder import build_phase_train_plan

    p0 = build_phase_train_plan(0, checkpoint_path=None, phase_end_steps=(1888, 3824), total_steps=5722)
    p1 = build_phase_train_plan(1, checkpoint_path="gs://x/checkpoint-1", phase_end_steps=(1888, 3824), total_steps=5722)
    p2 = build_phase_train_plan(2, checkpoint_path="gs://x/checkpoint-2", phase_end_steps=(1888, 3824), total_steps=5722)
    assert p0.initialize_from_checkpoint_path is None
    assert p1.initialize_from_checkpoint_path is not None
    assert p2.initialize_from_checkpoint_path is not None
    assert p1.reset_data_loader_on_init is False
    assert p2.reset_data_loader_on_init is False
    assert p1.cumulative_steps == 3824
    assert p2.cumulative_steps == 5722


def test_rollout_defaults_match_native_aligned_boundaries():
    three_cfg = eval_runner.EvaluateConfig(
        policy_artifact_path="x/policy_artifact.json",
        output_dir="tmp",
        n_replicates=1,
    )
    two_cfg = two_phase_eval.EvaluateTwoPhaseConfig(
        policy_artifact_path="x/policy_artifact.json",
        output_dir="tmp",
        n_replicates=1,
    )

    assert three_cfg.phase_end_steps == (1888, 3824)
    assert two_cfg.phase_end_steps == (1904,)


def test_simulated_epoching_train_respects_experiment_budget_override(monkeypatch):
    from experiments import defaults as defaults_module

    @dataclass(frozen=True)
    class _FakeDataConfig:
        target_budget: int | None = None
        experiment_budget: int | None = None
        simulated_epoch_subset_seed: int | None = None

    class _FakeTrainConfig:
        train_batch_size = 4
        num_train_steps = 10
        train_seq_len = 16

    captured = {}

    monkeypatch.setattr(
        defaults_module,
        "_prepare_data_config",
        lambda tokenized, use_default_validation: _FakeDataConfig(),
    )
    monkeypatch.setattr(defaults_module, "_validate_train_length", lambda train_seq_len, model_config: 32)

    def _fake_default_train(
        name,
        tokenized,
        model_config,
        train_config,
        tags,
        use_default_validation,
        eval_harness_tasks,
        wandb_name=None,
        eval_datasets_cache_path=None,
    ):
        captured["experiment_budget"] = tokenized.experiment_budget
        captured["target_budget"] = tokenized.target_budget
        return tokenized

    monkeypatch.setattr(defaults_module, "default_train", _fake_default_train)

    defaults_module.simulated_epoching_train(
        name="x",
        tokenized="unused",
        model_config=object(),
        train_config=_FakeTrainConfig(),
        target_budget=999,
        experiment_budget_override=123456,
    )

    assert captured["experiment_budget"] == 123456
    assert captured["target_budget"] == 999


def test_three_phase_rollout_builder_keeps_native_budget_and_schedule(monkeypatch):
    from experiments.domain_phase_mix.config import PhaseSchedule

    @dataclass(frozen=True)
    class _FakeOptimizerConfig:
        lr_schedule: object | None = None

    class _FakeStep:
        def __init__(self):
            self.output_path: str | None = None

        def with_output_path(self, output_path: str):
            self.output_path = output_path
            return self

    class _FakeExperiment:
        def __init__(self):
            self.phase_schedule = PhaseSchedule.from_boundaries([0.33, 0.67], names=["phase_0", "phase_1", "phase_2"])
            self.batch_size = 128
            self.mixture_block_size = 2048
            self.num_train_steps = 5722
            self.experiment_budget = 1499987968
            self.steps_per_eval = 1000
            self.optimizer_config = _FakeOptimizerConfig()
            self.resources = None
            self.eval_datasets_cache_path = None
            self.calls = []

        def create_training_step(self, **kwargs):
            self.calls.append(kwargs)
            return _FakeStep()

    fake_experiment = _FakeExperiment()
    monkeypatch.setattr(
        eval_runner,
        "create_three_phase_experiment",
        lambda name, eval_datasets_cache_path: fake_experiment,
    )

    step = eval_runner._build_training_step(
        run_namespace="ns",
        phase_plan=eval_runner.PhaseTrainPlan(
            phase_index=1,
            cumulative_steps=3824,
            initialize_from_checkpoint_path="gs://x/ckpt",
            reset_data_loader_on_init=False,
        ),
        actions_so_far=[0.455, 0.32],
        override_output_path="out/path",
        run_id=7,
        data_seed=11,
        global_total_steps=5722,
        tpu_type="v5p-8",
        eval_datasets_cache_path="gs://eval-cache",
    )

    call = fake_experiment.calls[0]
    assert step.output_path == "out/path"
    assert call["num_train_steps"] == 3824
    assert call["experiment_budget_override"] == 1499987968
    assert call["steps_per_eval"] == 1000
    assert call["initialize_from_checkpoint_path"] == "gs://x/ckpt"
    assert call["reset_data_loader_on_init"] is False
    assert call["data_seed"] == 11
    assert call["weight_config"].phase_weights["phase_0"]["starcoder"] == 0.455
    assert call["weight_config"].phase_weights["phase_1"]["starcoder"] == 0.32
    assert call["weight_config"].phase_weights["phase_2"]["starcoder"] == 0.32


def test_two_phase_rollout_builder_keeps_native_budget_and_schedule(monkeypatch):
    from experiments.domain_phase_mix.config import PhaseSchedule

    @dataclass(frozen=True)
    class _FakeOptimizerConfig:
        lr_schedule: object | None = None

    class _FakeStep:
        def __init__(self):
            self.output_path: str | None = None

        def with_output_path(self, output_path: str):
            self.output_path = output_path
            return self

    class _FakeExperiment:
        def __init__(self):
            self.phase_schedule = PhaseSchedule.from_boundaries([0.5], names=["phase_0", "phase_1"])
            self.batch_size = 128
            self.mixture_block_size = 2048
            self.num_train_steps = 3814
            self.experiment_budget = 999817216
            self.steps_per_eval = 1000
            self.optimizer_config = _FakeOptimizerConfig()
            self.resources = None
            self.eval_datasets_cache_path = None
            self.calls = []

        def create_training_step(self, **kwargs):
            self.calls.append(kwargs)
            return _FakeStep()

    fake_experiment = _FakeExperiment()
    monkeypatch.setattr(two_phase_eval, "create_two_phase_experiment", lambda name: fake_experiment)

    step = two_phase_eval._build_training_step(
        run_namespace="ns",
        phase_plan=eval_runner.PhaseTrainPlan(
            phase_index=0,
            cumulative_steps=1904,
            initialize_from_checkpoint_path=None,
            reset_data_loader_on_init=True,
        ),
        actions_so_far=[0.05],
        override_output_path="out/path",
        run_id=3,
        data_seed=17,
        global_total_steps=3814,
        tpu_type="v5p-8",
        eval_datasets_cache_path="gs://eval-cache",
    )

    call = fake_experiment.calls[0]
    assert step.output_path == "out/path"
    assert call["num_train_steps"] == 1904
    assert call["experiment_budget_override"] == 999817216
    assert call["steps_per_eval"] == 1000
    assert call["data_seed"] == 17
    assert call["weight_config"].phase_weights["phase_0"]["starcoder"] == 0.05
    assert call["weight_config"].phase_weights["phase_1"]["starcoder"] == 0.05


def test_three_phase_rollout_reuses_one_data_seed_per_replicate(monkeypatch, tmp_path):
    captured_calls: list[dict[str, int]] = []

    @dataclass(frozen=True)
    class _FakeArtifact:
        state_keys: tuple[str, ...] = ()

    def _fake_build_training_step(**kwargs):
        captured_calls.append(
            {
                "run_id": kwargs["run_id"],
                "data_seed": kwargs["data_seed"],
                "phase_index": kwargs["phase_plan"].phase_index,
            }
        )
        return object()

    monkeypatch.setattr(eval_runner, "_resolve_prefix", lambda prefix: prefix or str(tmp_path / "prefix"))
    monkeypatch.setattr(eval_runner, "_load_policy_artifact_cached", lambda _: _FakeArtifact())
    monkeypatch.setattr(eval_runner, "_initial_policy_state", lambda *args, **kwargs: {})
    monkeypatch.setattr(eval_runner, "_policy_predict_action", lambda *args, **kwargs: 0.32)
    monkeypatch.setattr(eval_runner, "_build_training_step", _fake_build_training_step)
    monkeypatch.setattr(eval_runner, "_artifact_state_defaults", lambda artifact: {})

    results = eval_runner.evaluate_policy(
        eval_runner.EvaluateConfig(
            policy_artifact_path="x/policy_artifact.json",
            output_dir=str(tmp_path / "out"),
            n_replicates=1,
            dry_run=True,
        )
    )

    assert results["phase_0_starcoder"].tolist() == [0.32]
    assert [call["phase_index"] for call in captured_calls] == [0, 1, 2]
    assert [call["run_id"] for call in captured_calls] == [0, 1, 2]
    assert [call["data_seed"] for call in captured_calls] == [0, 0, 0]


def test_two_phase_rollout_reuses_one_data_seed_per_replicate(monkeypatch, tmp_path):
    captured_calls: list[dict[str, int]] = []

    def _fake_build_training_step(**kwargs):
        captured_calls.append(
            {
                "run_id": kwargs["run_id"],
                "data_seed": kwargs["data_seed"],
                "phase_index": kwargs["phase_plan"].phase_index,
            }
        )
        return object()

    monkeypatch.setattr(two_phase_eval.shared, "_resolve_prefix", lambda prefix: prefix or str(tmp_path / "prefix"))
    monkeypatch.setattr(two_phase_eval.shared, "_initial_policy_state", lambda *args, **kwargs: {})
    monkeypatch.setattr(two_phase_eval.shared, "_policy_predict_action", lambda *args, **kwargs: 0.05)
    monkeypatch.setattr(two_phase_eval, "_build_training_step", _fake_build_training_step)
    monkeypatch.setattr(
        two_phase_eval,
        "_hypothetical_state_for_decision",
        lambda **kwargs: {},
    )

    results = two_phase_eval.evaluate_policy(
        two_phase_eval.EvaluateTwoPhaseConfig(
            policy_artifact_path="x/policy_artifact.json",
            output_dir=str(tmp_path / "out"),
            n_replicates=1,
            dry_run=True,
        )
    )

    assert results["phase_0_starcoder"].tolist() == [0.05]
    assert [call["phase_index"] for call in captured_calls] == [0, 1]
    assert [call["run_id"] for call in captured_calls] == [0, 1]
    assert [call["data_seed"] for call in captured_calls] == [0, 0]


def test_local_fallback_preflight(monkeypatch):
    from experiments.domain_phase_mix.offline_rl.evaluate_policy_three_phase_starcoder import (
        EvaluateConfig,
        _should_force_local_dry_run,
    )

    module_path = "experiments.domain_phase_mix.offline_rl.evaluate_policy_three_phase_starcoder.marin_region"
    monkeypatch.setattr(module_path, lambda: None)
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


def test_action_grid_roundtrip_is_bounded():
    grid = build_action_grid(0.05, 0.95, 21)
    idx, value = discretize_action(0.271, grid)
    assert 0 <= idx < 21
    assert 0.05 <= value <= 0.95
    assert np.isclose(value, grid[idx])


def test_build_pooled_transition_dataset_stage_tables(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    runs = pd.DataFrame(
        [
            {
                "wandb_run_id": "two-a",
                "run_name": "two-a",
                "source_experiment": "two_phase_starcoder_4",
                "run_family": "two_phase_starcoder",
                "local_run_id": 1,
                "num_phases_total": 2,
                "total_steps": 3814,
                "phase_boundaries_json": "[1904]",
                "phase_0_starcoder": 0.2,
                "phase_1_starcoder": 0.3,
                "eval/paloma/dolma_100_programing_languages/bpb": 0.82,
            },
            {
                "wandb_run_id": "three-a",
                "run_name": "three-a",
                "source_experiment": "three_phase_starcoder_1",
                "run_family": "three_phase_starcoder",
                "local_run_id": 2,
                "num_phases_total": 3,
                "total_steps": 5722,
                "phase_boundaries_json": "[1888, 3833]",
                "phase_0_starcoder": 0.1,
                "phase_1_starcoder": 0.4,
                "phase_2_starcoder": 0.7,
                "eval/paloma/dolma_100_programing_languages/bpb": 0.76,
            },
        ]
    )
    history = pd.DataFrame(
        [
            {
                "wandb_run_id": "two-a",
                "step": 0,
                "train/loss": 2.5,
                "eval/loss": np.nan,
                "eval/paloma/dolma_100_programing_languages/bpb": np.nan,
                "optim/learning_rate": 0.001,
                "optim/adam_lr": 0.001,
                "throughput/examples_per_second": 10.0,
                "throughput/tokens_per_second": 100.0,
                "grad/norm/total": 1.0,
                "total_tokens": 0.0,
            },
            {
                "wandb_run_id": "two-a",
                "step": 1904,
                "train/loss": 2.0,
                "eval/loss": 2.1,
                "eval/paloma/dolma_100_programing_languages/bpb": 0.9,
                "optim/learning_rate": 0.0008,
                "optim/adam_lr": 0.0008,
                "throughput/examples_per_second": 10.5,
                "throughput/tokens_per_second": 110.0,
                "grad/norm/total": 0.9,
                "total_tokens": 1.0,
            },
            {
                "wandb_run_id": "two-a",
                "step": 3814,
                "train/loss": 1.8,
                "eval/loss": 1.9,
                "eval/paloma/dolma_100_programing_languages/bpb": 0.82,
                "optim/learning_rate": 0.0004,
                "optim/adam_lr": 0.0004,
                "throughput/examples_per_second": 11.0,
                "throughput/tokens_per_second": 120.0,
                "grad/norm/total": 0.8,
                "total_tokens": 2.0,
            },
            {
                "wandb_run_id": "three-a",
                "step": 0,
                "train/loss": 2.6,
                "eval/loss": np.nan,
                "eval/paloma/dolma_100_programing_languages/bpb": np.nan,
                "optim/learning_rate": 0.0011,
                "optim/adam_lr": 0.0011,
                "throughput/examples_per_second": 9.0,
                "throughput/tokens_per_second": 90.0,
                "grad/norm/total": 1.2,
                "total_tokens": 0.0,
            },
            {
                "wandb_run_id": "three-a",
                "step": 1888,
                "train/loss": 2.2,
                "eval/loss": 2.3,
                "eval/paloma/dolma_100_programing_languages/bpb": 1.0,
                "optim/learning_rate": 0.0009,
                "optim/adam_lr": 0.0009,
                "throughput/examples_per_second": 9.5,
                "throughput/tokens_per_second": 95.0,
                "grad/norm/total": 1.0,
                "total_tokens": 1.0,
            },
            {
                "wandb_run_id": "three-a",
                "step": 3833,
                "train/loss": 2.0,
                "eval/loss": 2.05,
                "eval/paloma/dolma_100_programing_languages/bpb": 0.9,
                "optim/learning_rate": 0.0007,
                "optim/adam_lr": 0.0007,
                "throughput/examples_per_second": 9.8,
                "throughput/tokens_per_second": 98.0,
                "grad/norm/total": 0.95,
                "total_tokens": 2.0,
            },
            {
                "wandb_run_id": "three-a",
                "step": 5722,
                "train/loss": 1.7,
                "eval/loss": 1.8,
                "eval/paloma/dolma_100_programing_languages/bpb": 0.76,
                "optim/learning_rate": 0.0003,
                "optim/adam_lr": 0.0003,
                "throughput/examples_per_second": 10.1,
                "throughput/tokens_per_second": 101.0,
                "grad/norm/total": 0.7,
                "total_tokens": 3.0,
            },
        ]
    )

    runs.to_parquet(input_dir / "runs.parquet", index=False)
    history.to_parquet(input_dir / "history_wide.parquet", index=False)

    decisions, _feature_manifest, manifest = build_pooled_transition_dataset(
        BuildPooledTransitionConfig(
            input_dir=str(input_dir),
            output_dir=str(tmp_path / "output"),
            dataset_config=PooledDatasetConfig(n_cv_folds=2),
        )
    )
    assert manifest["n_runs"] == 2
    assert set(decisions[decisions["decision_index"] == 0]["run_family"]) == {
        "two_phase_starcoder",
        "three_phase_starcoder",
    }
    assert set(decisions[decisions["decision_index"] == 1]["run_family"]) == {
        "two_phase_starcoder",
        "three_phase_starcoder",
    }
    assert set(decisions[decisions["decision_index"] == 2]["run_family"]) == {"three_phase_starcoder"}
    assert "decision_index" in {
        row["feature_name"] for row in json.loads((tmp_path / "output" / "feature_manifest.json").read_text())["rows"]
    }
    assert {"decision_0.parquet", "decision_1.parquet", "decision_2.parquet"} <= {
        path.name for path in (tmp_path / "output").iterdir()
    }


class _ArrayPolicy:
    def __init__(self, name: str, action_idx: int):
        self.name = name
        self.action_idx = action_idx

    def predict_action_indices(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.action_idx, dtype=np.int64)


class _StateAwarePolicy:
    def __init__(self, name: str):
        self.name = name

    def predict_action_indices(self, frame: pd.DataFrame) -> np.ndarray:
        preferred = frame["decision_feature"].to_numpy(dtype=np.float32)
        return np.where(preferred > 1.5, 2, 1).astype(np.int64)


def test_ope_prefers_better_policy():
    action_grid = np.asarray([0.05, 0.5, 0.95], dtype=np.float32)
    rows = []
    for episode_id in range(12):
        preferred = 2 if episode_id % 2 == 0 else 1
        for decision_index in range(3):
            for action_idx, action_value in enumerate(action_grid):
                reward = 1.0 if action_idx == preferred else 0.0
                rows.append(
                    {
                        "wandb_run_id": f"run-{episode_id}-{action_idx}-{decision_index}",
                        "episode_id": f"episode-{episode_id}-{action_idx}",
                        "step_in_episode": 0,
                        "decision_index": decision_index,
                        "run_family": "three_phase_starcoder",
                        "action_idx": action_idx,
                        "action_grid_value": float(action_value),
                        "action_starcoder": float(action_value),
                        "done": True,
                        "reward_dense_raw": reward,
                        "final_objective": -reward,
                        "decision_feature": float(preferred),
                        "num_phases_total": 3.0,
                        "remaining_decisions": float(2 - decision_index),
                        "budget_frac_consumed": float(decision_index) / 3.0,
                        "budget_frac_remaining": 1.0 - float(decision_index) / 3.0,
                    }
                )
    frame = pd.DataFrame(rows)
    feature_keys = (
        "decision_feature",
        "decision_index",
        "num_phases_total",
        "remaining_decisions",
        "budget_frac_consumed",
        "budget_frac_remaining",
    )
    train_episode_ids = tuple(f"episode-{idx}-" for idx in range(6))
    val_episode_ids = tuple(f"episode-{idx}-" for idx in range(6, 12))
    train = frame[frame["episode_id"].str.startswith(train_episode_ids)].reset_index(drop=True)
    val = frame[frame["episode_id"].str.startswith(val_episode_ids)].reset_index(drop=True)
    behavior = fit_behavior_policy(train, feature_keys, action_count=3, random_state=0)
    reward_models = fit_reward_models(
        train,
        feature_keys,
        random_state=0,
        final_model_weight=1.0,
        reward_bonus_weight=0.0,
    )
    good_policy = _StateAwarePolicy("good")
    bad_policy = _ArrayPolicy("bad", 0)

    fqe_good = fit_fqe_model(train, feature_keys, good_policy, action_grid, gamma=1.0, random_state=0, iterations=8)
    fqe_bad = fit_fqe_model(train, feature_keys, bad_policy, action_grid, gamma=1.0, random_state=0, iterations=8)

    assert direct_method_value(val, good_policy, reward_models) >= direct_method_value(val, bad_policy, reward_models)
    assert initial_state_value(val, good_policy, fqe_good, action_grid) >= initial_state_value(
        val, bad_policy, fqe_bad, action_grid
    )
    assert doubly_robust_estimate(val, good_policy, behavior, fqe_good, action_grid, gamma=1.0) >= (
        doubly_robust_estimate(val, bad_policy, behavior, fqe_bad, action_grid, gamma=1.0)
    )

    diagnostics_df, summary = policy_diagnostics(
        frame=val,
        policy=_ArrayPolicy("collapsed", 0),
        behavior_policy=behavior,
        action_grid=(0.05, 0.5, 0.95),
        support_threshold=0.4,
    )
    assert not diagnostics_df.empty
    assert summary["boundary_rate"] == 1.0


def test_offline_policy_bench_smoke(tmp_path):
    input_dir = tmp_path / "dataset"
    output_dir = tmp_path / "bench"
    input_dir.mkdir()

    action_grid = [0.05, 0.5, 0.95]
    feature_keys = [
        "decision_index",
        "num_phases_total",
        "remaining_decisions",
        "budget_frac_consumed",
        "budget_frac_remaining",
        "last_obj_bpb",
        "prev_action_starcoder",
    ]
    rows = []
    for run_index in range(8):
        run_family = "three_phase_starcoder" if run_index < 4 else "two_phase_starcoder"
        n_decisions = 3 if run_family == "three_phase_starcoder" else 2
        fold = run_index % 2
        preferred = 2 if run_index % 2 == 0 else 1
        final_reward = 0.0
        prev_action = 0.5
        for decision_index in range(n_decisions):
            action_idx = preferred if decision_index == 0 else 1
            action_value = action_grid[action_idx]
            reward = 1.0 if action_idx == preferred else 0.0
            final_reward += reward
            rows.append(
                {
                    "wandb_run_id": f"run-{run_index}",
                    "episode_id": f"run-{run_index}",
                    "step_in_episode": decision_index,
                    "cv_fold": fold,
                    "run_family": run_family,
                    "decision_index": decision_index,
                    "action_idx": action_idx,
                    "action_grid_value": action_value,
                    "action_starcoder": action_value,
                    "done": decision_index == n_decisions - 1,
                    "reward_dense_raw": reward,
                    "final_objective": float(n_decisions - final_reward),
                    "last_obj_bpb": float(1.0 - 0.1 * decision_index),
                    "prev_action_starcoder": prev_action,
                    "num_phases_total": float(n_decisions),
                    "remaining_decisions": float(n_decisions - decision_index - 1),
                    "budget_frac_consumed": float(decision_index) / float(n_decisions),
                    "budget_frac_remaining": 1.0 - float(decision_index) / float(n_decisions),
                }
            )
            prev_action = action_value
    decisions = pd.DataFrame(rows)
    decisions.to_parquet(input_dir / "decisions.parquet", index=False)
    with (input_dir / "feature_manifest.json").open("w") as f:
        json.dump({"selected_feature_keys": feature_keys}, f)
    with (input_dir / "dataset_manifest.json").open("w") as f:
        json.dump({"action_grid": action_grid, "n_cv_folds": 2}, f)

    summary = run_offline_policy_bench(
        OfflinePolicyBenchConfig(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            fqe_iterations=2,
            d3rlpy_steps=2,
            d3rlpy_steps_per_epoch=1,
            discrete_iql_epochs=2,
            discrete_iql_batch_size=8,
            include_continuous_cql=False,
            device="cpu",
        )
    )
    assert (output_dir / "fold_scores.csv").exists()
    assert (output_dir / "policy_summary.csv").exists()
    assert (output_dir / "artifacts" / "discrete_cql.json").exists()
    assert summary["best_dr_method"] is not None
