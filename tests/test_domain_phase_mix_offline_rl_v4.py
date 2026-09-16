# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.offline_rl.build_pooled_dense_policy_dataset_v4 import (
    BuildPooledDenseDatasetV4Config,
    build_pooled_dense_policy_dataset_v4,
)
from experiments.domain_phase_mix.offline_rl.build_three_phase_dense_policy_dataset import SEQUENCE_CHANNELS
from experiments.domain_phase_mix.offline_rl.train_three_phase_policy_bench_v4 import (
    ThreePhasePolicyBenchV4Config,
    run_three_phase_policy_bench_v4,
)


def _small_pooled_dense_inputs(tmp_path: Path) -> Path:
    input_dir = tmp_path / "pooled_dense_input"
    input_dir.mkdir(parents=True)
    runs = pd.DataFrame(
        [
            {
                "wandb_run_id": "three-a",
                "run_name": "three/three-a",
                "source_experiment": "exp-three-a",
                "run_family": "three_phase_starcoder",
                "local_run_id": 1,
                "num_phases_total": 3,
                "total_steps": 90,
                "phase_boundaries_json": json.dumps([30, 60]),
                "eval/paloma/dolma_100_programing_languages/bpb": 0.85,
                "phase_0_starcoder": 0.0,
                "phase_1_starcoder": 0.3,
                "phase_2_starcoder": 0.25,
            },
            {
                "wandb_run_id": "three-b",
                "run_name": "three/three-b",
                "source_experiment": "exp-three-b",
                "run_family": "three_phase_starcoder",
                "local_run_id": 2,
                "num_phases_total": 3,
                "total_steps": 90,
                "phase_boundaries_json": json.dumps([30, 60]),
                "eval/paloma/dolma_100_programing_languages/bpb": 0.88,
                "phase_0_starcoder": 0.05,
                "phase_1_starcoder": 0.35,
                "phase_2_starcoder": 0.2,
            },
            {
                "wandb_run_id": "three-c",
                "run_name": "three/three-c",
                "source_experiment": "exp-three-c",
                "run_family": "three_phase_starcoder",
                "local_run_id": 3,
                "num_phases_total": 3,
                "total_steps": 90,
                "phase_boundaries_json": json.dumps([30, 60]),
                "eval/paloma/dolma_100_programing_languages/bpb": 1.12,
                "phase_0_starcoder": 0.85,
                "phase_1_starcoder": 0.8,
                "phase_2_starcoder": 0.75,
            },
            {
                "wandb_run_id": "three-d",
                "run_name": "three/three-d",
                "source_experiment": "exp-three-d",
                "run_family": "three_phase_starcoder",
                "local_run_id": 4,
                "num_phases_total": 3,
                "total_steps": 90,
                "phase_boundaries_json": json.dumps([30, 60]),
                "eval/paloma/dolma_100_programing_languages/bpb": 1.08,
                "phase_0_starcoder": 0.75,
                "phase_1_starcoder": 0.7,
                "phase_2_starcoder": 0.65,
            },
            {
                "wandb_run_id": "two-a",
                "run_name": "two/two-a",
                "source_experiment": "exp-two-a",
                "run_family": "two_phase_starcoder",
                "local_run_id": 5,
                "num_phases_total": 2,
                "total_steps": 60,
                "phase_boundaries_json": json.dumps([30]),
                "eval/paloma/dolma_100_programing_languages/bpb": 0.91,
                "phase_0_starcoder": 0.02,
                "phase_1_starcoder": 0.28,
                "phase_2_starcoder": np.nan,
            },
            {
                "wandb_run_id": "two-b",
                "run_name": "two/two-b",
                "source_experiment": "exp-two-b",
                "run_family": "two_phase_starcoder",
                "local_run_id": 6,
                "num_phases_total": 2,
                "total_steps": 60,
                "phase_boundaries_json": json.dumps([30]),
                "eval/paloma/dolma_100_programing_languages/bpb": 1.15,
                "phase_0_starcoder": 0.9,
                "phase_1_starcoder": 0.85,
                "phase_2_starcoder": np.nan,
            },
        ]
    )
    history_rows = []
    for _, run in runs.iterrows():
        total_steps = int(run["total_steps"])
        scale = 1.0 if "a" in str(run["wandb_run_id"]) or "b" in str(run["wandb_run_id"]) else 1.2
        for step in range(0, total_steps + 1, 10):
            progress = step / max(total_steps, 1)
            history_rows.append(
                {
                    "wandb_run_id": run["wandb_run_id"],
                    "source_experiment": run["source_experiment"],
                    "local_run_id": run["local_run_id"],
                    "run_name": run["run_name"],
                    "step": step,
                    "train/loss": 4.5 * scale - 1.2 * progress,
                    "optim/learning_rate": max(0.0, 0.02 - 0.019 * progress),
                    "optim/adam_lr": max(0.0, 0.008 - 0.0076 * progress),
                    "grad/norm/total": 0.8 * scale + 0.4 * progress,
                    "params/norm/total": 10.0 * scale + 0.8 * progress,
                    "eval/loss": 4.7 * scale - 1.3 * progress,
                    "eval/paloma/dolma_100_programing_languages/bpb": (
                        float(run["eval/paloma/dolma_100_programing_languages/bpb"]) + (1.0 - progress) * 0.5
                    ),
                }
            )
    pd.DataFrame(history_rows).to_parquet(input_dir / "history_dense_wide.parquet", index=False)
    runs.to_parquet(input_dir / "runs.parquet", index=False)
    return input_dir


def test_build_pooled_dense_policy_dataset_includes_two_and_three_phase_rows(tmp_path: Path):
    input_dir = _small_pooled_dense_inputs(tmp_path)
    output_dir = tmp_path / "pooled_dense_output"
    decisions, sequence_arrays, manifest = build_pooled_dense_policy_dataset_v4(
        BuildPooledDenseDatasetV4Config(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            n_cv_folds=2,
            window_steps=20,
            window_bins=4,
            pretrain_stride=10,
        )
    )
    assert len(decisions) == 16
    assert manifest["run_family_counts"]["three_phase_starcoder"] == 12
    assert manifest["run_family_counts"]["two_phase_starcoder"] == 4
    assert sequence_arrays["sequences"].shape == (16, 4, len(SEQUENCE_CHANNELS))
    with np.load(output_dir / "pretrain_sequences.npz", allow_pickle=False) as payload:
        assert len(payload["row_ids"]) >= len(decisions)


def test_run_three_phase_policy_bench_v4_smoke(tmp_path: Path):
    input_dir = _small_pooled_dense_inputs(tmp_path)
    dataset_dir = tmp_path / "pooled_dense_dataset"
    build_pooled_dense_policy_dataset_v4(
        BuildPooledDenseDatasetV4Config(
            input_dir=str(input_dir),
            output_dir=str(dataset_dir),
            n_cv_folds=2,
            window_steps=20,
            window_bins=4,
            pretrain_stride=10,
        )
    )
    output_dir = tmp_path / "bench_v4"
    summary = run_three_phase_policy_bench_v4(
        ThreePhasePolicyBenchV4Config(
            input_dir=str(dataset_dir),
            output_dir=str(output_dir),
            pretrain_epochs=0,
            finetune_epochs=4,
            batch_size=8,
            learning_rate=1e-3,
            q_max_iter=16,
            fqe_iterations=2,
            device="cpu",
        )
    )
    assert "dynamic_q_planner_v4_pooled" in summary["artifacts"]
    policy_summary = pd.read_csv(output_dir / "policy_summary.csv")
    assert "dynamic_q_planner_v4_pooled" in set(policy_summary["method"])
