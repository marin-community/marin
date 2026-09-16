# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd

from experiments.domain_phase_mix.offline_rl.build_pooled_dense_policy_dataset_v4 import (
    BuildPooledDenseDatasetV4Config,
    build_pooled_dense_policy_dataset_v4,
)
from experiments.domain_phase_mix.offline_rl.train_three_phase_policy_bench_v5 import (
    ThreePhasePolicyBenchV5Config,
    run_three_phase_policy_bench_v5,
)
from tests.test_domain_phase_mix_offline_rl_v4 import _small_pooled_dense_inputs


def test_run_three_phase_policy_bench_v5_smoke(tmp_path: Path):
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
    output_dir = tmp_path / "bench_v5"
    report = run_three_phase_policy_bench_v5(
        ThreePhasePolicyBenchV5Config(
            input_dir=str(dataset_dir),
            output_dir=str(output_dir),
            fqe_iterations=2,
            q_max_iter=16,
            dense_direct_reward_bonus_weight=0.05,
            dense_direct_support_lambda=0.02,
            hybrid_direct_alpha=1.0,
        )
    )
    assert report["best_method_by_summary"] in {
        "legacy_outcome_planner",
        "dense_direct_v5",
        "hybrid_q_direct_v5",
        "fixed_best_schedule",
        "discrete_bc",
    }
    policy_summary = pd.read_csv(output_dir / "policy_summary.csv")
    assert {"legacy_outcome_planner", "dense_direct_v5", "hybrid_q_direct_v5"}.issubset(set(policy_summary["method"]))
    comparison = pd.read_csv(output_dir / "comparison_report.csv")
    assert {"dense_direct_v5", "hybrid_q_direct_v5"} == set(comparison["method"])
