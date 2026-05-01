# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from experiments.domain_phase_mix.exploratory.plot_starcoder_optima_validation import (
    _select_actual_metric_candidates,
)
from experiments.domain_phase_mix.exploratory.three_phase_starcoder_feature_bayes_linear_wsds_repeat import (
    _default_workers,
    _load_proxy_launch_plan,
    _parse_subset_sizes,
)
from experiments.domain_phase_mix.starcoder_metadata import THREE_PHASE_STARCODER, load_starcoder_dataset
from experiments.domain_phase_mix.static_batch_selection import retrospective_generic_selection
from experiments.domain_phase_mix.three_phase_starcoder_experiment import (
    PHASE_NAMES,
    create_three_phase_experiment,
    create_three_phase_wsds_experiment,
    create_three_phase_wsds_optimizer_config,
    resolve_three_phase_wsds_cycle_points,
)


def test_three_phase_wsds_optimizer_resolves_aligned_cycles():
    experiment = create_three_phase_wsds_experiment(name="wsds_test")
    baseline = create_three_phase_experiment(name="baseline_test")
    optimizer = create_three_phase_wsds_optimizer_config()

    assert resolve_three_phase_wsds_cycle_points() == [1888, 3824]
    assert optimizer.cycles == [1888, 3824]
    assert experiment.optimizer_config.cycles == [1888, 3824]
    assert optimizer.lr_schedule == "cosine"
    assert optimizer.warmup == 0.01
    assert optimizer.decay == 0.1
    assert optimizer.rewarmup == 0.0
    assert experiment.phase_schedule.phase_names == PHASE_NAMES
    assert experiment.phase_schedule.phase_names == baseline.phase_schedule.phase_names
    assert [domain.name for domain in experiment.domains] == [domain.name for domain in baseline.domains]


def test_feature_bayes_linear_prefix_chain_matches_max_k_selection():
    spec, _ = load_starcoder_dataset(THREE_PHASE_STARCODER, target_col="eval/paloma/dolma_100_programing_languages/bpb")
    spec = spec.subset(list(range(16)))

    selection_k8 = retrospective_generic_selection(spec, method="feature_bayes_linear", k=8, seed=0)
    selection_k4 = retrospective_generic_selection(spec, method="feature_bayes_linear", k=4, seed=0)

    assert selection_k8.selected_indices[:4] == selection_k4.selected_indices


def test_select_actual_metric_candidates_prefers_exact_run_name_over_suffix_match():
    exact_run = "t3s-wsds-val-test/fbl_wsds_k016_optimum"

    selected = _select_actual_metric_candidates(
        run_names=[exact_run],
        exact_candidates={
            exact_run: [
                {
                    "actual_bpb": 0.901,
                    "wandb_state": "finished",
                    "wandb_url": "https://wandb.ai/example/exact",
                    "wandb_run_name": exact_run,
                    "created_at": "2026-03-11T10:00:00",
                }
            ]
        },
        suffix_candidates={
            exact_run: [
                {
                    "actual_bpb": 0.812,
                    "wandb_state": "finished",
                    "wandb_url": "https://wandb.ai/example/suffix",
                    "wandb_run_name": "pcx/dm/t3s-suffix/fbl_wsds_k016_optimum",
                    "created_at": "2026-03-11T10:05:00",
                }
            ]
        },
    )

    assert selected[exact_run]["wandb_run_name"] == exact_run
    assert selected[exact_run]["actual_bpb"] == 0.901


def test_three_phase_wsds_repeat_smoke(tmp_path: Path):
    script = (
        Path(__file__).resolve().parent.parent
        / "experiments"
        / "domain_phase_mix"
        / "exploratory"
        / "three_phase_starcoder_feature_bayes_linear_wsds_repeat.py"
    )
    output_dir = tmp_path / "wsds_repeat"
    subset_sizes = "4,8"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(output_dir),
            "--stage",
            "all",
            "--subset-sizes",
            subset_sizes,
            "--row-limit",
            "16",
            "--workers",
            str(min(2, _default_workers())),
            "--dry-run-launches",
            "--skip-wandb",
            "--collect-from-source-observed",
            "--dsre-fit-seeds",
            "1",
            "--dsre-restarts",
            "2",
            "--dsre-maxiter",
            "40",
            "--opt-search-points",
            "128",
            "--opt-restarts",
            "2",
            "--opt-maxiter",
            "40",
        ],
        check=True,
    )

    expected_files = [
        output_dir / "proxy_selection_plan.csv",
        output_dir / "proxy_launch_plan.json",
        output_dir / "proxy_results.csv",
        output_dir / "fit_status.json",
        output_dir / "curve_points.csv",
        output_dir / "predicted_optima.csv",
        output_dir / "predicted_optima.jsonl",
        output_dir / "three_phase_starcoder_validation_launch_plan.json",
        output_dir / "plots" / "three_phase_starcoder_feature_bayes_linear_wsds_predicted_vs_actual_bpb.png",
    ]
    for path in expected_files:
        assert path.exists(), path

    plan = _load_proxy_launch_plan(output_dir)
    assert plan["policy"] == "feature_bayes_linear_wsds"
    assert plan["subset_sizes"] == [4, 8]
    assert plan["n_runs"] == 8

    proxy_results = pd.read_csv(output_dir / "proxy_results.csv")
    predicted_optima = pd.read_csv(output_dir / "predicted_optima.csv")
    validation_plan = json.loads((output_dir / "three_phase_starcoder_validation_launch_plan.json").read_text())
    fit_status = json.loads((output_dir / "fit_status.json").read_text())

    assert len(proxy_results) == 8
    assert proxy_results["actual_bpb"].notna().all()
    assert predicted_optima["subset_size"].tolist() == [4, 8]
    assert validation_plan["n_runs"] == 2
    assert [run["subset_size"] for run in validation_plan["runs"]] == [4, 8]
    assert fit_status["ready_subset_sizes"] == [4, 8]

    with (output_dir / "predicted_optima.jsonl").open() as handle:
        first_record = json.loads(handle.readline())
    assert first_record["policy"] == "feature_bayes_linear_wsds"
    assert _parse_subset_sizes(subset_sizes, max_size=16) == (4, 8)
