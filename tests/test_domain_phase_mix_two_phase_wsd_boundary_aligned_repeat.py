# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.static_batch_selection import retrospective_generic_selection
from experiments.domain_phase_mix.starcoder_metadata import TWO_PHASE_STARCODER, load_starcoder_dataset
from experiments.domain_phase_mix.two_phase_starcoder_experiment import (
    BATCH_SIZE,
    EXPERIMENT_BUDGET,
    PHASE_BOUNDARIES,
    SEQ_LEN,
    create_two_phase_experiment,
    create_two_phase_wsd_boundary_aligned_experiment,
    create_two_phase_wsd_boundary_aligned_optimizer_config,
    resolve_two_phase_wsd_boundary_schedule,
)


def test_two_phase_wsd_boundary_schedule_matches_aligned_phase_transition():
    baseline = create_two_phase_experiment(name="baseline_test")
    wsd_experiment = create_two_phase_wsd_boundary_aligned_experiment(name="wsd_test")
    schedule = resolve_two_phase_wsd_boundary_schedule(phase_schedule=baseline.phase_schedule)
    optimizer = create_two_phase_wsd_boundary_aligned_optimizer_config(phase_schedule=baseline.phase_schedule)

    expected_total_steps = EXPERIMENT_BUDGET // (BATCH_SIZE * SEQ_LEN)
    expected_boundary = baseline.phase_schedule.phases[1].get_start_step_aligned(
        expected_total_steps,
        BATCH_SIZE,
        2048,
    )

    assert PHASE_BOUNDARIES == [0.5]
    assert schedule.total_steps == expected_total_steps
    assert schedule.boundary_step == expected_boundary == 1904
    assert schedule.decay_steps == schedule.total_steps - schedule.boundary_step == 1910
    assert schedule.warmup_steps == int(expected_total_steps * 0.01) == 38
    assert optimizer.warmup == schedule.warmup_steps
    assert optimizer.decay == schedule.decay_steps
    assert optimizer.lr_schedule == "cosine"
    assert optimizer.rewarmup == 0.0
    assert optimizer.cycles is None
    assert optimizer.cycle_length is None
    assert optimizer.haps is None
    assert wsd_experiment.phase_schedule.phase_names == baseline.phase_schedule.phase_names
    assert [domain.name for domain in wsd_experiment.domains] == [domain.name for domain in baseline.domains]


def test_two_phase_wsd_boundary_schedule_has_single_plateau_then_decay():
    schedule = resolve_two_phase_wsd_boundary_schedule()
    optimizer = create_two_phase_wsd_boundary_aligned_optimizer_config()
    schedule_fn = optimizer.lr_scheduler(schedule.total_steps)

    max_lr = float(optimizer.learning_rate)
    plateau_step = (schedule.warmup_steps + schedule.boundary_step) // 2
    decay_probe = min(schedule.boundary_step + 64, schedule.total_steps - 1)

    assert np.isclose(schedule_fn(0), 0.0)
    assert np.isclose(schedule_fn(schedule.warmup_steps), max_lr)
    assert np.isclose(schedule_fn(plateau_step), max_lr)
    assert np.isclose(schedule_fn(schedule.boundary_step - 1), max_lr)
    assert schedule_fn(schedule.boundary_step) <= schedule_fn(schedule.boundary_step - 1)
    assert schedule_fn(decay_probe) < max_lr
    assert schedule_fn(schedule.total_steps - 1) < schedule_fn(decay_probe)


def test_two_phase_feature_bayes_linear_prefix_chain_matches_max_k_selection():
    spec, _ = load_starcoder_dataset(TWO_PHASE_STARCODER, target_col="eval/paloma/dolma_100_programing_languages/bpb")
    spec = spec.subset(list(range(32)))

    selection_k16 = retrospective_generic_selection(spec, method="feature_bayes_linear", k=16, seed=0)

    for prefix_size in (4, 6, 8, 10, 12):
        selection = retrospective_generic_selection(spec, method="feature_bayes_linear", k=prefix_size, seed=0)
        assert selection_k16.selected_indices[:prefix_size] == selection.selected_indices


def test_two_phase_wsd_boundary_aligned_repeat_smoke(tmp_path: Path):
    script = (
        Path(__file__).resolve().parent.parent
        / "experiments"
        / "domain_phase_mix"
        / "exploratory"
        / "two_phase_starcoder_feature_bayes_linear_wsd_boundary_aligned_repeat.py"
    )
    output_dir = tmp_path / "two_phase_wsd_boundary_aligned_repeat"
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
            "2",
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
        output_dir / "two_phase_starcoder_validation_launch_plan.json",
        output_dir
        / "plots"
        / "two_phase_starcoder_feature_bayes_linear_wsd_boundary_aligned_predicted_vs_actual_bpb.png",
    ]
    for path in expected_files:
        assert path.exists(), path

    plan = json.loads((output_dir / "proxy_launch_plan.json").read_text())
    assert plan["policy"] == "feature_bayes_linear_wsd_boundary_aligned"
    assert plan["subset_sizes"] == [4, 8]
    assert plan["n_runs"] == 8

    proxy_results = pd.read_csv(output_dir / "proxy_results.csv")
    predicted_optima = pd.read_csv(output_dir / "predicted_optima.csv")
    validation_plan = json.loads((output_dir / "two_phase_starcoder_validation_launch_plan.json").read_text())
    fit_status = json.loads((output_dir / "fit_status.json").read_text())

    assert len(proxy_results) == 8
    assert proxy_results["actual_bpb"].notna().all()
    assert predicted_optima["subset_size"].tolist() == [4, 8]
    assert validation_plan["n_runs"] == 2
    assert [run["subset_size"] for run in validation_plan["runs"]] == [4, 8]
    assert fit_status["ready_subset_sizes"] == [4, 8]

    with (output_dir / "predicted_optima.jsonl").open() as handle:
        first_record = json.loads(handle.readline())
    assert first_record["policy"] == "feature_bayes_linear_wsd_boundary_aligned"
