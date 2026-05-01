# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import base64
import subprocess
import sys
from pathlib import Path

import pandas as pd

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.exploratory import launch_starcoder_optima_validation as launch_validation


def test_launch_starcoder_optima_validation_dry_run(tmp_path: Path):
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()

    selector_summary = pd.DataFrame(
        [
            {
                "dataset": "two_phase_starcoder",
                "mode": "retrospective",
                "policy": "feature_maximin_observed",
                "dsre_median_regret@1": 0.02,
                "committee_mean_regret@1": 0.03,
            },
            {
                "dataset": "two_phase_starcoder",
                "mode": "retrospective",
                "policy": "feature_dpp_observed",
                "dsre_median_regret@1": 0.01,
                "committee_mean_regret@1": 0.02,
            },
            {
                "dataset": "two_phase_starcoder",
                "mode": "retrospective",
                "policy": "feature_bayes_linear_observed",
                "dsre_median_regret@1": 0.03,
                "committee_mean_regret@1": 0.04,
            },
        ]
    )
    selector_summary.to_csv(benchmark_dir / "selector_summary.csv", index=False)

    records = []
    for subset_size, run_id in ((4, 101), (10, 102)):
        records.append(
            {
                "dataset": "two_phase_starcoder",
                "mode": "retrospective",
                "policy": "feature_dpp_observed",
                "subset_size": subset_size,
                "selector_seed": 0,
                "evaluation_model": "DS-RE-CEQ",
                "weight_config": (
                    WeightConfig(
                        run_id=run_id,
                        phase_weights={
                            "phase_0": {"nemotron_full": 0.9, "starcoder": 0.1},
                            "phase_1": {"nemotron_full": 0.4, "starcoder": 0.6},
                        },
                    ).to_dict()
                ),
                "predicted_objective": 0.1 * subset_size,
                "nearest_observed_idx": 0,
                "nearest_observed_distance": 0.0,
            }
        )
    with (benchmark_dir / "predicted_optima.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    script = (
        Path(__file__).resolve().parent.parent
        / "experiments"
        / "domain_phase_mix"
        / "exploratory"
        / "launch_starcoder_optima_validation.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--benchmark-output-dir",
            str(benchmark_dir),
            "--dataset",
            "two_phase_starcoder",
            "--dry-run",
        ],
        check=True,
    )

    plan = json.loads((benchmark_dir / "two_phase_starcoder_validation_launch_plan.json").read_text())
    assert plan["dataset"] == "two_phase_starcoder"
    assert plan["policy"] == "feature_dpp_observed"
    assert plan["n_runs"] == 2
    assert [run["subset_size"] for run in plan["runs"]] == [4, 10]
    assert [run["weight_config"]["run_id"] for run in plan["runs"]] == [95004, 95010]


def test_launch_starcoder_optima_validation_from_inline_plan(tmp_path: Path):
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    launch_plan = {
        "dataset": "three_phase_starcoder",
        "benchmark_output_dir": str(benchmark_dir),
        "policy": "feature_bayes_linear_observed",
        "name_prefix": "pinlin_calvin_xu/data_mixture/three_phase_starcoder_selector_validation/full_20260308_rerun2",
        "n_runs": 2,
        "runs": [
            {
                "subset_size": 4,
                "selector_seed": 0,
                "run_name": "feature_bayes_linear_k004_optimum",
                "weight_config": (
                    WeightConfig(
                        run_id=98104,
                        phase_weights={
                            "phase_0": {"nemotron_full": 1.0, "starcoder": 0.0},
                            "phase_1": {"nemotron_full": 0.7, "starcoder": 0.3},
                            "phase_2": {"nemotron_full": 0.4, "starcoder": 0.6},
                        },
                    ).to_dict()
                ),
            },
            {
                "subset_size": 16,
                "selector_seed": 0,
                "run_name": "feature_bayes_linear_k016_optimum",
                "weight_config": (
                    WeightConfig(
                        run_id=98116,
                        phase_weights={
                            "phase_0": {"nemotron_full": 0.9, "starcoder": 0.1},
                            "phase_1": {"nemotron_full": 0.5, "starcoder": 0.5},
                            "phase_2": {"nemotron_full": 0.2, "starcoder": 0.8},
                        },
                    ).to_dict()
                ),
            },
        ],
    }
    encoded = base64.b64encode(json.dumps(launch_plan).encode("utf-8")).decode("utf-8")

    script = (
        Path(__file__).resolve().parent.parent
        / "experiments"
        / "domain_phase_mix"
        / "exploratory"
        / "launch_starcoder_optima_validation.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset",
            "three_phase_starcoder",
            "--launch-plan-json-base64",
            encoded,
            "--dry-run",
        ],
        check=True,
    )

    plan = json.loads((benchmark_dir / "three_phase_starcoder_validation_launch_plan.json").read_text())
    assert plan["dataset"] == "three_phase_starcoder"
    assert plan["policy"] == "feature_bayes_linear_observed"
    assert plan["n_runs"] == 2
    assert [run["subset_size"] for run in plan["runs"]] == [4, 16]
    assert [run["weight_config"]["run_id"] for run in plan["runs"]] == [98104, 98116]
    assert len(plan["name_prefix"]) <= 64


def test_launch_starcoder_optima_validation_from_inline_plan_missing_benchmark_dir(tmp_path: Path):
    missing_benchmark_dir = tmp_path / "missing-benchmark"
    launch_plan = {
        "dataset": "two_phase_starcoder",
        "benchmark_output_dir": str(missing_benchmark_dir),
        "policy": "feature_bayes_linear_observed",
        "name_prefix": "pinlin_calvin_xu/data_mixture/two_phase_starcoder_selector_validation/full_20260308_rerun2",
        "n_runs": 1,
        "runs": [
            {
                "subset_size": 4,
                "selector_seed": 0,
                "run_name": "feature_bayes_linear_k004_optimum",
                "weight_config": (
                    WeightConfig(
                        run_id=97104,
                        phase_weights={
                            "phase_0": {"nemotron_full": 1.0, "starcoder": 0.0},
                            "phase_1": {"nemotron_full": 0.7, "starcoder": 0.3},
                        },
                    ).to_dict()
                ),
            },
        ],
    }
    encoded = base64.b64encode(json.dumps(launch_plan).encode("utf-8")).decode("utf-8")

    script = (
        Path(__file__).resolve().parent.parent
        / "experiments"
        / "domain_phase_mix"
        / "exploratory"
        / "launch_starcoder_optima_validation.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset",
            "two_phase_starcoder",
            "--launch-plan-json-base64",
            encoded,
            "--dry-run",
        ],
        check=True,
        cwd=tmp_path,
    )

    plan = json.loads((tmp_path / "two_phase_starcoder_validation_launch_plan.json").read_text())
    assert plan["dataset"] == "two_phase_starcoder"
    assert plan["n_runs"] == 1


def test_safe_name_prefix_respects_wandb_limit():
    name_prefix = "pinlin_calvin_xu/data_mixture/three_phase_starcoder_selector_validation/full_20260308_rerun2"
    run_names = ["feature_bayes_linear_k128_optimum"]
    truncated = launch_validation._safe_name_prefix(name_prefix, run_names=run_names)
    assert len(truncated) + 1 + len(run_names[0]) <= 64
    assert truncated != name_prefix


def test_run_validation_specs_clears_executor_cli(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    class DummyExperiment:
        def create_weight_configs_step(self, configs, summary, seed, name_prefix):
            return {
                "kind": "weight_configs",
                "configs": configs,
                "summary": summary,
                "seed": seed,
                "name_prefix": name_prefix,
            }

        def create_training_step(self, weight_config, name_prefix, run_name, data_seed):
            return {
                "kind": "train",
                "weight_config": weight_config,
                "name_prefix": name_prefix,
                "run_name": run_name,
                "data_seed": data_seed,
            }

    monkeypatch.setattr(
        launch_validation,
        "_dataset_launch_config",
        lambda dataset: launch_validation.DatasetLaunchConfig(
            dataset=dataset,
            tokenizer_cache_base="gs://tokenizers",
            eval_datasets_cache_path="gs://eval",
            tokenizer_name="test/tokenizer",
            eval_tasks=("eval/task",),
        ),
    )
    monkeypatch.setattr(launch_validation, "_create_experiment", lambda dataset, *, name_prefix: DummyExperiment())
    monkeypatch.setattr(
        launch_validation,
        "create_cache_tokenizer_step",
        lambda **kwargs: {"kind": "cache_tokenizer", **kwargs},
    )
    monkeypatch.setattr(
        launch_validation,
        "create_cache_eval_datasets_step",
        lambda **kwargs: {"kind": "cache_eval", **kwargs},
    )

    def fake_executor_main(config, *, steps, description):
        captured["config"] = config
        captured["argv"] = sys.argv[:]
        captured["steps"] = steps
        captured["description"] = description

    monkeypatch.setattr(launch_validation, "executor_main", fake_executor_main)

    spec = launch_validation.ValidationRunSpec(
        dataset="two_phase_starcoder",
        subset_size=4,
        policy="feature_bayes_linear_observed",
        selector_seed=0,
        weight_config=WeightConfig(
            run_id=97104,
            phase_weights={
                "phase_0": {"nemotron_full": 1.0, "starcoder": 0.0},
                "phase_1": {"nemotron_full": 0.8, "starcoder": 0.2},
            },
        ),
        run_name="feature_bayes_linear_k004_optimum",
    )

    original_argv = sys.argv[:]
    sys.argv = [
        "launch_starcoder_optima_validation.py",
        "--dataset",
        "two_phase_starcoder",
        "--launch-plan-json-base64",
        "inline",
    ]
    try:
        plan_path = launch_validation._run_validation_specs(
            dataset="two_phase_starcoder",
            benchmark_output_dir=tmp_path,
            name_prefix="pinlin_calvin_xu/data_mixture/two_phase_starcoder_selector_validation/test",
            policy="feature_bayes_linear_observed",
            specs=[spec],
            data_seed=0,
            dry_run=False,
        )
    finally:
        sys.argv = original_argv

    assert plan_path == tmp_path / "two_phase_starcoder_validation_launch_plan.json"
    assert captured["argv"] == ["launch_starcoder_optima_validation.py"]
    assert captured["config"].max_concurrent == 4
    assert len(captured["steps"]) == 4
