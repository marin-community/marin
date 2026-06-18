# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import pytest

from experiments.domain_phase_mix import launch_300m_dclm_core_evals as launcher
from experiments.domain_phase_mix import build_300m_dclm_proportional_noise_state as proportional_noise
from experiments.evals import dclm_core


def _mcq_eval_row(
    *, eval_key: str = "dclm300m_000_mcq_signal_baseline_proportional_native_smooth_canary"
) -> launcher.DCLMEvalSpec:
    return launcher.DCLMEvalSpec(
        eval_key=eval_key,
        mode=launcher.TaskMode.MCQ.value,
        panel="signal_300m_6b",
        run_name="baseline_proportional",
        registry_key="signal:baseline_proportional",
        source_experiment="qsplit240_300m_6b",
        cohort="signal",
        checkpoint_root="gs://marin-us-east5/checkpoints/baseline",
        expected_checkpoint_step=22887,
        hf_checkpoint_count=1,
        hf_checkpoint_latest="gs://marin-us-east5/checkpoints/baseline/hf/step-22887",
        hf_checkpoint_latest_step=22887,
        has_exact_hf_checkpoint=True,
        checkpoint_region="us-east5",
        is_region_local=True,
        existing_artifact_count=0,
        existing_tasks="",
        missing_task_count=1,
        missing_tasks="boolq_10shot",
        has_all_tasks=False,
        task_aliases="boolq_10shot",
        launch_tpu_type="v5p-8",
        launch_tpu_region="us-east5",
        launch_tpu_zone="us-east5-a",
        eligible=True,
        launch_decision="launch",
        step_name="evaluation/lm_evaluation_harness_levanter/lmeval_debug_dclm300m_000_mcq_signal_baseline",
        result_path="gs://marin-us-east5/eval/result",
    )


def test_split_state_rows_by_task_alias_creates_one_short_eval_step_per_alias() -> None:
    row = _mcq_eval_row()
    row = launcher.replace(
        row,
        eval_key="dclm300m_000_mcq_signal_baseline",
        missing_task_count=2,
        missing_tasks="boolq_10shot;hellaswag_10shot",
        task_aliases="boolq_10shot;hellaswag_10shot",
    )

    split_rows = launcher.split_state_rows_by_task_alias([row])

    assert [split_row.missing_tasks for split_row in split_rows] == ["boolq_10shot", "hellaswag_10shot"]
    assert [split_row.missing_task_count for split_row in split_rows] == [1, 1]
    assert [split_row.eval_key for split_row in split_rows] == [
        "dclm300m_000_mcq_signal_baseline_boolq_10shot",
        "dclm300m_000_mcq_signal_baseline_hellaswag_10shot",
    ]
    assert all("lmeval_debug_" in split_row.step_name for split_row in split_rows)

    steps, results_by_key = launcher.build_eval_steps(
        name_prefix="unit-test/dclm-native-smooth",
        state_rows=split_rows,
        max_eval_instances=20,
        eval_datasets_cache_path=None,
    )

    assert len(steps) == 2
    assert sorted(results_by_key) == sorted(split_row.eval_key for split_row in split_rows)
    assert [step.config.evals.value[0].task_alias for step in steps] == ["boolq_10shot", "hellaswag_10shot"]


def test_alias_split_metric_rows_merge_back_to_one_checkpoint(monkeypatch) -> None:
    row = launcher.replace(
        _mcq_eval_row(),
        eval_key="dclm300m_000_mcq_signal_baseline",
        missing_task_count=2,
        missing_tasks="boolq_10shot;hellaswag_10shot",
        task_aliases="boolq_10shot;hellaswag_10shot",
    )
    split_rows = launcher.split_state_rows_by_task_alias([row])

    def fake_read_eval_metrics(path: str) -> tuple[dict[str, float], str | None]:
        if "boolq" in path:
            return {"lm_eval/boolq_10shot/acc": 0.7}, None
        if "hellaswag" in path:
            return {"lm_eval/hellaswag_10shot/acc_norm": 0.8}, None
        raise AssertionError(path)

    monkeypatch.setattr(launcher, "_read_eval_metrics", fake_read_eval_metrics)
    records = launcher._metric_rows_from_result_paths(
        split_rows,
        {split_row.eval_key: f"gs://marin-us-east5/results/{split_row.eval_key}" for split_row in split_rows},
    )
    merged = launcher._merge_records_by_checkpoint(records)

    assert len(merged) == 1
    assert merged[0]["lm_eval/boolq_10shot/acc"] == 0.7
    assert merged[0]["lm_eval/hellaswag_10shot/acc_norm"] == 0.8
    assert merged[0]["task_aliases"] == "boolq_10shot;hellaswag_10shot"


def test_dclm_core_inventory_accounts_for_all_paper_tasks() -> None:
    tasks = dclm_core.dclm_core_tasks()

    assert len(tasks) == 22
    assert {task.alias for task in tasks} == {
        "agieval_lsat_ar_3shot",
        "arc_easy_10shot",
        "arc_challenge_10shot",
        "bb_qa_wikidata_10shot",
        "bb_dyck_languages_10shot",
        "bb_operators_10shot",
        "bb_repeat_copy_logic_10shot",
        "bb_cs_algorithms_10shot",
        "bb_language_identification_10shot",
        "boolq_10shot",
        "commonsense_qa_10shot",
        "copa_0shot",
        "coqa_0shot",
        "hellaswag_0shot",
        "hellaswag_10shot",
        "jeopardy_10shot",
        "lambada_0shot",
        "openbookqa_0shot",
        "piqa_10shot",
        "squad_10shot",
        "winograd_0shot",
        "winogrande_0shot",
    }
    assert all(task.status == dclm_core.TaskStatus.LAUNCHABLE for task in tasks)
    assert len(dclm_core.launchable_task_aliases()) == 22
    assert dclm_core.task_by_alias("arc_easy_10shot").primary_metric == "acc_norm"
    assert dclm_core.task_by_alias("agieval_lsat_ar_3shot").primary_metric == "acc_norm"
    assert dclm_core.task_by_alias("commonsense_qa_10shot").random_baseline == 0.403
    assert dclm_core.task_by_alias("boolq_10shot").random_baseline == 0.62
    assert dclm_core.task_by_alias("bb_language_identification_10shot").mode == dclm_core.TaskMode.MCQ
    assert dclm_core.task_by_alias("bb_language_identification_10shot").random_baseline == 0.25
    assert dclm_core.task_by_alias("coqa_0shot").mode == dclm_core.TaskMode.GENERATION
    assert dclm_core.task_by_alias("squad_10shot").mode == dclm_core.TaskMode.GENERATION
    assert dclm_core.task_by_alias("jeopardy_10shot").task_kwargs is not None
    assert dclm_core.task_by_alias("winograd_0shot").task_kwargs is not None


def test_dclm_bigbench_generation_tasks_use_filtered_exact_match_configs() -> None:
    for alias in (
        "bb_qa_wikidata_10shot",
        "bb_dyck_languages_10shot",
        "bb_operators_10shot",
        "bb_repeat_copy_logic_10shot",
        "bb_cs_algorithms_10shot",
    ):
        task = dclm_core.task_by_alias(alias)

        assert task.task_kwargs is not None
        assert task.task_kwargs["generation_kwargs"] == {"max_gen_toks": 128}
        assert task.task_kwargs["filter_list"] == [
            {
                "name": "strip-then-match",
                "filter": [
                    {"function": "remove_whitespace"},
                    {"function": "take_first"},
                ],
            }
        ]
        assert "exact_match,strip-then-match" in task.metric_candidates
        assert "exact_match,none" not in task.metric_candidates


def test_custom_task_data_files_are_bundle_relative() -> None:
    for alias in ("jeopardy_10shot", "winograd_0shot"):
        task = dclm_core.task_by_alias(alias)
        assert task.task_kwargs is not None
        data_files = task.task_kwargs["dataset_kwargs"]["data_files"]

        assert not Path(data_files).is_absolute()
        assert str(data_files).startswith("experiments/scaling_law_sweeps/dclm_core/custom_tasks/")


def test_centered_accuracy_uses_primary_hard_metric_and_counts_missing_tasks() -> None:
    metrics = {
        "lm_eval/boolq_10shot/acc": 0.75,
        "lm_eval/hellaswag_0shot/acc_norm": 0.70,
        "lm_eval/coqa_0shot/f1": 0.40,
    }

    centered = dclm_core.dclm_core_centered_accuracy(metrics)

    assert centered["lm_eval/dclm_core/boolq_10shot/centered_accuracy"] == pytest.approx((0.75 - 0.62) / 0.38)
    assert centered["lm_eval/dclm_core/hellaswag_0shot/centered_accuracy"] == 0.60
    assert centered["lm_eval/dclm_core/coqa_0shot/centered_accuracy"] == 0.40
    assert centered["lm_eval/dclm_core/task_count"] == 3.0
    assert centered["lm_eval/dclm_core/missing_task_count"] == 19.0
    assert centered["lm_eval/dclm_core/centered_accuracy_macro"] == pytest.approx(
        ((0.75 - 0.62) / 0.38 + 0.60 + 0.40) / 3.0
    )


def test_centered_accuracy_accepts_actual_lm_eval_task_names_for_generation_tasks() -> None:
    metrics = {
        "lm_eval/bigbench_qa_wikidata_generate_until/exact_match,none": 0.0,
        "lm_eval/bigbench_qa_wikidata_generate_until/exact_match,strip-then-match": 0.30,
        "lm_eval/jeopardy/exact_match,strip-then-match": 0.25,
        "lm_eval/squad_completion/contains": 0.40,
    }

    centered = dclm_core.dclm_core_centered_accuracy(metrics)

    assert centered["lm_eval/dclm_core/bb_qa_wikidata_10shot/raw_score"] == 0.30
    assert centered["lm_eval/dclm_core/jeopardy_10shot/raw_score"] == 0.25
    assert centered["lm_eval/dclm_core/squad_10shot/raw_score"] == 0.40
    assert centered["lm_eval/dclm_core/task_count"] == 3.0


def test_centered_accuracy_rejects_unfiltered_bigbench_generation_exact_match_none() -> None:
    metrics = {
        "lm_eval/bigbench_qa_wikidata_generate_until/exact_match,none": 0.0,
    }

    centered = dclm_core.dclm_core_centered_accuracy(metrics, task_aliases=("bb_qa_wikidata_10shot",))

    assert "lm_eval/dclm_core/bb_qa_wikidata_10shot/raw_score" not in centered
    assert centered["lm_eval/dclm_core/task_count"] == 0.0
    assert centered["lm_eval/dclm_core/missing_task_count"] == 1.0


def test_metric_coverage_by_root_requires_alias_exact_metrics_for_skip(tmp_path) -> None:
    csv_path = tmp_path / "metrics.csv"
    pd.DataFrame.from_records(
        [
            {
                "checkpoint_root": "gs://marin-us-east5/checkpoints/a",
                "lm_eval/boolq_10shot/acc": 0.75,
                "lm_eval/arc_easy/acc_norm": 0.50,
                "lm_eval/hellaswag_0shot/acc_norm": pd.NA,
            },
            {
                "checkpoint_root": "gs://marin-us-east5/checkpoints/b",
                "lm_eval/boolq_10shot/acc": pd.NA,
                "lm_eval/hellaswag_0shot/acc_norm": 0.65,
            },
        ]
    ).to_csv(csv_path, index=False)

    coverage = launcher._metric_coverage_by_root(
        [csv_path],
        ("boolq_10shot", "arc_easy_10shot", "hellaswag_0shot"),
    )

    assert coverage["gs://marin-us-east5/checkpoints/a"] == {"boolq_10shot"}
    assert coverage["gs://marin-us-east5/checkpoints/b"] == {"hellaswag_0shot"}


def test_build_state_rows_marks_missing_dclm_core_tasks_for_launch(monkeypatch, tmp_path) -> None:
    metrics_csv = tmp_path / "existing.csv"
    pd.DataFrame.from_records(
        [
            {
                "checkpoint_root": "gs://marin-us-east5/checkpoints/a",
                "lm_eval/boolq_10shot/acc": 0.75,
            }
        ]
    ).to_csv(metrics_csv, index=False)

    monkeypatch.setattr(launcher, "METRICS_WIDE_CSV", metrics_csv)
    monkeypatch.setattr(launcher, "MERGED_RESULTS_CSV", tmp_path / "missing.csv")
    monkeypatch.setattr(
        launcher,
        "_dclm_candidate_records",
        lambda included_panels, included_run_names=None: [
            SimpleNamespace(
                panel="signal_300m_6b",
                run_name="candidate_a",
                registry_key="signal:candidate_a",
                source_experiment="source",
                cohort="signal",
                checkpoint_root="gs://marin-us-east5/checkpoints/a",
                expected_checkpoint_step=22887,
            )
        ],
    )
    monkeypatch.setattr(
        launcher,
        "_exact_hf_checkpoint",
        lambda checkpoint_root, expected_step: f"{checkpoint_root}/hf/step-{expected_step}",
    )

    rows = launcher.build_state_rows(
        default_tpu_type="v5p-8",
        default_tpu_region="us-east5",
        default_tpu_zone="us-east5-a",
        eval_key_suffix="canary",
        mode=launcher.DCLMEvalMode.ALL,
        task_aliases=("boolq_10shot", "hellaswag_0shot"),
        included_run_names=None,
        included_panels={"signal_300m_6b"},
    )

    assert len(rows) == 1
    assert rows[0].existing_tasks == "boolq_10shot"
    assert rows[0].missing_tasks == "hellaswag_0shot"
    assert rows[0].launch_decision == "launch"
    assert rows[0].launch_tpu_region == "us-east5"
    assert rows[0].launch_tpu_zone == "us-east5-a"


def test_build_state_rows_force_launch_runs_requested_tasks_even_when_hard_metrics_exist(monkeypatch, tmp_path) -> None:
    metrics_csv = tmp_path / "existing.csv"
    pd.DataFrame.from_records(
        [
            {
                "checkpoint_root": "gs://marin-us-east5/checkpoints/a",
                "lm_eval/boolq_10shot/acc": 0.75,
            }
        ]
    ).to_csv(metrics_csv, index=False)

    monkeypatch.setattr(launcher, "METRICS_WIDE_CSV", metrics_csv)
    monkeypatch.setattr(launcher, "MERGED_RESULTS_CSV", tmp_path / "missing.csv")
    monkeypatch.setattr(
        launcher,
        "_dclm_candidate_records",
        lambda included_panels, included_run_names=None: [
            SimpleNamespace(
                panel="signal_300m_6b",
                run_name="candidate_a",
                registry_key="signal:candidate_a",
                source_experiment="source",
                cohort="signal",
                checkpoint_root="gs://marin-us-east5/checkpoints/a",
                expected_checkpoint_step=22887,
            )
        ],
    )
    monkeypatch.setattr(
        launcher,
        "_exact_hf_checkpoint",
        lambda checkpoint_root, expected_step: f"{checkpoint_root}/hf/step-{expected_step}",
    )

    rows = launcher.build_state_rows(
        default_tpu_type="v5p-8",
        default_tpu_region="us-east5",
        default_tpu_zone="us-east5-a",
        eval_key_suffix="native_smooth_canary",
        mode=launcher.DCLMEvalMode.MCQ,
        task_aliases=("boolq_10shot",),
        included_run_names=None,
        included_panels={"signal_300m_6b"},
        force_launch=True,
    )

    assert len(rows) == 1
    assert rows[0].existing_tasks == "boolq_10shot"
    assert rows[0].missing_tasks == "boolq_10shot"
    assert rows[0].launch_decision == "launch"


def test_build_state_rows_defers_cross_region_checkpoints(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(launcher, "METRICS_WIDE_CSV", tmp_path / "missing.csv")
    monkeypatch.setattr(launcher, "MERGED_RESULTS_CSV", tmp_path / "missing_merged.csv")
    monkeypatch.setattr(
        launcher,
        "_dclm_candidate_records",
        lambda included_panels, included_run_names=None: [
            SimpleNamespace(
                panel="signal_300m_6b",
                run_name="candidate_central",
                registry_key="signal:candidate_central",
                source_experiment="source",
                cohort="signal",
                checkpoint_root="gs://marin-us-central2/checkpoints/c",
                expected_checkpoint_step=22887,
            )
        ],
    )
    monkeypatch.setattr(
        launcher,
        "_exact_hf_checkpoint",
        lambda checkpoint_root, expected_step: f"{checkpoint_root}/hf/step-{expected_step}",
    )

    rows = launcher.build_state_rows(
        default_tpu_type="v5p-8",
        default_tpu_region="us-east5",
        default_tpu_zone="us-east5-a",
        eval_key_suffix="",
        mode=launcher.DCLMEvalMode.MCQ,
        task_aliases=("boolq_10shot",),
        included_run_names=None,
        included_panels={"signal_300m_6b"},
    )

    assert rows[0].eligible is False
    assert rows[0].launch_decision == "defer_checkpoint_region_mismatch"


def test_build_state_rows_defaults_can_filter_to_signal_panel(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        launcher,
        "build_qsplit300m_run_specs",
        lambda: [
            SimpleNamespace(
                run_name="signal_a",
            )
        ],
    )
    monkeypatch.setattr(
        launcher,
        "_resolve_qsplit300m_checkpoint_root",
        lambda run_name: f"gs://marin-us-east5/checkpoints/{run_name}",
    )
    monkeypatch.setattr(
        launcher,
        "_candidate_records",
        lambda: [
            SimpleNamespace(
                panel="fixed_seed_noise_300m_6b",
                run_name="noise_a",
                registry_key="noise:a",
                source_experiment="source",
                cohort="seed_sweep",
                checkpoint_root="gs://marin-us-east5/checkpoints/noise",
                expected_checkpoint_step=22887,
            ),
        ],
    )

    rows = launcher._dclm_candidate_records({"signal_300m_6b"}, included_run_names={"signal_a"})

    assert {row.run_name for row in rows} == {"signal_a"}


def test_build_state_rows_splits_generation_and_mcq_modes(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(launcher, "METRICS_WIDE_CSV", tmp_path / "missing.csv")
    monkeypatch.setattr(launcher, "MERGED_RESULTS_CSV", tmp_path / "missing_merged.csv")
    monkeypatch.setattr(
        launcher,
        "_dclm_candidate_records",
        lambda included_panels, included_run_names=None: [
            SimpleNamespace(
                panel="signal_300m_6b",
                run_name="candidate_a",
                registry_key="signal:candidate_a",
                source_experiment="source",
                cohort="signal",
                checkpoint_root="gs://marin-us-east5/checkpoints/a",
                expected_checkpoint_step=22887,
            )
        ],
    )
    monkeypatch.setattr(
        launcher,
        "_exact_hf_checkpoint",
        lambda checkpoint_root, expected_step: f"{checkpoint_root}/hf/step-{expected_step}",
    )

    rows = launcher.build_state_rows(
        default_tpu_type="v5p-8",
        default_tpu_region="us-east5",
        default_tpu_zone="us-east5-a",
        eval_key_suffix="canary",
        mode=launcher.DCLMEvalMode.ALL,
        task_aliases=("jeopardy_10shot", "coqa_0shot", "bb_language_identification_10shot", "boolq_10shot"),
        included_run_names=None,
        included_panels={"signal_300m_6b"},
    )

    rows_by_mode = {row.mode: row for row in rows}
    assert set(rows_by_mode) == {"generation", "mcq"}
    assert rows_by_mode["generation"].task_aliases == "coqa_0shot;jeopardy_10shot"
    assert rows_by_mode["mcq"].task_aliases == "bb_language_identification_10shot;boolq_10shot"


def test_build_state_rows_smooth_mode_uses_levanter_rows_for_mcq_and_generation(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(launcher, "METRICS_WIDE_CSV", tmp_path / "missing.csv")
    monkeypatch.setattr(launcher, "MERGED_RESULTS_CSV", tmp_path / "missing_merged.csv")
    monkeypatch.setattr(
        launcher,
        "_dclm_candidate_records",
        lambda included_panels, included_run_names=None: [
            SimpleNamespace(
                panel="signal_300m_6b",
                run_name="candidate_a",
                registry_key="signal:candidate_a",
                source_experiment="source",
                cohort="signal",
                checkpoint_root="gs://marin-us-east5/checkpoints/a",
                expected_checkpoint_step=22887,
            )
        ],
    )

    rows = launcher.build_state_rows(
        default_tpu_type="v5p-8",
        default_tpu_region="us-east5",
        default_tpu_zone="us-east5-a",
        eval_key_suffix="smooth_all22",
        mode=launcher.DCLMEvalMode.SMOOTH,
        task_aliases=("boolq_10shot", "bb_qa_wikidata_10shot", "coqa_0shot"),
        included_run_names=None,
        included_panels={"signal_300m_6b"},
        assume_exact_hf_checkpoints=True,
        force_launch=True,
    )

    rows_by_mode = {row.mode: row for row in rows}
    assert set(rows_by_mode) == {"generation_smooth", "mcq"}
    assert rows_by_mode["mcq"].task_aliases == "boolq_10shot"
    assert rows_by_mode["generation_smooth"].task_aliases == "bb_qa_wikidata_10shot;coqa_0shot"
    assert all("/lm_evaluation_harness_levanter/lmeval_debug_" in row.step_name for row in rows)


def test_build_eval_steps_runs_generation_smooth_as_levanter_loglikelihood() -> None:
    row = launcher.DCLMEvalSpec(
        eval_key="dclm300m_000_generation_smooth_candidate_a",
        mode="generation_smooth",
        panel="signal_300m_6b",
        run_name="candidate_a",
        registry_key="signal:candidate_a",
        source_experiment="source",
        cohort="signal",
        checkpoint_root="gs://marin-us-east5/checkpoints/a",
        expected_checkpoint_step=22887,
        hf_checkpoint_count=1,
        hf_checkpoint_latest="gs://marin-us-east5/checkpoints/a/hf/step-22887",
        hf_checkpoint_latest_step=22887,
        has_exact_hf_checkpoint=True,
        checkpoint_region="us-east5",
        is_region_local=True,
        existing_artifact_count=0,
        existing_tasks="",
        missing_task_count=3,
        missing_tasks="bb_qa_wikidata_10shot;coqa_0shot;squad_10shot",
        has_all_tasks=False,
        task_aliases="bb_qa_wikidata_10shot;coqa_0shot;squad_10shot",
        launch_tpu_type="v5p-8",
        launch_tpu_region="us-east5",
        launch_tpu_zone="us-east5-a",
        eligible=True,
        launch_decision="launch",
        step_name=(
            "evaluation/lm_evaluation_harness_levanter/"
            "lmeval_debug_dclm300m_000_generation_smooth_candidate_a"
        ),
        result_path="executor_output:dclm300m_000_generation_smooth_candidate_a",
    )

    steps, _results_by_key = launcher.build_eval_steps(
        name_prefix="unit-test/dclm-smooth",
        state_rows=[row],
        max_eval_instances=20,
        eval_datasets_cache_path=None,
        log_samples=True,
        sample_log_all=True,
        sample_smooth_metrics=True,
        drop_samples_after_metrics=True,
        use_wandb_tracker=False,
    )

    assert len(steps) == 1
    assert steps[0].name == (
        "evaluation/lm_evaluation_harness_levanter/"
        "lmeval_debug_dclm300m_000_generation_smooth_candidate_a"
    )
    assert steps[0].config.log_samples is True
    assert steps[0].config.sample_log_all is True
    assert steps[0].config.sample_smooth_metrics is True
    assert steps[0].config.drop_samples_after_metrics is True
    assert steps[0].config.use_wandb_tracker is False
    evals = steps[0].config.evals.value
    assert [eval_config.task_alias for eval_config in evals] == [
        "bb_qa_wikidata_10shot",
        "coqa_0shot",
        "squad_10shot",
    ]
    assert all(eval_config.task_kwargs["output_type"] == "loglikelihood" for eval_config in evals)
    assert all("generation_kwargs" not in eval_config.task_kwargs for eval_config in evals)
    assert all("filter_list" not in eval_config.task_kwargs for eval_config in evals)
    assert all(eval_config.task_kwargs["metric_list"][0]["metric"] == "perplexity" for eval_config in evals)
    coqa_eval = evals[1]
    assert coqa_eval.name == launcher.COQA_GENERATION_SMOOTH_TASK
    assert coqa_eval.task_kwargs["dataset_path"] == "EleutherAI/coqa"
    assert coqa_eval.task_kwargs["output_type"] == "loglikelihood"
    assert "process_results" not in coqa_eval.task_kwargs
    assert "generation_kwargs" not in coqa_eval.task_kwargs
    assert "{{story}}" in coqa_eval.task_kwargs["doc_to_text"]
    assert coqa_eval.task_kwargs["doc_to_target"] == "{{answers.input_text[(questions.input_text | length) - 1]}}"
    squad_eval = evals[2]
    assert squad_eval.name == launcher.SQUAD_GENERATION_SMOOTH_TASK
    assert squad_eval.task_kwargs["dataset_path"] == "hazyresearch/based-squad"
    assert squad_eval.task_kwargs["doc_to_text"] == "{{text}}"
    assert squad_eval.task_kwargs["doc_to_target"] == "{{value}}"


def test_build_eval_steps_threads_native_smooth_sample_logging_to_mcq_rows() -> None:
    row = _mcq_eval_row()

    steps, _results_by_key = launcher.build_eval_steps(
        name_prefix="unit-test/dclm-native-smooth",
        state_rows=[row],
        max_eval_instances=20,
        eval_datasets_cache_path=None,
        log_samples=True,
        sample_log_all=True,
        max_logged_samples_per_task=20,
        sample_smooth_metrics=True,
        drop_samples_after_metrics=True,
    )

    assert len(steps) == 1
    assert steps[0].config.log_samples is True
    assert steps[0].config.sample_log_all is True
    assert steps[0].config.max_logged_samples_per_task.value == 20
    assert steps[0].config.sample_smooth_metrics is True
    assert steps[0].config.drop_samples_after_metrics is True
    assert steps[0].config.max_eval_instances.value == 20
    assert steps[0].config.resource_config.preemptible is True
    assert steps[0].fn.resources.preemptible is True


def test_build_eval_steps_can_disable_wandb_tracker_for_mcq_rows() -> None:
    steps, _results_by_key = launcher.build_eval_steps(
        name_prefix="unit-test/dclm-native-smooth",
        state_rows=[_mcq_eval_row()],
        max_eval_instances=20,
        eval_datasets_cache_path=None,
        use_wandb_tracker=False,
    )

    assert len(steps) == 1
    assert steps[0].config.use_wandb_tracker is False


def test_build_eval_steps_can_require_non_preemptible_child_eval_resources() -> None:
    steps, _results_by_key = launcher.build_eval_steps(
        name_prefix="unit-test/dclm-native-smooth",
        state_rows=[_mcq_eval_row()],
        max_eval_instances=20,
        eval_datasets_cache_path=None,
        child_preemptible=False,
    )

    assert len(steps) == 1
    assert steps[0].config.resource_config.preemptible is False
    assert steps[0].fn.resources.preemptible is False
    assert steps[0].config.resource_config.regions == ["us-east5"]
    assert steps[0].config.resource_config.zone == "us-east5-a"


def test_build_eval_steps_can_reuse_existing_eval_dataset_cache_without_cache_step(monkeypatch) -> None:
    checked_paths = []

    def record_cache_manifest_check(path: str) -> None:
        checked_paths.append(path)

    monkeypatch.setattr(launcher, "_ensure_eval_dataset_cache_manifest", record_cache_manifest_check)

    steps, _results_by_key = launcher.build_eval_steps(
        name_prefix="unit-test/dclm-native-smooth",
        state_rows=[_mcq_eval_row()],
        max_eval_instances=20,
        eval_datasets_cache_path="gs://marin-us-east5/raw/eval-datasets/300m-dclm-core-v1",
        reuse_existing_eval_dataset_cache=True,
    )

    assert checked_paths == ["gs://marin-us-east5/raw/eval-datasets/300m-dclm-core-v1"]
    assert len(steps) == 1
    assert "cache_eval_datasets" not in steps[0].name
    assert steps[0].config.eval_datasets_cache_path.value == (
        "gs://marin-us-east5/raw/eval-datasets/300m-dclm-core-v1"
    )
    assert steps[0].config.eval_datasets_cache_dependency is None


def test_signal_candidate_records_include_qsplit_olmix_and_stratified_without_csv(monkeypatch) -> None:
    monkeypatch.setattr(
        launcher,
        "build_qsplit300m_run_specs",
        lambda: [
            SimpleNamespace(run_name="baseline_proportional"),
            SimpleNamespace(run_name="run_00002"),
        ],
    )
    monkeypatch.setattr(
        launcher,
        "_resolve_qsplit300m_checkpoint_root",
        lambda run_name: f"gs://marin-us-east5/checkpoints/{run_name}",
    )

    rows = launcher._signal_candidate_records()

    assert {row.run_name for row in rows} == {
        "baseline_proportional",
        "run_00002",
        "baseline_olmix_loglinear_uncheatable_bpb",
        "baseline_stratified",
    }
    stratified = next(row for row in rows if row.run_name == "baseline_stratified")
    assert stratified.checkpoint_root.startswith("gs://marin-us-east5/")
    assert all(row.cohort == "signal" for row in rows)


def test_signal_candidate_records_use_completed_csv_checkpoint_roots(monkeypatch, tmp_path) -> None:
    completed_csv = tmp_path / "completed.csv"
    pd.DataFrame.from_records(
        [
            {
                "run_name": "baseline_proportional",
                "checkpoint_root": "gs://marin-us-east5/checkpoints/completed/baseline_proportional-abc",
            },
            {
                "run_name": "baseline_olmix_loglinear_uncheatable_bpb",
                "checkpoint_root": "gs://marin-us-east5/checkpoints/completed/olmix-def",
            },
        ]
    ).to_csv(completed_csv, index=False)
    monkeypatch.setattr(launcher, "QSPLIT300M_COMPLETED_CSV", completed_csv)
    monkeypatch.setattr(
        launcher,
        "build_qsplit300m_run_specs",
        lambda: [SimpleNamespace(run_name="baseline_proportional")],
    )
    monkeypatch.setattr(
        launcher,
        "_resolve_qsplit300m_checkpoint_root",
        lambda run_name: pytest.fail(f"unexpected GCS checkpoint resolution for {run_name}"),
    )

    rows = launcher._signal_candidate_records({"baseline_proportional", "baseline_olmix_loglinear_uncheatable_bpb"})

    by_name = {row.run_name: row for row in rows}
    assert by_name["baseline_proportional"].checkpoint_root == (
        "gs://marin-us-east5/checkpoints/completed/baseline_proportional-abc"
    )
    assert by_name["baseline_olmix_loglinear_uncheatable_bpb"].checkpoint_root == (
        "gs://marin-us-east5/checkpoints/completed/olmix-def"
    )


def test_resolved_qsplit_checkpoint_root_strips_levanter_checkpoint_suffix(monkeypatch) -> None:
    monkeypatch.setattr(
        launcher,
        "resolve_latest_checkpoint_root",
        lambda **kwargs: "gs://marin-us-east5/checkpoints/example/run_00001-abc123/checkpoints",
    )

    assert launcher._resolve_qsplit300m_checkpoint_root("run_00001") == (
        "gs://marin-us-east5/checkpoints/example/run_00001-abc123"
    )


def test_build_state_rows_can_assume_exact_hf_checkpoints_without_gcs_probe(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(launcher, "METRICS_WIDE_CSV", tmp_path / "missing.csv")
    monkeypatch.setattr(launcher, "MERGED_RESULTS_CSV", tmp_path / "missing_merged.csv")
    monkeypatch.setattr(
        launcher,
        "_dclm_candidate_records",
        lambda included_panels, included_run_names=None: [
            SimpleNamespace(
                panel="signal_300m_6b",
                run_name="candidate_a",
                registry_key="signal:candidate_a",
                source_experiment="source",
                cohort="signal",
                checkpoint_root="gs://marin-us-east5/checkpoints/a",
                expected_checkpoint_step=22887,
            )
        ],
    )
    monkeypatch.setattr(
        launcher,
        "_exact_hf_checkpoint",
        lambda checkpoint_root, expected_step: pytest.fail("unexpected exact HF checkpoint probe"),
    )

    rows = launcher.build_state_rows(
        default_tpu_type="v5p-8",
        default_tpu_region="us-east5",
        default_tpu_zone="us-east5-a",
        eval_key_suffix="full",
        mode=launcher.DCLMEvalMode.MCQ,
        task_aliases=("boolq_10shot",),
        included_run_names=None,
        included_panels={"signal_300m_6b"},
        assume_exact_hf_checkpoints=True,
    )

    assert rows[0].has_exact_hf_checkpoint is True
    assert rows[0].hf_checkpoint_latest == "gs://marin-us-east5/checkpoints/a/hf/step-22887"
    assert rows[0].launch_decision == "launch"


def test_collect_eval_results_merges_existing_metrics_for_skipped_tasks(monkeypatch, tmp_path) -> None:
    metrics_csv = tmp_path / "existing.csv"
    pd.DataFrame.from_records(
        [
            {
                "checkpoint_root": "gs://marin-us-east5/checkpoints/a",
                "lm_eval/boolq_10shot/acc": 0.75,
            }
        ]
    ).to_csv(metrics_csv, index=False)
    monkeypatch.setattr(launcher, "METRICS_WIDE_CSV", metrics_csv)
    monkeypatch.setattr(launcher, "MERGED_RESULTS_CSV", tmp_path / "missing_merged.csv")
    monkeypatch.setattr(
        launcher,
        "_scan_eval_statuses",
        lambda prefixes: {
            "dclm300m_000_mcq_candidate_a": {
                "eval_key": "dclm300m_000_mcq_candidate_a",
                "output_path": "gs://marin-us-east5/results/a",
                "prefix": prefixes[0],
                "status": launcher.STATUS_SUCCESS,
                "status_path": "gs://marin-us-east5/results/a/.executor_status",
            }
        },
    )
    monkeypatch.setattr(
        launcher,
        "_read_eval_metrics",
        lambda path: (
            {
                "lm_eval/arc_easy_10shot/acc_norm": 0.50,
                "lm_eval/arc_easy_10shot/native_margin": 0.125,
                "lm_eval/arc_easy_10shot/native_gold_bpb": 1.75,
            },
            "",
        ),
    )
    rows = [
        launcher.DCLMEvalSpec(
            eval_key="dclm300m_000_mcq_candidate_a",
            mode="mcq",
            panel="signal_300m_6b",
            run_name="candidate_a",
            registry_key="signal:candidate_a",
            source_experiment="source",
            cohort="signal",
            checkpoint_root="gs://marin-us-east5/checkpoints/a",
            expected_checkpoint_step=22887,
            hf_checkpoint_count=1,
            hf_checkpoint_latest="gs://marin-us-east5/checkpoints/a/hf/step-22887",
            hf_checkpoint_latest_step=22887,
            has_exact_hf_checkpoint=True,
            checkpoint_region="us-east5",
            is_region_local=True,
            existing_artifact_count=1,
            existing_tasks="boolq_10shot",
            missing_task_count=1,
            missing_tasks="arc_easy_10shot",
            has_all_tasks=False,
            task_aliases="arc_easy_10shot",
            launch_tpu_type="v5p-8",
            launch_tpu_region="us-east5",
            launch_tpu_zone="us-east5-a",
            eligible=True,
            launch_decision="launch",
            step_name="evaluation/lm_evaluation_harness_levanter/lmeval_debug_dclm300m_000_mcq_candidate_a",
            result_path="executor_output:dclm300m_000_mcq_candidate_a",
        )
    ]

    frame = launcher.collect_eval_results_from_prefixes(
        prefixes=["gs://marin-us-east5/prefix"],
        state_rows=rows,
        output_csv=tmp_path / "collected.csv",
    )

    row = frame.iloc[0]
    assert row["run_name"] == "candidate_a"
    assert row["lm_eval/dclm_core/boolq_10shot/raw_score"] == pytest.approx(0.75)
    assert row["lm_eval/dclm_core/arc_easy_10shot/raw_score"] == pytest.approx(0.50)
    assert row["lm_eval/arc_easy_10shot/native_margin"] == pytest.approx(0.125)
    assert row["lm_eval/arc_easy_10shot/native_gold_bpb"] == pytest.approx(1.75)
    assert row["lm_eval/dclm_core/task_count"] == 2.0


def test_write_gapfill_state_from_results_uses_unique_alias_exact_rows(tmp_path) -> None:
    results_csv = tmp_path / "results.csv"
    pd.DataFrame.from_records(
        [
            {
                "checkpoint_root": "gs://marin-us-east5/checkpoints/a",
                "run_name": "candidate_a",
                "lm_eval/arc_easy/acc_norm": 0.50,
                "lm_eval/dclm_core/boolq_10shot/raw_score": 0.75,
            }
        ]
    ).to_csv(results_csv, index=False)
    state_rows = [
        launcher.DCLMEvalSpec(
            eval_key="dclm300m_000_mcq_candidate_a",
            mode="mcq",
            panel="signal_300m_6b",
            run_name="candidate_a",
            registry_key="signal:candidate_a",
            source_experiment="source",
            cohort="signal",
            checkpoint_root="gs://marin-us-east5/checkpoints/a",
            expected_checkpoint_step=22887,
            hf_checkpoint_count=1,
            hf_checkpoint_latest="gs://marin-us-east5/checkpoints/a/hf/step-22887",
            hf_checkpoint_latest_step=22887,
            has_exact_hf_checkpoint=True,
            checkpoint_region="us-east5",
            is_region_local=True,
            existing_artifact_count=0,
            existing_tasks="",
            missing_task_count=3,
            missing_tasks="boolq_10shot;arc_easy_10shot;piqa_10shot",
            has_all_tasks=False,
            task_aliases="boolq_10shot;arc_easy_10shot;piqa_10shot",
            launch_tpu_type="v5p-8",
            launch_tpu_region="us-east5",
            launch_tpu_zone="us-east5-a",
            eligible=True,
            launch_decision="launch",
            step_name="evaluation/lm_evaluation_harness_levanter/lmeval_debug_dclm300m_000_mcq_candidate_a",
            result_path="executor_output:dclm300m_000_mcq_candidate_a",
        )
    ]

    rows = launcher.write_gapfill_state_from_results(
        state_rows=state_rows,
        results_csv=results_csv,
        output_csv=tmp_path / "gapfill.csv",
        eval_key_suffix="gap20260609",
    )

    assert len(rows) == 1
    assert rows[0].eval_key == "dclm300m_000_mcq_candidate_a_gap20260609"
    assert rows[0].missing_tasks == "arc_easy_10shot;piqa_10shot"
    assert rows[0].task_aliases == "arc_easy_10shot;piqa_10shot"
    assert rows[0].step_name.endswith("lmeval_debug_dclm300m_000_mcq_candidate_a_gap20260609")


def test_write_native_smooth_gapfill_state_requires_native_metric_rows(tmp_path) -> None:
    results_csv = tmp_path / "results.csv"
    pd.DataFrame.from_records(
        [
            {
                "checkpoint_root": "gs://marin-us-east5/checkpoints/a",
                "run_name": "candidate_a",
                "lm_eval/boolq_10shot/acc": 0.75,
                "lm_eval/arc_easy_10shot/native_margin": 0.125,
                "lm_eval/piqa_10shot/bpb": 0.90,
            }
        ]
    ).to_csv(results_csv, index=False)
    state_rows = [
        launcher.DCLMEvalSpec(
            eval_key="dclm300m_000_mcq_candidate_a",
            mode="mcq",
            panel="signal_300m_6b",
            run_name="candidate_a",
            registry_key="signal:candidate_a",
            source_experiment="source",
            cohort="signal",
            checkpoint_root="gs://marin-us-east5/checkpoints/a",
            expected_checkpoint_step=22887,
            hf_checkpoint_count=1,
            hf_checkpoint_latest="gs://marin-us-east5/checkpoints/a/hf/step-22887",
            hf_checkpoint_latest_step=22887,
            has_exact_hf_checkpoint=True,
            checkpoint_region="us-east5",
            is_region_local=True,
            existing_artifact_count=0,
            existing_tasks="",
            missing_task_count=4,
            missing_tasks="boolq_10shot;arc_easy_10shot;piqa_10shot;coqa_0shot",
            has_all_tasks=False,
            task_aliases="boolq_10shot;arc_easy_10shot;piqa_10shot;coqa_0shot",
            launch_tpu_type="v5p-8",
            launch_tpu_region="us-east5",
            launch_tpu_zone="us-east5-a",
            eligible=True,
            launch_decision="launch",
            step_name="evaluation/lm_evaluation_harness_levanter/lmeval_debug_dclm300m_000_mcq_candidate_a",
            result_path="executor_output:dclm300m_000_mcq_candidate_a",
        )
    ]

    rows = launcher.write_native_smooth_gapfill_state_from_results(
        state_rows=state_rows,
        results_csv=results_csv,
        output_csv=tmp_path / "native_smooth_gapfill.csv",
        eval_key_suffix="native_smooth_gap20260610",
    )

    assert len(rows) == 1
    assert rows[0].eval_key == "dclm300m_000_mcq_candidate_a_native_smooth_gap20260610"
    assert rows[0].missing_tasks == "boolq_10shot"
    assert rows[0].task_aliases == "boolq_10shot"


def test_write_native_smooth_gapfill_state_counts_metrics_from_running_status_rows(tmp_path) -> None:
    results_csv = tmp_path / "results.csv"
    pd.DataFrame.from_records(
        [
            {
                "checkpoint_root": "gs://marin-us-east5/checkpoints/a",
                "run_name": "candidate_a",
                "collection_status": "executor_status_RUNNING",
                "lm_eval/boolq_10shot/bpb": 0.50,
                "lm_eval/arc_easy_10shot/native_margin": 0.125,
            }
        ]
    ).to_csv(results_csv, index=False)
    state_rows = [
        launcher.DCLMEvalSpec(
            eval_key="dclm300m_000_mcq_candidate_a",
            mode="mcq",
            panel="signal_300m_6b",
            run_name="candidate_a",
            registry_key="signal:candidate_a",
            source_experiment="source",
            cohort="signal",
            checkpoint_root="gs://marin-us-east5/checkpoints/a",
            expected_checkpoint_step=22887,
            hf_checkpoint_count=1,
            hf_checkpoint_latest="gs://marin-us-east5/checkpoints/a/hf/step-22887",
            hf_checkpoint_latest_step=22887,
            has_exact_hf_checkpoint=True,
            checkpoint_region="us-east5",
            is_region_local=True,
            existing_artifact_count=0,
            existing_tasks="",
            missing_task_count=2,
            missing_tasks="boolq_10shot;arc_easy_10shot",
            has_all_tasks=False,
            task_aliases="boolq_10shot;arc_easy_10shot",
            launch_tpu_type="v5p-8",
            launch_tpu_region="us-east5",
            launch_tpu_zone="us-east5-a",
            eligible=True,
            launch_decision="launch",
            step_name="evaluation/lm_evaluation_harness_levanter/lmeval_debug_dclm300m_000_mcq_candidate_a",
            result_path="executor_output:dclm300m_000_mcq_candidate_a",
        )
    ]

    rows = launcher.write_native_smooth_gapfill_state_from_results(
        state_rows=state_rows,
        results_csv=results_csv,
        output_csv=tmp_path / "native_smooth_gapfill.csv",
        eval_key_suffix="native_smooth_gap20260610",
    )

    assert rows == []


def test_write_smooth_gapfill_state_from_results_includes_generation_smooth_rows(tmp_path) -> None:
    results_csv = tmp_path / "results.csv"
    pd.DataFrame.from_records(
        [
            {
                "checkpoint_root": "gs://marin-us-east5/checkpoints/a",
                "run_name": "candidate_a",
                "lm_eval/boolq_10shot/native_gold_bpb,none": 0.75,
                "lm_eval/bb_qa_wikidata_10shot/exact_match,strip-then-match": 0.0,
            }
        ]
    ).to_csv(results_csv, index=False)
    state_rows = [
        launcher.DCLMEvalSpec(
            eval_key="dclm300m_000_mcq_candidate_a",
            mode="mcq",
            panel="signal_300m_6b",
            run_name="candidate_a",
            registry_key="signal:candidate_a",
            source_experiment="source",
            cohort="signal",
            checkpoint_root="gs://marin-us-east5/checkpoints/a",
            expected_checkpoint_step=22887,
            hf_checkpoint_count=1,
            hf_checkpoint_latest="gs://marin-us-east5/checkpoints/a/hf/step-22887",
            hf_checkpoint_latest_step=22887,
            has_exact_hf_checkpoint=True,
            checkpoint_region="us-east5",
            is_region_local=True,
            existing_artifact_count=0,
            existing_tasks="",
            missing_task_count=1,
            missing_tasks="boolq_10shot",
            has_all_tasks=False,
            task_aliases="boolq_10shot",
            launch_tpu_type="v5p-8",
            launch_tpu_region="us-east5",
            launch_tpu_zone="us-east5-a",
            eligible=True,
            launch_decision="launch",
            step_name="evaluation/lm_evaluation_harness_levanter/lmeval_debug_dclm300m_000_mcq_candidate_a",
            result_path="executor_output:dclm300m_000_mcq_candidate_a",
        ),
        launcher.DCLMEvalSpec(
            eval_key="dclm300m_000_generation_smooth_candidate_a",
            mode="generation_smooth",
            panel="signal_300m_6b",
            run_name="candidate_a",
            registry_key="signal:candidate_a",
            source_experiment="source",
            cohort="signal",
            checkpoint_root="gs://marin-us-east5/checkpoints/a",
            expected_checkpoint_step=22887,
            hf_checkpoint_count=1,
            hf_checkpoint_latest="gs://marin-us-east5/checkpoints/a/hf/step-22887",
            hf_checkpoint_latest_step=22887,
            has_exact_hf_checkpoint=True,
            checkpoint_region="us-east5",
            is_region_local=True,
            existing_artifact_count=0,
            existing_tasks="",
            missing_task_count=2,
            missing_tasks="bb_qa_wikidata_10shot;coqa_0shot",
            has_all_tasks=False,
            task_aliases="bb_qa_wikidata_10shot;coqa_0shot",
            launch_tpu_type="v5p-8",
            launch_tpu_region="us-east5",
            launch_tpu_zone="us-east5-a",
            eligible=True,
            launch_decision="launch",
            step_name=(
                "evaluation/lm_evaluation_harness_levanter/"
                "lmeval_debug_dclm300m_000_generation_smooth_candidate_a"
            ),
            result_path="executor_output:dclm300m_000_generation_smooth_candidate_a",
        ),
    ]

    rows = launcher.write_smooth_gapfill_state_from_results(
        state_rows=state_rows,
        results_csv=results_csv,
        output_csv=tmp_path / "smooth_gapfill.csv",
        eval_key_suffix="smooth_gap20260613",
    )

    assert len(rows) == 1
    assert rows[0].mode == "generation_smooth"
    assert rows[0].missing_tasks == "bb_qa_wikidata_10shot;coqa_0shot"
    assert rows[0].step_name.endswith("lmeval_debug_dclm300m_000_generation_smooth_candidate_a_smooth_gap20260613")


def test_merge_records_by_checkpoint_unions_mode_metrics_and_task_metadata() -> None:
    merged = launcher._merge_records_by_checkpoint(
        [
            {
                "eval_key": "dclm300m_000_extractive_candidate_a",
                "mode": "extractive",
                "run_name": "candidate_a",
                "checkpoint_root": "gs://marin-us-east5/checkpoints/a",
                "task_aliases": "coqa_0shot",
                "missing_tasks": "coqa_0shot",
                "collection_status": "collected",
                "collection_error": "",
                "lm_eval/coqa_0shot/f1": 0.4,
            },
            {
                "eval_key": "dclm300m_000_mcq_candidate_a",
                "mode": "mcq",
                "run_name": "candidate_a",
                "checkpoint_root": "gs://marin-us-east5/checkpoints/a",
                "task_aliases": "boolq_10shot",
                "missing_tasks": "boolq_10shot",
                "collection_status": "collected",
                "collection_error": "",
                "lm_eval/boolq_10shot/acc": 0.75,
            },
        ]
    )

    assert len(merged) == 1
    row = merged[0]
    assert row["mode"] == "extractive;mcq"
    assert row["eval_key"] == "dclm300m_000_extractive_candidate_a;dclm300m_000_mcq_candidate_a"
    assert row["task_aliases"] == "coqa_0shot;boolq_10shot"
    assert row["missing_tasks"] == "coqa_0shot;boolq_10shot"
    assert row["lm_eval/coqa_0shot/f1"] == 0.4
    assert row["lm_eval/boolq_10shot/acc"] == 0.75
    assert row["lm_eval/dclm_core/task_count"] == 2.0
    assert row["lm_eval/dclm_core/centered_accuracy_macro"] == pytest.approx((0.4 + (0.75 - 0.62) / 0.38) / 2.0)


def test_generation_steps_enable_tokenized_requests_for_prompt_truncation() -> None:
    row = launcher.DCLMEvalSpec(
        eval_key="dclm300m_000_generation_candidate_a",
        mode="generation",
        panel="signal_300m_6b",
        run_name="candidate_a",
        registry_key="signal:candidate_a",
        source_experiment="source",
        cohort="signal",
        checkpoint_root="gs://marin-us-east5/checkpoints/a",
        expected_checkpoint_step=22887,
        hf_checkpoint_count=1,
        hf_checkpoint_latest="gs://marin-us-east5/checkpoints/a/hf/step-22887",
        hf_checkpoint_latest_step=22887,
        has_exact_hf_checkpoint=True,
        checkpoint_region="us-east5",
        is_region_local=True,
        existing_artifact_count=0,
        existing_tasks="",
        missing_task_count=1,
        missing_tasks="bb_repeat_copy_logic_10shot",
        has_all_tasks=False,
        task_aliases="bb_repeat_copy_logic_10shot",
        launch_tpu_type="v5p-8",
        launch_tpu_region="us-east5",
        launch_tpu_zone="us-east5-a",
        eligible=True,
        launch_decision="launch",
        step_name="evaluation/lm_evaluation_harness/dclm300m_000_generation_candidate_a",
        result_path="executor_output:dclm300m_000_generation_candidate_a",
    )

    steps, _ = launcher.build_eval_steps(
        name_prefix="test_dclm",
        state_rows=[row],
        max_eval_instances=20,
        eval_datasets_cache_path=None,
    )

    assert len(steps) == 1
    assert steps[0].config.engine_kwargs["tokenized_requests"] is True
    assert steps[0].config.engine_kwargs["max_num_batched_tokens"] == 4096
    assert steps[0].config.engine_kwargs["max_model_len"] == 2048
    assert steps[0].config.engine_kwargs["max_length"] == 2048
    assert steps[0].config.engine_kwargs["truncate"] is True
    assert steps[0].config.engine_kwargs["max_gen_toks"] == 128
    assert steps[0].config.generation_params == {"max_gen_toks": 128}
    assert steps[0].config.evals[0].task_kwargs["filter_list"][0]["name"] == "strip-then-match"


def test_extractive_steps_keep_string_prompts_with_bounded_generation_length() -> None:
    row = launcher.DCLMEvalSpec(
        eval_key="dclm300m_000_extractive_candidate_a",
        mode="extractive",
        panel="signal_300m_6b",
        run_name="candidate_a",
        registry_key="signal:candidate_a",
        source_experiment="source",
        cohort="signal",
        checkpoint_root="gs://marin-us-east5/checkpoints/a",
        expected_checkpoint_step=22887,
        hf_checkpoint_count=1,
        hf_checkpoint_latest="gs://marin-us-east5/checkpoints/a/hf/step-22887",
        hf_checkpoint_latest_step=22887,
        has_exact_hf_checkpoint=True,
        checkpoint_region="us-east5",
        is_region_local=True,
        existing_artifact_count=0,
        existing_tasks="",
        missing_task_count=1,
        missing_tasks="coqa_0shot",
        has_all_tasks=False,
        task_aliases="coqa_0shot",
        launch_tpu_type="v5p-8",
        launch_tpu_region="us-east5",
        launch_tpu_zone="us-east5-a",
        eligible=True,
        launch_decision="launch",
        step_name="evaluation/lm_evaluation_harness/dclm300m_000_extractive_candidate_a",
        result_path="executor_output:dclm300m_000_extractive_candidate_a",
    )

    steps, _ = launcher.build_eval_steps(
        name_prefix="test_dclm",
        state_rows=[row],
        max_eval_instances=20,
        eval_datasets_cache_path=None,
    )

    assert len(steps) == 1
    assert "tokenized_requests" not in steps[0].config.engine_kwargs
    assert steps[0].config.engine_kwargs["max_gen_toks"] == 32
    assert steps[0].config.generation_params == {"max_gen_toks": 32}


def test_dclm_launcher_defaults_use_maximal_east5_parallelism() -> None:
    assert launcher.DEFAULT_TPU_TYPE == "v5p-8"
    assert launcher.DEFAULT_TPU_REGION == "us-east5"
    assert launcher.DEFAULT_TPU_ZONE == "us-east5-a"
    assert launcher.DEFAULT_MAX_CONCURRENT >= 512
    assert launcher.DCLM_GENERATION_PARAMS["max_gen_toks"] == 128
    assert launcher.DCLM_EXTRACTIVE_GENERATION_PARAMS["max_gen_toks"] == 32
    assert launcher.DCLM_EXTRACTIVE_ENGINE_KWARGS["max_gen_toks"] == 32
    assert launcher.DCLM_GENERATION_ENGINE_KWARGS["max_gen_toks"] == 128
    assert launcher.DCLM_GENERATION_ENGINE_KWARGS["max_model_len"] == 2048
    assert launcher.DCLM_GENERATION_ENGINE_KWARGS["max_length"] == 2048
    assert launcher.DCLM_GENERATION_ENGINE_KWARGS["max_num_batched_tokens"] == 4096
    assert launcher.DCLM_GENERATION_ENGINE_KWARGS["tokenized_requests"] is True


def test_proportional_noise_dclm_candidates_from_completed_rows(tmp_path: Path) -> None:
    matrix_csv = tmp_path / "noise.csv"
    pd.DataFrame.from_records(
        [
            {
                "run_name": "propvar_300m_6b_trainer_seed_10000",
                "registry_run_key": "300m_6b:seed_sweep:source:seed10000",
                "source_experiment": "pinlin/source",
                "checkpoint_root": "gs://marin-us-east5/checkpoints/seed10000",
                "target_final_checkpoint_step": 22887,
                "status": "completed",
            },
            {
                "run_name": "propvar_300m_6b_trainer_seed_10001",
                "registry_run_key": "300m_6b:seed_sweep:source:seed10001",
                "source_experiment": "pinlin/source",
                "checkpoint_root": "gs://marin-us-east5/checkpoints/seed10001/",
                "target_final_checkpoint_step": 22887,
                "status": "completed",
            },
        ]
    ).to_csv(matrix_csv, index=False)

    candidates = proportional_noise.proportional_noise_candidates(matrix_csv, allow_incomplete=True)

    assert [candidate.run_name for candidate in candidates] == [
        "propvar_300m_6b_trainer_seed_10000",
        "propvar_300m_6b_trainer_seed_10001",
    ]
    assert candidates[0].panel == "proportional_variable_subset_noise_300m_6b"
    assert candidates[0].cohort == "proportional_noise"
    assert candidates[0].checkpoint_root == "gs://marin-us-east5/checkpoints/seed10000"
    assert candidates[1].checkpoint_root == "gs://marin-us-east5/checkpoints/seed10001"


def test_proportional_noise_dclm_combined_state_splits_hard_and_smooth(tmp_path: Path) -> None:
    matrix_csv = tmp_path / "noise.csv"
    pd.DataFrame.from_records(
        [
            {
                "run_name": "propvar_300m_6b_trainer_seed_10000",
                "registry_run_key": "300m_6b:seed_sweep:source:seed10000",
                "source_experiment": "pinlin/source",
                "checkpoint_root": "gs://marin-us-east5/checkpoints/seed10000",
                "target_final_checkpoint_step": 22887,
                "status": "completed",
            }
        ]
    ).to_csv(matrix_csv, index=False)

    rows = proportional_noise.build_combined_state_rows(
        input_csv=matrix_csv,
        eval_key_suffix="unit",
        tpu_type="v5p-8",
        tpu_region="us-east5",
        tpu_zone="us-east5-a",
        split_task_alias_rows=True,
        allow_incomplete=True,
    )

    assert len(rows) == 30
    assert len({row.eval_key for row in rows}) == 30
    assert {row.launch_decision for row in rows} == {"launch"}
    assert {row.launch_tpu_region for row in rows} == {"us-east5"}
    assert {row.launch_tpu_zone for row in rows} == {"us-east5-a"}
    assert {row.expected_checkpoint_step for row in rows} == {22887}
    assert all(row.missing_task_count == 1 for row in rows)
    assert sum(row.mode == "generation_smooth" for row in rows) == 8
    assert sum(row.mode == "mcq" for row in rows) == 14
    assert sum(row.mode == "generation" for row in rows) == 8
    assert all(row.hf_checkpoint_latest.endswith("/hf/step-22887") for row in rows)
