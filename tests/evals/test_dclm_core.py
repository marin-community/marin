# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marin.evaluation.evaluation_config import convert_to_levanter_task_config

from experiments.evals import dclm_core
from experiments.evals.task_configs import DCLM_CORE_V2_TASKS


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
    assert dclm_core.task_by_alias("arc_easy_10shot").primary_metric == "acc_norm"
    assert dclm_core.task_by_alias("boolq_10shot").random_baseline == 0.62
    assert dclm_core.task_by_alias("commonsense_qa_10shot").random_baseline == 0.403


def test_custom_data_tasks_are_not_launchable_until_provenance_is_resolved() -> None:
    launchable_aliases = set(dclm_core.launchable_task_aliases())

    assert "jeopardy_10shot" not in launchable_aliases
    assert "winograd_0shot" not in launchable_aliases
    assert dclm_core.task_by_alias("jeopardy_10shot").status == dclm_core.TaskStatus.REQUIRES_CUSTOM_TASK
    assert dclm_core.task_by_alias("winograd_0shot").status == dclm_core.TaskStatus.REQUIRES_CUSTOM_TASK
    with pytest.raises(ValueError, match="not launchable"):
        dclm_core.task_by_alias("jeopardy_10shot").eval_config()


def test_bigbench_generation_tasks_are_not_launchable_until_filtered_metrics_are_supported() -> None:
    launchable_aliases = set(dclm_core.launchable_task_aliases())

    assert "bb_qa_wikidata_10shot" not in launchable_aliases
    assert dclm_core.task_by_alias("bb_qa_wikidata_10shot").status == dclm_core.TaskStatus.REQUIRES_FILTERED_GENERATION
    with pytest.raises(ValueError, match="not launchable"):
        dclm_core.task_by_alias("bb_qa_wikidata_10shot").eval_config()


def test_task_configs_exports_only_launchable_dclm_core_v2_tasks() -> None:
    aliases = {task.task_alias for task in DCLM_CORE_V2_TASKS}

    assert len(DCLM_CORE_V2_TASKS) == 15
    assert aliases == set(dclm_core.launchable_task_aliases())
    assert "bb_qa_wikidata_10shot" not in aliases
    assert "jeopardy_10shot" not in aliases
    assert "winograd_0shot" not in aliases


def test_launchable_eval_configs_convert_to_levanter_task_configs() -> None:
    task = dclm_core.task_by_alias("boolq_10shot")

    [config] = convert_to_levanter_task_config([task.eval_config()])

    assert config.task == "boolq"
    assert config.task_alias == "boolq_10shot"


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


def test_bigbench_generation_scoring_prefers_filtered_exact_match() -> None:
    metrics = {
        "lm_eval/bigbench_qa_wikidata_generate_until/exact_match,none": 0.0,
        "lm_eval/bigbench_qa_wikidata_generate_until/exact_match,strip-then-match": 0.30,
    }

    centered = dclm_core.dclm_core_centered_accuracy(metrics, task_aliases=("bb_qa_wikidata_10shot",))

    assert centered["lm_eval/dclm_core/bb_qa_wikidata_10shot/raw_score"] == 0.30
    assert centered["lm_eval/dclm_core/bb_qa_wikidata_10shot/centered_accuracy"] == 0.30
    assert centered["lm_eval/dclm_core/task_count"] == 1.0


def test_bigbench_generation_scoring_rejects_unfiltered_exact_match_none() -> None:
    metrics = {
        "lm_eval/bigbench_qa_wikidata_generate_until/exact_match,none": 0.0,
    }

    centered = dclm_core.dclm_core_centered_accuracy(metrics, task_aliases=("bb_qa_wikidata_10shot",))

    assert "lm_eval/dclm_core/bb_qa_wikidata_10shot/raw_score" not in centered
    assert centered["lm_eval/dclm_core/task_count"] == 0.0
    assert centered["lm_eval/dclm_core/missing_task_count"] == 1.0


def test_bigbench_generation_scoring_rejects_post_strip_unfiltered_exact_match() -> None:
    metrics = {
        "lm_eval/bigbench_qa_wikidata_generate_until/exact_match": 0.0,
    }

    centered = dclm_core.dclm_core_centered_accuracy(metrics, task_aliases=("bb_qa_wikidata_10shot",))

    assert "lm_eval/dclm_core/bb_qa_wikidata_10shot/raw_score" not in centered
    assert centered["lm_eval/dclm_core/task_count"] == 0.0
    assert centered["lm_eval/dclm_core/missing_task_count"] == 1.0
