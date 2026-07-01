# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DCLM Core v2 task inventory and centered-accuracy scoring helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from marin.evaluation.evaluation_config import EvalTaskConfig


class TaskStatus(StrEnum):
    """Implementation status for one DCLM Core task."""

    LAUNCHABLE = "launchable"
    REQUIRES_CUSTOM_TASK = "requires_custom_task"
    REQUIRES_FILTERED_GENERATION = "requires_filtered_generation"


class TaskMode(StrEnum):
    """Evaluation backend mode for one DCLM Core task."""

    EXTRACTIVE = "extractive"
    MCQ = "mcq"
    GENERATION = "generation"


BIGBENCH_FILTERED_EXACT_MATCH_METRICS = (
    "exact_match,strip-then-match",
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "em",
    "f1",
)

EXACT_MATCH_METRICS = (
    "exact_match",
    "exact_match,none",
    "exact_match,strip-then-match",
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "em",
    "f1",
)


@dataclass(frozen=True)
class DCLMCoreTask:
    """One DCLM Core v2 task and its DCLM-centered scoring metadata."""

    name: str
    alias: str
    num_fewshot: int
    mode: TaskMode
    random_baseline: float
    primary_metric: str
    status: TaskStatus = TaskStatus.LAUNCHABLE
    task_kwargs: dict[str, Any] | None = None
    metric_candidates: tuple[str, ...] = ()
    metric_task_names: tuple[str, ...] = ()

    def eval_config(self) -> EvalTaskConfig:
        """Return the Marin eval task config for launchable tasks."""
        if self.status != TaskStatus.LAUNCHABLE:
            raise ValueError(f"DCLM Core task {self.alias} is not launchable: {self.status}")
        return EvalTaskConfig(
            self.name,
            self.num_fewshot,
            task_alias=self.alias,
            task_kwargs=self.task_kwargs,
        )

    def metric_paths(self) -> tuple[str, ...]:
        """Return flattened lm-eval metric columns that can score this task."""
        metrics = self.metric_candidates or (self.primary_metric,)
        task_names = (self.alias, self.name, *self.metric_task_names)
        paths: list[str] = []
        seen: set[str] = set()
        for task_name in task_names:
            for metric in metrics:
                path = f"lm_eval/{task_name}/{metric}"
                if path in seen:
                    continue
                seen.add(path)
                paths.append(path)
        return tuple(paths)


def _bigbench_generate_until_task_kwargs(dataset_name: str) -> dict[str, Any]:
    return {
        "dataset_path": "hails/bigbench",
        "dataset_name": dataset_name,
        "test_split": "default",
        "output_type": "generate_until",
        "doc_to_text": "inputs",
        "doc_to_target": "{{targets[0]}}",
        "metric_list": [
            {
                "metric": "exact_match",
                "aggregation": "mean",
                "higher_is_better": True,
                "ignore_punctuation": True,
            }
        ],
        "metadata": {
            "version": 1.0,
            "source": "lm-eval BigBench generate-until template adapted for DCLM Core scoring",
        },
    }


def dclm_core_tasks() -> tuple[DCLMCoreTask, ...]:
    """Return the 22-task DCLM Core v2 inventory from the DCLM paper."""
    return (
        DCLMCoreTask("agieval_lsat_ar", "agieval_lsat_ar_3shot", 3, TaskMode.MCQ, 0.25, "acc_norm"),
        DCLMCoreTask("arc_easy", "arc_easy_10shot", 10, TaskMode.MCQ, 0.25, "acc_norm"),
        DCLMCoreTask("arc_challenge", "arc_challenge_10shot", 10, TaskMode.MCQ, 0.25, "acc_norm"),
        DCLMCoreTask(
            "bigbench_qa_wikidata_generate_until",
            "bb_qa_wikidata_10shot",
            10,
            TaskMode.GENERATION,
            0.0,
            "exact_match",
            status=TaskStatus.REQUIRES_FILTERED_GENERATION,
            task_kwargs=_bigbench_generate_until_task_kwargs("qa_wikidata_zero_shot"),
            metric_candidates=BIGBENCH_FILTERED_EXACT_MATCH_METRICS,
        ),
        DCLMCoreTask(
            "bigbench_dyck_languages_generate_until",
            "bb_dyck_languages_10shot",
            10,
            TaskMode.GENERATION,
            0.0,
            "exact_match",
            status=TaskStatus.REQUIRES_FILTERED_GENERATION,
            task_kwargs=_bigbench_generate_until_task_kwargs("dyck_languages_zero_shot"),
            metric_candidates=BIGBENCH_FILTERED_EXACT_MATCH_METRICS,
        ),
        DCLMCoreTask(
            "bigbench_operators_generate_until",
            "bb_operators_10shot",
            10,
            TaskMode.GENERATION,
            0.0,
            "exact_match",
            status=TaskStatus.REQUIRES_FILTERED_GENERATION,
            task_kwargs=_bigbench_generate_until_task_kwargs("operators_zero_shot"),
            metric_candidates=BIGBENCH_FILTERED_EXACT_MATCH_METRICS,
        ),
        DCLMCoreTask(
            "bigbench_repeat_copy_logic_generate_until",
            "bb_repeat_copy_logic_10shot",
            10,
            TaskMode.GENERATION,
            0.0,
            "exact_match",
            status=TaskStatus.REQUIRES_FILTERED_GENERATION,
            task_kwargs=_bigbench_generate_until_task_kwargs("repeat_copy_logic_zero_shot"),
            metric_candidates=BIGBENCH_FILTERED_EXACT_MATCH_METRICS,
        ),
        DCLMCoreTask(
            "bigbench_cs_algorithms_generate_until",
            "bb_cs_algorithms_10shot",
            10,
            TaskMode.GENERATION,
            0.0,
            "exact_match",
            status=TaskStatus.REQUIRES_FILTERED_GENERATION,
            task_kwargs=_bigbench_generate_until_task_kwargs("cs_algorithms_zero_shot"),
            metric_candidates=BIGBENCH_FILTERED_EXACT_MATCH_METRICS,
        ),
        DCLMCoreTask(
            "bigbench_language_identification_multiple_choice",
            "bb_language_identification_10shot",
            10,
            TaskMode.MCQ,
            0.25,
            "acc",
        ),
        DCLMCoreTask("boolq", "boolq_10shot", 10, TaskMode.MCQ, 0.62, "acc"),
        DCLMCoreTask("commonsense_qa", "commonsense_qa_10shot", 10, TaskMode.MCQ, 0.403, "acc"),
        DCLMCoreTask("copa", "copa_0shot", 0, TaskMode.MCQ, 0.5, "acc"),
        DCLMCoreTask(
            "coqa",
            "coqa_0shot",
            0,
            TaskMode.GENERATION,
            0.0,
            "f1",
            metric_candidates=("f1", "em", *EXACT_MATCH_METRICS),
        ),
        # DCLM Core counts HellaSwag at both 0-shot and 10-shot.
        DCLMCoreTask("hellaswag", "hellaswag_0shot", 0, TaskMode.MCQ, 0.25, "acc_norm"),
        DCLMCoreTask("hellaswag", "hellaswag_10shot", 10, TaskMode.MCQ, 0.25, "acc_norm"),
        DCLMCoreTask(
            "jeopardy",
            "jeopardy_10shot",
            10,
            TaskMode.GENERATION,
            0.0,
            "exact_match",
            status=TaskStatus.REQUIRES_CUSTOM_TASK,
            metric_candidates=EXACT_MATCH_METRICS,
        ),
        DCLMCoreTask("lambada_openai", "lambada_0shot", 0, TaskMode.MCQ, 0.0, "acc"),
        DCLMCoreTask("openbookqa", "openbookqa_0shot", 0, TaskMode.MCQ, 0.25, "acc_norm"),
        DCLMCoreTask("piqa", "piqa_10shot", 10, TaskMode.MCQ, 0.5, "acc_norm"),
        DCLMCoreTask(
            "squad_completion",
            "squad_10shot",
            10,
            TaskMode.GENERATION,
            0.0,
            "contains",
            metric_candidates=("contains", "exact", "exact_match", "f1"),
        ),
        DCLMCoreTask(
            "winograd",
            "winograd_0shot",
            0,
            TaskMode.MCQ,
            0.5,
            "acc",
            status=TaskStatus.REQUIRES_CUSTOM_TASK,
        ),
        DCLMCoreTask("winogrande", "winogrande_0shot", 0, TaskMode.MCQ, 0.5, "acc"),
    )


def task_by_alias(alias: str) -> DCLMCoreTask:
    """Return a DCLM Core task by alias."""
    by_alias = {task.alias: task for task in dclm_core_tasks()}
    try:
        return by_alias[alias]
    except KeyError as exc:
        raise ValueError(f"Unknown DCLM Core task alias: {alias}") from exc


def dclm_core_task_aliases() -> tuple[str, ...]:
    """Return aliases for all DCLM Core v2 tasks, including custom-task gaps."""
    return tuple(task.alias for task in dclm_core_tasks())


def launchable_task_aliases() -> tuple[str, ...]:
    """Return aliases for DCLM Core tasks with safe, currently supported launch configs."""
    return tuple(task.alias for task in dclm_core_tasks() if task.status == TaskStatus.LAUNCHABLE)


def task_aliases_for_mode(mode: TaskMode) -> tuple[str, ...]:
    """Return launchable aliases for one evaluator mode."""
    return tuple(task.alias for task in dclm_core_tasks() if task.status == TaskStatus.LAUNCHABLE and task.mode == mode)


def eval_tasks_for_aliases(task_aliases: tuple[str, ...]) -> list[EvalTaskConfig]:
    """Return EvalTaskConfig objects in the requested alias order."""
    return [task_by_alias(alias).eval_config() for alias in task_aliases]


def _metric_value_for_task(metrics: dict[str, Any], task: DCLMCoreTask) -> float | None:
    for path in task.metric_paths():
        value = metrics.get(path)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            return numeric
    return None


def dclm_core_centered_accuracy(
    metrics: dict[str, Any],
    task_aliases: tuple[str, ...] | None = None,
) -> dict[str, float]:
    r"""Compute DCLM-centered task accuracies and their macro average.

    Centered accuracy is \((a-r)/(1-r)\), where \(a\) is a task's hard
    accuracy-like score and \(r\) is its DCLM random baseline. The macro average
    uses observed tasks only and reports ``missing_task_count``; do not compare a
    partial macro directly to a fully covered DCLM Core score.
    """
    aliases = task_aliases or dclm_core_task_aliases()
    scores: dict[str, float] = {}
    values: list[float] = []
    for alias in aliases:
        task = task_by_alias(alias)
        raw_value = _metric_value_for_task(metrics, task)
        if raw_value is None:
            continue
        denominator = 1.0 - task.random_baseline
        if denominator <= 0.0:
            raise ValueError(f"Invalid random baseline for {alias}: {task.random_baseline}")
        centered = (raw_value - task.random_baseline) / denominator
        scores[f"lm_eval/dclm_core/{alias}/centered_accuracy"] = centered
        scores[f"lm_eval/dclm_core/{alias}/raw_score"] = raw_value
        values.append(centered)

    scores["lm_eval/dclm_core/task_count"] = float(len(values))
    scores["lm_eval/dclm_core/missing_task_count"] = float(len(aliases) - len(values))
    scores["lm_eval/dclm_core/centered_accuracy_macro"] = float(sum(values) / len(values)) if values else math.nan
    return scores
