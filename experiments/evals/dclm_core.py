# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DCLM Core task inventory and centered-accuracy scoring helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from marin.evaluation.evaluation_config import EvalTaskConfig


class TaskStatus(StrEnum):
    """Implementation status for one DCLM Core task."""

    LAUNCHABLE = "launchable"
    REQUIRES_CUSTOM_TASK = "requires_custom_task"


class TaskMode(StrEnum):
    """Evaluation backend mode for a DCLM Core task."""

    EXTRACTIVE = "extractive"
    MCQ = "mcq"
    GENERATION = "generation"


CUSTOM_TASKS_DIR = Path("experiments/scaling_law_sweeps/dclm_core/custom_tasks")


@dataclass(frozen=True)
class DCLMCoreTask:
    """One DCLM Core task as used for 300M completion accounting."""

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
        """Return the Marin eval task config for this task."""
        if self.status != TaskStatus.LAUNCHABLE:
            raise ValueError(f"Task {self.alias} is not launchable: {self.status}")
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
                if path not in seen:
                    seen.add(path)
                    paths.append(path)
        return tuple(paths)


EXACT_MATCH_METRICS = (
    "exact_match",
    "exact_match,strip-then-match",
    "exact_match,none",
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "em",
    "f1",
)

FILTERED_EXACT_MATCH_METRICS = (
    "exact_match,strip-then-match",
    "exact_match",
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "em",
    "f1",
)


def _custom_data_file(*parts: str) -> str:
    return str(CUSTOM_TASKS_DIR.joinpath(*parts))


def _bigbench_generate_until_task_kwargs(dataset_name: str) -> dict[str, Any]:
    """Return a DCLM-safe BigBench generation task spec.

    The pinned lm-eval BigBench template left task outputs unfiltered. That
    makes ordinary completions such as ``" 17"`` fail against gold ``"17"``.
    Use the same left-strip filter pattern as the custom Jeopardy task while
    preserving internal whitespace for sequence-copy tasks.
    """
    return {
        "dataset_path": "hails/bigbench",
        "dataset_name": dataset_name,
        "test_split": "default",
        "output_type": "generate_until",
        "doc_to_text": "inputs",
        "doc_to_target": "{{targets[0]}}",
        "generation_kwargs": {"max_gen_toks": 128},
        "filter_list": [
            {
                "name": "strip-then-match",
                "filter": [
                    {"function": "remove_whitespace"},
                    {"function": "take_first"},
                ],
            }
        ],
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
            "source": "lm-eval BigBench generate-until template with DCLM-safe filtering",
        },
    }


def _jeopardy_task_kwargs() -> dict[str, Any]:
    return {
        "dataset_path": "json",
        "dataset_kwargs": {"data_files": _custom_data_file("jeopardy", "jeopardy_all.jsonl")},
        "test_split": "train",
        "output_type": "generate_until",
        "doc_to_text": "{{context}}\nAnswer: ",
        "doc_to_target": "{{continuation}}",
        "generation_kwargs": {"until": ["\n", "\n\n"], "do_sample": False, "temperature": 0.0},
        "filter_list": [
            {
                "name": "strip-then-match",
                "filter": [
                    {"function": "remove_whitespace"},
                    {"function": "take_first"},
                ],
            }
        ],
        "metric_list": [
            {
                "metric": "exact_match",
                "aggregation": "mean",
                "higher_is_better": True,
                "ignore_case": True,
                "ignore_punctuation": True,
            }
        ],
        "metadata": {
            "version": 1.0,
            "source": "mosaicml/llm-foundry@v0.9.0",
            "dclm_meta_row": "jeopardy,world knowledge,language modeling,10,2117,0",
        },
    }


def _winograd_task_kwargs() -> dict[str, Any]:
    return {
        "dataset_path": "json",
        "dataset_kwargs": {"data_files": _custom_data_file("winograd", "wsc273.jsonl")},
        "test_split": "train",
        "output_type": "multiple_choice",
        "doc_to_text": "{{prefix}}",
        "doc_to_target": "{{gold}}",
        "doc_to_choice": "{{choices}}",
        "metric_list": [
            {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
            {"metric": "acc_norm", "aggregation": "mean", "higher_is_better": True},
        ],
        "metadata": {
            "version": 1.0,
            "source": "cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WSCollection.xml",
            "dclm_meta_row": "winograd,language understanding,schema,0,273,50",
        },
    }


def dclm_core_tasks() -> tuple[DCLMCoreTask, ...]:
    """Return the 22-task DCLM Core inventory from the paper."""
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
            task_kwargs=_bigbench_generate_until_task_kwargs("qa_wikidata_zero_shot"),
            metric_candidates=FILTERED_EXACT_MATCH_METRICS,
        ),
        DCLMCoreTask(
            "bigbench_dyck_languages_generate_until",
            "bb_dyck_languages_10shot",
            10,
            TaskMode.GENERATION,
            0.0,
            "exact_match",
            task_kwargs=_bigbench_generate_until_task_kwargs("dyck_languages_zero_shot"),
            metric_candidates=FILTERED_EXACT_MATCH_METRICS,
        ),
        DCLMCoreTask(
            "bigbench_operators_generate_until",
            "bb_operators_10shot",
            10,
            TaskMode.GENERATION,
            0.0,
            "exact_match",
            task_kwargs=_bigbench_generate_until_task_kwargs("operators_zero_shot"),
            metric_candidates=FILTERED_EXACT_MATCH_METRICS,
        ),
        DCLMCoreTask(
            "bigbench_repeat_copy_logic_generate_until",
            "bb_repeat_copy_logic_10shot",
            10,
            TaskMode.GENERATION,
            0.0,
            "exact_match",
            task_kwargs=_bigbench_generate_until_task_kwargs("repeat_copy_logic_zero_shot"),
            metric_candidates=FILTERED_EXACT_MATCH_METRICS,
        ),
        DCLMCoreTask(
            "bigbench_cs_algorithms_generate_until",
            "bb_cs_algorithms_10shot",
            10,
            TaskMode.GENERATION,
            0.0,
            "exact_match",
            task_kwargs=_bigbench_generate_until_task_kwargs("cs_algorithms_zero_shot"),
            metric_candidates=FILTERED_EXACT_MATCH_METRICS,
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
        DCLMCoreTask("hellaswag", "hellaswag_0shot", 0, TaskMode.MCQ, 0.25, "acc_norm"),
        DCLMCoreTask("hellaswag", "hellaswag_10shot", 10, TaskMode.MCQ, 0.25, "acc_norm"),
        DCLMCoreTask(
            "jeopardy",
            "jeopardy_10shot",
            10,
            TaskMode.GENERATION,
            0.0,
            "exact_match",
            task_kwargs=_jeopardy_task_kwargs(),
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
            task_kwargs=_winograd_task_kwargs(),
        ),
        DCLMCoreTask("winogrande", "winogrande_0shot", 0, TaskMode.MCQ, 0.5, "acc"),
    )


def task_by_alias(alias: str) -> DCLMCoreTask:
    """Return a DCLM task by alias."""
    by_alias = {task.alias: task for task in dclm_core_tasks()}
    try:
        return by_alias[alias]
    except KeyError as exc:
        raise ValueError(f"Unknown DCLM Core task alias: {alias}") from exc


def dclm_core_task_aliases() -> tuple[str, ...]:
    """Return aliases for all DCLM Core tasks, including custom-task gaps."""
    return tuple(task.alias for task in dclm_core_tasks())


def launchable_task_aliases() -> tuple[str, ...]:
    """Return aliases for DCLM Core tasks that the current lm-eval stack can launch."""
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

    Centered accuracy is \((a-r)/(1-r)\), where \(a\) is the task's hard
    accuracy-like score and \(r\) is the task-specific random baseline.
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
