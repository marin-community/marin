# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Sequence
from dataclasses import dataclass, field

from fray.cluster import ResourceConfig
from levanter.eval_harness import TaskConfig
from marin.execution.executor import InputName

# Wandb project name for evaluations. Controlled via WANDB_PROJECT env var.
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "marin")
_LEVANTER_TASK_KWARG_ALIASES = {
    "doc_to_target": "doct_to_target",
}
_SUPPORTED_LEVANTER_TASK_KWARGS = frozenset(TaskConfig.__dataclass_fields__) - {
    "task",
    "task_alias",
    "num_fewshot",
}


@dataclass(frozen=True)
class EvalTaskConfig:
    name: str
    """Name of the evaluation task."""

    num_fewshot: int
    """Number of few-shot examples to evaluate on."""

    task_alias: str | None = None
    """Alias for the task name."""

    task_kwargs: dict | None = None
    """Additional keyword arguments specifically for this task."""


@dataclass(frozen=True)
class EvaluationConfig:
    evaluator: str
    """Name of the evaluator to run."""

    resource_config: ResourceConfig
    """
    Resources to allocate for the eval step (passed to @remote).
    """

    model_name: str | None
    """
    Can be a name of the model in Hugging Face (e.g, google/gemma-2b) or
    a name given to the model checkpoint (e.g., $RUN/$CHECKPOINT).

    If None, the model_path should be provided and the name will be imputed from the path,
     using Levanter's path conventions. (i.e. $RUN/hf/step-$STEP --> $RUN-$STEP)
    """

    evaluation_path: str = "tmp/output"
    """
    Where to write results to. Can be a local path (e.g., /path/to/output) or
    a path on GCS (e.g., gs://bucket/path/to/output).
    """

    evals: list[EvalTaskConfig] = field(default_factory=list)
    """
    List of specific evals within an evaluation harness to run. This would be a list of
    tasks in for EleutherAI's lm-evaluation-harness or a list of evals from HELM (e.g., mmlu, lite, etc.).
    See https://github.com/stanford-crfm/helm/tree/main/src/helm/benchmark/presentation, or
    https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks
    for the full list.
    """

    model_path: str | None = None
    """
    Optional: Path to the model. Can be a path on GCS.
    """

    discover_latest_checkpoint: bool = False
    """
    Whether to discover the latest HF checkpoint in the model path.
    """

    max_eval_instances: int | None = None
    """
    Maximum number of evaluation instances to run.
    """

    engine_kwargs: dict | None = None
    """
    Additional keyword arguments to pass to the vLLM engine.
    """

    generation_params: dict | None = None
    """
    Additional keyword arguments passed to the vLLM sampling params engine
    """

    apply_chat_template: bool = False
    """
    Whether or not this model was trained with a Chat Template in the tokenizer
    """

    wandb_tags: list[str] | None = None
    """
    Tags to add to the wandb run.
    """

    base_eval_run_name: str | None = None
    """Custom base name for wandb runs. If set, wandb runs will be named
    evalchemy-{base_eval_run_name}[-step{N}]-{task}-seed{S}."""

    eval_datasets_cache_path: str | None = None
    """
    Optional GCS path to pre-cached evaluation datasets for Levanter lm-eval.
    """

    eval_datasets_cache_dependency: InputName | str | None = None
    """
    Optional executor-only dependency marker for the eval dataset cache step.

    This field is not read at runtime. It exists so Executor can see an
    `InputName` and block the evaluation step on a cache-population step.
    """


def convert_to_levanter_task_config(tasks: Sequence[EvalTaskConfig]) -> list[TaskConfig]:
    """Convert Marin eval task configs into Levanter task configs.

    Supported ``task_kwargs`` are mapped into ``TaskConfig`` so callers can pass
    inline task-spec fields for tasks whose registered YAML needs patching.
    """
    return [
        TaskConfig(
            task=task.name,
            num_fewshot=task.num_fewshot,
            task_alias=task.task_alias,
            **_convert_task_kwargs_for_levanter(task),
        )
        for task in tasks
    ]


def _convert_task_kwargs_for_levanter(task: EvalTaskConfig) -> dict[str, object]:
    if not task.task_kwargs:
        return {}

    converted_kwargs: dict[str, object] = {}
    unsupported_keys: list[str] = []
    for key, value in task.task_kwargs.items():
        mapped_key = _LEVANTER_TASK_KWARG_ALIASES.get(key, key)
        if mapped_key not in _SUPPORTED_LEVANTER_TASK_KWARGS:
            unsupported_keys.append(key)
            continue
        converted_kwargs[mapped_key] = value

    if unsupported_keys:
        raise ValueError(
            f"Unsupported Levanter task kwargs for {task.task_alias or task.name}: {sorted(unsupported_keys)}"
        )

    return converted_kwargs
