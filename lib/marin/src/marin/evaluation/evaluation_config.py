# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluation task configuration shared across evaluators."""

import os
from collections.abc import Sequence
from dataclasses import dataclass

from levanter.eval_harness import TaskConfig

# Wandb project name for evaluations. Controlled via WANDB_PROJECT env var.
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "marin")


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


def convert_to_levanter_task_config(tasks: Sequence[EvalTaskConfig]) -> list[TaskConfig]:
    """Convert a list of EvalTaskConfig to a list of TaskConfig that Levanter's eval_harness expects."""
    return [
        TaskConfig(
            task=task.name,
            num_fewshot=task.num_fewshot,
            task_alias=task.task_alias,
        )
        for task in tasks
    ]
