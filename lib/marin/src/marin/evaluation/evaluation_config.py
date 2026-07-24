# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task configuration shared by Evalchemy and Levanter evaluation."""

import os
from collections.abc import Sequence
from dataclasses import dataclass

from levanter.eval_harness import TaskConfig

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "marin")


@dataclass(frozen=True)
class EvalTaskConfig:
    """One lm-eval task and its prompt/scoring behavior."""

    name: str
    num_fewshot: int
    task_alias: str | None = None
    task_kwargs: dict | None = None
    generation: bool = False
    """Whether the task scores generated text rather than prompt loglikelihoods."""
    unsafe_code: bool = False
    """Whether scoring executes model-generated code."""
    completion_only: bool = False
    """Whether generation must use the completions API even for chat-template models."""


def convert_to_levanter_task_config(tasks: Sequence[EvalTaskConfig]) -> list[TaskConfig]:
    return [
        TaskConfig(
            task=task.name,
            num_fewshot=task.num_fewshot,
            task_alias=task.task_alias,
            **(task.task_kwargs or {}),
        )
        for task in tasks
    ]
