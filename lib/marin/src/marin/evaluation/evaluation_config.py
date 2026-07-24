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
    metadata: dict | None = None
    """lm-eval task metadata (``--metadata`` JSON). RULER reads ``max_seq_lengths`` from here to size
    its haystacks; the eval client routes a metadata-carrying task through lm-eval's own CLI, which
    merges the served tokenizer (from ``--model_args``) into this metadata."""


def convert_to_levanter_task_config(tasks: Sequence[EvalTaskConfig]) -> list[TaskConfig]:
    for task in tasks:
        if task.metadata is not None:
            # metadata (RULER's max_seq_lengths/tokenizer) is honored only by the Evalchemy path; the
            # Levanter harness has no channel for it, so fail loudly rather than run a misconfigured task.
            raise ValueError(f"task {task.name!r} sets metadata, which the Levanter eval path cannot apply")
    return [
        TaskConfig(
            task=task.name,
            num_fewshot=task.num_fewshot,
            task_alias=task.task_alias,
            **(task.task_kwargs or {}),
        )
        for task in tasks
    ]
