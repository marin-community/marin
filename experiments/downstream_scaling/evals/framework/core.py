# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Core composition API for downstream-scaling evals."""

from __future__ import annotations

from typing import Protocol

from marin.execution.executor import ExecutorStep, InputName, MirroredValue, output_path_of

from experiments.downstream_scaling.evals.framework.schema import COMPLETIONS_FILENAME, PROMPTS_FILENAME


class EvalTask(Protocol):
    def make_prompts_step(self) -> ExecutorStep:
        """Return the stable prompt step for this configured task."""
        ...

    def make_grade_step(
        self,
        *,
        name: str,
        prompts_path: str | InputName | MirroredValue,
        completions_path: str | InputName | MirroredValue,
    ) -> ExecutorStep:
        """Return a step whose output directory contains grades.jsonl.gz."""
        ...


class CompletionAlgorithm(Protocol):
    def make_completions_step(
        self,
        *,
        name: str,
        model_path: str | InputName | MirroredValue,
        prompts_path: str | InputName | MirroredValue,
    ) -> ExecutorStep:
        """Return the final step that writes completions.jsonl.gz."""
        ...


def make_eval_step(
    *,
    name: str,
    model_path: str | InputName | MirroredValue,
    task: EvalTask,
    alg: CompletionAlgorithm,
) -> ExecutorStep:
    prompts = task.make_prompts_step()
    prompts_path = output_path_of(prompts) / PROMPTS_FILENAME

    completions = alg.make_completions_step(
        name=f"{name}/completions",
        model_path=model_path,
        prompts_path=prompts_path,
    )

    return task.make_grade_step(
        name=f"{name}/grade",
        prompts_path=prompts_path,
        completions_path=output_path_of(completions) / COMPLETIONS_FILENAME,
    )
