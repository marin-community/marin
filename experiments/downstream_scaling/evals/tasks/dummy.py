# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dummy task implementation for downstream-scaling eval smoke tests."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import fsspec
from marin.execution.executor import ExecutorStep, InputName, MirroredValue, this_output_path, versioned
from marin.execution.remote import remote

from experiments.downstream_scaling.evals.framework.schema import (
    grades_file,
    prompts_file,
    read_completion_rows,
    read_prompt_rows,
)
from experiments.downstream_scaling.evals.utils import version_path

logger = logging.getLogger(__name__)

DUMMY_PROMPT_PREFIX = "Dummy prompt"


@dataclass(frozen=True)
class DummyTaskConfig:
    n_prompts: int = 16
    prompt_prefix: str = DUMMY_PROMPT_PREFIX


@dataclass(frozen=True)
class DummyPromptsConfig:
    output_path: str
    n_prompts: int
    prompt_prefix: str


@dataclass(frozen=True)
class DummyGradeConfig:
    output_path: str
    prompts_path: str
    completions_path: str


@dataclass(frozen=True)
class DummyTask:
    config: DummyTaskConfig

    def make_prompts_step(self) -> ExecutorStep:
        return ExecutorStep(
            name="downstream_scaling/evals/prompts/dummy",
            fn=remote(write_dummy_prompts, pip_dependency_groups=["eval"]),
            config=DummyPromptsConfig(
                output_path=this_output_path(),
                n_prompts=versioned(self.config.n_prompts),  # type: ignore[arg-type]
                prompt_prefix=versioned(self.config.prompt_prefix),  # type: ignore[arg-type]
            ),
        )

    def make_grade_step(
        self,
        *,
        name: str,
        prompts_path: str | InputName | MirroredValue,
        completions_path: str | InputName | MirroredValue,
    ) -> ExecutorStep:
        return ExecutorStep(
            name=name,
            fn=remote(grade_dummy, pip_dependency_groups=["eval"]),
            config=DummyGradeConfig(
                output_path=this_output_path(),
                prompts_path=version_path(prompts_path),  # type: ignore[arg-type]
                completions_path=version_path(completions_path),  # type: ignore[arg-type]
            ),
        )


def write_dummy_prompts(config: DummyPromptsConfig) -> None:
    rows = [
        {
            "id": f"dummy/test/{i}",
            "prompt": f"{config.prompt_prefix} {i}\nReturn anything.",
            "ground_truth": "always_correct",
            "metadata": {
                "index": i,
                "source": "dummy",
                "split": "test",
            },
        }
        for i in range(config.n_prompts)
    ]

    path = prompts_file(config.output_path)
    with fsspec.open(path, "wt", compression="gzip") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d dummy prompts to %s", len(rows), path)


def grade_dummy(config: DummyGradeConfig) -> None:
    prompt_ids = [row["id"] for row in read_prompt_rows(config.prompts_path)]
    completion_rows_by_id = {row["id"]: row for row in read_completion_rows(config.completions_path)}

    path = grades_file(config.output_path)
    with fsspec.open(path, "wt", compression="gzip") as f:
        for prompt_id in prompt_ids:
            completion_row = completion_rows_by_id[prompt_id]
            row = {
                "id": prompt_id,
                "grades": [
                    {
                        "score": 1.0,
                        "metadata": {
                            "correct": True,
                            "grader": "dummy",
                        },
                    }
                    for _ in completion_row["completions"]
                ],
            }
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote dummy grade rows to %s", path)
