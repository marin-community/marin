# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MATH-500 task implementation for downstream-scaling evals."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import fsspec
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, InputName, MirroredValue, this_output_path, versioned
from marin.execution.remote import remote
from zephyr import Dataset, ZephyrContext

from experiments.downstream_scaling.evals.framework.schema import (
    grades_file,
    prompts_file,
    read_prompt_rows,
)
from experiments.downstream_scaling.evals.utils import version_path

logger = logging.getLogger(__name__)

MATH500_PROMPT_PREFIX = (
    "How many r's are in strawberry? Write your answer in \\boxed{} format.\n\n"
    "Let's spell the word out and number all the letters: "
    "1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. "
    "We have r's at positions 3, 8, and 9. \\boxed{3}\n\n"
)
MATH500_PROMPT_SUFFIX = " Write your answer in \\boxed{} format."


@dataclass(frozen=True)
class Math500TaskConfig:
    prompt_prefix: str = MATH500_PROMPT_PREFIX
    prompt_suffix: str = MATH500_PROMPT_SUFFIX
    n_problems: int | None = None
    grade_workers: int = 32


@dataclass(frozen=True)
class Math500PromptsConfig:
    output_path: str
    prompt_prefix: str
    prompt_suffix: str
    n_problems: int | None


@dataclass(frozen=True)
class Math500GradeConfig:
    output_path: str
    prompts_path: str
    completions_path: str
    num_workers: int


@dataclass(frozen=True)
class Math500Task:
    config: Math500TaskConfig

    def make_prompts_step(self) -> ExecutorStep:
        return ExecutorStep(
            name="downstream_scaling/evals/prompts/math500",
            fn=remote(write_math500_prompts, pip_dependency_groups=["eval"]),
            config=Math500PromptsConfig(
                output_path=this_output_path(),
                prompt_prefix=versioned(self.config.prompt_prefix),  # type: ignore[arg-type]
                prompt_suffix=versioned(self.config.prompt_suffix),  # type: ignore[arg-type]
                n_problems=versioned(self.config.n_problems),  # type: ignore[arg-type]
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
            fn=remote(grade_math500, pip_dependency_groups=["math"]),
            config=Math500GradeConfig(
                output_path=this_output_path(),
                prompts_path=version_path(prompts_path),  # type: ignore[arg-type]
                completions_path=version_path(completions_path),  # type: ignore[arg-type]
                num_workers=self.config.grade_workers,
            ),
        )


def write_math500_prompts(config: Math500PromptsConfig) -> None:
    from datasets import load_dataset

    dataset = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    if config.n_problems is not None:
        dataset = dataset[: config.n_problems]

    rows = []
    for i, raw in enumerate(dataset):
        row = dict(raw)
        problem = row["problem"]
        rows.append(
            {
                "id": f"math500/test/{i}",
                "prompt": config.prompt_prefix + problem + config.prompt_suffix,
                "ground_truth": row["answer"],
                "metadata": {
                    "problem": problem,
                    "solution": row["solution"],
                    "subject": row.get("subject", ""),
                    "level": row.get("level", 0),
                    "unique_id": row.get("unique_id", ""),
                    "split": "test",
                },
            }
        )

    path = prompts_file(config.output_path)
    with fsspec.open(path, "wt", compression="gzip") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d MATH-500 prompts to %s", len(rows), path)


def _grade_math500_shard(items, shard_info):
    from marin.rl.environments.tinker_environments.math_env import safe_grade
    from marin.rl.environments.tinker_environments.math_grading import extract_boxed

    for item in items:
        try:
            extraction = extract_boxed(item["completion"])
        except ValueError:
            extraction = None
        correct = False if extraction is None else safe_grade(extraction, item["ground_truth"], grader="sympy")
        yield {
            "id": item["id"],
            "completion_index": item["completion_index"],
            "grade": {
                "score": 1.0 if correct else 0.0,
                "metadata": {
                    "extraction": extraction,
                    "correct": correct,
                },
            },
        }


def grade_math500(config: Math500GradeConfig) -> None:
    prompts_by_id = {row["id"]: row for row in read_prompt_rows(config.prompts_path)}

    def flatten(item):
        prompt = prompts_by_id[item["id"]]
        for i, completion in enumerate(item["completions"]):
            yield {
                "id": item["id"],
                "completion_index": i,
                "completion": completion["text"],
                "ground_truth": prompt["ground_truth"],
            }

    path = grades_file(config.output_path)
    pipeline = (
        Dataset.from_files(config.completions_path)
        .load_jsonl()
        .flat_map(flatten)
        .reshard(config.num_workers)
        .map_shard(_grade_math500_shard)
        .group_by(
            key=lambda rec: rec["id"],
            reducer=lambda prompt_id, items: {
                "id": prompt_id,
                "grades": [item["grade"] for item in items],
            },
            sort_by=lambda rec: rec["completion_index"],
            num_output_shards=1,
        )
        .write_jsonl(path, skip_existing=True)
    )
    ZephyrContext(
        name="grade-math500",
        max_workers=config.num_workers,
        coordinator_resources=ResourceConfig(cpu=0.1, ram="1g", preemptible=True),
    ).execute(pipeline)
    logger.info("Wrote MATH-500 grade rows to %s", path)
