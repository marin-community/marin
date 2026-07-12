# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HumanEval task implementation for downstream-scaling evals."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import fsspec
from fray.cluster import ResourceConfig
from thalas.execution.executor import ExecutorStep, InputName, MirroredValue
from thalas.execution.remote import remote
from thalas.execution.types import this_output_path, versioned
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from experiments.downstream_scaling.evals.framework.schema import (
    grades_file,
    prompts_file,
    read_prompt_rows,
)
from experiments.downstream_scaling.evals.utils import version_path

logger = logging.getLogger(__name__)

HUMANEVAL_TASK_NAME = "humaneval"
HUMANEVAL_FILTER_NAME = "create_test"
HF_ALLOW_CODE_EVAL = "HF_ALLOW_CODE_EVAL"


@dataclass(frozen=True)
class HumanEvalTaskConfig:
    num_fewshot: int = 10
    fewshot_seed: int = 1234
    n_problems: int | None = None
    grade_workers: int = 32


@dataclass(frozen=True)
class HumanEvalPromptsConfig:
    output_path: str
    num_fewshot: int
    fewshot_seed: int
    n_problems: int | None


@dataclass(frozen=True)
class HumanEvalGradeConfig:
    output_path: str
    prompts_path: str
    completions_path: str
    num_workers: int


@dataclass(frozen=True)
class HumanEvalTask:
    config: HumanEvalTaskConfig

    def make_prompts_step(self) -> ExecutorStep:
        return ExecutorStep(
            name="downstream_scaling/evals/prompts/humaneval",
            fn=remote(write_humaneval_prompts, pip_dependency_groups=["eval"]),
            config=HumanEvalPromptsConfig(
                output_path=this_output_path(),
                num_fewshot=versioned(self.config.num_fewshot),  # type: ignore[arg-type]
                fewshot_seed=versioned(self.config.fewshot_seed),  # type: ignore[arg-type]
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
            fn=remote(grade_humaneval, pip_dependency_groups=["eval"]),
            config=HumanEvalGradeConfig(
                output_path=this_output_path(),
                prompts_path=version_path(prompts_path),  # type: ignore[arg-type]
                completions_path=version_path(completions_path),  # type: ignore[arg-type]
                num_workers=self.config.grade_workers,
            ),
        )


def _allow_code_eval() -> None:
    os.environ[HF_ALLOW_CODE_EVAL] = "1"


def write_humaneval_prompts(config: HumanEvalPromptsConfig) -> None:
    _allow_code_eval()

    import lm_eval.tasks  # noqa: PLC0415  # optional dep: lm_eval

    task = lm_eval.tasks.get_task_dict([HUMANEVAL_TASK_NAME])[HUMANEVAL_TASK_NAME]
    task.set_fewshot_seed(config.fewshot_seed)
    docs = list(task.test_docs())
    if config.n_problems is not None:
        docs = docs[: config.n_problems]

    rows = []
    for doc in docs:
        task_id = doc["task_id"]
        rows.append(
            {
                "id": f"humaneval/test/{task_id}",
                "prompt": task.fewshot_context(doc, num_fewshot=config.num_fewshot),
                "ground_truth": doc["canonical_solution"],
                "metadata": {
                    "task_id": task_id,
                    "raw_prompt": doc["prompt"],
                    "entry_point": doc["entry_point"],
                    "test": doc["test"],
                    "canonical_solution": doc["canonical_solution"],
                    "split": "test",
                    "num_fewshot": config.num_fewshot,
                    "fewshot_seed": config.fewshot_seed,
                },
            }
        )

    path = prompts_file(config.output_path)
    with fsspec.open(path, "wt", compression="gzip") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d HumanEval prompts to %s", len(rows), path)


def _grade_humaneval_shard(items, shard_info):
    _allow_code_eval()

    import lm_eval.tasks  # noqa: PLC0415  # optional dep: lm_eval
    from lm_eval.api.instance import Instance  # noqa: PLC0415  # optional dep: lm_eval

    task = lm_eval.tasks.get_task_dict([HUMANEVAL_TASK_NAME])[HUMANEVAL_TASK_NAME]

    for item in items:
        doc = {
            "task_id": item["task_id"],
            "prompt": item["raw_prompt"],
            "entry_point": item["entry_point"],
            "test": item["test"],
            "canonical_solution": item["canonical_solution"],
        }
        inst = Instance(
            request_type="generate_until",
            doc=doc,
            arguments=("", {}),
            idx=item["completion_index"],
            task_name=HUMANEVAL_TASK_NAME,
        )
        inst.resps = [item["completion"]]
        task._instances = [inst]
        task.apply_filters()

        filtered = inst.filtered_resps[HUMANEVAL_FILTER_NAME]
        score = float(task.process_results(doc, [filtered])["pass@1"])
        passed = bool(score)
        yield {
            "id": item["id"],
            "completion_index": item["completion_index"],
            "grade": {
                "score": score,
                "metadata": {
                    "passed": passed,
                    "pass_at_1": score,
                    "completion_index": item["completion_index"],
                },
            },
        }


def grade_humaneval(config: HumanEvalGradeConfig) -> None:
    prompts_by_id = {row["id"]: row for row in read_prompt_rows(config.prompts_path)}

    def flatten(item):
        prompt = prompts_by_id[item["id"]]
        metadata: dict[str, Any] = prompt["metadata"]
        for i, completion in enumerate(item["completions"]):
            yield {
                "id": item["id"],
                "completion_index": i,
                "completion": completion["text"],
                "task_id": metadata["task_id"],
                "raw_prompt": metadata["raw_prompt"],
                "entry_point": metadata["entry_point"],
                "test": metadata["test"],
                "canonical_solution": metadata["canonical_solution"],
            }

    path = grades_file(config.output_path)
    pipeline = (
        Dataset.from_files(config.completions_path)
        .load_jsonl()
        .flat_map(flatten)
        .reshard(config.num_workers)
        .map_shard(_grade_humaneval_shard)
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
        name="grade-humaneval",
        max_workers=config.num_workers,
        coordinator_resources=ResourceConfig(cpu=0.1, ram="1g", preemptible=True),
    ).execute(pipeline)
    logger.info("Wrote HumanEval grade rows to %s", path)
