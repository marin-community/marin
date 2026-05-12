# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GSM8K task implementation for downstream-scaling evals."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True)
class GSM8KTaskConfig:
    num_fewshot: int = 5
    fewshot_seed: int = 1234
    n_problems: int | None = None
    grade_workers: int = 32


@dataclass(frozen=True)
class GSM8KPromptsConfig:
    output_path: str
    num_fewshot: int
    fewshot_seed: int
    n_problems: int | None


@dataclass(frozen=True)
class GSM8KGradeConfig:
    output_path: str
    prompts_path: str
    completions_path: str
    num_workers: int


@dataclass(frozen=True)
class GSM8KTask:
    config: GSM8KTaskConfig

    def make_prompts_step(self) -> ExecutorStep:
        return ExecutorStep(
            name="downstream_scaling/evals/prompts/gsm8k",
            fn=remote(write_gsm8k_prompts, pip_dependency_groups=["eval"]),
            config=GSM8KPromptsConfig(
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
            fn=remote(grade_gsm8k, pip_dependency_groups=["eval"]),
            config=GSM8KGradeConfig(
                output_path=this_output_path(),
                prompts_path=version_path(prompts_path),  # type: ignore[arg-type]
                completions_path=version_path(completions_path),  # type: ignore[arg-type]
                num_workers=self.config.grade_workers,
            ),
        )


def write_gsm8k_prompts(config: GSM8KPromptsConfig) -> None:
    import lm_eval.tasks

    task = lm_eval.tasks.get_task_dict(["gsm8k"])["gsm8k"]
    task.set_fewshot_seed(config.fewshot_seed)
    docs = list(task.test_docs())
    if config.n_problems is not None:
        docs = docs[: config.n_problems]

    answer_pattern = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)
    rows = []
    for i, doc in enumerate(docs):
        match = answer_pattern.search(doc["answer"])
        if not match:
            raise ValueError(f"GSM8K problem {i} has no '####' answer marker")
        rows.append(
            {
                "id": f"gsm8k/test/{i}",
                "prompt": task.fewshot_context(doc, num_fewshot=config.num_fewshot),
                "ground_truth": match.group(1).replace(",", ""),
                "metadata": {
                    "problem": doc["question"],
                    "solution": doc["answer"],
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
    logger.info("Wrote %d GSM8K prompts to %s", len(rows), path)


def _grade_gsm8k_shard(items, shard_info):
    import lm_eval.tasks
    from lm_eval.api.instance import Instance

    task = lm_eval.tasks.get_task_dict(["gsm8k"])["gsm8k"]
    filter_names = [f.name for f in task._filters]

    for item in items:
        doc = {"question": item["problem"], "answer": f"#### {item['ground_truth']}"}
        inst = Instance(
            request_type="generate_until",
            doc=doc,
            arguments=("", {}),
            idx=item["completion_index"],
            task_name="gsm8k",
        )
        inst.resps = [item["completion"]]
        task._instances = [inst]
        task.apply_filters()

        metadata: dict[str, Any] = {}
        score = 0.0
        for name in filter_names:
            key = name.replace("-", "_")
            filtered = inst.filtered_resps[name]
            correct = bool(task.process_results(doc, [filtered])["exact_match"])
            metadata[f"extraction_{key}"] = filtered
            metadata[f"correct_{key}"] = correct
            if key == "flexible_extract":
                score = 1.0 if correct else 0.0
        if "correct_flexible_extract" not in metadata and filter_names:
            score = 1.0 if metadata[f"correct_{filter_names[0].replace('-', '_')}"] else 0.0

        yield {
            "id": item["id"],
            "completion_index": item["completion_index"],
            "grade": {
                "score": score,
                "metadata": metadata,
            },
        }


def grade_gsm8k(config: GSM8KGradeConfig) -> None:
    prompts_by_id = {row["id"]: row for row in read_prompt_rows(config.prompts_path)}

    def flatten(item):
        prompt = prompts_by_id[item["id"]]
        for i, completion in enumerate(item["completions"]):
            yield {
                "id": item["id"],
                "completion_index": i,
                "completion": completion["text"],
                "problem": prompt["metadata"]["problem"],
                "ground_truth": prompt["ground_truth"],
            }

    path = grades_file(config.output_path)
    pipeline = (
        Dataset.from_files(config.completions_path)
        .load_jsonl()
        .flat_map(flatten)
        .reshard(config.num_workers)
        .map_shard(_grade_gsm8k_shard)
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
        name="grade-gsm8k",
        max_workers=config.num_workers,
        coordinator_resources=ResourceConfig(cpu=0.1, ram="1g", preemptible=True),
    ).execute(pipeline)
    logger.info("Wrote GSM8K grade rows to %s", path)
