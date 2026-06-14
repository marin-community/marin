# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Masked-solution GSM8K task with an independent mask per rollout.

Unlike :mod:`gsm8k_masked`, which masks each problem's solution once and shares
that single masked prompt across all rollouts, this task emits ``n_masks``
independently masked prompt variants per problem. Paired with the IID algorithm
at ``n_samples=1``, every rollout sees its own freshly sampled mask. The grade
step regroups the per-variant completions back into one row per problem so the
output shape matches :mod:`gsm8k_masked`.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass

import fsspec
from fray.cluster import ResourceConfig
from levanter.compat.hf_checkpoints import load_tokenizer
from marin.execution.executor import ExecutorStep, InputName, MirroredValue
from marin.execution.remote import remote
from marin.execution.types import this_output_path, versioned
from zephyr import Dataset, ZephyrContext

from experiments.downstream_scaling.evals.framework.schema import (
    grades_file,
    prompts_file,
    read_prompt_rows,
)
from experiments.downstream_scaling.evals.tasks.gsm8k import GSM8KGradeConfig, _grade_gsm8k_shard
from experiments.downstream_scaling.evals.tasks.gsm8k_masked import ANSWER_PATTERN, _mask_solution
from experiments.downstream_scaling.evals.utils import version_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MaskedGSM8KIIDTaskConfig:
    tokenizer_path: str | InputName | MirroredValue
    num_fewshot: int = 5
    fewshot_seed: int = 1234
    n_problems: int | None = None
    n_masks: int = 32
    mask_fraction: float = 0.5
    mask_text: str = "<mask>"
    grade_workers: int = 32


@dataclass(frozen=True)
class MaskedGSM8KIIDPromptsConfig:
    output_path: str
    tokenizer_path: str
    num_fewshot: int
    fewshot_seed: int
    n_problems: int | None
    n_masks: int
    mask_fraction: float
    mask_text: str


@dataclass(frozen=True)
class MaskedGSM8KIIDTask:
    config: MaskedGSM8KIIDTaskConfig

    def make_prompts_step(self) -> ExecutorStep:
        return ExecutorStep(
            name="downstream_scaling/evals/prompts/masked_gsm8k_iid",
            fn=remote(
                write_masked_gsm8k_iid_prompts,
                resources=ResourceConfig.with_cpu(cpu=1, ram="4g"),
                pip_dependency_groups=["eval"],
            ),
            config=MaskedGSM8KIIDPromptsConfig(
                output_path=this_output_path(),
                tokenizer_path=version_path(self.config.tokenizer_path),  # type: ignore[arg-type]
                num_fewshot=versioned(self.config.num_fewshot),  # type: ignore[arg-type]
                fewshot_seed=versioned(self.config.fewshot_seed),  # type: ignore[arg-type]
                n_problems=versioned(self.config.n_problems),  # type: ignore[arg-type]
                n_masks=versioned(self.config.n_masks),  # type: ignore[arg-type]
                mask_fraction=versioned(self.config.mask_fraction),  # type: ignore[arg-type]
                mask_text=versioned(self.config.mask_text),  # type: ignore[arg-type]
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
            fn=remote(grade_gsm8k_masked_iid, pip_dependency_groups=["eval"]),
            config=GSM8KGradeConfig(
                output_path=this_output_path(),
                prompts_path=version_path(prompts_path),  # type: ignore[arg-type]
                completions_path=version_path(completions_path),  # type: ignore[arg-type]
                num_workers=self.config.grade_workers,
            ),
        )


def write_masked_gsm8k_iid_prompts(config: MaskedGSM8KIIDPromptsConfig) -> None:
    import lm_eval.tasks

    logger.info("Loading tokenizer from %s", config.tokenizer_path)
    tokenizer = load_tokenizer(config.tokenizer_path)
    task = lm_eval.tasks.get_task_dict(["gsm8k"])["gsm8k"]
    task.set_fewshot_seed(config.fewshot_seed)
    docs = list(task.test_docs())
    if config.n_problems is not None:
        docs = docs[: config.n_problems]

    path = prompts_file(config.output_path)
    with fsspec.open(path, "wt", compression="gzip") as f:
        for i, doc in enumerate(docs):
            problem_id = f"masked_gsm8k/test/{i}"
            n_fewshot_samples = (
                config.num_fewshot + 1 if task.config.fewshot_split == task.config.test_split else config.num_fewshot
            )
            fewshot_docs = [sample for sample in task.sampler.sample(n_fewshot_samples) if sample != doc][
                : config.num_fewshot
            ]

            match = ANSWER_PATTERN.search(doc["answer"])
            if not match:
                raise ValueError(f"GSM8K problem {i} has no '####' answer marker")
            ground_truth = match.group(1).replace(",", "")

            for s in range(config.n_masks):
                prompt_id = f"{problem_id}/sample/{s}"

                prompt = ""
                for j, fewshot_doc in enumerate(fewshot_docs):
                    solution = task.doc_to_target(fewshot_doc).split("####", maxsplit=1)[0].rstrip()
                    masked_solution = _mask_solution(
                        tokenizer,
                        solution,
                        config.mask_fraction,
                        config.mask_text,
                        random.Random(f"{config.fewshot_seed}:{prompt_id}:fewshot:{j}"),
                    )
                    prompt += task.doc_to_text(fewshot_doc) + task.config.target_delimiter + masked_solution + "\n"
                    prompt += task.doc_to_target(fewshot_doc) + task.config.fewshot_delimiter

                solution = task.doc_to_target(doc).split("####", maxsplit=1)[0].rstrip()
                masked_solution = _mask_solution(
                    tokenizer,
                    solution,
                    config.mask_fraction,
                    config.mask_text,
                    random.Random(f"{config.fewshot_seed}:{prompt_id}:target"),
                )
                prompt += task.doc_to_text(doc) + task.config.target_delimiter + masked_solution + "\n"

                row = {
                    "id": prompt_id,
                    "prompt": prompt,
                    "ground_truth": ground_truth,
                    "metadata": {
                        "problem_id": problem_id,
                        "sample_index": s,
                        "problem": doc["question"],
                        "solution": doc["answer"],
                        "split": "test",
                        "num_fewshot": config.num_fewshot,
                        "fewshot_seed": config.fewshot_seed,
                        "mask_fraction": config.mask_fraction,
                        "mask_text": config.mask_text,
                    },
                }
                f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d masked-IID GSM8K prompts to %s", len(docs) * config.n_masks, path)


def grade_gsm8k_masked_iid(config: GSM8KGradeConfig) -> None:
    prompts_by_id = {row["id"]: row for row in read_prompt_rows(config.prompts_path)}

    def flatten(item):
        prompt = prompts_by_id[item["id"]]
        problem_id = prompt["metadata"]["problem_id"]
        sample_index = prompt["metadata"]["sample_index"]
        for completion in item["completions"]:
            yield {
                "id": problem_id,
                "completion_index": sample_index,
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
            reducer=lambda problem_id, items: {
                "id": problem_id,
                "grades": [item["grade"] for item in items],
            },
            sort_by=lambda rec: rec["completion_index"],
            num_output_shards=1,
        )
        .write_jsonl(path, skip_existing=True)
    )
    ZephyrContext(
        name="grade-gsm8k-masked-iid",
        max_workers=config.num_workers,
        coordinator_resources=ResourceConfig(cpu=0.1, ram="1g", preemptible=True),
    ).execute(pipeline)
    logger.info("Wrote masked-IID GSM8K grade rows to %s", path)
