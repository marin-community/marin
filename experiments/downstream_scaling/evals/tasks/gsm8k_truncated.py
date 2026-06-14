# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Truncated-solution GSM8K task for downstream-scaling evals."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import fsspec
from fray.cluster import ResourceConfig
from levanter.compat.hf_checkpoints import load_tokenizer
from marin.execution.executor import ExecutorStep, InputName, MirroredValue
from marin.execution.remote import remote
from marin.execution.types import this_output_path, versioned

from experiments.downstream_scaling.evals.framework.schema import prompts_file
from experiments.downstream_scaling.evals.tasks.gsm8k import GSM8KTask
from experiments.downstream_scaling.evals.utils import version_path

logger = logging.getLogger(__name__)

ANSWER_PATTERN = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class TruncatedGSM8KTaskConfig:
    tokenizer_path: str | InputName | MirroredValue
    num_fewshot: int = 5
    fewshot_seed: int = 1234
    n_problems: int | None = None
    truncate_fraction: float = 0.5
    grade_workers: int = 32


@dataclass(frozen=True)
class TruncatedGSM8KPromptsConfig:
    output_path: str
    tokenizer_path: str
    num_fewshot: int
    fewshot_seed: int
    n_problems: int | None
    truncate_fraction: float


@dataclass(frozen=True)
class TruncatedGSM8KTask:
    config: TruncatedGSM8KTaskConfig

    make_grade_step = GSM8KTask.make_grade_step

    def make_prompts_step(self) -> ExecutorStep:
        return ExecutorStep(
            name="downstream_scaling/evals/prompts/truncated_gsm8k",
            fn=remote(
                write_truncated_gsm8k_prompts,
                resources=ResourceConfig.with_cpu(cpu=1, ram="4g"),
                pip_dependency_groups=["eval"],
            ),
            config=TruncatedGSM8KPromptsConfig(
                output_path=this_output_path(),
                tokenizer_path=version_path(self.config.tokenizer_path),  # type: ignore[arg-type]
                num_fewshot=versioned(self.config.num_fewshot),  # type: ignore[arg-type]
                fewshot_seed=versioned(self.config.fewshot_seed),  # type: ignore[arg-type]
                n_problems=versioned(self.config.n_problems),  # type: ignore[arg-type]
                truncate_fraction=versioned(self.config.truncate_fraction),  # type: ignore[arg-type]
            ),
        )


def _truncate_solution(tokenizer, text: str, truncate_fraction: float) -> str:
    if not 0.0 <= truncate_fraction <= 1.0:
        raise ValueError(f"truncate_fraction must be in [0.0, 1.0], got {truncate_fraction}")

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    keep_count = round(len(token_ids) * (1.0 - truncate_fraction))
    return tokenizer.decode(token_ids[:keep_count], skip_special_tokens=False)


def write_truncated_gsm8k_prompts(config: TruncatedGSM8KPromptsConfig) -> None:
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
            prompt_id = f"truncated_gsm8k/test/{i}"
            n_samples = (
                config.num_fewshot + 1 if task.config.fewshot_split == task.config.test_split else config.num_fewshot
            )
            fewshot_docs = [sample for sample in task.sampler.sample(n_samples) if sample != doc][: config.num_fewshot]

            prompt = ""
            for fewshot_doc in fewshot_docs:
                prompt += task.doc_to_text(fewshot_doc) + task.config.target_delimiter
                prompt += task.doc_to_target(fewshot_doc) + task.config.fewshot_delimiter

            solution = task.doc_to_target(doc).split("####", maxsplit=1)[0].rstrip()
            truncated_solution = _truncate_solution(tokenizer, solution, config.truncate_fraction)
            prompt += task.doc_to_text(doc) + task.config.target_delimiter + truncated_solution + "\n"

            match = ANSWER_PATTERN.search(doc["answer"])
            if not match:
                raise ValueError(f"GSM8K problem {i} has no '####' answer marker")
            row = {
                "id": prompt_id,
                "prompt": prompt,
                "ground_truth": match.group(1).replace(",", ""),
                "metadata": {
                    "problem": doc["question"],
                    "solution": doc["answer"],
                    "split": "test",
                    "num_fewshot": config.num_fewshot,
                    "fewshot_seed": config.fewshot_seed,
                    "truncate_fraction": config.truncate_fraction,
                },
            }
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d truncated GSM8K prompts to %s", len(docs), path)
