# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Chat-templated HumanEval advisor prompts for downstream-scaling evals."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import fsspec
from fray.cluster import ResourceConfig
from levanter.compat.hf_checkpoints import load_tokenizer
from thalas.execution.executor import ExecutorStep, InputName, MirroredValue
from thalas.execution.remote import remote
from thalas.execution.types import this_output_path, versioned

from experiments.downstream_scaling.evals.framework.schema import prompts_file
from experiments.downstream_scaling.evals.tasks.gsm8k_chat import render_advisor_prompt
from experiments.downstream_scaling.evals.tasks.humaneval import (
    HUMANEVAL_TASK_NAME,
    HumanEvalTask,
    _allow_code_eval,
)
from experiments.downstream_scaling.evals.utils import discover_hf_checkpoints, version_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatHumanEvalTaskConfig:
    tokenizer_path: str | InputName | MirroredValue
    n_problems: int | None = None
    grade_workers: int = 32


@dataclass(frozen=True)
class ChatHumanEvalPromptsConfig:
    output_path: str
    tokenizer_path: str
    n_problems: int | None


def _load_humaneval_task():
    import lm_eval.tasks  # noqa: PLC0415  # optional dep: lm_eval

    return lm_eval.tasks.get_task_dict([HUMANEVAL_TASK_NAME])[HUMANEVAL_TASK_NAME]


@dataclass(frozen=True)
class ChatHumanEvalTask:
    config: ChatHumanEvalTaskConfig

    make_grade_step = HumanEvalTask.make_grade_step

    def make_prompts_step(self) -> ExecutorStep:
        return ExecutorStep(
            name="downstream_scaling/evals/prompts/humaneval_chat",
            fn=remote(
                write_humaneval_chat_prompts,
                resources=ResourceConfig.with_cpu(cpu=1, ram="4g"),
                pip_dependency_groups=["lm_eval"],
            ),
            config=ChatHumanEvalPromptsConfig(
                output_path=this_output_path(),
                tokenizer_path=version_path(self.config.tokenizer_path),  # type: ignore[arg-type]
                n_problems=versioned(self.config.n_problems),  # type: ignore[arg-type]
            ),
        )


def write_humaneval_chat_prompts(config: ChatHumanEvalPromptsConfig) -> None:
    _allow_code_eval()
    task = _load_humaneval_task()
    docs = list(task.test_docs())
    if config.n_problems is not None:
        docs = docs[: config.n_problems]

    tokenizer_path = discover_hf_checkpoints(config.tokenizer_path)[-1]
    logger.info("Loading tokenizer from %s", tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path)

    path = prompts_file(config.output_path)
    with fsspec.open(path, "wt", compression="gzip") as f:
        for doc in docs:
            messages = [
                {
                    "role": "user",
                    "content": (
                        "Write a solution to the following problem and make sure that it passes the tests:\n"
                        f"```python\n{doc['prompt']}\n```\n"
                    ),
                }
            ]
            task_id = doc["task_id"]
            row = {
                "id": f"humaneval/test/{task_id}",
                "prompt": render_advisor_prompt(tokenizer, messages, prefill=f"```python\n{doc['prompt']}"),
                "ground_truth": doc["canonical_solution"],
                "metadata": {
                    "task_id": task_id,
                    "raw_prompt": doc["prompt"],
                    "entry_point": doc["entry_point"],
                    "test": doc["test"],
                    "canonical_solution": doc["canonical_solution"],
                    "split": "test",
                },
            }
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d chat-templated HumanEval prompts to %s", len(docs), path)
