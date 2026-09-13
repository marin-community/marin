# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Chat-templated GSM8K advisor prompts for downstream-scaling evals."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import fsspec
from fray.cluster import ResourceConfig
from levanter.compat.hf_checkpoints import load_tokenizer
from thalas.execution.executor import ExecutorStep, InputName, MirroredValue
from thalas.execution.remote import remote
from thalas.execution.types import this_output_path, versioned

from experiments.downstream_scaling.evals.framework.schema import prompts_file
from experiments.downstream_scaling.evals.tasks.gsm8k import GSM8KTask, _load_gsm8k_task
from experiments.downstream_scaling.evals.utils import discover_hf_checkpoints, version_path

logger = logging.getLogger(__name__)

ANSWER_PATTERN = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class ChatGSM8KTaskConfig:
    tokenizer_path: str | InputName | MirroredValue
    n_problems: int | None = None
    grade_workers: int = 32


@dataclass(frozen=True)
class ChatGSM8KPromptsConfig:
    output_path: str
    tokenizer_path: str
    n_problems: int | None


def render_advisor_prompt(tokenizer: Any, messages: list[dict[str, str]], prefill: str) -> str:
    """Chat-template messages, then continue from prefill byte-exactly.

    The joint-decode worker tokenizes prompts with special tokens enabled,
    which adds one leading BOS. Chat-templated text already contains its BOS,
    so remove it here. Append the prefill directly because chat-template
    continuation helpers trim trailing whitespace.
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text.removeprefix(tokenizer.bos_token or "") + prefill


@dataclass(frozen=True)
class ChatGSM8KTask:
    config: ChatGSM8KTaskConfig

    make_grade_step = GSM8KTask.make_grade_step

    def make_prompts_step(self) -> ExecutorStep:
        return ExecutorStep(
            name="downstream_scaling/evals/prompts/gsm8k_chat",
            fn=remote(
                write_gsm8k_chat_prompts,
                resources=ResourceConfig.with_cpu(cpu=1, ram="4g"),
                pip_dependency_groups=["lm_eval"],
            ),
            config=ChatGSM8KPromptsConfig(
                output_path=this_output_path(),
                tokenizer_path=version_path(self.config.tokenizer_path),  # type: ignore[arg-type]
                n_problems=versioned(self.config.n_problems),  # type: ignore[arg-type]
            ),
        )


def write_gsm8k_chat_prompts(config: ChatGSM8KPromptsConfig) -> None:
    tokenizer_path = discover_hf_checkpoints(config.tokenizer_path)[-1]
    logger.info("Loading tokenizer from %s", tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path)

    task = _load_gsm8k_task()
    docs = list(task.test_docs())
    if config.n_problems is not None:
        docs = docs[: config.n_problems]

    path = prompts_file(config.output_path)
    with fsspec.open(path, "wt", compression="gzip") as f:
        for i, doc in enumerate(docs):
            match = ANSWER_PATTERN.search(doc["answer"])
            if not match:
                raise ValueError(f"GSM8K problem {i} has no '####' answer marker")

            messages = [
                {
                    "role": "user",
                    "content": f"{doc['question']}\n\nEnd your answer with: #### <number>",
                }
            ]
            row = {
                "id": f"gsm8k/test/{i}",
                "prompt": render_advisor_prompt(tokenizer, messages, prefill="Answer:"),
                "ground_truth": match.group(1).replace(",", ""),
                "metadata": {
                    "problem": doc["question"],
                    "solution": doc["answer"],
                    "split": "test",
                },
            }
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d chat-templated GSM8K prompts to %s", len(docs), path)
