# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reference-conditioned GSM8K chat prompts for instruction-tuned advisors."""

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
from experiments.downstream_scaling.evals.tasks.gsm8k import GSM8KTask, _load_gsm8k_task
from experiments.downstream_scaling.evals.tasks.gsm8k_chat import ANSWER_PATTERN, render_advisor_prompt
from experiments.downstream_scaling.evals.utils import discover_hf_checkpoints, version_path

logger = logging.getLogger(__name__)

TWO_SOLUTION_MESSAGES: tuple[dict[str, str], ...] = (
    {
        "role": "user",
        "content": (
            "Generate two independent, correct solutions to this problem.\n\n"
            "{question}\n\nEnd each solution with: #### <number>"
        ),
    },
)
TWO_SOLUTION_PREFILL = "Solution 1: {reference_solution}\nSolution 2:"

FOLLOWUP_MESSAGES: tuple[dict[str, str], ...] = (
    {"role": "user", "content": "{question}\n\nEnd your answer with: #### <number>"},
    {"role": "assistant", "content": "Answer: {reference_solution}"},
    {"role": "user", "content": "Great. Can you solve it another way?"},
)
FOLLOWUP_PREFILL = "Answer:"


@dataclass(frozen=True)
class ReferenceChatGSM8KTaskConfig:
    tokenizer_path: str | InputName | MirroredValue
    n_problems: int | None = None
    grade_workers: int = 32


@dataclass(frozen=True)
class ReferenceChatGSM8KPromptsConfig:
    output_path: str
    tokenizer_path: str
    n_problems: int | None
    message_templates: tuple[dict[str, str], ...]
    prefill_template: str


@dataclass(frozen=True)
class TwoSolutionChatGSM8KTask:
    config: ReferenceChatGSM8KTaskConfig

    make_grade_step = GSM8KTask.make_grade_step

    def make_prompts_step(self) -> ExecutorStep:
        return _make_prompts_step(
            name="downstream_scaling/evals/prompts/gsm8k_chat_two_solution",
            config=self.config,
            message_templates=TWO_SOLUTION_MESSAGES,
            prefill_template=TWO_SOLUTION_PREFILL,
        )


@dataclass(frozen=True)
class FollowupChatGSM8KTask:
    config: ReferenceChatGSM8KTaskConfig

    make_grade_step = GSM8KTask.make_grade_step

    def make_prompts_step(self) -> ExecutorStep:
        return _make_prompts_step(
            name="downstream_scaling/evals/prompts/gsm8k_chat_followup",
            config=self.config,
            message_templates=FOLLOWUP_MESSAGES,
            prefill_template=FOLLOWUP_PREFILL,
        )


def _make_prompts_step(
    *,
    name: str,
    config: ReferenceChatGSM8KTaskConfig,
    message_templates: tuple[dict[str, str], ...],
    prefill_template: str,
) -> ExecutorStep:
    return ExecutorStep(
        name=name,
        fn=remote(
            write_gsm8k_chat_reference_prompts,
            resources=ResourceConfig.with_cpu(cpu=1, ram="4g"),
            pip_dependency_groups=["lm_eval"],
        ),
        config=ReferenceChatGSM8KPromptsConfig(
            output_path=this_output_path(),
            tokenizer_path=version_path(config.tokenizer_path),  # type: ignore[arg-type]
            n_problems=versioned(config.n_problems),  # type: ignore[arg-type]
            message_templates=versioned(message_templates),  # type: ignore[arg-type]
            prefill_template=versioned(prefill_template),  # type: ignore[arg-type]
        ),
    )


def write_gsm8k_chat_reference_prompts(config: ReferenceChatGSM8KPromptsConfig) -> None:
    """Write advisor prompts with a reference solution and an open assistant prefill."""
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

            fields = {"question": doc["question"], "reference_solution": doc["answer"]}
            messages = [
                {"role": template["role"], "content": template["content"].format(**fields)}
                for template in config.message_templates
            ]
            row = {
                "id": f"gsm8k/test/{i}",
                "prompt": render_advisor_prompt(tokenizer, messages, prefill=config.prefill_template.format(**fields)),
                "ground_truth": match.group(1).replace(",", ""),
                "metadata": {
                    "problem": doc["question"],
                    "solution": doc["answer"],
                    "split": "test",
                },
            }
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d reference-conditioned GSM8K prompts to %s", len(docs), path)
