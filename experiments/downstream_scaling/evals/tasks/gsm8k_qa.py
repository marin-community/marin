# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Zero-shot GSM8K Q+A eval task — canonical lm-eval prompt format.

The prompt format is byte-identical to lm-evaluation-harness's `doc_to_text`:

    Question: <q>\n
    Answer:

The grader is shared with the parent `GSM8KTask` since the answer format
(chain-of-thought + `<<x=y>>` annotations + `#### N` final marker) is the
same; `lm_eval`'s `flexible-extract` / `strict-match` filters handle zero-shot
output identically to few-shot output.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import fsspec
from marin.execution.executor import ExecutorStep, this_output_path, versioned
from marin.execution.remote import remote

from experiments.downstream_scaling.evals.framework.schema import prompts_file
from experiments.downstream_scaling.evals.tasks.gsm8k import GSM8KTask

logger = logging.getLogger(__name__)

ANSWER_PATTERN = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class GSM8KQATaskConfig:
    n_problems: int | None = 256
    grade_workers: int = 32


@dataclass(frozen=True)
class GSM8KQAPromptsConfig:
    output_path: str
    n_problems: int | None


def write_gsm8k_qa_prompts(config: GSM8KQAPromptsConfig) -> None:
    import lm_eval.tasks

    task = lm_eval.tasks.get_task_dict(["gsm8k"])["gsm8k"]
    docs = list(task.test_docs())
    if config.n_problems is not None:
        docs = docs[: config.n_problems]

    path = prompts_file(config.output_path)
    n_written = 0
    with fsspec.open(path, "wt", compression="gzip") as f:
        for i, doc in enumerate(docs):
            match = ANSWER_PATTERN.search(doc["answer"])
            if not match:
                raise ValueError(f"GSM8K problem {i} has no '####' answer marker")
            rec = {
                "id": f"gsm8k_qa/test/{i}",
                "prompt": f"Question: {doc['question']}\nAnswer:",
                "ground_truth": match.group(1).replace(",", ""),
                "metadata": {
                    "problem": doc["question"],
                    "solution": doc["answer"],
                    "split": "test",
                    "num_fewshot": 0,
                    "format": "qa_zeroshot",
                },
            }
            f.write(json.dumps(rec) + "\n")
            n_written += 1
    logger.info("Wrote %d GSM8K Q+A zero-shot prompts to %s", n_written, path)


@dataclass(frozen=True)
class GSM8KQATask:
    config: GSM8KQATaskConfig

    # `make_grade_step` reads `self.config.grade_workers`, which both task configs share.
    make_grade_step = GSM8KTask.make_grade_step

    def make_prompts_step(self) -> ExecutorStep:
        return ExecutorStep(
            name="downstream_scaling/evals/prompts/gsm8k_qa",
            fn=remote(write_gsm8k_qa_prompts, pip_dependency_groups=["eval"]),
            config=GSM8KQAPromptsConfig(
                output_path=this_output_path(),
                n_problems=versioned(self.config.n_problems),  # type: ignore[arg-type]
            ),
        )
