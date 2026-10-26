# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AIME and HLE chat prompts with Luke Bailey's stored full hints."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import fsspec
from fray.cluster import ResourceConfig
from levanter.compat.hf_checkpoints import load_tokenizer
from thalas.execution.executor import ExecutorStep, InputName, MirroredValue
from thalas.execution.remote import remote
from thalas.execution.types import this_output_path, versioned

from experiments.downstream_scaling.evals.framework.schema import prompts_file
from experiments.downstream_scaling.evals.tasks.gsm8k_chat import render_advisor_prompt
from experiments.downstream_scaling.evals.utils import discover_hf_checkpoints, version_path

logger = logging.getLogger(__name__)

SOURCE_PREFIX = "gs://marin-us-central2/raw/posttrained_shared_teacher/20261023"
AIME_PROBLEMS_PATH = f"{SOURCE_PREFIX}/aime2025_2026.jsonl"
AIME_HINTS_PATH = f"{SOURCE_PREFIX}/answer_not_revealed.jsonl"
HLE_PROBLEMS_PATH = f"{SOURCE_PREFIX}/hle.jsonl"
HLE_HINTS_PATH = f"{SOURCE_PREFIX}/hle_answer_not_revealed_10.jsonl"
SAMPLES_PER_PROBLEM = 10

AIME_STUDENT_TEMPLATE = (
    "Put your final answer within \\boxed{{}}.\n"
    "The answer is an integer between 0 and 999 inclusive.\n\n"
    "Problem:\n{question}"
)
AIME_ADVISOR_TEMPLATE = (
    "You will be given a problem and a hint to the problem.\n"
    "Put your final answer within \\boxed{{}}.\n"
    "The answer is an integer between 0 and 999 inclusive.\n\n"
    "Problem:\n{question}\n\nHint:\n{hint}"
)
HLE_STUDENT_TEMPLATE = "Problem:\n{question}\n\nGive your final answer in the following format:\nAnswer: {{your answer}}"
HLE_ADVISOR_TEMPLATE = (
    "You will be given a problem and a hint to the problem. Use the hint if helpful.\n\n"
    "Problem:\n{question}\n\nHint:\n{hint}\n\n"
    "Give your final answer in the following format:\nAnswer: {{your answer}}"
)


@dataclass(frozen=True)
class HintedBenchmarkTaskConfig:
    tokenizer_path: str | InputName | MirroredValue


@dataclass(frozen=True)
class HintedBenchmarkPromptsConfig:
    output_path: str
    tokenizer_path: str
    problems_path: str
    hints_path: str
    benchmark: str
    message_template: str


@dataclass(frozen=True)
class AIMETask:
    config: HintedBenchmarkTaskConfig

    def make_prompts_step(self) -> ExecutorStep:
        return _make_prompts_step(
            name="downstream_scaling/evals/prompts/aime_shared_teacher/student",
            config=self.config,
            problems_path=AIME_PROBLEMS_PATH,
            hints_path=AIME_HINTS_PATH,
            benchmark="aime",
            message_template=AIME_STUDENT_TEMPLATE,
        )

    def make_grade_step(
        self,
        *,
        name: str,
        prompts_path: str | InputName | MirroredValue,
        completions_path: str | InputName | MirroredValue,
    ) -> ExecutorStep:
        raise NotImplementedError


@dataclass(frozen=True)
class AIMEAdvisorTask:
    config: HintedBenchmarkTaskConfig

    def make_prompts_step(self) -> ExecutorStep:
        return _make_prompts_step(
            name="downstream_scaling/evals/prompts/aime_shared_teacher/advisor",
            config=self.config,
            problems_path=AIME_PROBLEMS_PATH,
            hints_path=AIME_HINTS_PATH,
            benchmark="aime",
            message_template=AIME_ADVISOR_TEMPLATE,
        )

    def make_grade_step(
        self,
        *,
        name: str,
        prompts_path: str | InputName | MirroredValue,
        completions_path: str | InputName | MirroredValue,
    ) -> ExecutorStep:
        raise NotImplementedError


@dataclass(frozen=True)
class HLETask:
    config: HintedBenchmarkTaskConfig

    def make_prompts_step(self) -> ExecutorStep:
        return _make_prompts_step(
            name="downstream_scaling/evals/prompts/hle_shared_teacher/student",
            config=self.config,
            problems_path=HLE_PROBLEMS_PATH,
            hints_path=HLE_HINTS_PATH,
            benchmark="hle",
            message_template=HLE_STUDENT_TEMPLATE,
        )

    def make_grade_step(
        self,
        *,
        name: str,
        prompts_path: str | InputName | MirroredValue,
        completions_path: str | InputName | MirroredValue,
    ) -> ExecutorStep:
        raise NotImplementedError


@dataclass(frozen=True)
class HLEAdvisorTask:
    config: HintedBenchmarkTaskConfig

    def make_prompts_step(self) -> ExecutorStep:
        return _make_prompts_step(
            name="downstream_scaling/evals/prompts/hle_shared_teacher/advisor",
            config=self.config,
            problems_path=HLE_PROBLEMS_PATH,
            hints_path=HLE_HINTS_PATH,
            benchmark="hle",
            message_template=HLE_ADVISOR_TEMPLATE,
        )

    def make_grade_step(
        self,
        *,
        name: str,
        prompts_path: str | InputName | MirroredValue,
        completions_path: str | InputName | MirroredValue,
    ) -> ExecutorStep:
        raise NotImplementedError


def _make_prompts_step(
    *,
    name: str,
    config: HintedBenchmarkTaskConfig,
    problems_path: str,
    hints_path: str,
    benchmark: str,
    message_template: str,
) -> ExecutorStep:
    return ExecutorStep(
        name=name,
        fn=remote(
            write_hinted_benchmark_prompts,
            resources=ResourceConfig.with_cpu(cpu=1, ram="4g"),
            pip_dependency_groups=["lm_eval"],
        ),
        config=HintedBenchmarkPromptsConfig(
            output_path=this_output_path(),
            tokenizer_path=version_path(config.tokenizer_path),  # type: ignore[arg-type]
            problems_path=versioned(problems_path),  # type: ignore[arg-type]
            hints_path=versioned(hints_path),  # type: ignore[arg-type]
            benchmark=versioned(benchmark),  # type: ignore[arg-type]
            message_template=versioned(message_template),  # type: ignore[arg-type]
        ),
    )


def write_hinted_benchmark_prompts(config: HintedBenchmarkPromptsConfig) -> None:
    """Write ten paired prompt rows per problem, ordered by the stored hint rollouts."""
    with fsspec.open(config.problems_path, "rt") as f:
        problems = [json.loads(line) for line in f]
    hints_by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with fsspec.open(config.hints_path, "rt") as f:
        for line in f:
            hint = json.loads(line)
            hints_by_problem[hint["problem_id"]].append(hint)

    problem_ids = {problem["problem_id"] for problem in problems}
    if not problems or len(problem_ids) != len(problems) or problem_ids != set(hints_by_problem):
        raise ValueError("Problems must be nonempty, unique, and match the hint problem IDs")
    for problem_id, hints in hints_by_problem.items():
        if len(hints) != SAMPLES_PER_PROBLEM:
            raise ValueError(f"Problem {problem_id!r} has {len(hints)} hints; expected {SAMPLES_PER_PROBLEM}")
        hints.sort(key=lambda hint: (hint["rollout_id"], hint["full_hint"]))

    tokenizer_path = discover_hf_checkpoints(config.tokenizer_path)[-1]
    logger.info("Loading tokenizer from %s", tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path)
    path = prompts_file(config.output_path)
    with fsspec.open(path, "wt", compression="gzip") as f:
        for problem in problems:
            problem_id = problem["problem_id"]
            for sample_id, hint in enumerate(hints_by_problem[problem_id]):
                raw_prompt = config.message_template.format(
                    question=problem["question"].strip(), hint=hint["full_hint"].strip()
                )
                row = {
                    "id": f"{config.benchmark}/shared_teacher/{problem_id}/sample{sample_id}",
                    "prompt": render_advisor_prompt(tokenizer, [{"role": "user", "content": raw_prompt}], prefill=""),
                    "ground_truth": str(problem["answer"]),
                    "metadata": {
                        "problem": problem["question"],
                        "problem_id": problem_id,
                        "sample_id": sample_id,
                        "rollout_id": hint["rollout_id"],
                        "source": problem["source"],
                    },
                }
                f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d %s prompts to %s", len(problems) * SAMPLES_PER_PROBLEM, config.benchmark, path)
