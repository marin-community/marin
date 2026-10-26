# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Luke Bailey's hinted GSM8K prompts for the Delphi ladder.

Reproduces the GSM8K prompts that ``experiments/query_hinted_benchmarks/make_prompts.py``
in LukeBailey181/emergence-logit-mixing writes at ``SOURCE_REVISION``: the ``mask`` and
``truncate`` hint-block methods at hint levels 0.1 through 1.0, and the unhinted
``no_hint`` prompt. The prompt helpers below are copied from that revision's
``src/hinted/benchmarks/__init__.py`` and ``src/hinted/benchmarks/gsm8k_delphi.py``;
its two GSM8K data files are uploaded unchanged under ``SOURCE_PREFIX``.

GSM8K has one reference solution per problem, so every sample of a problem uses the
same prompt: the target hint is masked with seed ``f"{problem_id}|0"``, as Luke's
runner does for rollout 0, and does not depend on the sample index.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import fsspec
from fray.cluster import ResourceConfig
from thalas.execution.executor import ExecutorStep
from thalas.execution.remote import remote
from thalas.execution.types import this_output_path, versioned

from experiments.downstream_scaling.evals.framework.schema import prompts_file
from experiments.downstream_scaling.evals.tasks.gsm8k import GSM8KTask

logger = logging.getLogger(__name__)

SOURCE_REVISION = "56073be1dc97163b08121edbfc2b13e696b2aea7"
SOURCE_PREFIX = f"gs://marin-us-central2/raw/gsm8k_luke_hints/{SOURCE_REVISION}"
PROBLEMS_PATH = f"{SOURCE_PREFIX}/gsm8k_delphi.jsonl"
SOLUTIONS_PATH = f"{SOURCE_PREFIX}/gsm8k_solution_hints.jsonl"
HINT_LEVELS: tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(1, 11))
NO_HINT_LEVELS: tuple[float, ...] = (0.0,)
MASK_TOKEN = "<mask>"


class LukeHintMethod(StrEnum):
    MASK = "mask"
    TRUNCATE = "truncate"
    NO_HINT = "no_hint"


@dataclass(frozen=True)
class LukeHintsGSM8KTaskConfig:
    method: LukeHintMethod
    grade_workers: int = 32


@dataclass(frozen=True)
class LukeHintsGSM8KPromptsConfig:
    output_path: str
    problems_path: str
    solutions_path: str
    method: LukeHintMethod
    source_revision: str
    hint_levels: tuple[float, ...]


@dataclass(frozen=True)
class LukeHintsGSM8KTask:
    config: LukeHintsGSM8KTaskConfig

    make_grade_step = GSM8KTask.make_grade_step

    def make_prompts_step(self) -> ExecutorStep:
        method = self.config.method
        return ExecutorStep(
            name=f"downstream_scaling/evals/prompts/gsm8k_luke_hints/{method}",
            fn=remote(write_luke_hints_gsm8k_prompts, resources=ResourceConfig.with_cpu(cpu=1, ram="4g")),
            config=LukeHintsGSM8KPromptsConfig(
                output_path=this_output_path(),
                problems_path=versioned(PROBLEMS_PATH),  # type: ignore[arg-type]
                solutions_path=versioned(SOLUTIONS_PATH),  # type: ignore[arg-type]
                method=versioned(method),  # type: ignore[arg-type]
                source_revision=versioned(SOURCE_REVISION),  # type: ignore[arg-type]
                hint_levels=versioned(  # type: ignore[arg-type]
                    NO_HINT_LEVELS if method is LukeHintMethod.NO_HINT else HINT_LEVELS
                ),
            ),
        )


def _stable_int(*parts: object) -> int:
    """Process-stable hash (unlike hash()), for reproducible masking."""
    text = "||".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def mask_word(full_hint: str, fraction: float, *, seed: str, token: str) -> str:
    """Keep ceil(len * fraction) words visible, replacing the rest with ``token``.

    The masked subset is chosen by a hash-seeded RNG over (seed, fraction), and
    whitespace is preserved.
    """
    words = list(re.finditer(r"\S+", full_hint))
    keep = max(0, min(len(words), math.ceil(len(words) * fraction)))
    visible = set(range(len(words)))
    if len(words) > keep:
        rng = random.Random(_stable_int(seed, "mask_word", fraction))
        visible -= set(rng.sample(range(len(words)), len(words) - keep))

    parts: list[str] = []
    cursor = 0
    for idx, word in enumerate(words):
        parts.append(full_hint[cursor : word.start()])
        parts.append(word.group(0) if idx in visible else token)
        cursor = word.end()
    parts.append(full_hint[cursor:])
    return "".join(parts)


def truncate_word(full_hint: str, fraction: float) -> str:
    """Keep the first ceil(len * fraction) words, sliced from the original string."""
    words = list(re.finditer(r"\S+", full_hint))
    keep = max(0, min(len(words), math.ceil(len(words) * fraction)))
    if keep == 0:
        return ""
    if keep >= len(words):
        return full_hint
    return full_hint[: words[keep - 1].end()]


def redact_answer(text: str, answer: str) -> str:
    """Replace standalone occurrences of ``answer`` in ``text`` with ``<mask>``.

    Standalone means not embedded in a longer number. Both the plain and the
    comma-grouped spelling of the answer are covered.
    """
    plain = answer.strip().replace(",", "").replace("$", "")
    variants = {plain}
    sign, digits = (plain[0], plain[1:]) if plain.startswith("-") else ("", plain)
    if digits.isdigit() and len(digits) > 3:
        grouped = ""
        for i, ch in enumerate(reversed(digits)):
            if i and i % 3 == 0:
                grouped = "," + grouped
            grouped = ch + grouped
        variants.add(sign + grouped)
    pattern = "|".join(re.escape(v) for v in sorted(variants, key=len, reverse=True))
    return re.sub(rf"(?<![\d.])(?:{pattern})(?![\d.])", MASK_TOKEN, text)


def parse_exemplars(prefix: str) -> list[tuple[str, str, str]]:
    """Split the stored five-shot prefix into (question, working, answer) triples.

    Reassembling the triples must reproduce the stored prefix byte for byte.
    """
    # Split at block boundaries, not on every blank line: some workings contain blank lines.
    blocks = re.split(r"(?<=\n\n)(?=Question: )", prefix)
    triples = []
    for b in blocks:
        q, _, rest = b.partition("\nAnswer: ")
        assert q.startswith("Question: "), f"bad block start: {b[:40]!r}"
        working, _, tail = rest.rpartition("\n#### ")
        ans = tail.rstrip("\n")
        rebuilt = f"{q}\nAnswer: {working}\n#### {ans}\n\n"
        assert rebuilt == b, f"block reassembly mismatch: {b[:60]!r}"
        triples.append((q[len("Question: ") :], working, ans))
    assert len(triples) == 5, f"expected 5 exemplars, got {len(triples)}"
    return triples


def hinted_prompt(problem: dict[str, Any], full_hint: str, method: LukeHintMethod, level: float) -> str:
    """Hint-block prompt: every exemplar and the target show a hint at ``level``."""
    problem_id = problem["problem_id"]
    answer = str(problem["answer"])
    prefix = problem["metadata"]["exemplar_prefix"]
    parts = []
    for exemplar_index, (question, working, exemplar_answer) in enumerate(parse_exemplars(prefix)):
        if method is LukeHintMethod.MASK:
            fragment = mask_word(working, level, seed=f"{problem_id}|exemplar|{exemplar_index}", token=MASK_TOKEN)
        else:
            fragment = redact_answer(truncate_word(working, level), exemplar_answer)
        parts.append(f"Question: {question}\nHint: {fragment}\nAnswer: {working}\n#### {exemplar_answer}\n\n")
    if method is LukeHintMethod.MASK:
        shown = mask_word(full_hint, level, seed=f"{problem_id}|0", token=MASK_TOKEN)
    else:
        shown = redact_answer(truncate_word(full_hint, level), answer)
    parts.append(f"Question: {problem['question']}\nHint: {shown}\nAnswer:")
    return "".join(parts)


def question_prompt(problem: dict[str, Any]) -> str:
    """The stored five-shot prefix and the target question, ending at "Answer:"."""
    return f"{problem['metadata']['exemplar_prefix']}Question: {problem['question']}\nAnswer:"


def write_luke_hints_gsm8k_prompts(config: LukeHintsGSM8KPromptsConfig) -> None:
    """Write one method's prompts, iterating hint levels, then problems in source order."""
    with fsspec.open(config.problems_path, "rt") as f:
        problems = [json.loads(line) for line in f if line.strip()]
    with fsspec.open(config.solutions_path, "rt") as f:
        solutions = {row["problem_id"]: row["full_hint"] for row in (json.loads(line) for line in f if line.strip())}

    method = LukeHintMethod(config.method)
    path = prompts_file(config.output_path)
    with fsspec.open(path, "wt", compression="gzip") as f:
        for level in config.hint_levels:
            for problem in problems:
                problem_id = problem["problem_id"]
                prompt = (
                    question_prompt(problem)
                    if method is LukeHintMethod.NO_HINT
                    else hinted_prompt(problem, solutions[problem_id], method, level)
                )
                row = {
                    "id": f"gsm8k/luke/{method}/h{level:g}/{problem_id}",
                    "prompt": prompt,
                    "ground_truth": str(problem["answer"]),
                    "metadata": {
                        "problem": problem["question"],
                        "problem_id": problem_id,
                        "hint_method": str(method),
                        "hint_level": level,
                        "source_revision": config.source_revision,
                    },
                }
                f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d Luke-hinted GSM8K %s prompts to %s", len(problems) * len(config.hint_levels), method, path)
