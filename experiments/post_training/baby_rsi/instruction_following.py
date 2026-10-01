# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Overlay IFEval-style output constraints on problems whose answers are already verifiable.

Each problem keeps its question and reference answer and gains one or two formatting constraints,
such as an exact bullet count or a closing phrase, that a program can check. Constraints are stored
as IFEval stores them, a kind plus keyword arguments, in a JSON ``constraints`` column, and their
instructions are appended to the problem text. ``self_distill.grade_samples`` accepts a sample only
when its answer is correct and its final response, excluding the reasoning, satisfies every
constraint, so training rows teach the model to follow the format while still solving the problem.

The base problems are the programmatic finance problems, which need no teacher model.
"""

import json
import logging
import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import pyarrow as pa
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_owned_name
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.baby_rsi.finance import FINANCE_PROBLEM_SCHEMA, finance_problem
from experiments.post_training.baby_rsi.generation import MANIFEST_FILENAME, PROBLEMS_FILENAME, write_table

logger = logging.getLogger(__name__)

BULLET_PATTERN = re.compile(r"^\s*[*-]\s", re.MULTILINE)
BOXED_PATTERN = re.compile(r"\\boxed\{[^{}]*\}")
JSON_BLOCK_PATTERN = re.compile(r"```json\n(.*?)```\s*$", re.DOTALL)
TITLE_PATTERN = re.compile(r"<<[^<>\n]+>>")
PARAGRAPH_DIVIDER = "***"
END_PHRASES = ("That concludes the analysis.", "Let me know if you have any other questions.")
KEYWORDS = ("analysis", "result", "figure", "calculation")

INSTRUCTION_FOLLOWING_PROBLEM_SCHEMA = FINANCE_PROBLEM_SCHEMA.append(pa.field("constraints", pa.string()))


class ConstraintKind(StrEnum):
    BULLET_COUNT = "bullet_count"
    END_PHRASE = "end_phrase"
    NO_COMMAS = "no_commas"
    LOWERCASE = "lowercase"
    QUOTATION = "quotation"
    KEYWORD_FREQUENCY = "keyword_frequency"
    WORD_LIMIT = "word_limit"
    TITLE = "title"
    JSON_ANSWER = "json_answer"
    PARAGRAPH_COUNT = "paragraph_count"


@dataclass(frozen=True)
class ConstraintSpec:
    """One registered constraint kind.

    ``instruction`` is formatted with the sampled keyword arguments, and ``check`` receives the final
    response followed by the same arguments.
    """

    instruction: str
    check: Callable[..., bool]
    sample_kwargs: Callable[[random.Random], dict[str, Any]]


def _no_kwargs(_rng: random.Random) -> dict[str, Any]:
    return {}


def _has_bullets(response: str, count: int) -> bool:
    return len(BULLET_PATTERN.findall(response)) == count


def _ends_with(response: str, phrase: str) -> bool:
    return response.strip().endswith(phrase)


def _is_lowercase(response: str) -> bool:
    text = BOXED_PATTERN.sub("", response)
    return text == text.lower()


def _is_quoted(response: str) -> bool:
    text = response.strip()
    return len(text) > 1 and text[0] == '"' and text[-1] == '"'


def _has_keyword(response: str, keyword: str, count: int) -> bool:
    return len(re.findall(rf"\b{re.escape(keyword)}\b", response, re.IGNORECASE)) >= count


def _ends_with_json_answer(response: str) -> bool:
    match = JSON_BLOCK_PATTERN.search(response)
    if match is None:
        return False
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict) and list(value) == ["answer"] and isinstance(value["answer"], int | float)


def _has_paragraphs(response: str, count: int) -> bool:
    paragraphs = response.strip().split(PARAGRAPH_DIVIDER)
    return len(paragraphs) == count and all(paragraph.strip() for paragraph in paragraphs)


CONSTRAINTS: dict[ConstraintKind, ConstraintSpec] = {
    ConstraintKind.BULLET_COUNT: ConstraintSpec(
        "Your response must contain exactly {count} bullet points, written as markdown bullets such as:\n"
        "* This is a point.",
        _has_bullets,
        lambda rng: {"count": rng.choice((2, 3, 4))},
    ),
    ConstraintKind.END_PHRASE: ConstraintSpec(
        'Finish your response with this exact phrase: "{phrase}". No other words should follow it.',
        _ends_with,
        lambda rng: {"phrase": rng.choice(END_PHRASES)},
    ),
    ConstraintKind.NO_COMMAS: ConstraintSpec(
        "Do not use any commas in your response, including inside numbers.",
        lambda response: "," not in response,
        _no_kwargs,
    ),
    ConstraintKind.LOWERCASE: ConstraintSpec(
        "Write your entire response in lowercase letters. No capital letters are allowed outside \\boxed{{}}.",
        _is_lowercase,
        _no_kwargs,
    ),
    ConstraintKind.QUOTATION: ConstraintSpec(
        "Wrap your entire response in double quotation marks.",
        _is_quoted,
        _no_kwargs,
    ),
    ConstraintKind.KEYWORD_FREQUENCY: ConstraintSpec(
        'Use the word "{keyword}" at least {count} times in your response.',
        _has_keyword,
        lambda rng: {"keyword": rng.choice(KEYWORDS), "count": rng.choice((2, 3))},
    ),
    ConstraintKind.WORD_LIMIT: ConstraintSpec(
        "Answer with fewer than {limit} words.",
        lambda response, limit: len(response.split()) < limit,
        lambda rng: {"limit": rng.choice((100, 150, 200))},
    ),
    ConstraintKind.TITLE: ConstraintSpec(
        "Give your response a title wrapped in double angular brackets, such as <<quarterly review>>.",
        lambda response: TITLE_PATTERN.search(response) is not None,
        _no_kwargs,
    ),
    ConstraintKind.JSON_ANSWER: ConstraintSpec(
        'End your response with a ```json code block holding an object with the single key "answer" whose '
        "value is your final number.",
        _ends_with_json_answer,
        _no_kwargs,
    ),
    ConstraintKind.PARAGRAPH_COUNT: ConstraintSpec(
        "Your response must have exactly {count} paragraphs, separated from each other by the markdown divider ***.",
        _has_paragraphs,
        lambda rng: {"count": rng.choice((2, 3))},
    ),
}

# Pairs that no response can satisfy together: both claim the end of the response, or the closing
# phrases contain capital letters.
CONFLICTS = frozenset(
    frozenset(pair)
    for pair in (
        (ConstraintKind.END_PHRASE, ConstraintKind.QUOTATION),
        (ConstraintKind.END_PHRASE, ConstraintKind.JSON_ANSWER),
        (ConstraintKind.QUOTATION, ConstraintKind.JSON_ANSWER),
        (ConstraintKind.END_PHRASE, ConstraintKind.LOWERCASE),
    )
)


def sample_constraints(rng: random.Random) -> list[dict[str, Any]]:
    """Draw one or two compatible constraints as IFEval-style ``{"kind", "kwargs"}`` records."""
    kinds = [rng.choice(list(ConstraintKind))]
    if rng.random() < 0.5:
        compatible = [kind for kind in ConstraintKind if kind != kinds[0] and {kind, kinds[0]} not in CONFLICTS]
        kinds.append(rng.choice(compatible))
    return [{"kind": str(kind), "kwargs": CONSTRAINTS[kind].sample_kwargs(rng)} for kind in kinds]


def constraint_instructions(constraints: list[dict[str, Any]]) -> str:
    lines = [CONSTRAINTS[ConstraintKind(c["kind"])].instruction.format(**c["kwargs"]) for c in constraints]
    return "Your final response must also follow these instructions:\n" + "\n".join(f"- {line}" for line in lines)


def follows_constraints(constraints: list[dict[str, Any]], response: str) -> bool:
    """Whether ``response`` satisfies every constraint record."""
    return all(CONSTRAINTS[ConstraintKind(c["kind"])].check(response, **c["kwargs"]) for c in constraints)


def instruction_following_problem(capability_id: str, index: int, seed: int) -> dict[str, Any]:
    """Finance problem ``index`` with one or two output constraints appended."""
    row = finance_problem(capability_id, index, seed)
    constraints = sample_constraints(random.Random(f"{seed}/{capability_id}/{index}/constraints"))
    return {
        **row,
        "problem": f"{row['problem']}\n\n{constraint_instructions(constraints)}",
        "constraints": json.dumps(constraints),
    }


@dataclass(frozen=True)
class InstructionFollowingProblemsConfig:
    output_path: str
    capability_id: str
    count: int
    seed: int


def write_instruction_following_problems(config: InstructionFollowingProblemsConfig) -> Artifact:
    """Write the constrained problems Parquet and a manifest."""
    rows = [instruction_following_problem(config.capability_id, index, config.seed) for index in range(config.count)]
    output = StoragePath(config.output_path)
    output.mkdirs()
    write_table(output, PROBLEMS_FILENAME, rows, INSTRUCTION_FOLLOWING_PROBLEM_SCHEMA)
    kinds: dict[str, int] = {}
    for row in rows:
        for constraint in json.loads(row["constraints"]):
            kinds[constraint["kind"]] = kinds.get(constraint["kind"], 0) + 1
    manifest = {
        "capability_id": config.capability_id,
        "generator": "programmatic finance problems with overlaid output constraints",
        "seed": config.seed,
        "problems": len(rows),
        "constraints": kinds,
    }
    (output / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("wrote %s constrained problems for %s", len(rows), config.capability_id)
    return Artifact(path=config.output_path)


def generate_instruction_following_problems(
    capability_id: str, *, version: str, count: int, seed: int
) -> ArtifactStep[Artifact]:
    """Build one CPU step that writes ``count`` constrained finance problems for a capability."""

    def build_config(ctx: StepContext) -> InstructionFollowingProblemsConfig:
        return InstructionFollowingProblemsConfig(
            output_path=ctx.output_path, capability_id=capability_id, count=count, seed=seed
        )

    return ArtifactStep(
        name=user_owned_name(f"documents/curriculum-sft/{capability_id}/instruction-following-problems"),
        version=version,
        artifact_type=Artifact,
        run=write_instruction_following_problems,
        build_config=build_config,
    )
