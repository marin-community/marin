# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LongBench v2 tasks with precompiled middle-truncated prompt variants.

The prompt, truncation rule and answer parser follow THUDM/LongBench at
2e00731f8d0bff23dc4325161044d0ed8af94c1e (``prompts/0shot.txt`` and ``pred.py``).
Upstream truncates each prompt to the model's own limit at run time. Here each
item instead carries the full prompt plus copies cut to a fixed ladder of
``o200k_base`` token budgets, keeping the first and last halves as upstream
does. A ``fit_context`` run sends the largest variant that fits the served
context, so every model answers every item and identical budgets yield
identical prompts.
"""

import re
from collections.abc import Iterator, Mapping
from typing import Any

from rigging.filesystem.storage_path import StoragePath

from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation.spec import ChatGeneration, ChatPromptVariant, EvaluationItem
from taskcompendium.importers.evalchemy.grader import source_package
from taskcompendium.importers.evalchemy.long_context import official_tokenizer
from taskcompendium.importers.evalchemy.scorers.longbench_v2 import LONGBENCH_V2_CHOICES
from taskcompendium.models import ConversationInput, TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import RawRow

LONGBENCH_V2_POPULATION = 503
LONGBENCH_V2_LENGTHS = ("short", "medium", "long")
LONGBENCH_V2_LENGTH_LABEL = "longbench.length"
LONGBENCH_V2_DIFFICULTY_LABEL = "longbench.difficulty"
LONGBENCH_V2_DOMAIN_LABEL = "longbench.domain"
LONGBENCH_V2_PROMPT = (
    "Please read the following text and answer the question below.\n\n"
    "<text>\n{context}\n</text>\n\n"
    "What is the correct answer to this question: {question}\n"
    "Choices:\n(A) {A}\n(B) {B}\n(C) {C}\n(D) {D}\n\n"
    'Format your response as follows: "The correct answer is (insert answer here)".'
)
# Upstream truncates to 120,000 tokens for 128K-context models. Rungs step by 1.25 around it,
# so a model whose tokenizer and output budget miss one rung loses at most a fifth of its window.
LONGBENCH_V2_TOKEN_BUDGETS = tuple(round(120_000 * 1.25**exponent) for exponent in range(-12, 10))


def read_longbench_v2(path: StoragePath, _context: ConversionContext) -> Iterator[dict]:
    """Stream the 465 MB source array one record at a time."""
    # ijson is a curation-only optional dependency; evaluator/grader environments do not need it.
    import ijson  # noqa: PLC0415

    with path.open("rb") as stream:
        yield from ijson.items(stream, "item")


def longbench_v2_prompt(row: Mapping[str, Any]) -> str:
    """Fill the upstream 0-shot template without substituting placeholders inside the context."""
    return LONGBENCH_V2_PROMPT.format(
        context=row["context"].strip(),
        question=row["question"].strip(),
        **{choice: row[f"choice_{choice}"].strip() for choice in LONGBENCH_V2_CHOICES},
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def convert_longbench_v2(row: RawRow, context: ConversionContext) -> TaskSpec:
    data = row.data
    if data["answer"] not in LONGBENCH_V2_CHOICES or data["length"] not in LONGBENCH_V2_LENGTHS:
        raise ValueError(f"LongBench v2 row {data['_id']} has answer {data['answer']!r}, length {data['length']!r}")
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=longbench_v2_prompt(data)),),
        package=source_package(
            "longbench_v2", {"expected": data["answer"]}, environment=required_grader_environment(context)
        ),
        # The context is already in the prompt; evidence keeps the row's metadata for review.
        evidence={key: value for key, value in data.items() if key != "context"},
    )
    return task.model_copy(
        update={
            "tags": ("eval", "source:longbench-v2", "benchmark:LongBenchV2", "long-context"),
            "labels": {
                LONGBENCH_V2_LENGTH_LABEL: data["length"],
                LONGBENCH_V2_DIFFICULTY_LABEL: data["difficulty"],
                LONGBENCH_V2_DOMAIN_LABEL: _slug(data["domain"]),
            },
        }
    )


def longbench_v2_item(task: TaskSpec, *, budgets: tuple[int, ...]) -> EvaluationItem:
    """Attach the full prompt and its middle-truncated copies, ranked smallest first.

    ``ChatPromptVariant.fewshot`` is the selection rank: runs try the highest rank first, and rank 0
    is the final fallback.
    """
    [message] = task.context.events
    assert isinstance(message, TextMessage)
    tokenizer = official_tokenizer()
    ids = tokenizer.encode(message.content, disallowed_special=())
    truncated = [
        tokenizer.decode(ids[: budget // 2] + ids[len(ids) - budget // 2 :])
        for budget in sorted(budgets)
        if budget < len(ids)
    ]
    variants = (
        *(
            ChatPromptVariant(
                context=ConversationInput(events=(TextMessage(role="user", content=prompt),)), fewshot=rank
            )
            for rank, prompt in enumerate(truncated)
        ),
        ChatPromptVariant(context=task.context, fewshot=len(truncated)),
    )
    return EvaluationItem(
        task_id=task.id,
        source_row=task.source.row,
        operation=ChatGeneration(prompt_selection="fit_context", response_selection="final_only", variants=variants),
        groups={
            "length": task.labels[LONGBENCH_V2_LENGTH_LABEL],
            "difficulty": task.labels[LONGBENCH_V2_DIFFICULTY_LABEL],
            "domain": task.labels[LONGBENCH_V2_DOMAIN_LABEL],
        },
    )
