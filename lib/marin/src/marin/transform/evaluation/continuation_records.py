# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared rendering for few-shot continuation PPL evals.

Both ``code_interpretation`` and ``prompt_format_sensitivity`` build the same
kind of supervised, target-only records: a prompt of finished support examples
plus one unfinished held-out query, scored only on the continuation the template
would append. The task/template/example shapes differ per eval, so the helpers
here are generic over the example type ``E`` and duck-type the small structural
surface they touch via the protocols below.
"""

from collections.abc import Callable
from typing import Protocol, TypeVar

E = TypeVar("E")


class ContinuationTask(Protocol[E]):
    key: str
    title: str
    description: str
    support_examples: tuple[E, ...]


class ContinuationTemplate(Protocol[E]):
    key: str
    description: str
    renderer: Callable[[E, bool], str]


def render_support_and_query(
    *,
    task: ContinuationTask[E],
    template: ContinuationTemplate[E],
    heldout: E,
    num_fewshot: int,
) -> str:
    """Render ``num_fewshot`` finished support examples followed by one unfinished held-out query."""
    if len(task.support_examples) != num_fewshot:
        raise ValueError(f"{task.key} must have exactly {num_fewshot} support examples")
    header = f"Task: {task.title}\nInstruction: {task.description}\nFormat: {template.description}"
    blocks = [header, *(template.renderer(example, True) for example in task.support_examples)]
    blocks.append(template.renderer(heldout, False))
    return "\n\n".join(blocks)


def render_continuation_target(*, template: ContinuationTemplate[E], heldout: E) -> str:
    """Return the suffix by which the finished render extends the unfinished held-out query.

    Scoring the target on this suffix trains only the continuation tokens.
    """
    unfinished = template.renderer(heldout, False)
    finished = template.renderer(heldout, True)
    if not finished.startswith(unfinished):
        raise ValueError(f"{template.key} renderer must extend its unfinished held-out query")
    return finished[len(unfinished) :]
