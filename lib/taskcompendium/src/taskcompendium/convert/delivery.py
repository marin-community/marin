# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Move a source's file-delivery wording into the assistant reply and record the rewrite.

Many sources were written for a shell agent that writes its answer to a file. A conversation task
asks for the same answer in its reply instead; the rewrite is kept as a normalization change, so the
audit retains the source's original wording.
"""

from collections.abc import Sequence

from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.models import NormalizationChange, NormalizedTask


def replace_phrases(text: str, phrases: Sequence[tuple[str, str]]) -> str:
    """Apply each ``(original, replacement)`` rewrite in order."""
    for original, replacement in phrases:
        text = text.replace(original, replacement)
    return text


def rewritten_task(task: TaskSpec, *, original: str, reason: str) -> TaskSpec | NormalizedTask:
    """``task``, with a change record when its first message differs from the source's ``original``.

    Stripping outer whitespace alone is not recorded as a change.
    """
    prompt = task.context.events[0]
    assert isinstance(prompt, TextMessage)
    if prompt.content.strip() == original.strip():
        return task
    change = NormalizationChange(field="instruction", reason=reason, original=original, replacement=prompt.content)
    return NormalizedTask(task, (change,))
