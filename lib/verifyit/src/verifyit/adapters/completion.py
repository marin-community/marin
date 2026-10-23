# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize provider completion fields before benchmark-specific extraction."""

import re
from dataclasses import dataclass
from enum import StrEnum

REASONING_END_MARKERS = ("</think>", "<|end_think|>")
BOX_START = re.compile(r"\\(?:boxed|fbox)\s*\{")


class CompletionStatus(StrEnum):
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    FAILED = "failed"


@dataclass(frozen=True)
class Completion:
    content: str
    reasoning: str = ""
    status: CompletionStatus = CompletionStatus.COMPLETED


def final_text(text: str) -> str:
    """Remove a preceding inline reasoning trace using the last end marker."""
    boundary = max(
        (index + len(marker) for marker in REASONING_END_MARKERS if (index := text.rfind(marker)) >= 0),
        default=0,
    )
    return text[boundary:]


def answer_text(completion: Completion) -> str:
    """Prefer final content, allowing completed reasoning-only answers."""
    if completion.content:
        return final_text(completion.content)
    if completion.status != CompletionStatus.COMPLETED:
        return ""
    return final_text(completion.reasoning)


def math_answer_text(completion: Completion) -> str:
    """Read final content or a completed reasoning-only answer containing a box."""
    if not completion.content and not BOX_START.search(completion.reasoning):
        return ""
    return answer_text(completion)
