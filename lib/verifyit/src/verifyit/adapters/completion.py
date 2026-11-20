# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize provider completion fields before benchmark-specific extraction."""

from dataclasses import dataclass

from verifyit.compat import StrEnum

REASONING_END_MARKERS = ("</think>", "<|end_think|>")
END_OF_TURN = (
    "<|im_end|>",
    "<|eot_id|>",
    "<|end_of_text|>",
    "<|endoftext|>",
    "</s>",
    "\nYou are an AI assistant",
    "\nQuestion:",
    "\nQ:",
    "\n[Question]",
    "\nUser:",
    "\nuser\n",
    "\nAssistant:",
)


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
