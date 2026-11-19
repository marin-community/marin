# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select scorer text without discarding the archived completion channels.

Preserves Marin Evalchemy's CompletionResponse normalization and MMLU-Pro's
content-first extraction policy. A length-limited reasoning trace is not a
complete final answer under the latter policy. ``final_only`` matches
Evalchemy's ``final_only`` content policy: graders such as LLM judges see the
final answer and never the reasoning, even when the final answer is empty.

A reply served without a reasoning parser can carry its reasoning inline,
between a chat template's delimiters. ``split_inline_reasoning`` moves that
reasoning into the reasoning channel so every selection policy means the same
thing for both serving modes. ``combine`` renders inline reasoning back as one
well-formed span, so scorers that find the answer after the closing delimiter
keep that boundary.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

from taskcompendium.evaluation.spec import ResponseSelection

INLINE_REASONING_KEY = "inline_reasoning"
"""Raw-message key recording the delimiters of reasoning split from the reply text."""


class ReasoningDelimiters(NamedTuple):
    start: str
    end: str


@dataclass(frozen=True)
class InlineReasoning:
    """A reply's text split at its inline reasoning span.

    ``closed`` is false for unfinished reasoning, whose answer is empty.
    """

    reasoning: str
    content: str
    closed: bool


def split_inline_reasoning(
    text: str, delimiters: ReasoningDelimiters, *, prompt_opens_span: bool
) -> InlineReasoning | None:
    """Separate a reply's inline reasoning from its final answer, or return ``None`` when it has none.

    Everything through the last closing delimiter is reasoning, with or without an opening delimiter: a
    model may omit the opener, and a chat template may open the span at the end of the prompt. A reply
    that opens a span without closing it is unfinished reasoning, as is a reply with no delimiter at all
    when ``prompt_opens_span``.
    """
    head, closer, answer = text.rpartition(delimiters.end)
    if closer:
        return InlineReasoning(_without_delimiters(head, delimiters), answer, closed=True)
    if prompt_opens_span or delimiters.start in text:
        return InlineReasoning(_without_delimiters(text, delimiters), "", closed=False)
    return None


def _without_delimiters(text: str, delimiters: ReasoningDelimiters) -> str:
    return text.replace(delimiters.start, "").replace(delimiters.end, "")


def inline_reasoning_message(
    message: Mapping[str, Any], split: InlineReasoning, delimiters: ReasoningDelimiters
) -> dict[str, Any]:
    """Return ``message`` with its inline reasoning moved to ``reasoning_content``."""
    return {
        **message,
        "content": split.content,
        "reasoning_content": split.reasoning,
        INLINE_REASONING_KEY: {"start": delimiters.start, "end": delimiters.end, "closed": split.closed},
    }


def completion_text(message: Mapping[str, Any], stop_reason: str, selection: ResponseSelection) -> str:
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if selection == "final_only":
        return content
    if selection == "final_or_complete_boxed_reasoning":
        return content or (
            reasoning if stop_reason != "length" and re.search(r"\\(?:boxed|fbox)\s*\{", reasoning) else ""
        )
    if selection == "final_or_complete_reasoning":
        return content or (reasoning if stop_reason != "length" else "")
    if selection == "combine":
        if inline := message.get(INLINE_REASONING_KEY):
            closing = inline["end"] + content if inline["closed"] else ""
            return inline["start"] + reasoning + closing
        return f"{reasoning}\n\n{content}" if content and reasoning else reasoning or content
    raise ValueError(f"Unknown completion response selection: {selection}")
