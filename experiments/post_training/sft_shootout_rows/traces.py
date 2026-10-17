# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn #207 canonical traces (rebuild_traces.py rows) into v0 chat-template tokens with per-author loss masks.

A trace row is one Harbor trial attempt: messages in order, every assistant turn tagged with its author (glm or v0)
and relay trigger, reasoning restored from the proxy's full reply as reasoning_content. Rendering uses v0's chat
template; reasoning goes into <|start_think|>...<|end_think|>. The template's generation blocks give one contiguous
span per assistant turn; a turn is trained only if its author is the teacher (glm). v0's turns stay in the sequence
as context with zero loss, which is what the relay arm needs; plain and RS rows have glm turns only.
"""

from dataclasses import dataclass

TEACHER = "glm"
STUDENT = "v0"
TEMPLATE_KEYS = ("role", "content", "reasoning_content", "tool_calls", "tool_call_id", "name")


@dataclass(frozen=True)
class RenderedTrace:
    ids: list[int]
    mask: list[int]
    authors: list[str]

    @property
    def supervised(self) -> int:
        return sum(self.mask[1:])


def template_messages(messages: list[dict]) -> tuple[list[dict], list[str]]:
    """Strip trace-only fields; return template messages and the author of each assistant turn, in order."""
    converted, authors = [], []
    for message in messages:
        if message["role"] == "assistant":
            author = message.get("author")
            if author not in (TEACHER, STUDENT):
                raise ValueError(f"Assistant turn without a known author: {author!r}")
            authors.append(author)
        converted.append({key: message[key] for key in TEMPLATE_KEYS if message.get(key) is not None})
    return converted, authors


def assistant_spans(mask: list[int]) -> list[tuple[int, int]]:
    """Contiguous runs of 1 in the template's assistant mask, as [start, end) pairs."""
    spans, start = [], None
    for index, value in enumerate([*mask, 0]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            spans.append((start, index))
            start = None
    return spans


def render_trace(tokenizer, messages: list[dict], tools: list[dict] | None) -> RenderedTrace:
    """Render with v0's template; train teacher turns only. Raises if spans and assistant turns do not align."""
    converted, authors = template_messages(messages)
    encoded = tokenizer.apply_chat_template(
        converted,
        tools=tools or None,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        enable_thinking=True,
    )
    ids, mask = list(encoded["input_ids"]), list(encoded["assistant_masks"])
    spans = assistant_spans(mask)
    if len(spans) != len(authors):
        raise ValueError(f"{len(spans)} assistant spans for {len(authors)} assistant turns")
    for (start, end), author in zip(spans, authors, strict=True):
        if author != TEACHER:
            mask[start:end] = [0] * (end - start)
    return RenderedTrace(ids=ids, mask=mask, authors=authors)
