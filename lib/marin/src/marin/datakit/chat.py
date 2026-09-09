# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit chat sources and conversion of source turns to Harmony messages."""

import json
import re
from dataclasses import dataclass

from openai_harmony import Author, Message, Role

from marin.execution.step_spec import StepSpec

_REASONING_PREFIX = re.compile(r"\A<\|start_think\|>(.*?)<\|end_think\|>(.*)\Z", re.DOTALL)


@dataclass(frozen=True)
class DatakitChatSource:
    """A source whose normalized artifact contains structured Harmony messages."""

    name: str
    normalize_steps: tuple[StepSpec, ...]
    rough_token_count_b: float

    @property
    def normalized(self) -> StepSpec:
        return self.normalize_steps[-1]


def to_harmony_messages(messages: list[dict]) -> list[dict]:
    """Convert validated source turns to Harmony's text-message representation.

    Source tool IDs resolve observations before conversion. Parallel calls are
    emitted in source order, followed by observations in that same call order,
    so repeated calls to the same function retain their association.
    """
    output: list[Message] = []
    pending_calls: list[tuple[str, str]] = []
    observations: dict[str, str] = {}
    for message in messages:
        role = Role(message["role"])
        content = message.get("content") or ""
        author = Author.new(role, message.get("name"))
        if role == Role.TOOL:
            observations[message["tool_call_id"]] = content
            if len(observations) == len(pending_calls):
                for call_id, name in pending_calls:
                    output.append(
                        Message.from_author_and_content(
                            Author.new(Role.TOOL, f"functions.{name}"), observations[call_id]
                        )
                        .with_channel("commentary")
                        .with_recipient("assistant")
                    )
                pending_calls = []
                observations = {}
            continue
        if role != Role.ASSISTANT:
            output.append(Message.from_author_and_content(author, content))
            continue

        reasoning = message.get("reasoning_content") or ""
        match = _REASONING_PREFIX.fullmatch(content)
        if match is not None:
            if reasoning:
                raise ValueError("Assistant reasoning is present in both content and reasoning_content")
            reasoning, content = match.groups()
        if reasoning.strip():
            output.append(Message.from_author_and_content(author, reasoning.strip()).with_channel("analysis"))
        calls = message.get("tool_calls") or []
        if content.strip():
            output.append(
                Message.from_author_and_content(author, content.strip()).with_channel("commentary" if calls else "final")
            )
        for call in calls:
            function = call["function"]
            arguments = json.loads(function["arguments"])
            output.append(
                Message.from_author_and_content(
                    author, json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                )
                .with_channel("commentary")
                .with_recipient(f"functions.{function['name']}")
            )
            pending_calls.append((call["id"], function["name"]))
    if observations:
        raise ValueError("A parallel tool-call batch is missing observations")
    return [message.to_dict() for message in output]
