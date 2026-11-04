# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert pi coding-agent transcripts into one-turn reasoning-markup tasks.

A pi transcript (``pi.txt``) is a JSON-lines event log. Its ``message_end`` events carry the
messages the model saw: the structured system prompt with the tools pi advertised, the user task,
the assistant replies with their raw reasoning markers and function calls, and the tool results.
Every assistant reply is a candidate: the conversation before it is a prompt, pi's tools are the
final tools, and the model answers with one assistant turn. The reply pi recorded is classified with
the reasoning-span rules, so a miner can keep the prompts that preceded a malformed span (hazards)
and matched prompts that preceded a well-formed one (controls).

Mining produces one row per kept prompt (``prefix_row``); the curation pipeline converts a row into a
task (``prefix_row_task``). Messages after pi's first context compaction are dropped, because the
compacted prompt the model saw is not in the transcript.

A task is graded by the ``reasoning_markup`` mode and also carries the ``reject`` markup policy. The
SkyRL worker strips reasoning from the text of a reply without function calls, so the policy's
token-level check covers that runtime while the grader covers runtimes that keep the markers.
"""

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from verifyit.json_objects import unique_object
from verifyit.reasoning_markup import ReasoningViolation, reasoning_span_violation
from verifyit.spec import ReasoningMarkupSpec

from taskcompendium.chat import chat_input
from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    ConversationEvent,
    ConversationToolCall,
    EnvironmentRequirements,
    FinalAction,
    FunctionDefinition,
    ReasoningMarkupPolicy,
    TaskSpec,
    TextMessage,
    ToolResult,
)
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow
from taskcompendium.submission import conversation_messages

MESSAGE_END = "message_end"
COMPACTION_START = "compaction_start"
SYSTEM_SECTION_SEPARATOR = "\n\n"
LENGTH_STOP_REASON = "length"
COMPLETED_STOP_REASON = "stop"
HAZARD_SPLIT = "hazard"
CONTROL_SPLIT = "control"
WELL_FORMED = "well_formed"


class UnsupportedTranscript(ValueError):
    """The transcript holds content these tasks cannot present to a model as text."""


@dataclass(frozen=True)
class PiReply:
    """One assistant reply pi recorded, located in the transcript's events."""

    position: int
    """Index of the reply in the transcript's events; the events before it are its prompt."""
    text: str
    """The reply text with its reasoning markers, before any function call."""
    stop_reason: str
    prompt_tokens: int | None
    """Prompt length pi's provider reported for this reply, when it reported usage."""
    violation: ReasoningViolation | None


@dataclass(frozen=True)
class PiTranscript:
    tools: tuple[FunctionDefinition, ...]
    events: tuple[ConversationEvent, ...]
    """Messages up to the first compaction, in model-visible order."""
    replies: tuple[PiReply, ...]
    compacted: bool
    """Whether events after a compaction were dropped."""
    truncated: bool
    """Whether the log ended in a record that was not complete JSON, such as a cut-off upload."""


def _text(parts: Any) -> str:
    if isinstance(parts, str):
        return parts
    if not isinstance(parts, list):
        raise UnsupportedTranscript("Message content must be text or content parts")
    unsupported = sorted({part.get("type") for part in parts if part.get("type") != "text"})
    if unsupported:
        raise UnsupportedTranscript(f"Unsupported message content: {', '.join(map(str, unsupported))}")
    return "".join(part["text"] for part in parts)


def _system_text(message: dict[str, Any]) -> str:
    sections = message.get("sections") or {}
    if sections:
        return SYSTEM_SECTION_SEPARATOR.join(sections.values())
    return _text(message["content"])


def _tool_definition(tool: dict[str, Any]) -> FunctionDefinition:
    return FunctionDefinition(name=tool["name"], description=tool.get("description"), parameters=tool["parameters"])


def _assistant_event(message: dict[str, Any]) -> tuple[ConversationEvent, str]:
    parts = message["content"]
    if not isinstance(parts, list):
        raise UnsupportedTranscript("Assistant content must be content parts")
    unsupported = sorted({part["type"] for part in parts if part["type"] not in ("text", "toolCall")})
    if unsupported:
        raise UnsupportedTranscript(f"Unsupported assistant content: {', '.join(unsupported)}")
    text = "".join(part["text"] for part in parts if part["type"] == "text")
    calls = tuple(
        ConversationToolCall(call_id=part["id"], name=part["name"], arguments=part["arguments"])
        for part in parts
        if part["type"] == "toolCall"
    )
    if not calls:
        return TextMessage(role="assistant", content=text), text
    return AssistantToolCalls(calls=calls, content=text or None), text


def _prompt_tokens(message: dict[str, Any]) -> int | None:
    usage = message.get("usage")
    if usage is None:
        return None
    return int(usage["input"]) + int(usage.get("cacheRead") or 0)


def parse_pi_transcript(lines: Iterable[str]) -> PiTranscript:
    """Read a pi event log into model-visible events and the replies it recorded.

    Messages before a compaction or a malformed record are kept; the rest of the log is dropped.
    Raises ``UnsupportedTranscript`` for content that is not text, such as images or provider
    reasoning parts.
    """
    tools: tuple[FunctionDefinition, ...] = ()
    events: list[ConversationEvent] = []
    replies: list[PiReply] = []
    compacted = False
    truncated = False
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line, object_pairs_hook=unique_object)
        except json.JSONDecodeError:
            truncated = True
            break
        kind = record.get("type")
        if kind == COMPACTION_START:
            compacted = True
            break
        if kind != MESSAGE_END:
            continue
        message = record["message"]
        role = message["role"]
        if role == "system":
            tools = tuple(_tool_definition(tool) for tool in message.get("toolsAdded") or ())
            events.append(TextMessage(role="system", content=_system_text(message)))
        elif role == "user":
            events.append(TextMessage(role="user", content=_text(message["content"])))
        elif role == "assistant":
            event, text = _assistant_event(message)
            stop_reason = LENGTH_STOP_REASON if message.get("stopReason") == LENGTH_STOP_REASON else COMPLETED_STOP_REASON
            replies.append(
                PiReply(
                    position=len(events),
                    text=text,
                    stop_reason=stop_reason,
                    prompt_tokens=_prompt_tokens(message),
                    violation=reasoning_span_violation(text, stop_reason),
                )
            )
            events.append(event)
        elif role == "toolResult":
            events.append(ToolResult(call_id=message["toolCallId"], content=_text(message["content"])))
        else:
            raise UnsupportedTranscript(f"Unsupported message role: {role}")
    return PiTranscript(
        tools=tools, events=tuple(events), replies=tuple(replies), compacted=compacted, truncated=truncated
    )


def prompt_messages(transcript: PiTranscript, reply: PiReply) -> list[dict[str, Any]]:
    """The chat messages a model receives for ``reply``'s prompt, before the chat template."""
    return conversation_messages(transcript.events[: reply.position])


def _tool_payload(tools: tuple[FunctionDefinition, ...]) -> list[dict[str, Any]]:
    return [tool.model_dump(exclude_none=True) for tool in tools]


def prefix_digest(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    """Identify a prompt by its messages and tools, so identical prompts from different runs collapse."""
    payload = {"messages": messages, "tools": tools}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prefix_row(
    transcript: PiTranscript,
    reply: PiReply,
    *,
    split: str,
    benchmark: str,
    model: str,
    run: str,
    trial: str,
) -> dict[str, Any]:
    """One source row for the prompt before ``reply``: its messages and tools, and where it came from.

    ``messages`` and ``tools`` are JSON text so rows with different tool sets share one Parquet schema.
    """
    if reply.position == 0 or not transcript.tools:
        raise ValueError("A prefix row needs a prompt and pi's advertised tools")
    messages = prompt_messages(transcript, reply)
    tools = _tool_payload(transcript.tools)
    return {
        "digest": prefix_digest(messages, tools),
        "messages": json.dumps(messages, ensure_ascii=False),
        "tools": json.dumps(tools, ensure_ascii=False),
        "split": split,
        "violation": reply.violation.value if reply.violation is not None else None,
        "prompt_tokens": reply.prompt_tokens,
        "benchmark": benchmark,
        "model": model,
        "run": run,
        "trial": trial,
        "position": reply.position,
        "reply": json.dumps(conversation_messages((transcript.events[reply.position],))[0], ensure_ascii=False),
    }


def prefix_row_task(row: RawRow, _context: ConversionContext) -> TaskSpec:
    """The one-turn task for a mined prefix row, graded on the reasoning markup of its reply."""
    data = row.data
    messages = json.loads(data["messages"], object_pairs_hook=unique_object)
    tools = json.loads(data["tools"], object_pairs_hook=unique_object)
    return TaskSpec(
        id=row.id,
        context=chat_input(messages),
        environment_requirements=EnvironmentRequirements(),
        final_tools=tuple(FunctionDefinition.model_validate(tool) for tool in tools),
        answer_type=AnswerType.NATIVE_ACTION,
        answer_format=FinalAction(),
        grader=verifyit_package(ReasoningMarkupSpec()).grader,
        reasoning_markup=ReasoningMarkupPolicy.REJECT,
        source=row.source,
        tags=(data["split"], data["benchmark"], data["model"], data["violation"] or WELL_FORMED),
    )
