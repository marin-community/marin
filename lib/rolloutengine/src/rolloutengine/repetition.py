# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Detect periodic response suffixes without depending on training reward policy."""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rolloutengine.agents.terminus.terminus_xml_plain_parser import TerminusXMLPlainParser

PART_SEPARATOR = "\x1e"
"""Joins the parts of one identity string; a character no action payload contains."""

XML_PARSER = TerminusXMLPlainParser()

_WAIT_COMMANDS = ("sleep", "ps", "wait", "jobs")
"""Shell commands that observe or wait without changing anything."""
_SHELL_OPERATORS = {"&&", "||", ";", "|", "&", ">", "<", ">>"}
"""Standalone shell tokens that disqualify an otherwise wait-only command."""


@dataclass(frozen=True)
class CommandBatch:
    """The Terminus commands parsed from one assistant message."""

    keystrokes: tuple[str, ...]
    durations: tuple[float, ...]

    @property
    def wait_only(self) -> bool:
        """Whether every command waits instead of acting: empty keystrokes poll the pane."""
        return all(keystrokes == "" for keystrokes in self.keystrokes)


def canonical_arguments(arguments: Any) -> str:
    """Serialize tool-call arguments so equal content compares equal regardless of key order."""
    if arguments is None:
        return ""
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except ValueError:
            return arguments.strip()
    else:
        parsed = arguments
    try:
        return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(parsed)


def normalize_text(text: str) -> str:
    """Collapse whitespace runs so actions that differ only in spacing compare equal."""
    return " ".join(text.split())


def parse_commands(content: Any) -> CommandBatch | None:
    """Parse Terminus commands from a message's text content, or return ``None``.

    The JSON form is read leniently straight from the recorded text, because the audit
    sees messages even when the agent's own parser rejected them; the XML form reuses
    the agent's parser, whose grammar is not worth a second implementation.
    """
    if not isinstance(content, str) or not content.strip():
        return None
    commands = _json_commands(content)
    if commands is not None:
        return commands
    result = XML_PARSER.parse_response(content)
    if result.commands:
        return CommandBatch(
            keystrokes=tuple(command.keystrokes for command in result.commands),
            durations=tuple(float(command.duration) for command in result.commands),
        )
    if not result.error:
        return CommandBatch(keystrokes=(), durations=())
    return None


def _json_commands(content: str) -> CommandBatch | None:
    try:
        parsed = json.loads(content)
    except ValueError:
        return None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("commands"), list):
        return None
    keystrokes = []
    durations = []
    for command in parsed["commands"]:
        if not isinstance(command, dict):
            return None
        value = command.get("keystrokes", "")
        if not isinstance(value, str):
            return None
        keystrokes.append(value)
        durations.append(float(command.get("duration") or 1.0))
    return CommandBatch(keystrokes=tuple(keystrokes), durations=tuple(durations))


def _tool_call_identities(message: Mapping[str, Any]) -> list[str]:
    identities = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, Mapping):
            continue
        function = call.get("function") if isinstance(call.get("function"), Mapping) else {}
        name = call.get("name") or function.get("name") or ""
        arguments = call.get("arguments", function.get("arguments"))
        identities.append(f"tool{PART_SEPARATOR}{name}{PART_SEPARATOR}{canonical_arguments(arguments)}")
    return identities


def action_identity(message: Mapping[str, Any]) -> str:
    """The canonical identity of one assistant action batch.

    A Terminus command batch is identified by its normalized keystrokes alone, so
    churn in the analysis or plan text and a changed wait duration do not make a
    new action. Tool calls are identified by name plus canonical JSON arguments.
    A message with neither is identified by its normalized text.
    """
    commands = parse_commands(message.get("content"))
    if commands is not None:
        keystrokes = (normalize_text(text) for text in commands.keystrokes)
        return f"terminus{PART_SEPARATOR}{PART_SEPARATOR.join(keystrokes)}"
    identities = _tool_call_identities(message)
    if identities:
        return PART_SEPARATOR.join(identities)
    content = message.get("content")
    text = content if isinstance(content, str) else json.dumps(content, sort_keys=True, default=str)
    return f"text{PART_SEPARATOR}{normalize_text(text or '')}"


def wait_only(message: Mapping[str, Any]) -> bool:
    """Whether an action batch only waits: empty Terminus keystrokes, a sleep, or a pure poll.

    The shell actions here are the tool-call arguments of the standard shell tool:
    a ``command`` argument that is a sleep or a process-state query acts on nothing.
    """
    commands = parse_commands(message.get("content"))
    if commands is not None:
        return commands.wait_only
    calls = _tool_call_identities(message)
    if not calls:
        return False
    return all(_is_wait_shell_call(call) for call in calls)


def _is_wait_shell_call(identity: str) -> bool:
    prefix = f"tool{PART_SEPARATOR}"
    if not identity.startswith(prefix):
        return False
    arguments = identity[len(prefix) :].partition(PART_SEPARATOR)[2]
    try:
        parsed = json.loads(arguments)
    except ValueError:
        return False
    command = parsed.get("command") if isinstance(parsed, Mapping) else None
    if not isinstance(command, str):
        return False
    words = command.split()
    if not words or words[0] not in _WAIT_COMMANDS:
        return False
    return not any(word in _SHELL_OPERATORS for word in words[1:])


def observation_identity(observations: Iterable[Mapping[str, Any]]) -> str:
    """The canonical identity of one observation batch: role and exact content of each message."""
    parts = []
    for message in observations:
        content = message.get("content")
        if not isinstance(content, str):
            content = json.dumps(content, sort_keys=True, separators=(",", ":"), default=str) if content else ""
        parts.append(f"{message.get('role', '')}{PART_SEPARATOR}{content}")
    return PART_SEPARATOR.join(parts)


@dataclass(frozen=True)
class ActionPair:
    """One completed (action batch, observation batch) turn of an attempt."""

    action: str
    observation: str
    wait: bool

    @classmethod
    def of(cls, message: Mapping[str, Any], observations: Sequence[Mapping[str, Any]]) -> "ActionPair":
        return cls(action_identity(message), observation_identity(observations), wait_only(message))


def detect_stall(pairs: Sequence[ActionPair], batches: int) -> bool:
    """Whether the last ``batches`` pairs are identical non-wait repeats of one action.

    Wait-only pairs never match: polling a long job or sleeping is legitimate, not a
    stall. The comparison requires ``batches >= 2`` and at least ``batches``
    completed pairs, so a short attempt never fires.
    """
    if batches < 2 or len(pairs) < batches:
        return False
    window = pairs[-batches:]
    if any(pair.wait for pair in window):
        return False
    first = window[0]
    return all(pair.action == first.action and pair.observation == first.observation for pair in window[1:])


def _active_segments(token_ids: Sequence[int], loss_mask: Sequence[int]) -> list[tuple[list[int], list[int]]]:
    if len(token_ids) != len(loss_mask):
        raise ValueError("response IDs and loss masks must have the same length")

    segments: list[tuple[list[int], list[int]]] = []
    segment_tokens: list[int] = []
    segment_positions: list[int] = []
    for position, (token_id, active) in enumerate(zip(token_ids, loss_mask, strict=True)):
        if active:
            segment_tokens.append(int(token_id))
            segment_positions.append(position)
        elif segment_tokens:
            segments.append((segment_tokens, segment_positions))
            segment_tokens = []
            segment_positions = []
    if segment_tokens:
        segments.append((segment_tokens, segment_positions))
    return segments


@dataclass(frozen=True)
class TailLoop:
    """Token coordinates for a periodic suffix within the last trainable segment."""

    start: int
    period: int
    end: int

    @property
    def repeat_start(self) -> int:
        return self.start + self.period


def detect_tail_loop(
    response_ids: Sequence[int],
    loss_mask: Sequence[int],
    *,
    max_period_tokens: int,
    tail_tokens: int,
    minimum_occurrences: int,
) -> TailLoop | None:
    """Find a repeating trainable suffix and return its original token coordinates."""
    segments = _active_segments(response_ids, loss_mask)
    if not segments:
        return None

    # Only a loop that survives to the final trainable suffix is charged. An
    # earlier loop followed by more assistant work is a recovery, not a failure.
    segment_tokens, segment_positions = segments[-1]
    tail = segment_tokens[-tail_tokens:]
    max_period = min(max_period_tokens, len(tail) // minimum_occurrences)
    period = next(
        (
            candidate
            for candidate in range(1, max_period + 1)
            if all(
                tail[-offset] == tail[-offset - candidate]
                for offset in range(1, candidate * (minimum_occurrences - 1) + 1)
            )
        ),
        None,
    )
    if period is None:
        return None

    periodic_start = len(segment_tokens) - period * minimum_occurrences
    while periodic_start > 0 and segment_tokens[periodic_start - 1] == segment_tokens[periodic_start - 1 + period]:
        periodic_start -= 1
    return TailLoop(segment_positions[periodic_start], period, segment_positions[-1] + 1)
