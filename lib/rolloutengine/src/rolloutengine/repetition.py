# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared action and observation identity for repetition detection.

One module defines what "the same action" and "the same observation" mean, so the
runtime stall stop (``recorder``), the loop audit (``vendor/skyrl/scripts``), and the
offline counterfactual all agree. Actions are message-level: a tool-call batch, a
Terminus command batch, or a plain text reply. Terminus command batches reuse the
agent's real parsers, so identity cannot drift from what executed.
"""

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
"""A command containing one of these does more than wait, whatever its first word."""


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
    stall. The comparison needs at least ``batches`` completed pairs, so a short
    attempt never fires.
    """
    if batches < 2 or len(pairs) < batches:
        return False
    window = pairs[-batches:]
    if any(pair.wait for pair in window):
        return False
    first = window[0]
    return all(pair.action == first.action and pair.observation == first.observation for pair in window[1:])


def maximal_runs(identities: Sequence[str]) -> list[int]:
    """The length of each maximal run of equal consecutive identities."""
    runs: list[int] = []
    previous = None
    length = 0
    for identity in identities:
        if identity == previous:
            length += 1
        else:
            if length:
                runs.append(length)
            previous = identity
            length = 1
    if length:
        runs.append(length)
    return runs


def repeat_turns(identities: Sequence[str]) -> int:
    """Turns inside runs beyond each run's first copy: the sum of run lengths minus one per run."""
    return sum(length - 1 for length in maximal_runs(identities))


def _command_call_identities(message: Mapping[str, Any]) -> list[str]:
    """Terminus commands as (name, canonical arguments) calls, matching the tool-call form."""
    commands = parse_commands(message.get("content"))
    if commands is None:
        return []
    return [
        "terminus" + PART_SEPARATOR + canonical_arguments({"keystrokes": keystrokes, "duration": duration})
        for keystrokes, duration in zip(commands.keystrokes, commands.durations, strict=True)
    ]


def duplicate_tool_call_fraction(messages: Sequence[Mapping[str, Any]]) -> float | None:
    """The share of calls whose (name, canonical arguments) pair appeared earlier in the trial.

    A call is a tool call or, on Terminus turns, one command of the command batch. ``None``
    when the trial makes no calls, so a call-free trial never dilutes the cohort statistic.
    """
    seen: set[str] = set()
    total = 0
    duplicates = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for identity in (*_tool_call_identities(message), *_command_call_identities(message)):
            total += 1
            if identity in seen:
                duplicates += 1
            seen.add(identity)
    if not total:
        return None
    return duplicates / total
