# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-memory navigation over solver transcripts for the rollout reviewer, and its ``suggest`` exit.

Transcripts are too large to dump (k trials of up to a model window each), so every tool returns a bounded slice.
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import override

from taskcompendium.models import TaskSpec
from taskcompendium.runtime.resources import resource_bytes

from forge.effects.context import Context
from forge.harness.protocol import ReturnRequest, Tool, ToolResult
from forge.harness.tools.base import ValidatedTool
from forge.model.review import RevisionKind, Suggestion, TrialExcerpt

GREP_CONTEXT = 120
GREP_MATCHES = 40


@dataclass(frozen=True, slots=True)
class ReviewedTrial:
    """One solver trial: a one-line JSON summary of its grade, and its transcript as one string per message."""

    summary: str
    messages: tuple[str, ...]


def render_message(message: Mapping[str, object]) -> str:
    role = message.get("role", "?")
    parts = [f"{role}:"]
    for key in ("reasoning_content", "reasoning"):
        if isinstance(reasoning := message.get(key), str) and reasoning:
            parts.append(f"[reasoning] {reasoning}")
    content = message.get("content")
    if content:
        parts.append(content if isinstance(content, str) else json.dumps(content, default=str))
    calls = message.get("tool_calls")
    for call in calls if isinstance(calls, list) else ():
        function = call.get("function", call) if isinstance(call, Mapping) else call
        parts.append(f"[tool call] {json.dumps(function, default=str)}")
    return "\n".join(parts)


def normalized(text: str) -> str:
    return " ".join(text.split())


@dataclass(frozen=True, slots=True)
class NoArgs:
    pass


@dataclass(frozen=True, slots=True)
class ListTrialsTool(ValidatedTool[NoArgs]):
    trials: Sequence[ReviewedTrial]
    name = "list_trials"
    description = "List every solver trial with its grade, cause, message count and transcript length in characters."
    args_type = NoArgs

    @override
    async def run(self, ctx: Context, args: NoArgs) -> ToolResult:
        return ToolResult(
            "\n".join(
                f"trial {index}: {trial.summary} messages={len(trial.messages)} chars={sum(map(len, trial.messages))}"
                for index, trial in enumerate(self.trials)
            )
        )


@dataclass(frozen=True, slots=True)
class ReadTrialArgs:
    trial: int
    start: int = 0
    end: int | None = None
    offset: int = 0


@dataclass(frozen=True, slots=True)
class ReadTrialTool(ValidatedTool[ReadTrialArgs]):
    trials: Sequence[ReviewedTrial]
    output_chars: int
    name = "read_trial"
    description = (
        "Read messages start..end (exclusive) of one trial's transcript; offset skips characters of the first "
        "message. Output is bounded; a cut reply names where to continue."
    )
    args_type = ReadTrialArgs

    @override
    async def run(self, ctx: Context, args: ReadTrialArgs) -> ToolResult:
        if not 0 <= args.trial < len(self.trials):
            return ToolResult(f"trial must be in 0..{len(self.trials) - 1}")
        messages = self.trials[args.trial].messages
        end = len(messages) if args.end is None else min(args.end, len(messages))
        if not 0 <= args.start <= end or args.offset < 0:
            return ToolResult(f"need 0 <= start <= end <= {len(messages)} and offset >= 0")
        budget, out = self.output_chars - 200, []
        for index in range(args.start, end):
            skip = args.offset if index == args.start else 0
            text = messages[index][skip:]
            header = f"--- message {index}" + (f" from character {skip}" if skip else "") + " ---\n"
            if len(header) + len(text) > budget:
                kept = max(0, budget - len(header))
                if kept:
                    out.append(header + text[:kept])
                out.append(f"[cut: continue with start={index}, offset={skip + kept}]")
                break
            out.append(header + text)
            budget -= len(header) + len(text)
        return ToolResult("\n".join(out) or "No messages in that range")


@dataclass(frozen=True, slots=True)
class GrepArgs:
    pattern: str
    trials: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class GrepTrialsTool(ValidatedTool[GrepArgs]):
    trials: Sequence[ReviewedTrial]
    name = "grep_trials"
    description = (
        "Search transcripts with a Python regular expression (case-insensitive), in the listed trials or all of "
        f"them. Returns up to {GREP_MATCHES} matches with trial, message index and surrounding text."
    )
    args_type = GrepArgs

    @override
    async def run(self, ctx: Context, args: GrepArgs) -> ToolResult:
        try:
            pattern = re.compile(args.pattern, re.IGNORECASE)
        except re.error as error:
            return ToolResult(f"Invalid pattern: {error}")
        selected = args.trials or tuple(range(len(self.trials)))
        if any(not 0 <= trial < len(self.trials) for trial in selected):
            return ToolResult(f"trials must be in 0..{len(self.trials) - 1}")
        lines: list[str] = []
        total = 0
        for trial in selected:
            for index, text in enumerate(self.trials[trial].messages):
                for match in pattern.finditer(text):
                    total += 1
                    if len(lines) < GREP_MATCHES:
                        start, end = max(0, match.start() - GREP_CONTEXT), match.end() + GREP_CONTEXT
                        lines.append(f"trial {trial} message {index}: {text[start:end]!r}")
        if total > len(lines):
            lines.append(f"[{total - len(lines)} more matches; narrow the pattern or the trials]")
        return ToolResult("\n".join(lines) or "No matches")


@dataclass(frozen=True, slots=True)
class ReadTaskFileArgs:
    path: str
    offset: int = 0


@dataclass(frozen=True, slots=True)
class ReadTaskFileTool(ValidatedTool[ReadTaskFileArgs]):
    task: TaskSpec
    output_chars: int
    name = "read_task_file"
    description = "Read a task resource (shared, worker or verifier file) by its manifest path, from offset characters."
    args_type = ReadTaskFileArgs

    @override
    async def run(self, ctx: Context, args: ReadTaskFileArgs) -> ToolResult:
        resources = self.task.resources
        files = {file.path: file for file in (*resources.all, *resources.worker, *resources.verifier)}
        if args.path not in files:
            return ToolResult(f"No task file {args.path!r}; files: {sorted(files)}")
        text = resource_bytes(files[args.path]).decode(errors="replace")[max(0, args.offset) :]
        limit = self.output_chars - 200
        cut = f"\n[cut: continue with offset={args.offset + limit}]" if len(text) > limit else ""
        return ToolResult(text[:limit] + cut)


@dataclass(frozen=True, slots=True)
class ExcerptArgs:
    trial: int
    excerpt: str


@dataclass(frozen=True, slots=True)
class RevisionArgs:
    kind: RevisionKind
    evidence: tuple[ExcerptArgs, ...]
    suggested_edit: str


@dataclass(frozen=True, slots=True)
class SuggestArgs:
    revisions: tuple[RevisionArgs, ...]


@dataclass(frozen=True, slots=True)
class Suggestions:
    revisions: tuple[Suggestion, ...]


@dataclass(frozen=True, slots=True)
class SuggestTool(ValidatedTool[SuggestArgs]):
    trials: Sequence[ReviewedTrial]
    name = "suggest"
    description = (
        "Finish the review with revisions for the author, or an empty list when the rollouts show nothing to fix. "
        "Each revision has a kind, evidence (trial index plus a verbatim excerpt of that trial's transcript) and a "
        "concrete suggested_edit to the task, controls or grader."
    )
    args_type = SuggestArgs

    @override
    async def run(self, ctx: Context, args: SuggestArgs) -> ToolResult | ReturnRequest[Suggestions]:
        revisions: list[Suggestion] = []
        for number, revision in enumerate(args.revisions):
            if not revision.evidence or not revision.suggested_edit.strip():
                return ToolResult(f"Invalid suggest: revision {number} needs evidence and a suggested_edit")
            for item in revision.evidence:
                if not 0 <= item.trial < len(self.trials):
                    return ToolResult(f"Invalid suggest: revision {number} cites trial {item.trial}, which is absent")
                quoted = normalized(item.excerpt)
                transcript = normalized("\n".join(self.trials[item.trial].messages))
                if not quoted or quoted not in transcript:
                    return ToolResult(
                        f"Invalid suggest: revision {number}'s excerpt does not occur verbatim in trial {item.trial}; "
                        "copy it from read_trial or grep_trials output"
                    )
            evidence = tuple(TrialExcerpt(item.trial, item.excerpt) for item in revision.evidence)
            revisions.append(Suggestion(revision.kind, evidence, revision.suggested_edit))
        return ReturnRequest("suggest", Suggestions(tuple(revisions)))


def reviewer_tools(trials: Sequence[ReviewedTrial], task: TaskSpec, output_chars: int) -> Mapping[str, Tool[object]]:
    tools: tuple[Tool[object], ...] = (
        ListTrialsTool(trials),
        ReadTrialTool(trials, output_chars),
        GrepTrialsTool(trials),
        ReadTaskFileTool(task, output_chars),
        SuggestTool(trials),
    )
    return {tool.schema.name: tool for tool in tools}
