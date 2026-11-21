# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Source: Forge marin-forge-passes, lib/forge/src/forge/harness/tools/solve_review.py (behaviour and tool text).

"""The solve-review's evidence: one index over recorded solver trials, their grades and the task's files.

Each tool is an ordered, bounded view of the index that emits no effect, so a replayed review makes no call, and no
reply exceeds ``output_chars``, so nothing spills to a workspace the machine-less reviewer does not have. Citation
checks read the same index: a citation is valid when it quotes text the reviewer could have read, opened or not.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import override

from taskforge.effects.context import Context
from taskforge.harness.protocol import ReturnRequest, Tool, ToolResult
from taskforge.harness.tools.base import ValidatedTool
from taskforge.model.content_hash import digest
from taskforge.model.review import Citation, ReviewEnd, ReviewFinding, SolveReview, TrialAccount

PAGE_ROOM = 200
"""Characters each reply keeps free for its headers and its continuation line."""
SEARCH_CONTEXT = 120


@dataclass(frozen=True, slots=True)
class Source:
    """One readable unit by its stable name: a trial's rendered messages and then its grade, or a task file's lines."""

    name: str
    summary: str
    parts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Evidence:
    """Every source in reading order; ``unpassed`` names the trials a report must account for."""

    sources: Mapping[str, Source]
    unpassed: frozenset[int]
    output_chars: int

    @property
    def version(self) -> str:
        return digest(tuple(self.sources.values()))


def normalized(text: str) -> str:
    return " ".join(text.split())


def bounded(lines: Sequence[str], chars: int, more: str) -> str:
    """``lines`` in order while they fit in ``chars``, then ``more`` with the count of lines left out."""
    out, budget = [], chars - PAGE_ROOM
    for index, line in enumerate(lines):
        if len(line) + 1 > budget:
            return "\n".join((*out, more.format(len(lines) - index)))
        out.append(line)
        budget -= len(line) + 1
    return "\n".join(out)


@dataclass(frozen=True, slots=True)
class NoArgs:
    pass


@dataclass(frozen=True, slots=True)
class ListTool(ValidatedTool[NoArgs]):
    evidence: Evidence
    name = "list_evidence"
    description = "List every evidence source (each solver trial, then each task file) with its size and summary."
    args_type = NoArgs

    @override
    async def run(self, ctx: Context, args: NoArgs) -> ToolResult:
        lines = [source.summary for source in self.evidence.sources.values()]
        return ToolResult(bounded(lines, self.evidence.output_chars, "[{} more sources; search narrows them]"))


@dataclass(frozen=True, slots=True)
class ReadArgs:
    source: str
    start: int = 0
    offset: int = 0


@dataclass(frozen=True, slots=True)
class ReadTool(ValidatedTool[ReadArgs]):
    evidence: Evidence
    name = "read_evidence"
    description = (
        "Read one source from part `start` (a trial's message, or a file's line; a trial's last part is the grader's "
        "complete output), `offset` characters into that part. A page is bounded and names where the next one starts."
    )
    args_type = ReadArgs

    @override
    async def run(self, ctx: Context, args: ReadArgs) -> ToolResult:
        source = self.evidence.sources.get(args.source)
        if source is None:
            return ToolResult(f"No source {args.source!r}; sources: {list(self.evidence.sources)}")
        if not 0 <= args.start < len(source.parts) or args.offset < 0:
            return ToolResult(f"need 0 <= start < {len(source.parts)} and offset >= 0")
        out, budget = [], self.evidence.output_chars - PAGE_ROOM
        for index in range(args.start, len(source.parts)):
            skip = args.offset if index == args.start else 0
            header = f"--- {source.name} #{index}" + (f" from character {skip}" if skip else "") + " ---\n"
            text = source.parts[index][skip:]
            if len(header) + len(text) > budget:
                kept = budget - len(header)
                out += [header + text[:kept], f"[page ends: read start={index} offset={skip + kept} next]"]
                break
            out.append(header + text)
            budget -= len(header) + len(text)
        else:
            out.append(f"[end of {source.name}]")
        return ToolResult("\n".join(out))


@dataclass(frozen=True, slots=True)
class SearchArgs:
    pattern: str
    sources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SearchTool(ValidatedTool[SearchArgs]):
    evidence: Evidence
    name = "search_evidence"
    description = (
        "Search the listed sources, or all of them, with a case-insensitive Python regular expression; each match "
        "names its source and part, with surrounding text (whitespace collapsed)."
    )
    args_type = SearchArgs

    @override
    async def run(self, ctx: Context, args: SearchArgs) -> ToolResult:
        try:
            pattern = re.compile(args.pattern, re.IGNORECASE)
        except re.error as error:
            return ToolResult(f"Invalid pattern: {error}")
        if unknown := [name for name in args.sources if name not in self.evidence.sources]:
            return ToolResult(f"No sources {unknown}; sources: {list(self.evidence.sources)}")
        lines = [
            f"{source.name} #{index}: {normalized(text[max(0, m.start() - SEARCH_CONTEXT) : m.end() + SEARCH_CONTEXT])}"
            for source in self.evidence.sources.values()
            if not args.sources or source.name in args.sources
            for index, text in enumerate(source.parts)
            for m in pattern.finditer(text)
        ]
        more = "[{} more matches; narrow the pattern or the sources]"
        return ToolResult(bounded(lines, self.evidence.output_chars, more) or "No matches")


def citation_problem(evidence: Evidence, citation: Citation) -> str | None:
    source = evidence.sources.get(citation.source)
    if source is None:
        return f"cites {citation.source!r}, which is not in the evidence"
    if not 0 <= citation.start <= citation.end < len(source.parts):
        return f"cites {citation.source} #{citation.start}..{citation.end}, outside its parts 0..{len(source.parts) - 1}"
    quote = normalized(citation.quote)
    if not quote or quote not in normalized("\n".join(source.parts[citation.start : citation.end + 1])):
        return f"its quote does not occur verbatim in {citation.source} #{citation.start}..{citation.end}"
    return None


def checked(
    evidence: Evidence, findings: Sequence[ReviewFinding]
) -> tuple[list[ReviewFinding], list[tuple[ReviewFinding, str]]]:
    kept, dropped = [], []
    for finding in findings:
        problems = [problem for c in finding.citations if (problem := citation_problem(evidence, c)) is not None]
        if problems:
            dropped.append((finding, "; ".join(problems)))
        else:
            kept.append(finding)
    return kept, dropped


def report(
    evidence: Evidence,
    findings: Sequence[ReviewFinding],
    accounts: Sequence[TrialAccount],
) -> ToolResult | ReturnRequest[SolveReview]:
    """The review, once its accounts cover every trial that did not pass; a finding whose citations fail is dropped
    with its reason, never trusted and never refused back to the reviewer."""
    accounted = [account.trial for account in accounts]
    trials = {source.name for source in evidence.sources.values()}
    problems = [f"trial {trial} is not a solver trial" for trial in accounted if f"trial-{trial}" not in trials]
    problems += [f"account for trial {trial} once" for trial in set(accounted) if accounted.count(trial) > 1]
    problems += [f"trial {trial} did not pass and needs an account" for trial in evidence.unpassed - set(accounted)]
    if problems:
        return ToolResult("Invalid report: " + "; ".join(problems) + ". Fix these and call report again.")
    kept, dropped = checked(evidence, findings)
    review = SolveReview(evidence.version, ReviewEnd.REPORTED, tuple(kept), tuple(dropped), tuple(accounts))
    return ReturnRequest("report", review)


@dataclass(frozen=True, slots=True)
class ReportArgs:
    findings: tuple[ReviewFinding, ...]
    accounts: tuple[TrialAccount, ...]


@dataclass(frozen=True, slots=True)
class ReportTool(ValidatedTool[ReportArgs]):
    evidence: Evidence
    name = "report"
    description = (
        "Finish the review. `findings` is empty when the trials show no task defect. Each finding has `kind` "
        "(ambiguous_prompt, grader_defect, answer_leak, environment_problem), `description` (what is wrong), `edit` "
        "(the concrete change to the task that removes it) and `citations`: one or more {source, start, end, quote}, "
        "where quote is text copied verbatim from parts start..end of that source. `accounts` gives every trial that "
        "did not pass one {trial, attribution, reason}: solver (the outcome reflects the solver's own work), defect (a "
        "finding you report produced it) or uninformative (it ended for reasons that say nothing about the task)."
    )
    args_type = ReportArgs

    @override
    async def run(self, ctx: Context, args: ReportArgs) -> ToolResult | ReturnRequest[SolveReview]:
        return report(self.evidence, args.findings, args.accounts)


def evidence_tools(evidence: Evidence, report_tool: type[ReportTool]) -> Mapping[str, Tool[object]]:
    tools = (ListTool(evidence), ReadTool(evidence), SearchTool(evidence), report_tool(evidence))
    return {tool.schema.name: tool for tool in tools}
