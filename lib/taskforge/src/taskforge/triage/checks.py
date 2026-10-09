# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic proposal checks: everything triage can decide without a model.

Each check reports PASS (the property holds), FAIL (it does not; a fact about the document, not an
opinion), or SKIP (the property does not apply to this proposal). SKIP never counts as PASS. A FATAL
failure rejects the proposal before any model call; ADVISORY failures are shown to the rubric.

Every check carries a comment recording the false positives and false negatives it is expected to
have, so a reader knows how far to trust a FAIL. Grounded proposals (``Grounding.GROUNDED``) run
the same checks; none yet verifies their citations against the cited repository.
"""

import itertools
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Protocol

from taskforge.proposal.model import (
    REQUIRED_HEADINGS,
    Environment,
    ProposalFormatError,
    TaskProposal,
    Verification,
    body_sections,
    parse,
    render,
)

GRADER_HEADING = "Grader design and controls"
BUILD_HEADING = "Build plan"
MIN_SECTION_WORDS = 40
MIN_PURPOSE_WORDS = 5
MIN_NULL_REASON_WORDS = 5
# The template's section placeholder; angle-bracket answer formats such as "<yes|no>" are legitimate.
SECTION_PLACEHOLDER = re.compile(r"<\s*(?:\.\.\.|\u2026)\s*>")
GLOB_CHARACTERS = "*?["

CODE_FILE = re.compile(r"\.(?:py|sh|js|ts|rs|go|rb|jl)\b")
# Evidence that a grader runs code: a code file, or naming an executable check or what it does.
EXECUTABLE_CHECK = re.compile(
    CODE_FILE.pattern + r"|\bpytest\b|\bunit tests?\b|\bscripts?\b|\bexecutable\b|\bprogram(?:matic|matically)?\b"
    r"|\bchecker\b|\bevaluator\b|\bpars(?:e|es|ed|ing|er)\b|\brecomput",
    re.IGNORECASE,
)
# Evidence of a judged rubric.
RUBRIC = re.compile(r"\brubric\b|\bjudge[ds]?\b|\bcriteri(?:on|a)\b", re.IGNORECASE)
# Evidence of an exact, numeric, or choice match.
SIMPLE_MATCH = re.compile(
    r"\bexact\b|\bmatch(?:es|ing)?\b|\btolerance\b|\bnumeric(?:al)?\b|\bnormali[sz]", re.IGNORECASE
)

ALL_COMBINATIONS = frozenset(itertools.product(Environment, Verification))


class Severity(StrEnum):
    FATAL = "fatal"
    ADVISORY = "advisory"


class CheckStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True)
class CheckResult:
    name: str
    severity: Severity
    status: CheckStatus
    reason: str

    @property
    def blocks(self) -> bool:
        """Whether this result rejects the proposal without a model call."""
        return self.status is CheckStatus.FAIL and self.severity is Severity.FATAL


@dataclass(frozen=True)
class CheckContext:
    """What checks need beyond the proposal.

    Attributes:
        allowed_combinations: Environment x verification pairings the downstream stages accept.
    """

    allowed_combinations: frozenset[tuple[Environment, Verification]]


class Check(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def severity(self) -> Severity: ...

    def run(self, p: TaskProposal, ctx: CheckContext) -> CheckResult: ...


Outcome = tuple[CheckStatus, str]


@dataclass(frozen=True)
class RuleCheck:
    """A ``Check`` whose logic is a function returning a status and its reason."""

    name: str
    severity: Severity
    rule: Callable[[TaskProposal, CheckContext], Outcome]

    def run(self, p: TaskProposal, ctx: CheckContext) -> CheckResult:
        status, reason = self.rule(p, ctx)
        return CheckResult(name=self.name, severity=self.severity, status=status, reason=reason)


def is_null(p: TaskProposal) -> bool:
    return p.header.null_reason is not None


def word_count(text: str) -> int:
    return len(text.split())


def is_template_placeholder(value: str) -> bool:
    """Whether a header value is still the template's "<...>" slot rather than content."""
    return value.startswith("<") and value.endswith(">")


# Expected false positives: none for documents produced by ``parse``; this fires on proposals built
# in code (or edited after parsing) whose canonical rendering no longer parses or no longer means the
# same thing, and on resource paths that escape the build root. Expected false negatives: anything
# the strict parser accepts, e.g. a well-formed id that names the wrong slot.
def header_schema_valid(p: TaskProposal, ctx: CheckContext) -> Outcome:
    try:
        reparsed = parse(render(p))
    except ProposalFormatError as error:
        return CheckStatus.FAIL, f"canonical rendering does not parse: {error}"
    if reparsed.header != p.header:
        return CheckStatus.FAIL, "header changes when rendered and re-parsed"
    escaping = [r for r in p.header.resources if PurePosixPath(r).is_absolute() or ".." in PurePosixPath(r).parts]
    if escaping:
        return CheckStatus.FAIL, f"resources must be relative paths inside the build root: {escaping}"
    names = [item.name for item in p.header.build]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        return CheckStatus.FAIL, f"build names repeat: {duplicates}"
    return CheckStatus.PASS, "header renders and re-parses to the same value"


# Expected false positives: none; the allowed set is configuration, so a FAIL means the caller's
# stages cannot take this pairing. Expected false negatives: a pairing that is allowed in general but
# wrong for this task (a judge where an exact match suffices); that is the rubric's environment_fit
# axis. Null proposals SKIP: they build nothing, so their pairing is moot.
def combination_allowed(p: TaskProposal, ctx: CheckContext) -> Outcome:
    if is_null(p):
        return CheckStatus.SKIP, "null proposal"
    pair = (p.header.environment, p.header.verification)
    if pair not in ctx.allowed_combinations:
        return CheckStatus.FAIL, f"{pair[0]} x {pair[1]} is not an allowed combination"
    return CheckStatus.PASS, f"{pair[0]} x {pair[1]} is allowed"


# Keyword evidence in the grader section (and, for executable checks, a code file among the
# resources). Expected false positives: a section that describes mechanical checks in none of the
# EXECUTABLE_CHECK words, or a rubric only as "scored items". The vocabulary goes beyond
# file/script/program/pytest because code graders are often described as a "checker", a "parse"
# or a "recompute", and a code file among the resources counts as evidence too.
# Expected false negatives: the vocabulary is broad ("parse", "criteria"), so a judge-only design
# that parses the answer passes as code, and keywords cannot read negation ("no rubric is used").
# Advisory for that reason.
def grader_matches_verification(p: TaskProposal, ctx: CheckContext) -> Outcome:
    if is_null(p):
        return CheckStatus.SKIP, "null proposal"
    grader = body_sections(p.body).get(GRADER_HEADING)
    if grader is None:
        return CheckStatus.SKIP, f"no '## {GRADER_HEADING}' section; required_sections reports it"
    executable = EXECUTABLE_CHECK.search(grader) is not None or any(CODE_FILE.search(r) for r in p.header.resources)
    rubric = RUBRIC.search(grader) is not None
    verification = p.header.verification
    missing: list[str] = []
    if verification in (Verification.CODE, Verification.COMPOSITE) and not executable:
        missing.append("an executable check (a code file, script, program, or test)")
    if verification in (Verification.JUDGE, Verification.COMPOSITE) and not rubric:
        missing.append("a judge rubric or criteria")
    if verification is Verification.SIMPLE and SIMPLE_MATCH.search(grader) is None:
        missing.append("an exact, numeric, or choice match rule")
    if missing:
        return CheckStatus.FAIL, f"verification is {verification} but the grader section names no {' or '.join(missing)}"
    return CheckStatus.PASS, f"grader section is consistent with {verification}"


# Expected false positives: a section that is short because it is genuinely simple (a reasoning
# task's Build plan can be a few lines); MIN_SECTION_WORDS is set well below what GLM writes (its
# documents run 20-30 KB). A FAIL on a template placeholder such as "<...>" is never a false positive.
# Expected false negatives: long but empty prose. Null proposals SKIP: their body may be abbreviated.
def required_sections(p: TaskProposal, ctx: CheckContext) -> Outcome:
    if is_null(p):
        return CheckStatus.SKIP, "null proposal"
    sections = body_sections(p.body)
    problems: list[str] = []
    for heading in REQUIRED_HEADINGS:
        text = sections.get(heading)
        if text is None:
            problems.append(f"'## {heading}' missing")
        elif SECTION_PLACEHOLDER.fullmatch(text):
            problems.append(f"'## {heading}' is still the template placeholder {text!r}")
        elif word_count(text) < MIN_SECTION_WORDS:
            problems.append(f"'## {heading}' has {word_count(text)} words, fewer than {MIN_SECTION_WORDS}")
    if problems:
        return CheckStatus.FAIL, "; ".join(problems)
    return CheckStatus.PASS, f"all {len(REQUIRED_HEADINGS)} required sections present with content"


# Expected false positives: a terse but real purpose ("CODATA constants table") under
# MIN_PURPOSE_WORDS. Expected false negatives: a long purpose that is vague. A proposal with no
# research items SKIPs: needing no lookup is legitimate for synthetic tasks.
def research_has_purpose(p: TaskProposal, ctx: CheckContext) -> Outcome:
    if not p.header.research:
        return CheckStatus.SKIP, "no research items"
    weak = [
        f"research[{i}]"
        for i, item in enumerate(p.header.research)
        if is_template_placeholder(item.purpose) or word_count(item.purpose) < MIN_PURPOSE_WORDS
    ]
    if weak:
        return (
            CheckStatus.FAIL,
            f"{weak} lack a substantive purpose (placeholder or fewer than {MIN_PURPOSE_WORDS} words)",
        )
    return CheckStatus.PASS, f"all {len(p.header.research)} research items state a purpose"


def resource_mentions(resource: str) -> tuple[str, ...]:
    """Strings whose presence in the build plan counts as referencing ``resource``.

    A literal path counts by itself, by its file name, or by its directory written with a trailing
    slash (``build/gold/``, which also matches ``build/gold/*``); a glob counts
    by the directory before its first wildcard, with or without the leading directories.
    """
    if not any(c in resource for c in GLOB_CHARACTERS):
        parent = str(PurePosixPath(resource).parent)
        directory = () if parent == "." else (f"{parent}/",)
        return (resource, PurePosixPath(resource).name, *directory)
    prefix = resource[: min(resource.index(c) for c in GLOB_CHARACTERS if c in resource)]
    directory = prefix.rsplit("/", 1)[0] if "/" in prefix else prefix
    if not directory:
        return (resource,)
    return (resource, f"{directory}/", f"{PurePosixPath(directory).name}/")


# Expected false positives: the build plan refers to a resource by description ("the answer key")
# rather than by path or file name. Expected false negatives: a common file name (README.md,
# task.md) mentioned in the build plan for a different file, and globs whose directory is mentioned
# for another reason. Advisory: a missing reference is a specificity gap, not an invalid task.
def resources_in_build_plan(p: TaskProposal, ctx: CheckContext) -> Outcome:
    if not p.header.resources:
        return CheckStatus.SKIP, "no resources declared"
    build_plan = body_sections(p.body).get(BUILD_HEADING)
    if build_plan is None:
        return CheckStatus.SKIP, f"no '## {BUILD_HEADING}' section; required_sections reports it"
    unreferenced = [r for r in p.header.resources if not any(m in build_plan for m in resource_mentions(r))]
    if unreferenced:
        return (
            CheckStatus.FAIL,
            f"{len(unreferenced)}/{len(p.header.resources)} resources never named in the build plan: {unreferenced}",
        )
    return CheckStatus.PASS, f"all {len(p.header.resources)} resources named in the build plan"


# Expected false positives: a null proposal that keeps a research or build note explaining why the
# slot failed; the format asks for empty lists, so this is a format violation, not an opinion.
# Expected false negatives: a reason of five or more words that is still not substantive. Non-null
# proposals SKIP.
def null_reason_only(p: TaskProposal, ctx: CheckContext) -> Outcome:
    if not is_null(p):
        return CheckStatus.SKIP, "not a null proposal"
    h = p.header
    reason = h.null_reason
    assert reason is not None
    extras = [
        name for name, value in (("research", h.research), ("build", h.build), ("resources", h.resources)) if value
    ]
    if extras:
        return CheckStatus.FAIL, f"null proposal still declares {extras}"
    if word_count(reason) < MIN_NULL_REASON_WORDS:
        return CheckStatus.FAIL, f"null_reason has fewer than {MIN_NULL_REASON_WORDS} words: {reason!r}"
    return CheckStatus.PASS, "null proposal carries a reason and nothing else"


CHECKS: tuple[Check, ...] = (
    RuleCheck("header_schema_valid", Severity.FATAL, header_schema_valid),
    RuleCheck("combination_allowed", Severity.FATAL, combination_allowed),
    RuleCheck("grader_matches_verification", Severity.ADVISORY, grader_matches_verification),
    RuleCheck("required_sections", Severity.FATAL, required_sections),
    RuleCheck("research_has_purpose", Severity.ADVISORY, research_has_purpose),
    RuleCheck("resources_in_build_plan", Severity.ADVISORY, resources_in_build_plan),
    RuleCheck("null_reason_only", Severity.FATAL, null_reason_only),
)

CHECKS_BY_NAME: Mapping[str, Check] = {c.name: c for c in CHECKS}


def run_checks(p: TaskProposal, checks: Sequence[Check], ctx: CheckContext) -> tuple[CheckResult, ...]:
    return tuple(check.run(p, ctx) for check in checks)
