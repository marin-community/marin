# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed proposal JSON with a Markdown body.

The JSON object holds ``header`` and ``body``. Parsing checks the environment,
research decision and citations, and the ordered Markdown section headings.
Canonical rendering sorts keys and normalizes body whitespace, so formatting
alone does not change a proposal's content digest. Generation uses the JSON
schema derived from these same dataclasses; semantic checks run after decoding.
"""

import json
import re
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, get_args

from pydantic import ConfigDict, Field, StringConstraints, TypeAdapter, ValidationError, with_config

from taskforge.content_hash import pretty_json, sha256_hex

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

CODE_FENCES = ("```", "~~~")
FENCE_INDENT = " \t"

REQUIRED_HEADINGS = (
    "Task",
    "Realism and workflow",
    "Research plan",
    "Build plan",
    "Grader design and controls",
    "Risks and null conditions",
)

RESEARCH_CITATION = re.compile(r"research\[(\d+)\]")

DOCUMENT_REPAIR_PROMPT = """Your previous reply, shown above, is not a valid proposal document:
{error}
Write the corrected document in full, keeping all valid content and changing only what the error requires."""


class SourceKind(StrEnum):
    CAPABILITY = "capability"
    REPO = "repo"


class Environment(StrEnum):
    """What the solver works in: a prompt only, a simulated shell, or a real container."""

    REASONING = "reasoning"
    SHELLSIM = "shellsim"
    CONTAINER = "container"


class SolverMachine(StrEnum):
    """The machine a reasoning or shellsim proposal's solver works on; a container proposal has its own image.

    No machine at all (the solver reads the instructions and answers in the conversation, with no tools
    and no files; legal only for a reasoning proposal), ShellSim, CPython 3.12 with its standard library
    on Taskforge's grader-base image, or CPython 3.12 with PyPI packages the build lists, which runs only
    on a host with the local backend.
    """

    NONE = "none"
    SHELLSIM = "shellsim"
    CPYTHON = "cpython"
    PACKAGES = "packages"


SOLVER_MACHINE_RULE = f"""solver_machine is `null` for a container task, which runs in its own image, and \
otherwise the solver's machine: `{SolverMachine.NONE}` (reasoning tasks only) when the solver needs no tools \
and no files, only the instructions and a reply; `{SolverMachine.SHELLSIM}` when reading, shell tools or a \
short Python script are enough; `{SolverMachine.CPYTHON}` when solving needs more computation than ShellSim's \
few seconds of Python, or a real clock; `{SolverMachine.PACKAGES}` when solving also needs PyPI packages \
(numpy, pandas, ...) and no system packages. The build keeps the machine the proposal declares."""
"""The rule for the ``solver_machine`` header value, for prompts that write or revise a proposal."""


class Verification(StrEnum):
    """How the private evaluator grades.

    ``answer_in_box``: a generic match of a final answer whose format is unambiguous (exact, numeric, math,
    choice). ``code``: the usual kind; the task states an exact answer structure (named fields, a units field
    per quantity, enumerated labels) that the grader parses strictly against build-computed ground truth, or
    the work is what only execution can judge (behavioural tests, running systems). ``decision_model``: a
    decision model compares the answer with the reference answer against a rubric, only where fields cannot
    capture the answer, and its fooling cases must earn no credit. ``llm_as_judge``: a model judge
    for deliverables whose prose is the product. ``composite``: a weighted list of the other kinds
    (``CompositePart``). ``composite`` is reserved: documents that carry it parse, but it is not offered to
    proposers (``OFFERED_VERIFICATIONS``) until a composite grader type exists, since a TaskSpec holds one
    grader.
    """

    ANSWER_IN_BOX = "answer_in_box"
    DECISION_MODEL = "decision_model"
    CODE = "code"
    LLM_AS_JUDGE = "llm_as_judge"
    COMPOSITE = "composite"


OfferedVerification = Literal[
    Verification.DECISION_MODEL, Verification.ANSWER_IN_BOX, Verification.CODE, Verification.LLM_AS_JUDGE
]
"""The kinds a proposer may choose: every kind but the reserved ``composite``."""
OFFERED_VERIFICATIONS: tuple[Verification, ...] = get_args(OfferedVerification)

COMPOSITE_PART_KINDS = tuple(kind for kind in Verification if kind is not Verification.COMPOSITE)
"""The kinds a composite part may be: every kind but composite, each as often as the proposal needs."""
COMPOSITE_MIN_PARTS = 2
COMPOSITE_WEIGHT_TOLERANCE = 1e-6
"""How far the parts' weights may sum from 1."""


class Grounding(StrEnum):
    """Whether the proposal's factual claims were checked against a real source before triage."""

    UNVERIFIED = "unverified"
    GROUNDED = "grounded"


class ResearchKind(StrEnum):
    WEB = "web"
    GITHUB = "github"


class ResearchNeed(StrEnum):
    """Whether the build depends on external facts or artifacts that must be looked up."""

    NEEDED = "needed"
    NOT_NEEDED = "not_needed"


class ProposalFormatError(ValueError):
    """The document does not match the proposal format; the message names the field."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProposalFormatError(f"duplicate key {key!r}")
        result[key] = value
    return result


@with_config(ConfigDict(extra="forbid"))
@dataclass(frozen=True)
class SourceRef:
    """Where the proposal came from: a catalog capability id or a repository, and its content hash."""

    kind: SourceKind
    ref: Text
    hash: Text


@with_config(ConfigDict(extra="forbid"))
@dataclass(frozen=True)
class ResearchItem:
    kind: ResearchKind
    purpose: Text


@with_config(ConfigDict(extra="forbid"))
@dataclass(frozen=True)
class BuildItem:
    """One named build artifact and a one-line description; the body explains it."""

    name: Text
    description: Text


@with_config(ConfigDict(extra="forbid"))
@dataclass(frozen=True)
class CompositePart:
    """One part of a composite verification: a grader of ``kind`` and its share of the reward.

    For now a composite is deliberately strict: a weighted list of the other kinds, repeats allowed, whose
    weights sum to 1. It is meant to become an arbitrary program over graders.
    """

    kind: Verification
    weight: Annotated[float, Field(strict=True, gt=0, le=1)]


@with_config(ConfigDict(extra="forbid"))
@dataclass(frozen=True)
class ProposalHeader:
    id: Text
    source: SourceRef
    environment: Environment
    solver_machine: SolverMachine | None
    verification: Verification
    composite: tuple[CompositePart, ...]
    grounding: Grounding
    research_need: ResearchNeed
    research_reason: Text
    research: tuple[ResearchItem, ...]
    build: tuple[BuildItem, ...]
    resources: tuple[Text, ...]
    null_reason: Text | None


@with_config(ConfigDict(extra="forbid"))
@dataclass(frozen=True)
class TaskProposal:
    header: ProposalHeader
    body: str

    @property
    def digest(self) -> str:
        return sha256_hex(render(self).encode())


def source_text(ref: SourceRef) -> str:
    """The canonical JSON of a source reference."""
    return pretty_json(ref)


def render(p: TaskProposal) -> str:
    """Return the canonical JSON of ``p``."""
    return pretty_json(replace(p, body=_canonical_body(p.body))) + "\n"


def _canonical_body(body: str) -> str:
    lines = [line.rstrip() for line in body.replace("\r\n", "\n").strip().split("\n")]
    return "\n".join(lines) + "\n"


def parse(text: str) -> TaskProposal:
    """Parse and validate a proposal document; raise ``ProposalFormatError`` naming the bad field."""
    try:
        proposal = _PROPOSAL.validate_python(json.loads(text, object_pairs_hook=_unique_object))
    except (json.JSONDecodeError, ValidationError) as error:
        raise ProposalFormatError(str(error)) from error
    header, body = proposal.header, proposal.body
    violation = solver_machine_violation(header.environment, header.solver_machine)
    if violation is not None:
        raise ProposalFormatError(f"solver_machine: {violation}")
    check_composite(header)
    _check_build(header)
    if header.null_reason is None:
        _check_headings(body)
    elif not body.strip():
        raise ProposalFormatError("body: a null proposal still needs a body explaining the null reason")
    _check_research(header, body)
    return TaskProposal(header=header, body=_canonical_body(body))


def _heading_lines(lines: list[str]) -> list[tuple[int, str]]:
    """Index and text of each level-two heading in ``lines``, ignoring lines inside fenced code blocks."""
    headings = []
    fence: str | None = None
    for i, line in enumerate(lines):
        marker = line.lstrip(FENCE_INDENT)[:3]
        if fence is None and marker in CODE_FENCES:
            fence = marker
        elif fence is not None and marker == fence:
            fence = None
        elif fence is None and line.startswith("## "):
            headings.append((i, line[3:].strip()))
    return headings


def _headings(body: str) -> list[str]:
    return [heading for _, heading in _heading_lines(body.split("\n"))]


def body_sections(body: str) -> dict[str, str]:
    """Map each level-two heading in ``body`` to the text under it, up to the next level-two heading.

    Uses the same fence-aware boundaries as ``parse``; a repeated heading's sections are joined.
    """
    lines = body.split("\n")
    headings = _heading_lines(lines)
    ends = [i for i, _ in headings[1:]] + [len(lines)]
    sections: dict[str, list[str]] = {}
    for (start, heading), end in zip(headings, ends, strict=True):
        sections.setdefault(heading, []).extend(lines[start + 1 : end])
    return {heading: "\n".join(text).strip() for heading, text in sections.items()}


def _check_headings(body: str) -> None:
    found = _headings(body)
    required = [h for h in found if h in REQUIRED_HEADINGS]
    missing = [h for h in REQUIRED_HEADINGS if h not in found]
    if missing:
        raise ProposalFormatError(f"body: missing required heading(s) {['## ' + h for h in missing]}")
    if required != list(REQUIRED_HEADINGS):
        raise ProposalFormatError(
            f"body: required headings must appear once each in the order {list(REQUIRED_HEADINGS)}; found {required}"
        )


def check_composite(header: ProposalHeader) -> None:
    """A composite lists at least ``COMPOSITE_MIN_PARTS`` parts whose weights sum to 1; other kinds list none.

    A null proposal may leave a composite's parts empty: it is never built.

    Raises:
        ProposalFormatError: the parts contradict the verification.
    """
    parts = header.composite
    if header.verification is not Verification.COMPOSITE:
        if parts:
            raise ProposalFormatError(f"composite: verification is {header.verification}, so composite must be []")
        return
    if not parts and header.null_reason is not None:
        return
    if len(parts) < COMPOSITE_MIN_PARTS:
        raise ProposalFormatError(f"composite: a composite verification lists at least {COMPOSITE_MIN_PARTS} parts")
    total = sum(part.weight for part in parts)
    if abs(total - 1) > COMPOSITE_WEIGHT_TOLERANCE:
        raise ProposalFormatError(f"composite: the parts' weights sum to {total}, not 1")


def _check_research(header: ProposalHeader, body: str) -> None:
    """The research decision agrees with the list, and a non-null body cites exactly the listed items."""
    count = len(header.research)
    if header.research_need is ResearchNeed.NEEDED and not count:
        raise ProposalFormatError("research: research_need is needed but research is empty")
    if header.research_need is ResearchNeed.NOT_NEEDED and count:
        raise ProposalFormatError(f"research: research_need is not_needed but research lists {count} items")
    if header.null_reason is not None:
        return
    cited = {int(index) for index in RESEARCH_CITATION.findall(body)}
    for index in sorted(cited):
        if index >= count:
            raise ProposalFormatError(f"body: cites research[{index}] but research has {count} items")
    for index in range(count):
        if index not in cited:
            raise ProposalFormatError(f"body: research[{index}] is never cited as research[{index}]")


def solver_machine_violation(environment: Environment, machine: SolverMachine | None) -> str | None:
    """Why ``machine`` cannot be the solver machine of an ``environment`` proposal, or ``None`` when it can.

    A container proposal declares no machine (``None``), any other declares one, and only a reasoning
    proposal may declare ``SolverMachine.NONE``.
    """
    if environment is Environment.CONTAINER:
        if machine is not None:
            return f"a container proposal runs in its own image and declares null, got {str(machine)!r}"
        return None
    if machine is None:
        return f"a {environment} proposal declares one of {[str(m) for m in SolverMachine]}, got null"
    if machine is SolverMachine.NONE and environment is not Environment.REASONING:
        return (
            f"only a reasoning proposal may declare {SolverMachine.NONE}; a {environment} "
            f"proposal declares one of {[str(m) for m in SolverMachine if m is not SolverMachine.NONE]}"
        )
    return None


def _check_build(header: ProposalHeader) -> None:
    for part in header.composite:
        if part.kind not in COMPOSITE_PART_KINDS or isinstance(part.weight, bool) or not 0 < part.weight <= 1:
            raise ProposalFormatError("composite: parts require a non-composite kind and weight in (0, 1]")
    names = [item.name for item in header.build]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ProposalFormatError(f"build: names repeat: {repeated}")
    escaping = [r for r in header.resources if PurePosixPath(r).is_absolute() or ".." in PurePosixPath(r).parts]
    if escaping:
        raise ProposalFormatError(f"resources: must be relative paths inside the build root: {escaping}")


_PROPOSAL = TypeAdapter(TaskProposal)


def document_request_fields(proposal_id: str, source: SourceRef) -> dict[str, Any]:
    """Constrain generation to the typed proposal schema with fixed identity fields."""
    schema = _PROPOSAL.json_schema()
    fields = schema["$defs"]["ProposalHeader"]["properties"]
    fields["id"] = {"const": proposal_id, "type": "string"}
    fields["source"] = {"const": json.loads(source_text(source))}
    fields["verification"] = {"enum": list(OFFERED_VERIFICATIONS), "type": "string"}
    fields["composite"] = {"type": "array", "maxItems": 0, "items": {"$ref": "#/$defs/CompositePart"}}
    return {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "TaskProposal",
                "schema": schema,
                "strict": True,
            },
        }
    }
