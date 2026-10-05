# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The TaskProposal document: a strictly validated YAML front matter plus a markdown body.

A proposal is written as::

    ---
    id: "d01.algebra.linear-transformations/3"
    source: {kind: capability, ref: "d01.algebra.linear-transformations", hash: "9f3a..."}
    environment: container
    verification: composite
    grounding: unverified
    research:
      - {kind: web, purpose: "two real lab handouts to model the deliverable on"}
    build:
      - fixtures: "30 matrices with known eigen-structure, generated and checked"
    resources: ["fixtures/*.json", "grader/check.py"]
    null_reason: null
    ---
    ## Task
    ...

The body must contain every heading in ``REQUIRED_HEADINGS`` as a level-two heading, once each,
in that order, unless ``null_reason`` is set, in which case the body may be abbreviated.
``render`` produces the canonical bytes: string scalars are JSON-quoted (valid YAML), keys appear
in a fixed order, and the body is stripped of trailing whitespace. ``digest`` hashes those bytes,
so a proposal's identity does not depend on how its author formatted the header.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import yaml

from taskforge.content_hash import sha256_hex

FRONT_MATTER_DELIMITER = "---"
CODE_FENCES = ("```", "~~~")

REQUIRED_HEADINGS = (
    "Task",
    "Realism and workflow",
    "Research plan",
    "Build plan",
    "Grader design and controls",
    "Risks and null conditions",
)

HEADER_KEYS = (
    "id",
    "source",
    "environment",
    "verification",
    "grounding",
    "research",
    "build",
    "resources",
    "null_reason",
)


class SourceKind(StrEnum):
    CAPABILITY = "capability"
    REPO = "repo"


class Environment(StrEnum):
    """What the solver works in: a prompt only, a simulated shell, or a real container."""

    REASONING = "reasoning"
    SHELLSIM = "shellsim"
    CONTAINER = "container"


class Verification(StrEnum):
    """How the private evaluator grades: exact match, executable checks, a judge rubric, or both."""

    SIMPLE = "simple"
    CODE = "code"
    JUDGE = "judge"
    COMPOSITE = "composite"


class Grounding(StrEnum):
    """Whether the proposal's factual claims were checked against a real source before triage."""

    UNVERIFIED = "unverified"
    GROUNDED = "grounded"


class ResearchKind(StrEnum):
    WEB = "web"
    GITHUB = "github"


class ProposalFormatError(ValueError):
    """The document does not match the proposal format; the message names the field."""


@dataclass(frozen=True)
class SourceRef:
    """Where the proposal came from: a catalog capability id or a repository, and its content hash."""

    kind: SourceKind
    ref: str
    hash: str


@dataclass(frozen=True)
class ResearchItem:
    kind: ResearchKind
    purpose: str


@dataclass(frozen=True)
class BuildItem:
    """One named build artifact and a one-line description; the body explains it."""

    name: str
    description: str


@dataclass(frozen=True)
class ProposalHeader:
    id: str
    source: SourceRef
    environment: Environment
    verification: Verification
    grounding: Grounding
    research: tuple[ResearchItem, ...]
    build: tuple[BuildItem, ...]
    resources: tuple[str, ...]
    null_reason: str | None


@dataclass(frozen=True)
class TaskProposal:
    header: ProposalHeader
    body: str

    @property
    def digest(self) -> str:
        return sha256_hex(render(self).encode())


def _quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def source_text(ref: SourceRef) -> str:
    """The canonical text of a ``source`` header value."""
    return f"{{kind: {ref.kind}, ref: {_quoted(ref.ref)}, hash: {_quoted(ref.hash)}}}"


def render(p: TaskProposal) -> str:
    """Return the canonical text of ``p``."""
    h = p.header
    lines = [
        FRONT_MATTER_DELIMITER,
        f"id: {_quoted(h.id)}",
        f"source: {source_text(h.source)}",
        f"environment: {h.environment}",
        f"verification: {h.verification}",
        f"grounding: {h.grounding}",
        "research:" if h.research else "research: []",
        *(f"  - {{kind: {item.kind}, purpose: {_quoted(item.purpose)}}}" for item in h.research),
        "build:" if h.build else "build: []",
        *(f"  - {_quoted(item.name)}: {_quoted(item.description)}" for item in h.build),
        f"resources: [{', '.join(_quoted(r) for r in h.resources)}]",
        f"null_reason: {'null' if h.null_reason is None else _quoted(h.null_reason)}",
        FRONT_MATTER_DELIMITER,
    ]
    return "\n".join(lines) + "\n" + _canonical_body(p.body)


def _canonical_body(body: str) -> str:
    lines = [line.rstrip() for line in body.replace("\r\n", "\n").strip().split("\n")]
    return "\n".join(lines) + "\n"


def parse(text: str) -> TaskProposal:
    """Parse and validate a proposal document; raise ``ProposalFormatError`` naming the bad field."""
    lines = text.replace("\r\n", "\n").split("\n")
    if lines[0].strip() != FRONT_MATTER_DELIMITER:
        raise ProposalFormatError(f"document must start with a '{FRONT_MATTER_DELIMITER}' front-matter line")
    closing = next((i for i in range(1, len(lines)) if lines[i].strip() == FRONT_MATTER_DELIMITER), None)
    if closing is None:
        raise ProposalFormatError(f"front matter has no closing '{FRONT_MATTER_DELIMITER}' line")
    try:
        raw = yaml.safe_load("\n".join(lines[1:closing]))
    except yaml.YAMLError as error:
        raise ProposalFormatError(f"front matter is not valid YAML: {error}") from error
    header = _header(raw)
    body = "\n".join(lines[closing + 1 :])
    if header.null_reason is None:
        _check_headings(body)
    elif not body.strip():
        raise ProposalFormatError("body: a null proposal still needs a body explaining the null reason")
    return TaskProposal(header=header, body=_canonical_body(body))


def _headings(body: str) -> list[str]:
    """Level-two headings of ``body``, ignoring lines inside fenced code blocks."""
    headings = []
    fence: str | None = None
    for line in body.split("\n"):
        marker = line.lstrip()[:3]
        if fence is None and marker in CODE_FENCES:
            fence = marker
        elif fence is not None and marker == fence:
            fence = None
        elif fence is None and line.startswith("## "):
            headings.append(line[3:].strip())
    return headings


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


def _header(raw: object) -> ProposalHeader:
    if not isinstance(raw, Mapping):
        raise ProposalFormatError(f"front matter must be a mapping, got {type(raw).__name__}")
    unknown = sorted(str(k) for k in raw if k not in HEADER_KEYS)
    if unknown:
        raise ProposalFormatError(f"front matter: unknown key(s) {unknown}; allowed keys are {list(HEADER_KEYS)}")
    missing = [k for k in HEADER_KEYS if k not in raw]
    if missing:
        raise ProposalFormatError(f"front matter: missing key(s) {missing}")
    null_reason = raw["null_reason"]
    return ProposalHeader(
        id=_text(raw["id"], "id"),
        source=_source(raw["source"]),
        environment=_enum(Environment, raw["environment"], "environment"),
        verification=_enum(Verification, raw["verification"], "verification"),
        grounding=_enum(Grounding, raw["grounding"], "grounding"),
        research=tuple(_research(item, f"research[{i}]") for i, item in enumerate(_list(raw["research"], "research"))),
        build=tuple(_build(item, f"build[{i}]") for i, item in enumerate(_list(raw["build"], "build"))),
        resources=tuple(_text(item, f"resources[{i}]") for i, item in enumerate(_list(raw["resources"], "resources"))),
        null_reason=None if null_reason is None else _text(null_reason, "null_reason"),
    )


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProposalFormatError(f"{field}: expected a non-empty string, got {value!r}")
    return value.strip()


def _enum[E: StrEnum](enum: type[E], value: object, field: str) -> E:
    allowed = [str(member) for member in enum]
    if value not in allowed:
        raise ProposalFormatError(f"{field}: {value!r} is not one of {allowed}")
    return enum(value)


def _list(value: object, field: str) -> list:
    if not isinstance(value, list):
        raise ProposalFormatError(f"{field}: expected a list, got {value!r}")
    return value


def _mapping(value: object, field: str, keys: tuple[str, ...]) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ProposalFormatError(f"{field}: expected a mapping with exactly the keys {list(keys)}, got {value!r}")
    return value


def _source(value: object) -> SourceRef:
    raw = _mapping(value, "source", ("kind", "ref", "hash"))
    return SourceRef(
        kind=_enum(SourceKind, raw["kind"], "source.kind"),
        ref=_text(raw["ref"], "source.ref"),
        hash=_text(raw["hash"], "source.hash"),
    )


def _research(value: object, field: str) -> ResearchItem:
    raw = _mapping(value, field, ("kind", "purpose"))
    return ResearchItem(
        kind=_enum(ResearchKind, raw["kind"], f"{field}.kind"), purpose=_text(raw["purpose"], f"{field}.purpose")
    )


def _build(value: object, field: str) -> BuildItem:
    if not isinstance(value, Mapping) or len(value) != 1:
        raise ProposalFormatError(f"{field}: expected a one-entry mapping 'name: description', got {value!r}")
    ((name, description),) = value.items()
    return BuildItem(name=_text(name, f"{field} name"), description=_text(description, f"{field}.{name}"))
