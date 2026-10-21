# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The TaskProposal document: a strictly validated YAML front matter plus a markdown body.

A proposal is written as::

    ---
    id: "d01.algebra.linear-transformations/3"
    source: {kind: capability, ref: "d01.algebra.linear-transformations", hash: "9f3a..."}
    grounding: unverified
    null_reason: null
    environment: container
    solver_machine: null
    verification: composite
    research_need: needed
    research_reason: "the deliverable should follow real lab handouts"
    research:
      - {kind: web, purpose: "two real lab handouts to model the deliverable on"}
    build:
      - "fixtures": "30 matrices with known eigen-structure, generated and checked"
    resources: ["fixtures/*.json", "grader/check.py"]
    ---
    ## Task
    ...

The body must contain every heading in ``REQUIRED_HEADINGS`` as a level-two heading, once each,
in that order, unless ``null_reason`` is set, in which case the body may be abbreviated.
``solver_machine`` pins the machine the solver works on: ``null`` for a container proposal, which runs
in the image its build publishes, and one of ``SolverMachine`` for any other, so no later stage chooses
it. ``none``, no machine at all, is legal only for a reasoning proposal.
Every ``resources`` entry is a relative path inside the build root (no leading ``/``, no ``..`` part),
and no two ``build`` items share a name.
``research_need`` must agree with ``research``: ``needed`` lists at least one item and ``not_needed``
none, and the body of a non-null proposal cites every item, and only listed items, as ``research[i]``.
``render`` produces the canonical bytes: string scalars are JSON-quoted (valid YAML), keys appear
in a fixed order, and the body is stripped of trailing whitespace. ``digest`` hashes those bytes,
so a proposal's identity does not depend on how its author formatted the header.

``document_regex`` constrains generation to that canonical layout: a ``structured_outputs.regex``
that fully matches ``render(p)`` for every valid ``p`` whose string values hold no ``"``, backslash
or newline and whose body, outside fenced code, starts with the first required heading, has no other
level-two heading, and has a line under every required heading but the last.
"""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

import yaml

from taskforge.content_hash import sha256_hex

FRONT_MATTER_DELIMITER = "---"
CODE_FENCES = ("```", "~~~")
# A fence marker may be indented by spaces and tabs only, which document_regex can express.
FENCE_INDENT = " \t"

REQUIRED_HEADINGS = (
    "Task",
    "Realism and workflow",
    "Research plan",
    "Build plan",
    "Grader design and controls",
    "Risks and null conditions",
)

# Authoring order: render writes the keys in this order and document_regex pins it.
HEADER_KEYS = (
    "id",
    "source",
    "grounding",
    "null_reason",
    "environment",
    "solver_machine",
    "verification",
    "research_need",
    "research_reason",
    "research",
    "build",
    "resources",
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


class ResearchNeed(StrEnum):
    """Whether the build depends on external facts or artifacts that must be looked up."""

    NEEDED = "needed"
    NOT_NEEDED = "not_needed"


class ProposalFormatError(ValueError):
    """The document does not match the proposal format; the message names the field."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """A safe loader that refuses a mapping that repeats a key, which ``safe_load`` resolves to the last value."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        seen = []
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(None, None, f"duplicate key {key!r}", key_node.start_mark)
            seen.append(key)
        return super().construct_mapping(node, deep=deep)


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
    solver_machine: SolverMachine | None
    verification: Verification
    grounding: Grounding
    research_need: ResearchNeed
    research_reason: str
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
        f"grounding: {h.grounding}",
        f"null_reason: {'null' if h.null_reason is None else _quoted(h.null_reason)}",
        f"environment: {h.environment}",
        f"solver_machine: {'null' if h.solver_machine is None else h.solver_machine}",
        f"verification: {h.verification}",
        f"research_need: {h.research_need}",
        f"research_reason: {_quoted(h.research_reason)}",
        "research:" if h.research else "research: []",
        *(f"  - {{kind: {item.kind}, purpose: {_quoted(item.purpose)}}}" for item in h.research),
        "build:" if h.build else "build: []",
        *(f"  - {_quoted(item.name)}: {_quoted(item.description)}" for item in h.build),
        f"resources: [{', '.join(_quoted(r) for r in h.resources)}]",
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
        raw = yaml.load("\n".join(lines[1:closing]), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ProposalFormatError(f"front matter is not valid YAML: {error}") from error
    header = _header(raw)
    body = "\n".join(lines[closing + 1 :])
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
    environment = _enum(Environment, raw["environment"], "environment")
    return ProposalHeader(
        id=_text(raw["id"], "id"),
        source=_source(raw["source"]),
        environment=environment,
        solver_machine=_solver_machine(raw["solver_machine"], environment),
        verification=_enum(Verification, raw["verification"], "verification"),
        grounding=_enum(Grounding, raw["grounding"], "grounding"),
        research_need=_enum(ResearchNeed, raw["research_need"], "research_need"),
        research_reason=_text(raw["research_reason"], "research_reason"),
        research=tuple(_research(item, f"research[{i}]") for i, item in enumerate(_list(raw["research"], "research"))),
        build=_build_items(raw["build"]),
        resources=_resources(raw["resources"]),
        null_reason=None if null_reason is None else _text(null_reason, "null_reason"),
    )


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


def _solver_machine(value: object, environment: Environment) -> SolverMachine | None:
    machine = None if value is None else _enum(SolverMachine, value, "solver_machine")
    violation = solver_machine_violation(environment, machine)
    if violation is not None:
        raise ProposalFormatError(f"solver_machine: {violation}")
    return machine


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


def _build_items(value: object) -> tuple[BuildItem, ...]:
    items = tuple(_build(item, f"build[{i}]") for i, item in enumerate(_list(value, "build")))
    names = [item.name for item in items]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ProposalFormatError(f"build: names repeat: {repeated}")
    return items


def _resources(value: object) -> tuple[str, ...]:
    resources = tuple(_text(item, f"resources[{i}]") for i, item in enumerate(_list(value, "resources")))
    escaping = [r for r in resources if PurePosixPath(r).is_absolute() or ".." in PurePosixPath(r).parts]
    if escaping:
        raise ProposalFormatError(f"resources: must be relative paths inside the build root: {escaping}")
    return resources


# The document regex. The server's regex dialect rejects escapes of non-metacharacters (``\ ``, ``\-``), so
# literals escape only REGEX_META rather than going through ``re.escape``.
REGEX_META = frozenset("\\.+*?()|[]{}^$")
QUOTED = r'"[^"\\\n]+"'


def _not_fence(char: str) -> str:
    """Text after a line's indent that does not start with three ``char``s, so it neither opens nor closes."""
    return rf"(?:{char}(?:[^{char}\n][^\n]*)?|{char}{char}(?:[^{char}\n][^\n]*)?)"


def _fence_patterns(char: str) -> tuple[str, str]:
    """A closed fenced block of ``char`` without its final newline, and an unclosed one running to the end."""
    marker = rf"[ \t]*{char * 3}[^\n]*"
    inner = rf"[ \t]*(?:[^ \t\n{char}][^\n]*|{_not_fence(char)})?"
    return rf"{marker}\n(?:{inner}\n)*{marker}", rf"{marker}(?:\n{inner})*"


# One body line outside fenced code that is neither a level-two heading nor a fence marker.
TEXT = (
    r"(?:[^ \t\n#`~][^\n]*|#[^#\n][^\n]*|##[^ \n][^\n]*|##?"
    rf"|{_not_fence('`')}|{_not_fence('~')}|[ \t]+(?:[^ \t\n`~][^\n]*|{_not_fence('`')}|{_not_fence('~')})?)"
)
_BACKTICK_BLOCK, _BACKTICK_OPEN = _fence_patterns("`")
_TILDE_BLOCK, _TILDE_OPEN = _fence_patterns("~")
# A body line, or a whole fenced block, inside which any line, a level-two heading included, is code.
BODY_LINE = rf"(?:{TEXT}|{_BACKTICK_BLOCK}|{_TILDE_BLOCK})"
LINE = rf"{BODY_LINE}?\n"
# The last section: EOS is reachable after any line, inside an unclosed fence too, so decoding does not
# run on to max_tokens.
END = rf"(?:{LINE})*(?:{BODY_LINE}|{_BACKTICK_OPEN}|{_TILDE_OPEN})?"


def _literal(text: str) -> str:
    return "".join(r"\n" if ch == "\n" else "\\" + ch if ch in REGEX_META else ch for ch in text)


def _alternation(enum: type[StrEnum]) -> str:
    return "(?:" + "|".join(_literal(str(member)) for member in enum) + ")"


def _header_rest() -> str:
    """The front matter after ``null_reason``, through the closing delimiter."""
    item = rf"  - \{{kind: {_alternation(ResearchKind)}, purpose: {QUOTED}\}}\n"
    with_machine = "(?:" + "|".join(_literal(str(m)) for m in SolverMachine if m is not SolverMachine.NONE) + ")"
    return (
        rf"(?:environment: {_literal(Environment.CONTAINER)}\nsolver_machine: null\n"
        rf"|environment: {_literal(Environment.REASONING)}\nsolver_machine: {_alternation(SolverMachine)}\n"
        rf"|environment: {_literal(Environment.SHELLSIM)}\nsolver_machine: {with_machine}\n)"
        rf"verification: {_alternation(Verification)}\n"
        rf"(?:research_need: {ResearchNeed.NEEDED}\nresearch_reason: {QUOTED}\nresearch:\n(?:{item})+"
        rf"|research_need: {ResearchNeed.NOT_NEEDED}\nresearch_reason: {QUOTED}\nresearch: \[\]\n)"
        rf"build:(?: \[\]\n|\n(?:  - {QUOTED}: {QUOTED}\n)+)"
        rf"resources: \[(?:{QUOTED}(?:, {QUOTED})*)?\]\n"
        rf"{_literal(FRONT_MATTER_DELIMITER)}\n"
    )


def document_regex(proposal_id: str, source: SourceRef) -> str:
    """A ``structured_outputs.regex`` over a whole proposal document.

    Front-matter keys, their order and enum values are pinned and the ``id`` and ``source`` lines are literal;
    ``solver_machine`` is ``null`` exactly for a container proposal, and ``none`` only for a reasoning proposal.
    A proposed document (``null_reason: null``) has every required heading, in order, each but the last with
    at least one line under it; a null document has a free body. Bodies are free text in which a level-two
    heading line appears only inside fenced code, and EOS is allowed after any line of the last section.
    """
    fixed = _literal(f"{FRONT_MATTER_DELIMITER}\nid: {_quoted(proposal_id)}\nsource: {source_text(source)}\n")
    sections = "".join(rf"## {_literal(h)}\n(?:{LINE})+" for h in REQUIRED_HEADINGS[:-1])
    sections += rf"## {_literal(REQUIRED_HEADINGS[-1])}\n{END}"
    rest = _header_rest()
    proposed = rf"null\n{rest}\n*{sections}"
    null = rf"{QUOTED}\n{rest}{END}"
    return rf"{fixed}grounding: {_alternation(Grounding)}\nnull_reason: (?:{proposed}|{null})"


def document_request_fields(proposal_id: str, source: SourceRef) -> dict[str, object]:
    """Request fields that constrain a completion to ``document_regex``."""
    return {"structured_outputs": {"regex": document_regex(proposal_id, source)}}
