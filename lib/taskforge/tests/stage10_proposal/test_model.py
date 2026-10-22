# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re

import pytest

from taskforge.stage10_proposal.model import (
    BuildItem,
    Environment,
    Grounding,
    ProposalFormatError,
    ProposalHeader,
    ResearchItem,
    ResearchKind,
    ResearchNeed,
    SolverMachine,
    SourceKind,
    SourceRef,
    TaskProposal,
    Verification,
    document_regex,
    parse,
    render,
)

BODY = """## Task
Find the kernel of the map.

## Realism and workflow
A lab handout.

## Research plan
### Sources
Two handouts, research[0].

## Build plan
Generate fixtures.

## Grader design and controls
check.py scores the JSON.

## Risks and null conditions
None known.
"""

# Hand-written by an author: unquoted scalars, block-style mappings, keys out of order, trailing spaces, CRLF.
AUTHORED = (
    "---\n"
    "id: d01.algebra.linear-transformations/3\n"
    "source:\n"
    "  kind: capability\n"
    "  ref: d01.algebra.linear-transformations\n"
    "  hash: 9f3a12\n"
    "environment: container\n"
    "solver_machine: null\n"
    "verification: composite\n"
    "grounding: unverified\n"
    "research_need: needed\n"
    "research_reason: the deliverable should follow real handouts\n"
    "research:\n"
    "  - kind: web\n"
    "    purpose: two real lab handouts   \n"
    "build:\n"
    "  - fixtures: 30 matrices with known eigen-structure\n"
    "  - grader: 'check.py: scores the JSON'\n"
    "resources: [fixtures/data.json, 'grader/*.py']\n"
    "null_reason: null\n"
    "---\n\n" + BODY.replace("\n", "\r\n") + "\n\n"
)

PROPOSAL_ID = "d01.algebra.linear-transformations/3"
SOURCE = SourceRef(SourceKind.CAPABILITY, "d01.algebra.linear-transformations", "9f3a12")


def _proposal(
    research_need: ResearchNeed = ResearchNeed.NEEDED,
    research: tuple[ResearchItem, ...] = (ResearchItem(ResearchKind.WEB, "two real lab handouts"),),
    build: tuple[BuildItem, ...] = (BuildItem("fixtures", "30 matrices"), BuildItem("grader", "check.py")),
    null_reason: str | None = None,
    body: str = BODY,
) -> TaskProposal:
    header = ProposalHeader(
        id=PROPOSAL_ID,
        source=SOURCE,
        grounding=Grounding.UNVERIFIED,
        null_reason=null_reason,
        environment=Environment.CONTAINER,
        solver_machine=None,
        verification=Verification.COMPOSITE,
        research_need=research_need,
        research_reason="the deliverable should follow real handouts",
        research=research,
        build=build,
        resources=("fixtures/data.json", "grader/*.py"),
    )
    return parse(render(TaskProposal(header, body)))


NOT_NEEDED = dict(research_need=ResearchNeed.NOT_NEEDED, research=(), body=BODY.replace(", research[0]", ""))
NULL = {**NOT_NEEDED, "build": (), "null_reason": "no realistic grader exists", "body": "No private evidence.\nNone.\n"}


def test_authored_document_parses_to_typed_header_and_canonical_form_is_stable():
    proposal = parse(AUTHORED)
    header = proposal.header
    assert header.environment is Environment.CONTAINER
    assert header.verification is Verification.COMPOSITE
    assert header.research_need is ResearchNeed.NEEDED
    assert header.research[0].kind is ResearchKind.WEB
    assert header.build == (
        BuildItem("fixtures", "30 matrices with known eigen-structure"),
        BuildItem("grader", "check.py: scores the JSON"),
    )
    assert header.resources == ("fixtures/data.json", "grader/*.py")

    canonical = render(proposal)
    assert parse(canonical) == proposal
    assert render(parse(canonical)) == canonical
    assert parse(canonical).digest == proposal.digest


@pytest.mark.parametrize(
    ("environment", "machine"),
    [
        *((Environment.REASONING, machine) for machine in SolverMachine),
        *((Environment.SHELLSIM, machine) for machine in SolverMachine if machine is not SolverMachine.NONE),
    ],
)
def test_an_imageless_proposal_declares_its_solver_machine(environment, machine):
    text = AUTHORED.replace("environment: container", f"environment: {environment}").replace(
        "solver_machine: null", f"solver_machine: {machine}"
    )

    proposal = parse(text)

    assert proposal.header.solver_machine is machine
    assert f"solver_machine: {machine}\n" in render(proposal)
    assert parse(render(proposal)) == proposal
    assert re.fullmatch(document_regex(PROPOSAL_ID, SOURCE), render(proposal))


def test_only_a_reasoning_proposal_declares_no_machine():
    text = AUTHORED.replace("environment: container", "environment: shellsim").replace(
        "solver_machine: null", "solver_machine: none"
    )

    with pytest.raises(ProposalFormatError, match="only a reasoning proposal may declare none"):
        parse(text)
    assert not re.fullmatch(document_regex(PROPOSAL_ID, SOURCE), text)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (("solver_machine: null\n", ""), "missing key(s) ['solver_machine']"),
        (("solver_machine: null", "solver_machine: cpython"), "a container proposal runs in its own image"),
        (("environment: container", "environment: reasoning"), "a reasoning proposal declares one of"),
        (("solver_machine: null", "solver_machine: container"), "solver_machine: 'container' is not one of"),
    ],
    ids=["missing", "container-with-a-machine", "imageless-without-one", "container-as-a-machine"],
)
def test_the_solver_machine_must_match_the_environment(edit, message):
    with pytest.raises(ProposalFormatError, match=re.escape(message)):
        parse(AUTHORED.replace(*edit))


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            ("fixtures/data.json, ", "../x, "),
            "resources: must be relative paths inside the build root: ['../x']",
        ),
        (
            ("fixtures/data.json, ", "/abs/data.json, "),
            "resources: must be relative paths inside the build root: ['/abs/data.json']",
        ),
        (("  - grader: 'check.py", "  - fixtures: 'check.py"), "build: names repeat: ['fixtures']"),
    ],
    ids=["parent-dir", "absolute", "repeated-build-name"],
)
def test_parse_refuses_resources_outside_the_build_root_and_repeated_build_names(edit, message):
    with pytest.raises(ProposalFormatError, match=re.escape(message)):
        parse(AUTHORED.replace(*edit))


def test_digest_changes_with_content():
    proposal = parse(AUTHORED)
    edited = parse(AUTHORED.replace("Find the kernel", "Find the image"))
    assert edited.digest != proposal.digest


@pytest.mark.parametrize(
    ("edit", "field"),
    [
        (("null_reason: null\n", "null_reason: null\nsummary: x\n"), "summary"),
        (("verification: composite", "verification: rubric"), "verification"),
        (("research_need: needed", "research_need: maybe"), "research_need"),
        (("  - kind: web", "  - kind: arxiv"), "research[0].kind"),
        (("## Build plan", "## Construction"), "## Build plan"),
        (("hash: 9f3a12", "hash: 12345"), "source.hash"),
    ],
)
def test_strict_parse_names_the_offending_field(edit, field):
    with pytest.raises(ProposalFormatError, match=re.escape(field)):
        parse(AUTHORED.replace(*edit))


FENCED_EXAMPLE = "Find the kernel of the map.\nThe deliverable looks like:\n```markdown\n## Task\n## Build plan\n```\n"


def test_headings_inside_code_fences_are_not_section_headings():
    proposal = parse(AUTHORED.replace("Find the kernel of the map.\r\n", FENCED_EXAMPLE))
    assert "## Build plan\n```" in proposal.body
    assert re.fullmatch(document_regex(PROPOSAL_ID, SOURCE), render(proposal))


@pytest.mark.parametrize(
    "edit",
    [("id: d01", "id: d02.other/1\nid: d01"), ("  hash: 9f3a12\n", "  hash: 9f3a12\n  ref: elsewhere\n")],
    ids=["header", "nested"],
)
def test_a_repeated_front_matter_key_is_refused(edit):
    with pytest.raises(ProposalFormatError, match="duplicate key"):
        parse(AUTHORED.replace(*edit))


def test_null_proposal_may_have_abbreviated_body():
    text = AUTHORED.split("---\n\n")[0].replace("null_reason: null", "null_reason: no realistic grader exists")
    proposal = parse(text + "---\nNo private evidence can separate good from bad answers.\n")
    assert proposal.header.null_reason == "no realistic grader exists"
    assert parse(render(proposal)) == proposal


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            ("research:\n  - kind: web\n    purpose: two real lab handouts   \n", "research: []\n"),
            "research: research_need is needed but research is empty",
        ),
        (
            ("research_need: needed", "research_need: not_needed"),
            "research: research_need is not_needed but research lists 1 items",
        ),
        (("Generate fixtures.", "Generate fixtures from research[1]."), "body: cites research[1] but research has 1"),
        (("Two handouts, research[0].", "Two handouts."), "body: research[0] is never cited as research[0]"),
    ],
    ids=["needed-empty", "not_needed-with-items", "citation-out-of-range", "item-never-cited"],
)
def test_research_consistency_errors(edit, message):
    with pytest.raises(ProposalFormatError, match=re.escape(message)):
        parse(AUTHORED.replace(*edit))


def test_null_proposal_skips_citation_rules():
    head = AUTHORED.split("---\n\n")[0].replace("null_reason: null", "null_reason: no realistic grader exists")
    proposal = parse(head + "---\nNo private evidence; research[3] would not help.\n")
    assert proposal.header.research_need is ResearchNeed.NEEDED

    header_rule = head.replace("research_need: needed", "research_need: not_needed")
    with pytest.raises(ProposalFormatError, match="research_need is not_needed"):
        parse(header_rule + "---\nNo private evidence.\n")


@pytest.mark.parametrize("variant", [{}, NOT_NEEDED, NULL], ids=["needed", "not_needed", "null"])
def test_document_regex_matches_canonical_render(variant):
    assert re.fullmatch(document_regex(PROPOSAL_ID, SOURCE), render(_proposal(**variant)))


@pytest.mark.parametrize(
    "edit",
    [
        ("environment: container", 'environment: "shellsim"'),
        ("solver_machine: null", "solver_machine: shellsim"),
        ('build:\n  - "fixtures": "30 matrices"\n  - "grader": "check.py"\n', 'build: {"fixtures": "30 matrices"}\n'),
        ('hash: "9f3a12"', 'hash: "9f3a"'),
        ("/3", "/4"),
        ('"grader/*.py"]\n---\n', '"grader/*.py"]\n'),
        ('research:\n  - {kind: web, purpose: "two real lab handouts"}\n', "research: []\n"),
        ("## Build plan\n", ""),
    ],
    ids=[
        "quoted-enum",
        "container-with-a-machine",
        "build-mapping",
        "hash-copy",
        "changed-id",
        "missing-closing-delimiter",
        "needed-with-empty-research",
        "missing-heading",
    ],
)
def test_document_regex_rejects_drift(edit: tuple[str, str]):
    old, new = edit
    canonical = render(_proposal())
    assert old in canonical
    assert not re.fullmatch(document_regex(PROPOSAL_ID, SOURCE), canonical.replace(old, new, 1))


def test_document_regex_rejects_a_fence_that_swallows_the_next_heading():
    canonical = render(_proposal())
    unclosed = canonical.replace("Find the kernel of the map.\n", "Find the kernel of the map.\n```\n")
    with pytest.raises(ProposalFormatError, match="missing required heading"):
        parse(unclosed)
    assert not re.fullmatch(document_regex(PROPOSAL_ID, SOURCE), unclosed)


def test_document_regex_rejects_headings_out_of_order():
    swapped = (
        render(_proposal())
        .replace("## Research plan", "## SWAP")
        .replace("## Build plan", "## Research plan")
        .replace("## SWAP", "## Build plan")
    )
    assert not re.fullmatch(document_regex(PROPOSAL_ID, SOURCE), swapped)


@pytest.mark.parametrize(
    ("variant", "last_section_start"),
    [({}, "## Risks and null conditions\n"), (NULL, "\n---\n")],
    ids=["proposed", "null"],
)
def test_document_regex_allows_eos_after_any_body_line(variant, last_section_start):
    canonical = render(_proposal(**variant))
    start = canonical.index(last_section_start) + len(last_section_start)
    line_ends = [i + 1 for i in range(start, len(canonical)) if canonical[i] == "\n"]
    assert line_ends
    regex = document_regex(PROPOSAL_ID, SOURCE)
    for end in [start, *line_ends]:
        assert re.fullmatch(regex, canonical[:end]), canonical[:end]
