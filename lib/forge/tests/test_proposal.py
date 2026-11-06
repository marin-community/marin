# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Proposal documents: JSON round trip, canonical digest and the parser's refusals.

Invariants: proposal structure and research claims agree.
"""

import json
from collections.abc import Callable
from dataclasses import replace

import pytest

from forge.model.proposal import (
    BuildItem,
    Environment,
    Grounding,
    ProposalFormatError,
    ProposalHeader,
    ResearchItem,
    ResearchKind,
    ResearchNeed,
    SourceKind,
    SourceRef,
    TaskProposal,
    Verification,
    body_sections,
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

SOURCE = SourceRef(SourceKind.CAPABILITY, "d01.algebra.linear-transformations", "9f3a12")
PROPOSAL_ID = "d01.algebra.linear-transformations/3"


@pytest.fixture
def proposal() -> TaskProposal:
    return TaskProposal(
        ProposalHeader(
            id=PROPOSAL_ID,
            source=SOURCE,
            environment=Environment.CONTAINER,
            verification=Verification.CODE,
            grounding=Grounding.UNVERIFIED,
            research_need=ResearchNeed.NEEDED,
            research_reason="Follow real handouts",
            research=(ResearchItem(ResearchKind.WEB, "two real lab handouts"),),
            build=(BuildItem("fixtures", "30 matrices"), BuildItem("grader", "check.py")),
            resources=("fixtures/data.json", "grader/*.py"),
            null_reason=None,
            composite=(),
        ),
        BODY,
    )


def test_json_round_trip_preserves_typed_content_and_digest(proposal: TaskProposal) -> None:
    authored = json.loads(render(proposal))
    authored["body"] = authored["body"].replace("\n", "\r\n") + "  \n"
    authored["header"]["research"][0]["purpose"] += "   "
    parsed = parse(json.dumps(authored, sort_keys=True, indent=4))
    assert parsed == proposal
    assert parsed.header.source.kind is SourceKind.CAPABILITY
    assert parsed.header.build == proposal.header.build
    assert render(parse(render(parsed))) == render(parsed)
    assert parsed.digest == proposal.digest
    assert replace(parsed, body=BODY.replace("kernel", "image")).digest != parsed.digest


def test_escaped_strings_and_fenced_headings_survive_round_trip(proposal: TaskProposal) -> None:
    body = BODY.replace(
        "Find the kernel of the map.", 'Use "quoted" Unicode λ and path C:\\tmp.\n```markdown\n## Build plan\n```'
    )
    expected = replace(
        proposal, body=body, header=replace(proposal.header, build=(BuildItem('a"b', "line one\nline two\\three"),))
    )
    parsed = parse(render(expected))
    assert parsed == expected
    assert "## Build plan" in body_sections(parsed.body)["Task"]


def forged_key(raw: str, key: str) -> str:
    return raw.replace(f'"{key}":', f'"{key}": "forged", "{key}":', 1)


def with_header(raw: str, key: str, value: object) -> str:
    document = json.loads(raw)
    document["header"][key] = value
    return json.dumps(document)


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda raw: with_header(raw, "unexpected", 1), None),
        (lambda raw: with_header(raw, "source", {"kind": "capability", "ref": "x", "hash": 7}), None),
        (lambda raw: forged_key(raw, "id"), r"duplicate.*key"),
        (lambda raw: forged_key(raw, "hash"), r"duplicate.*key"),
        (lambda raw: with_header(raw, "resources", ["../private/key"]), "resources"),
        (lambda raw: with_header(raw, "resources", ["/absolute/path"]), "resources"),
        (lambda raw: with_header(raw, "build", [{"name": "same", "description": d} for d in ("a", "b")]), "build"),
    ],
    ids=["unknown-field", "wrong-type", "duplicate-id", "duplicate-hash", "escape", "absolute", "same-artifact-name"],
)
def test_a_header_with_unknown_forged_or_escaping_fields_is_refused(
    proposal: TaskProposal, edit: Callable[[str], str], match: str
) -> None:
    with pytest.raises(ProposalFormatError, match=match):
        parse(edit(render(proposal)))


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (BODY.replace("## Build plan", "## Construction"), "body"),
        (BODY.replace("## Build plan", "## Research plan"), "body"),
        (BODY.replace("Find the kernel of the map.", "Find the kernel.\n```"), "body"),
        (BODY.replace(", research[0]", ""), "research"),
        (BODY + "Use research[1].\n", "research"),
    ],
    ids=["unknown-heading", "repeated-heading", "open-fence", "uncited-research", "undeclared-research"],
)
def test_a_body_with_malformed_sections_or_research_references_is_refused(
    proposal: TaskProposal, body: str, match: str
) -> None:
    with pytest.raises(ProposalFormatError, match=match):
        parse(render(replace(proposal, body=body)))


def test_research_decision_and_list_agree_even_for_null_proposals(proposal: TaskProposal) -> None:
    invalid = replace(proposal, header=replace(proposal.header, research_need=ResearchNeed.NOT_NEEDED))
    with pytest.raises(ProposalFormatError, match="research_need"):
        parse(render(invalid))
    null = replace(invalid, header=replace(invalid.header, null_reason="No private evidence"), body="No task.\n")
    with pytest.raises(ProposalFormatError, match="research_need"):
        parse(render(null))
    valid = replace(null, header=replace(null.header, research=()))
    assert parse(render(valid)) == valid
    assert parse(
        render(
            replace(
                proposal,
                header=replace(proposal.header, null_reason="No private evidence"),
                body="No private evidence; research[3] would not help.\n",
            )
        )
    ).header.null_reason


def test_composite_round_trip_checks_weights(proposal: TaskProposal) -> None:
    raw = json.loads(render(proposal))
    raw["header"].update(
        verification="composite",
        composite=[
            {"kind": "code", "weight": 0.25},
            {"kind": "decision_model", "weight": 0.75},
        ],
    )
    composite = parse(json.dumps(raw))
    assert composite.header.verification is Verification.COMPOSITE
    assert [part.weight for part in composite.header.composite] == [0.25, 0.75]
    assert parse(render(composite)) == composite
    raw["header"]["composite"][0]["weight"] = 0.5
    with pytest.raises(ProposalFormatError, match="not 1"):
        parse(json.dumps(raw))


def test_existing_catalog_drops_obsolete_solver_machine(proposal: TaskProposal) -> None:
    parsed = parse(with_header(render(proposal), "solver_machine", "packages"))
    assert parsed == proposal
    assert "solver_machine" not in json.loads(render(parsed))["header"]
