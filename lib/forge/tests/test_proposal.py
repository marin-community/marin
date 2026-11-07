# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Proposal documents: JSON round trip, canonical digest and the parser's refusals.

Invariants: the task runs on the solver machine it declares (``solver_machine_violation``).
"""

import json
from collections.abc import Callable
from dataclasses import replace

import jsonschema
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
    SolverMachine,
    SourceKind,
    SourceRef,
    TaskProposal,
    Verification,
    body_sections,
    parse,
    render,
)
from forge.model.serialization import proposal_request_fields as document_request_fields

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
            solver_machine=None,
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


def test_json_round_trip_preserves_typed_content_and_canonical_digest(proposal: TaskProposal) -> None:
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
    schema = document_request_fields(PROPOSAL_ID, SOURCE)["response_format"]["json_schema"]["schema"]
    jsonschema.validate(json.loads(render(parsed)), schema)


COMPATIBLE = [
    *((Environment.REASONING, machine) for machine in SolverMachine),
    *((Environment.SHELLSIM, machine) for machine in SolverMachine if machine is not SolverMachine.NONE),
]
INCOMPATIBLE = [(Environment.CONTAINER, SolverMachine.CPYTHON), (Environment.SHELLSIM, SolverMachine.NONE)]


@pytest.mark.parametrize("environment,machine", [*COMPATIBLE, *INCOMPATIBLE, (Environment.REASONING, None)])
def test_the_solver_machine_must_fit_the_environment(
    proposal: TaskProposal, environment: Environment, machine: SolverMachine | None
) -> None:
    expected = replace(proposal, header=replace(proposal.header, environment=environment, solver_machine=machine))
    if (environment, machine) in COMPATIBLE:
        assert parse(render(expected)) == expected
    else:
        with pytest.raises(ProposalFormatError, match="solver_machine"):
            parse(render(expected))


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


def test_generation_schema_pins_identity(proposal: TaskProposal) -> None:
    schema = document_request_fields(PROPOSAL_ID, SOURCE)["response_format"]["json_schema"]["schema"]
    raw = json.loads(render(proposal))
    jsonschema.validate(raw, schema)
    raw["header"]["source"]["hash"] = "wrong hash"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(raw, schema)


def test_composite_round_trip_remains_readable_but_is_not_offered_to_generation(proposal: TaskProposal) -> None:
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
    schema = document_request_fields(PROPOSAL_ID, SOURCE)["response_format"]["json_schema"]["schema"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(raw, schema)
    raw["header"]["composite"][0]["weight"] = 0.5
    with pytest.raises(ProposalFormatError, match="not 1"):
        parse(json.dumps(raw))
