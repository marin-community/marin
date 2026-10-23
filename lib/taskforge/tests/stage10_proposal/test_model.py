# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import jsonschema
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
    body_sections,
    document_request_fields,
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
def proposal():
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


def test_json_round_trip_preserves_typed_content_and_canonical_digest(proposal):
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


def test_escaped_strings_and_fenced_headings_survive_round_trip(proposal):
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


@pytest.mark.parametrize(
    "environment,machine",
    [
        *((Environment.REASONING, machine) for machine in SolverMachine),
        *((Environment.SHELLSIM, machine) for machine in SolverMachine if machine is not SolverMachine.NONE),
    ],
)
def test_imageless_proposal_round_trip_preserves_machine(proposal, environment, machine):
    expected = replace(proposal, header=replace(proposal.header, environment=environment, solver_machine=machine))
    assert parse(render(expected)) == expected


@pytest.mark.parametrize(
    "environment,machine",
    [
        (Environment.CONTAINER, SolverMachine.CPYTHON),
        (Environment.SHELLSIM, SolverMachine.NONE),
        (Environment.REASONING, None),
    ],
)
def test_incompatible_environment_and_machine_are_rejected(proposal, environment, machine):
    invalid = replace(proposal, header=replace(proposal.header, environment=environment, solver_machine=machine))
    with pytest.raises(ProposalFormatError, match="solver_machine"):
        parse(render(invalid))


@pytest.mark.parametrize("key,value", [("unexpected", 1), ("source", {"kind": "capability", "ref": "x", "hash": 7})])
def test_json_header_contract_rejects_unknown_fields_and_wrong_types(proposal, key, value):
    raw = json.loads(render(proposal))
    raw["header"][key] = value
    with pytest.raises(ProposalFormatError):
        parse(json.dumps(raw))


@pytest.mark.parametrize("key", ["id", "hash"])
def test_duplicate_json_keys_cannot_replace_identity(proposal, key):
    raw = render(proposal)
    raw = raw.replace(f'"{key}":', f'"{key}": "forged", "{key}":', 1)
    with pytest.raises(ProposalFormatError, match="duplicate key"):
        parse(raw)


@pytest.mark.parametrize(
    "body",
    [
        BODY.replace("## Build plan", "## Construction"),
        BODY.replace("## Build plan", "## Research plan"),
        BODY.replace("Find the kernel of the map.", "Find the kernel.\n```"),
    ],
)
def test_malformed_section_boundaries_are_rejected(proposal, body):
    with pytest.raises(ProposalFormatError, match="body"):
        parse(render(replace(proposal, body=body)))


@pytest.mark.parametrize("body", [BODY.replace(", research[0]", ""), BODY + "Use research[1].\n"])
def test_research_references_must_cover_exactly_the_declared_items(proposal, body):
    with pytest.raises(ProposalFormatError, match="research"):
        parse(render(replace(proposal, body=body)))


def test_research_decision_and_list_agree_even_for_null_proposals(proposal):
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


def test_generation_schema_pins_identity(proposal):
    schema = document_request_fields(PROPOSAL_ID, SOURCE)["response_format"]["json_schema"]["schema"]
    raw = json.loads(render(proposal))
    jsonschema.validate(raw, schema)
    raw["header"]["source"]["hash"] = "wrong hash"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(raw, schema)


def test_composite_round_trip_remains_readable_but_is_not_offered_to_generation(proposal):
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


@pytest.mark.parametrize(
    "field,value",
    [
        ("resources", ["../private/key"]),
        ("resources", ["/absolute/path"]),
        ("build", [{"name": "same", "description": "first"}, {"name": "same", "description": "second"}]),
    ],
)
def test_build_manifest_cannot_escape_root_or_replace_named_artifact(proposal, field, value):
    raw = json.loads(render(proposal))
    raw["header"][field] = value
    with pytest.raises(ProposalFormatError, match=field):
        parse(json.dumps(raw))
