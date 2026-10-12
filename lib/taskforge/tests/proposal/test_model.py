# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from taskforge.proposal.model import (
    BuildItem,
    Environment,
    ProposalFormatError,
    ResearchKind,
    Verification,
    parse,
    render,
)

BODY = """## Task
Find the kernel of the map.

## Realism and workflow
A lab handout.

## Research plan
### Sources
Two handouts.

## Build plan
Generate fixtures.

## Grader design and controls
check.py scores the JSON.

## Risks and null conditions
None known.
"""

# Hand-written by an author: unquoted scalars, block-style mappings, trailing spaces, CRLF.
AUTHORED = (
    "---\n"
    "id: d01.algebra.linear-transformations/3\n"
    "source:\n"
    "  kind: capability\n"
    "  ref: d01.algebra.linear-transformations\n"
    "  hash: 9f3a12\n"
    "environment: container\n"
    "verification: composite\n"
    "grounding: unverified\n"
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


def test_authored_document_parses_to_typed_header_and_canonical_form_is_stable():
    proposal = parse(AUTHORED)
    header = proposal.header
    assert header.environment is Environment.CONTAINER
    assert header.verification is Verification.COMPOSITE
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


def test_digest_changes_with_content():
    proposal = parse(AUTHORED)
    edited = parse(AUTHORED.replace("Find the kernel", "Find the image"))
    assert edited.digest != proposal.digest


@pytest.mark.parametrize(
    ("edit", "field"),
    [
        (("null_reason: null\n", "null_reason: null\nsummary: x\n"), "summary"),
        (("verification: composite", "verification: rubric"), "verification"),
        (("  - kind: web", "  - kind: arxiv"), "research[0].kind"),
        (("## Build plan", "## Construction"), "## Build plan"),
        (("hash: 9f3a12", "hash: 12345"), "source.hash"),
    ],
)
def test_strict_parse_names_the_offending_field(edit, field):
    with pytest.raises(ProposalFormatError, match=field.replace("[", r"\[").replace("]", r"\]")):
        parse(AUTHORED.replace(*edit))


def test_headings_inside_code_fences_are_not_section_headings():
    example = "Find the kernel of the map.\nThe deliverable looks like:\n```markdown\n## Task\n## Build plan\n```\n"
    proposal = parse(AUTHORED.replace("Find the kernel of the map.\r\n", example))
    assert "## Build plan\n```" in proposal.body


def test_null_proposal_may_have_abbreviated_body():
    text = AUTHORED.split("---\n\n")[0].replace("null_reason: null", "null_reason: no realistic grader exists")
    proposal = parse(text + "---\nNo private evidence can separate good from bad answers.\n")
    assert proposal.header.null_reason == "no realistic grader exists"
    assert parse(render(proposal)) == proposal
