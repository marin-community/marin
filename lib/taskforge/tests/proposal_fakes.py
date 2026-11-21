# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A sample task proposal shared by the Taskforge tests."""

from taskforge.model.proposal import (
    BuildItem,
    Environment,
    Grounding,
    ProposalHeader,
    ResearchItem,
    ResearchKind,
    ResearchNeed,
    SourceKind,
    SourceRef,
    TaskProposal,
    Verification,
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


def sample_proposal() -> TaskProposal:
    """A container proposal with research and build items, the shared fixture across Taskforge tests."""
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
