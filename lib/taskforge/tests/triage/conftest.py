# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A small valid proposal document that tests edit one property at a time."""

from dataclasses import replace

import pytest

from taskforge.proposal.model import REQUIRED_HEADINGS, TaskProposal, parse

FILLER = "The builder writes this section in full detail with concrete inputs, outputs, and checks. " * 6

SECTIONS = {
    "Grader design and controls": (
        "The private evaluator runs `grader/check.py`, which parses the answer and compares each value. " + FILLER
    ),
    "Build plan": "Session S1 writes `grader/check.py` and the fixtures under `fixtures/`. " + FILLER,
}


def document(verification: str = "code", resources: str = '["grader/check.py", "fixtures/*.json"]') -> str:
    body = "\n".join(f"## {h}\n{SECTIONS.get(h, FILLER)}\n" for h in REQUIRED_HEADINGS)
    return f"""---
id: "d01.cap/1"
source: {{kind: capability, ref: "d01.cap", hash: "abc"}}
environment: reasoning
verification: {verification}
grounding: unverified
research:
  - {{kind: web, purpose: "two real lab handouts to model the deliverable on"}}
build:
  - "grader": "the private checker and its tests"
resources: {resources}
null_reason: null
---
{body}"""


def with_section(p: TaskProposal, heading: str, text: str) -> TaskProposal:
    body = "\n".join(f"## {h}\n{text if h == heading else SECTIONS.get(h, FILLER)}\n" for h in REQUIRED_HEADINGS)
    return replace(p, body=body)


@pytest.fixture
def proposal() -> TaskProposal:
    return parse(document())


@pytest.fixture
def make_document():
    return document


@pytest.fixture
def section_replaced():
    return with_section
