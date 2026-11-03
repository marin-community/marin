# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Real proposals of the 2026-11-02 n=25 curriculum run, one per offered verification kind, parse and triage.

Each fixture in ``n25_proposals/`` is a ``slot-NN.proposal.json`` under
``s3://marin-us-east-02a/marin/taskforge/proposals/curriculum-v3-n25-20261102/run/`` (``code``:
``d21.soil.diagnosis.interpretation/slot-11``, ``answer_in_box``: ``d12.8.2/slot-10``, ``decision_model``:
``d10.7.3/slot-12``, ``llm_as_judge``: ``d45.electronic_rework/slot-19``), cut below 2 KB: its header verbatim but
for empty ``build`` and ``resources``, and each body section a contiguous window of the section's own text as long
as triage's ``MIN_SECTION_WORDS``, the grader section's window naming the evidence its kind needs.
"""

import asyncio
from pathlib import Path

import pytest

from taskforge.stage10_proposal.model import OFFERED_VERIFICATIONS, Verification, parse
from taskforge.stage20_triage.checks import CheckStatus
from taskforge.stage20_triage.preflight import preflight
from taskforge.stage20_triage.verdict import TriageDecision

FIXTURES = Path(__file__).parent / "n25_proposals"


@pytest.mark.parametrize("kind", OFFERED_VERIFICATIONS)
def test_a_run_proposal_of_each_offered_kind_parses_and_triage_accepts_it(kind: Verification):
    text = (FIXTURES / f"{kind}.json").read_text()
    assert len(text.encode()) < 2048

    proposal = parse(text)
    verdict = asyncio.run(preflight(proposal))

    assert proposal.header.verification is kind and proposal.header.composite == ()
    assert verdict.decision is TriageDecision.ACCEPT, verdict.reasons
    checks = {check.name: check.status for check in verdict.structural}
    assert checks["required_sections"] is CheckStatus.PASS
    assert checks["grader_matches_verification"] is CheckStatus.PASS
