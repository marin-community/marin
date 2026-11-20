# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Proposal documents as the driver decodes them: JSON round trip, canonical digest, legacy headers."""

import json
from dataclasses import replace

from lib.taskforge.tests.proposal_fakes import BODY

from taskforge.model.proposal import TASK_PROPOSAL, BuildItem, TaskProposal, render


def test_json_round_trip_preserves_typed_content_and_digest(proposal: TaskProposal) -> None:
    authored = json.loads(render(proposal))
    authored["body"] = authored["body"].replace("\n", "\r\n") + "  \n"
    authored["header"]["research"][0]["purpose"] += "   "
    decoded = TASK_PROPOSAL.validate_json(json.dumps(authored, sort_keys=True, indent=4))
    assert decoded.header == proposal.header
    assert render(decoded) == render(proposal)
    assert decoded.digest == proposal.digest
    assert replace(decoded, body=BODY.replace("kernel", "image")).digest != decoded.digest


def test_escaped_strings_and_fenced_headings_survive_round_trip(proposal: TaskProposal) -> None:
    body = BODY.replace(
        "Find the kernel of the map.", 'Use "quoted" Unicode λ and path C:\\tmp.\n```markdown\n## Build plan\n```'
    )
    expected = replace(
        proposal, body=body, header=replace(proposal.header, build=(BuildItem('a"b', "line one\nline two\\three"),))
    )
    assert TASK_PROPOSAL.validate_json(render(expected)) == expected


def test_existing_catalog_drops_obsolete_solver_machine(proposal: TaskProposal) -> None:
    document = json.loads(render(proposal))
    document["header"]["solver_machine"] = "packages"
    decoded = TASK_PROPOSAL.validate_json(json.dumps(document))
    assert decoded == proposal
    assert "solver_machine" not in json.loads(render(decoded))["header"]
