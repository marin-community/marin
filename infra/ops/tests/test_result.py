# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from ops_workflow.result import EscalationSeverity, OpsOutcome, parse_ops_result

CASE_ID = "8c592336-b43c-4a5a-88bc-1f13dd861680"
TURN_ID = "40fea6a1-10a1-4d42-983e-a8dbfc3f971b"


def _content(**updates: object) -> str:
    payload: dict[str, object] = {
        "schema_version": 2,
        "case_id": CASE_ID,
        "ops_turn_id": TURN_ID,
        "outcome": "action_recommended",
        "summary": "The node image filesystem is still above its safe operating threshold.",
        "evidence": [{"claim": "Disk usage is 94%", "source": "kubectl describe node/g5bea54"}],
        "action_taken": "none",
        "recommended_next_step": "Inspect non-image disk consumers on the node.",
        "escalation": {"severity": "error", "reason": "Automated image cleanup freed no space."},
    }
    payload.update(updates)
    return json.dumps(payload)


def test_parse_ops_result_accepts_bounded_escalation_contract():
    result = parse_ops_result(_content(), case_id=CASE_ID, turn_id=TURN_ID)

    assert result.outcome == OpsOutcome.ACTION_RECOMMENDED
    assert result.escalation is not None
    assert result.escalation.severity == EscalationSeverity.ERROR
    assert result.escalation.reason == "Automated image cleanup freed no space."


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"ops_turn_id": "f55b69df-bd3d-425e-8e00-924f208f08eb"}, "does not match the active turn"),
        ({"outcome": "no_action"}, "only action_recommended or blocked"),
        ({"action_taken": "restarted node"}, "action_taken must be 'none'"),
    ],
)
def test_parse_ops_result_rejects_unsafe_or_misattributed_escalation(updates: dict[str, object], expected: str):
    with pytest.raises(ValueError, match=expected):
        parse_ops_result(_content(**updates), case_id=CASE_ID, turn_id=TURN_ID)
