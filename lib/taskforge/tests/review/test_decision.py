# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections import Counter

import pytest
from taskcompendium.grading_result import Outcome

from taskforge.build.step import StepRole
from taskforge.review.decision import (
    Accept,
    Reject,
    RejectKind,
    Repair,
    RepairBrief,
    Retry,
    load_decision,
    write_decision,
)
from taskforge.spec.controls import (
    REJECTION_CEILING,
    Control,
    ControlCategory,
    ControlKind,
    Expectation,
    Transcript,
    reply,
)
from taskforge.validate.calibration import Finding, FindingKind
from taskforge.validate.outcome import Cause

LEAK = Finding(
    kind=FindingKind.LEAK_PASSED,
    detail="read /workspace/.answer and submitted 395",
    roles=(StepRole.FIXTURES, StepRole.ENVIRONMENT),
    new_controls=(
        Control(
            id="adv-leak-0",
            kind=ControlKind.NEGATIVE,
            category=ControlCategory.REWARD_HACK,
            author="adversary/leak/0",
            payload=Transcript((reply("395"),)),
            expect=Expectation(Outcome.GRADED, reward_max=REJECTION_CEILING),
        ),
    ),
)


def decisions(summary):
    return [
        Accept(summary=summary()),
        Reject(
            kind=RejectKind.HOST,
            reasons=("machine_unsupported: 8",),
            summary=summary(causes=Counter({Cause.MACHINE_UNSUPPORTED: 8})),
        ),
        Reject(kind=RejectKind.BUDGET, reasons=("build revisions exhausted",), summary=None),
        Repair(
            program_digest="abc", brief=RepairBrief(findings=(LEAK,), failure="fix the leak"), invalidate=("fixtures",)
        ),
        Retry(cause=Cause.TOKEN_CONTRACT, count=2),
    ]


def test_every_decision_round_trips_through_decision_json(tmp_path, summary):
    for index, decision in enumerate(decisions(summary)):
        path = tmp_path / f"decision-{index}.json"
        write_decision(path, decision)
        assert load_decision(path) == decision


def test_unknown_decision_type_is_refused(tmp_path):
    path = tmp_path / "decision.json"
    path.write_text('{"type": "Escalate"}')
    with pytest.raises(ValueError, match="Escalate"):
        load_decision(path)
