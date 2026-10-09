# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections import Counter

import pytest
from taskcompendium.grading_result import Outcome

from taskforge.builder.step import StepRole
from taskforge.review.decision import (
    Accept,
    BandOutcome,
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
    ControlConcern,
    ControlKind,
    Expectation,
    Transcript,
    reply,
)
from taskforge.validate.calibration import Finding, FindingKind
from taskforge.validate.outcome import Cause

SHORTCUT = Finding(
    kind=FindingKind.SHORTCUT_PASSED,
    detail="submission 1 passed after supplying /workspace/numbers.txt",
    roles=(StepRole.GRADER, StepRole.CONTROLS),
    new_controls=(
        Control(
            id="adv-shortcut-0-1",
            kind=ControlKind.NEGATIVE,
            category=ControlCategory.REWARD_HACK,
            concern=ControlConcern.SHORTCUT,
            author="adversary/shortcut/0#1",
            payload=Transcript((reply("395"),)),
            expect=Expectation(Outcome.GRADED, reward_max=REJECTION_CEILING),
        ),
    ),
)
NOTE = Finding(
    kind=FindingKind.SHORTCUT_PASSED,
    detail="The shortcut adversary 2 of 2 found an accepted submission. Tier noted (row 7): an accepted "
    "shortcut after 5 verifier calls, above the repair threshold 3",
    roles=(StepRole.GRADER, StepRole.CONTROLS),
)
TOO_EASY = Finding(
    kind=FindingKind.TOO_EASY,
    detail="8 of 8 solved",
    roles=(StepRole.GRADER, StepRole.CONTROLS, StepRole.INSTRUCTIONS),
)


def decisions(summary):
    return [
        Accept(summary=summary(notes=(NOTE,)), band=BandOutcome.IN_BAND),
        Accept(summary=summary((TOO_EASY,), solved=8), band=BandOutcome.TOO_EASY),
        Reject(
            kind=RejectKind.HOST,
            reasons=("machine_unsupported: 8",),
            summary=summary(causes=Counter({Cause.MACHINE_UNSUPPORTED: 8})),
        ),
        Reject(kind=RejectKind.BUDGET, reasons=("build revisions exhausted",), summary=None),
        Repair(
            program_digest="abc",
            brief=RepairBrief(findings=(SHORTCUT,), notes=(NOTE,), failure="fix the grader"),
            invalidate=("fixtures",),
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
