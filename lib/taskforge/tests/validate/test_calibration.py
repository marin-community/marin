# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""summarize: band findings over complete evidence, decisive findings regardless, adversary passes as controls."""

from collections import Counter

import pytest
from rigging.timing import ExponentialBackoff
from taskcompendium.execution import TaskExecution
from taskcompendium.grading_result import Outcome as GradeStatus
from taskcompendium.models import TextMessage
from taskcompendium.submission import PlainText

from taskforge.build.step import StepRole
from taskforge.ledger.jsonl import JsonlLedger
from taskforge.spec.controls import REJECTION_CEILING, ControlKind, Transcript
from taskforge.validate.adversary import AdversaryRole
from taskforge.validate.calibration import DECISIVE, FindingKind, RoleStats, load_summary, summarize, write_summary
from taskforge.validate.evidence import Complete, Incomplete
from taskforge.validate.outcome import Cause, Outcome, TrialKind, Ungraded
from taskforge.validate.run import ValidationEvidence, control_outcome
from taskforge.validate.trials import Deadlines, EngineSettings, TrialPlan, run_trial

DIGEST = "ab" * 32
UNAVAILABLE = Ungraded(Cause.MODEL_UNAVAILABLE, "router drained", None)
SETUP_FAILED = Ungraded(Cause.TASK_SETUP, "setup command exited 3", None)


@pytest.fixture
def answer(tmp_path, math_task, fakes):
    """Runs the math task once with a scripted final reply and returns the outcome."""
    count = 0

    async def run(reply: str) -> Outcome:
        nonlocal count
        count += 1
        plan = TrialPlan(
            item_id="item",
            round=0,
            kind=TrialKind.SOLVER,
            k=1,
            deadlines=Deadlines(agent_timeout=30, attempt_timeout=60),
            max_retries=0,
            token_contract_retries=0,
            retry_backoff=ExponentialBackoff(initial=0.001, maximum=0.001),
            evidence_dir=tmp_path,
            ledger=JsonlLedger(tmp_path / "ledger"),
            first_attempt=0,
        )
        settings = EngineSettings({}, {}, 4, 10, 10, (PlainText(id="plain"),))
        return await run_trial(
            math_task, TaskExecution(), plan, settings, fakes.script_model([fakes.text(reply)]), str(count)
        )

    return run


def evidence(controls=(), solver=(), adversaries=None) -> ValidationEvidence:
    return ValidationEvidence(DIGEST, tuple(controls), tuple(solver), adversaries or {})


async def test_band_findings_need_complete_evidence_with_k_graded_solver_trials(answer, rounds):
    policy = rounds.policy(k=3)
    right, wrong = await answer("395"), await answer("391")

    too_easy = summarize(evidence(solver=(right, right, right)), policy)
    too_hard = summarize(evidence(solver=(wrong, wrong, wrong)), policy)
    calibrated = summarize(evidence(solver=(right, wrong, wrong)), policy)
    incomplete = summarize(evidence(solver=(wrong, wrong, UNAVAILABLE)), policy)
    short = summarize(evidence(solver=(wrong, wrong)), policy)

    assert [f.kind for f in too_easy.findings] == [FindingKind.TOO_EASY]
    assert too_easy.findings[0].roles == (StepRole.GRADER, StepRole.CONTROLS, StepRole.INSTRUCTIONS)
    assert [f.kind for f in too_hard.findings] == [FindingKind.TOO_HARD]
    assert not too_easy.decisive and not too_hard.decisive
    assert calibrated.findings == () and calibrated.calibrated and calibrated.solve_rate == pytest.approx(1 / 3)
    assert incomplete.status == Incomplete(Counter({Cause.MODEL_UNAVAILABLE: 1}))
    assert incomplete.findings == () and not incomplete.calibrated
    assert isinstance(short.status, Complete) and short.findings == () and short.solver.graded == 2


async def test_decisive_findings_are_reported_when_other_trials_are_ungraded(answer, rounds, math_controls):
    correct = next(c for c in math_controls if c.kind is ControlKind.POSITIVE)
    violated = control_outcome(correct, await answer("391"))
    adversaries = {AdversaryRole.LEAK: (SETUP_FAILED, SETUP_FAILED)}

    summary = summarize(evidence((violated,), (UNAVAILABLE,), adversaries), rounds.policy(k=3))

    assert [f.kind for f in summary.findings] == [FindingKind.CONTROL_VIOLATED, FindingKind.TASK_DEFECT]
    assert summary.decisive == summary.findings and {f.kind for f in summary.findings} <= DECISIVE
    violation, defect = summary.findings
    assert violation.roles == (StepRole.GRADER, StepRole.CONTROLS) and "control/correct" in violation.detail
    assert defect.roles == (StepRole.ENVIRONMENT, StepRole.FIXTURES)
    assert "adversary/leak/0" in defect.detail and "adversary/leak/1" in defect.detail
    assert summary.controls_violated == ("correct",)
    assert summary.status == Incomplete(Counter({Cause.MODEL_UNAVAILABLE: 1, Cause.TASK_SETUP: 2}))


async def test_shortcut_and_leak_passes_become_negative_controls_and_sentinels_are_counted(answer, rounds):
    adversaries = {
        AdversaryRole.SHORTCUT: (await answer("395"), await answer("NO_SHORTCUT_FOUND")),
        AdversaryRole.LEAK: (await answer("NO_LEAK_FOUND"), await answer("NO_LEAK_FOUND")),
        AdversaryRole.AMBIGUITY: (await answer("395"), await answer("391")),
    }

    summary = summarize(evidence(adversaries=adversaries), rounds.policy(adversary_k=2))

    assert [f.kind for f in summary.findings] == [FindingKind.SHORTCUT_PASSED, FindingKind.AMBIGUOUS]
    shortcut, ambiguous = summary.findings
    assert ambiguous.new_controls == () and ambiguous.roles == (StepRole.INSTRUCTIONS,)
    (control,) = shortcut.new_controls
    assert control.kind is ControlKind.NEGATIVE and control.author == "adversary/shortcut/0"
    assert control.expect.status is GradeStatus.GRADED and control.expect.reward_max == REJECTION_CEILING
    assert control.payload == Transcript((TextMessage(role="assistant", content="395"),))
    assert summary.roles == {
        AdversaryRole.SHORTCUT: RoleStats(required=2, graded=2, passes=1, sentinel_replies=1),
        AdversaryRole.LEAK: RoleStats(required=2, graded=2, passes=0, sentinel_replies=2),
        AdversaryRole.AMBIGUITY: RoleStats(required=2, graded=2, passes=1, sentinel_replies=0),
    }


async def test_a_summary_reads_back_from_calibration_json(tmp_path, answer, rounds, math_controls):
    correct = next(c for c in math_controls if c.kind is ControlKind.POSITIVE)
    summary = summarize(
        evidence(
            (control_outcome(correct, await answer("391")),),
            (await answer("395"), UNAVAILABLE),
            {AdversaryRole.SHORTCUT: (await answer("395"),)},
        ),
        rounds.policy(k=2, adversary_k=1),
    )
    path = tmp_path / "calibration.json"

    write_summary(path, summary)

    assert load_summary(path) == summary
