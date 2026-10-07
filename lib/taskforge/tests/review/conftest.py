# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drafts and calibration summaries for the review tests."""

from collections import Counter
from collections.abc import Callable

import pytest
from taskcompendium.environment import EnvironmentKind, ExitCodeReward
from taskcompendium.execution import StageExecution
from taskcompendium.grading import numeric_answer
from taskcompendium.models import AnswerType, Source, StageRewardStrategy
from taskcompendium.submission import PlainText

from taskforge.build.run import Provenance, TaskDraft
from taskforge.build.step import Blob, CacheStatus, StepRecord, StepRole
from taskforge.spec.draft import assemble, environment, shell_verifier, stage, staged, task_execution
from taskforge.validate.adversary import AdversaryRole
from taskforge.validate.calibration import CalibrationBand, CalibrationSummary, DefectTier, Finding, RoleStats
from taskforge.validate.evidence import Complete, Incomplete, RewardStats
from taskforge.validate.outcome import Cause

SOURCE = Source(dataset="taskforge-review-tests", revision="1", row="0", importer_revision="1")
BAND = CalibrationBand(min_solve_rate=0.125, max_solve_rate=0.875)
K = 8
STEPS = (
    ("sources", StepRole.SOURCES),
    ("machine", StepRole.ENVIRONMENT),
    ("fixtures", StepRole.FIXTURES),
    ("grader", StepRole.GRADER),
    ("controls", StepRole.CONTROLS),
    ("instruction", StepRole.INSTRUCTIONS),
    ("assemble", StepRole.ASSEMBLE),
)


def provenance(steps: tuple[tuple[str, StepRole], ...] = STEPS) -> Provenance:
    blob = Blob(digest="0" * 64, size=0)
    return Provenance(
        item_id="item-1",
        proposal_digest="p" * 64,
        program_digest="program-digest-1",
        sdk_version="1",
        model="glm",
        policy_digest="policy",
        round=0,
        steps=tuple(
            StepRecord(name=name, role=role, key=f"key-{name}", status=CacheStatus.MISS, output=blob, resources=())
            for name, role in steps
        ),
        resources=(),
    )


@pytest.fixture
def draft() -> TaskDraft:
    task = assemble(
        "review-math",
        "What is 17 * 23 + 4? Reply with only the number.",
        AnswerType.NUMBER,
        environment(EnvironmentKind.NULL),
        numeric_answer("395", tolerance_abs=0, tolerance_rel=0),
        SOURCE,
        execution=task_execution(),
    )
    return TaskDraft(
        task=task,
        execution=task_execution(),
        convention=PlainText(id="plain_text"),
        controls=(),
        provenance=provenance(),
    )


@pytest.fixture
def staged_draft() -> TaskDraft:
    check = shell_verifier(("sh", "-c", '[ "$(cat /workspace/answer)" = 12 ]'), ExitCodeReward(), timeout=5)
    execution = task_execution(stages={"one": StageExecution(), "two": StageExecution()})
    task = assemble(
        "review-staged",
        "Write 12 to /workspace/answer.",
        AnswerType.FILE,
        environment(EnvironmentKind.SHELLSIM),
        staged(StageRewardStrategy.FINAL),
        SOURCE,
        execution=execution,
        stages=(stage("one", check), stage("two", check, instruction="Again.")),
    )
    return TaskDraft(
        task=task, execution=execution, convention=PlainText(id="plain_text"), controls=(), provenance=provenance()
    )


def role_stats(passes: int) -> RoleStats:
    """Two graded trials of one role: ``passes`` noted passes, the rest given up."""
    return RoleStats(
        required=2,
        graded=2,
        passes=passes,
        gave_up=2 - passes,
        exhausted=0,
        output_tokens=3000,
        tiers={DefectTier.REPAIR: 0, DefectTier.NOTED: passes, DefectTier.NONE: 2 - passes},
    )


def make_summary(
    findings: tuple[Finding, ...] = (),
    causes: Counter[Cause] | None = None,
    solved: int = 4,
    notes: tuple[Finding, ...] = (),
) -> CalibrationSummary:
    status = Incomplete(causes) if causes else Complete()
    graded = K - (sum(causes.values()) if causes else 0)
    return CalibrationSummary(
        task_digest="t" * 64,
        policy_digest="v" * 64,
        band=BAND,
        status=status,
        k=K,
        solver=RewardStats(graded=graded, mean_reward=solved / graded if graded else None, solved=solved, timed_out=0),
        solve_rate=solved / graded if graded else None,
        controls_met=("pos-0", "neg-0"),
        controls_violated=(),
        controls_ungraded=(),
        roles={role: role_stats(passes=0) for role in AdversaryRole},
        findings=findings,
        assessments=(),
        notes=notes,
    )


@pytest.fixture
def summary() -> Callable[..., CalibrationSummary]:
    """Calibration summaries with every field but the ones a test is about filled consistently."""
    return make_summary
