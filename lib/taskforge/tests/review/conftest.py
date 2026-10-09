# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drafts and calibration summaries for the review tests."""

from collections import Counter
from collections.abc import Callable

import pytest
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend
from taskcompendium.models import AnswerType, PlainText, Source
from verifyit.spec import NumericSpec

from taskforge.builder.run import Provenance, TaskDraft
from taskforge.builder.step import Blob, CacheStatus, StepRecord, StepRole
from taskforge.review.rules import BandChoice, BandRule, BandRules
from taskforge.sandbox.factories import MachineHost
from taskforge.spec.draft import answer_grader, assemble, lower, session
from taskforge.validate.adversary import AdversaryRole, ClaimKind
from taskforge.validate.calibration import CalibrationBand, CalibrationSummary, DefectTier, Finding, RoleStats
from taskforge.validate.evidence import Complete, Incomplete, RewardStats
from taskforge.validate.outcome import Cause

SOURCE = Source(dataset="taskforge-review-tests", revision="1", row="0", importer_revision="1")
BAND = CalibrationBand(min_solve_rate=0.125, max_solve_rate=0.875)
K = 8
SESSION = session(
    max_turns=4,
    model_turn_timeout=None,
    command_timeout=10,
    tool_turn_timeout=20,
    total_turn_timeout=None,
    attempt_timeout=None,
    verifier_timeout=30,
    cleanup_timeout=10,
)
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
        PlainText(),
        answer_grader(NumericSpec(expected="395", tolerance_abs=0, tolerance_rel=0)),
        SOURCE,
        environment=None,
    )
    lowered = lower(
        task,
        host=MachineHost.LAPTOP,
        task_machine=None,
        verifier_machine=None,
        session=SESSION,
        factories={Backend.SHELLSIM.value: ShellSimMachineFactory()},
    )
    return TaskDraft(task, lowered, (), provenance())


def role_stats(passes: int) -> RoleStats:
    """Two graded trials of one role: ``passes`` claimed shortcuts noted past the threshold, the rest
    reporting no shortcut."""
    return RoleStats(
        required=2,
        graded=2,
        passes=passes,
        submissions=8,
        budget_spent=0,
        claims={ClaimKind.SHORTCUT: passes, ClaimKind.NO_SHORTCUT: 2 - passes, ClaimKind.NONE: 0},
        failed_audits=0,
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
def rules() -> BandRules:
    """Taskforge's own band rules: one repair per kind, then reject."""
    return BandRules(too_easy=BandRule(1, BandChoice.REJECT), too_hard=BandRule(1, BandChoice.REJECT))


@pytest.fixture
def summary() -> Callable[..., CalibrationSummary]:
    """Calibration summaries with every field but the ones a test is about filled consistently."""
    return make_summary
