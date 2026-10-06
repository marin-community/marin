# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drafts and calibration summaries for the review tests.

``validate.calibration`` and ``validate.adversary`` belong to layer 08, which is not on this branch's
base yet. Until it is, ``_install_design_calibration`` registers test-local modules under those names,
built from the dataclass definitions in design section 3.5, so the tests import the real module paths.
Integration deletes the shim; the tests stay as written.
"""

import importlib.util
import sys
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import ModuleType

import pytest
from taskcompendium.environment import EnvironmentKind, ExitCodeReward
from taskcompendium.execution import StageExecution
from taskcompendium.grading import numeric_answer
from taskcompendium.models import AnswerType, Source, StageRewardStrategy
from taskcompendium.submission import PlainText

from taskforge.build.run import Provenance, TaskDraft
from taskforge.build.step import Blob, CacheStatus, StepRecord, StepRole
from taskforge.spec.controls import Control
from taskforge.spec.draft import assemble, environment, shell_verifier, stage, staged, task_execution
from taskforge.validate.evidence import Complete, Incomplete, RewardStats
from taskforge.validate.outcome import Cause


def _install_design_calibration() -> None:
    if importlib.util.find_spec("taskforge.validate.calibration") is not None:
        return

    class AdversaryRole(StrEnum):
        SHORTCUT = "shortcut"
        LEAK = "leak"
        AMBIGUITY = "ambiguity"

    @dataclass(frozen=True)
    class CalibrationBand:
        min_solve_rate: float
        max_solve_rate: float

    class FindingKind(StrEnum):
        CONTROL_VIOLATED = "control_violated"
        SHORTCUT_PASSED = "shortcut_passed"
        LEAK_PASSED = "leak_passed"
        AMBIGUOUS = "ambiguous"
        TASK_DEFECT = "task_defect"
        TOO_EASY = "too_easy"
        TOO_HARD = "too_hard"

    decisive_kinds = frozenset(
        {
            FindingKind.CONTROL_VIOLATED,
            FindingKind.SHORTCUT_PASSED,
            FindingKind.LEAK_PASSED,
            FindingKind.AMBIGUOUS,
            FindingKind.TASK_DEFECT,
        }
    )
    finding_roles = {
        FindingKind.CONTROL_VIOLATED: (StepRole.GRADER, StepRole.CONTROLS),
        FindingKind.SHORTCUT_PASSED: (StepRole.GRADER, StepRole.CONTROLS),
        FindingKind.LEAK_PASSED: (StepRole.FIXTURES, StepRole.ENVIRONMENT, StepRole.INSTRUCTIONS),
        FindingKind.AMBIGUOUS: (StepRole.INSTRUCTIONS,),
        FindingKind.TOO_EASY: (StepRole.GRADER, StepRole.CONTROLS, StepRole.INSTRUCTIONS),
        FindingKind.TOO_HARD: (StepRole.INSTRUCTIONS, StepRole.FIXTURES),
    }

    @dataclass(frozen=True)
    class Finding:
        kind: FindingKind
        detail: str
        roles: tuple[StepRole, ...]
        new_controls: tuple[Control, ...] = ()

    @dataclass(frozen=True)
    class RoleStats:
        required: int
        graded: int
        passes: int
        sentinel_replies: int

    @dataclass(frozen=True)
    class CalibrationSummary:
        task_digest: str
        policy_digest: str
        band: CalibrationBand
        status: Complete | Incomplete
        k: int
        solver: RewardStats
        solve_rate: float | None
        controls_met: tuple[str, ...]
        controls_violated: tuple[str, ...]
        controls_ungraded: tuple[tuple[str, Cause], ...]
        roles: Mapping[AdversaryRole, RoleStats]
        findings: tuple[Finding, ...]

        @property
        def decisive(self) -> tuple[Finding, ...]:
            return tuple(finding for finding in self.findings if finding.kind in decisive_kinds)

        @property
        def calibrated(self) -> bool:
            return isinstance(self.status, Complete) and not self.findings

    adversary = ModuleType("taskforge.validate.adversary")
    adversary.AdversaryRole = AdversaryRole
    calibration = ModuleType("taskforge.validate.calibration")
    for name, value in {
        "CalibrationBand": CalibrationBand,
        "FindingKind": FindingKind,
        "DECISIVE": decisive_kinds,
        "FINDING_ROLES": finding_roles,
        "Finding": Finding,
        "RoleStats": RoleStats,
        "CalibrationSummary": CalibrationSummary,
    }.items():
        setattr(calibration, name, value)
    for module in (adversary, calibration):
        for value in vars(module).values():
            if isinstance(value, type):
                value.__module__ = module.__name__
        sys.modules[module.__name__] = module


_install_design_calibration()

from taskforge.validate.adversary import AdversaryRole  # noqa: E402 - after the layer-08 shim
from taskforge.validate.calibration import (  # noqa: E402
    CalibrationBand,
    CalibrationSummary,
    Finding,
    RoleStats,
)

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


def make_summary(
    findings: tuple[Finding, ...] = (),
    causes: Counter[Cause] | None = None,
    solved: int = 4,
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
        roles={role: RoleStats(required=2, graded=2, passes=0, sentinel_replies=2) for role in AdversaryRole},
        findings=findings,
    )


@pytest.fixture
def summary() -> Callable[..., CalibrationSummary]:
    """Calibration summaries with every field but the ones a test is about filled consistently."""
    return make_summary
