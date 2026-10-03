# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in episode controls for mock calendar and shell runtime checks.

Tests and prototypes call these controls explicitly; the pinned source graph
does not use them.
"""

import asyncio
from dataclasses import replace

from taskcompendium.grading import Outcome
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.models import CheckResult, CheckStatus, CheckSuite, VerificationReport
from taskcompendium.pipeline.verification import PLAIN
from taskcompendium.runtime.calendar import calendar_controls
from taskcompendium.runtime.controls import Control, tool_turn
from taskcompendium.runtime.episode import ScriptedActor, run_episode
from taskcompendium.runtime.models import EnvironmentFactory, Termination
from taskcompendium.runtime.shell import CONTROL_PATH, OUTPUT_PATH, ShellFactory
from taskcompendium.verifiers.base import GradingAttempt
from taskcompendium.verifiers.dispatch import resolve_custom_verifier
from taskcompendium.verifiers.runtime import CalendarStateVerifier, CaptureOutputVerifier


async def check_episodes(task: TaskSpec, factory: EnvironmentFactory, *, max_steps: int) -> VerificationReport:
    verifier = resolve_custom_verifier(task.verifier)
    if isinstance(verifier, CalendarStateVerifier):
        controls = calendar_controls(verifier)
    elif isinstance(verifier, CaptureOutputVerifier) and isinstance(factory, ShellFactory):
        wrong = (
            "__incorrect_record__"
            if verifier.expected_output.strip() != "__incorrect_record__"
            else "__another_record__"
        )
        controls = (
            Control("noop", (), 0.0),
            Control("reference", (tool_turn("Bash", {"command": f"bash {CONTROL_PATH}"}),), 1.0),
            Control("perturbed", (tool_turn("Bash", {"command": f"printf '%s\\n' '{wrong}' > {OUTPUT_PATH}"}),), 0.0),
            Control("reset", (tool_turn("Bash", {"command": f"bash {CONTROL_PATH}"}),), 1.0),
        )
    else:
        raise ValueError("No episode controls for this verifier and runtime")
    results, rollouts = [], []
    for control in controls:
        # The scripted oracle alone gets the private witness files.
        bound_factory = factory
        if isinstance(factory, ShellFactory) and control.name in {"reference", "reset"}:
            bound_factory = replace(factory, mounted_roles=("worker", "oracle"))
        rollout = await run_episode(
            task, ScriptedActor(control.responses), bound_factory, max_steps=max_steps, control=control.name
        )
        rollouts.append(rollout)
        if rollout.termination == Termination.INFRA_ERROR:
            results.append(CheckResult(check=control.name, status=CheckStatus.INFRA_ERROR, detail=rollout.detail))
            continue
        grade = verifier.grade(GradingAttempt(PLAIN, rollout.events, rollout.evidence()))
        passed = (
            rollout.termination == Termination.FINAL_MESSAGE
            and grade.status == Outcome.GRADED
            and grade.reward == control.expected_reward
        )
        status = (
            CheckStatus.INFRA_ERROR
            if grade.status == Outcome.INFRA_ERROR
            else CheckStatus.PASS if passed else CheckStatus.FAIL
        )
        results.append(
            CheckResult(
                check=control.name,
                status=status,
                detail=f"{rollout.termination}; {grade.status}: reward={grade.reward}; {grade.error or ''}",
            )
        )
    return VerificationReport(results, tuple(rollouts))


def episode_suite(factory: EnvironmentFactory, *, max_steps: int) -> CheckSuite:
    return CheckSuite(
        id="episode-controls",
        revision="1",
        parameters={"runtime": factory.identity, "max_steps": max_steps},
        run=lambda task: asyncio.run(check_episodes(task, factory, max_steps=max_steps)),
    )
