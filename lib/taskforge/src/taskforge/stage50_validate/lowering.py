# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lower a built task for the host validation runs on.

A draft holds the semantic ``TaskSpec`` only, so validation picks the machines and the limits. The
task machine and a grader's verifier machine get ``TASK_MACHINE``: no network, the factory's limits,
the image's user. The turn budget and deadlines of ``LIMITS`` are placeholders that ``TrialPlan.deadlines``
and ``EngineSettings.apply`` replace for every trial; its verifier deadline, ``VERIFIER_TIMEOUT``,
stands. The shell command timeout is not here: ``EngineSettings`` registers it with the shell harness.
"""

from collections.abc import Mapping

from rolloutengine.models import TaskRolloutSpec
from shellbox.machine import MachineFactory
from taskcompendium.models import TaskSpec

from taskforge.sandbox.factories import SANDBOX_SCHEDULING_TIMEOUT, MachineHost, grading_environment
from taskforge.spec import draft
from taskforge.spec.draft import SHELL_CAPABILITY
from taskforge.stage40_build_with_program.sdk import GRADER_TIMEOUT

STARTUP_TIMEOUT = 600.0
"""Seconds the task machine and the verifier machine may take to start."""
VERIFIER_TIMEOUT = SANDBOX_SCHEDULING_TIMEOUT + GRADER_TIMEOUT
"""The trial's grading deadline: a verifier machine's scheduling budget, then a template grader's own timeout.

It stays behind the factory's scheduling timeout so a sandbox that never starts is the factory's
``TimeoutError``, which validation classifies, and not the deadline.
"""
CLEANUP_TIMEOUT = 120.0
TASK_MACHINE = draft.machine(startup_timeout=STARTUP_TIMEOUT)
LIMITS = draft.rollout_limits(
    max_turns=64,
    model_turn_timeout=None,
    tool_turn_timeout=None,
    total_turn_timeout=None,
    attempt_timeout=None,
    verifier_timeout=VERIFIER_TIMEOUT,
    cleanup_timeout=CLEANUP_TIMEOUT,
)


def lower(task: TaskSpec, host: MachineHost, factories: Mapping[str, MachineFactory]) -> TaskRolloutSpec:
    """The ``TaskRolloutSpec`` that runs ``task`` on ``host``'s ``factories``.

    A task with the shell capability gets a task machine, and a grader with an environment gets a
    verifier machine, both with ``TASK_MACHINE`` settings.

    Raises:
        ValueError: RolloutEngine rejects the lowered spec.
        NotImplementedError: RolloutEngine cannot run the lowered spec.
    """
    shell = SHELL_CAPABILITY in task.environment_requirements.capabilities
    return draft.lower(
        task,
        host=host,
        task_machine=TASK_MACHINE if shell else None,
        verifier_machine=None if grading_environment(task) is None else TASK_MACHINE,
        limits=LIMITS,
        factories=factories,
    )
