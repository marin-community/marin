# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lower a built task for the host validation runs on.

Validation supplies the limits once here. The task and verifier machines share the build stage's
startup settings; the grading deadline includes the grader's own timeout. The resulting messages-trace
spec is passed unchanged to probes and trials.
"""

from collections.abc import Mapping

from rolloutengine.models import RolloutLimits, TaskRolloutSpec, TraceMode
from shellbox.machine import MachineFactory
from taskcompendium.models import TaskSpec

from taskforge.sandbox.factories import MachineHost, grading_environment
from taskforge.spec import draft
from taskforge.spec.draft import GRADING_MACHINE, SHELL_CAPABILITY


def lower(
    task: TaskSpec, host: MachineHost, factories: Mapping[str, MachineFactory], limits: RolloutLimits
) -> TaskRolloutSpec:
    """The ``TaskRolloutSpec`` that runs ``task`` on ``host``'s ``factories``.

    A task with the shell capability gets a task machine, and a grader with an environment gets a
    verifier machine, both with ``GRADING_MACHINE`` settings.

    Raises:
        ValueError: RolloutEngine rejects the lowered spec.
        NotImplementedError: RolloutEngine cannot run the lowered spec.
    """
    shell = SHELL_CAPABILITY in task.environment_requirements.capabilities
    spec = draft.lower(
        task,
        host=host,
        task_machine=GRADING_MACHINE if shell else None,
        verifier_machine=None if grading_environment(task) is None else GRADING_MACHINE,
        limits=limits.model_copy(update={"verifier_timeout": draft.verifier_timeout(task.grader)}),
        factories=factories,
    )

    return spec.model_copy(update={"trace": TraceMode.MESSAGES})
