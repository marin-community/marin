# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The context a build step receives, and the exceptions a build raises when it stops.

A build ends in one of three ways besides a draft. A step raises ``BuildFailure`` when the task
cannot be built as written, which the next round's author can fix. ``BuildInfrastructureFailure``
says the host failed instead (``infrastructure.py`` classifies host errors), so no revision of the
program can fix it. ``BuildExpired`` says a build, step or grader ran past its clock, which is the
fault of neither until someone reads the record, so a driver holds the workpiece.
"""

from dataclasses import dataclass
from enum import StrEnum

from taskforge.llm.client import ModelClient
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.recording import CallLedger
from taskforge.stage40_build_with_program.services import RLEnvBuildServices


class BuildFailure(Exception):
    """Raised by a step when the task cannot be built as written.

    A step raises ``BuildFailure(message)``. The runner then re-raises it with ``step`` set to that
    step's name, so the next round's author learns where the build stopped.
    """

    def __init__(self, message: str, step: str | None = None):
        super().__init__(message if step is None else f"{step}: {message}")
        self.message = message
        self.step = step


class BuildInfrastructureFailure(Exception):
    """The build host failed (a sandbox did not start, an image did not pull, a pod died), not the program.

    Not a ``BuildFailure``: the original host error is in the ``__cause__`` chain.

    Attributes:
        cause: An ``infrastructure.InfrastructureCause`` value, the label of the failure.
        message: The host's account of the failure.
    """

    def __init__(self, cause: str, message: str):
        super().__init__(f"{cause}: {message}")
        self.cause = cause
        self.message = message


class ExpiredClock(StrEnum):
    """The clock a build ran past."""

    WALL = "wall"
    """The build's wall clock, ``build_process.run_build_in_process``'s ``wall_timeout``."""
    STEP = "step"
    """One step's clock, ``run.STEP_TIMEOUT``."""
    GRADER = "grader"
    """The grader command's own timeout, on the reference answer, in both of ``grade_candidate``'s gradings."""


class BuildExpired(Exception):
    """A build, step or grader ran past its clock.

    Whether Taskforge was slow or the program was is unknown until someone reads the record, so this is
    neither the program's failure nor the host's: the coordinator marks the workpiece HELD at its current
    stage and spends no revision. A step raises ``BuildExpired(message, clock)`` and the runner
    re-raises it with ``step`` set to that step's name; ``str()`` is ``message``.
    """

    def __init__(self, message: str, clock: ExpiredClock, step: str | None = None):
        super().__init__(message)
        self.message = message
        self.clock = clock
        self.step = step


@dataclass(frozen=True)
class RLEnvBuildContext:
    """The first argument of every step: the workpiece and round being built, the model, and the host.

    ``run_program`` sets ``step`` to the running step's name, so the model calls and tool calls a step
    makes are recorded under it. A step calls ``context.model.complete_structured(messages, Output,
    context.sampling)``, or the SDK's ``structured``, which records the call.

    Attributes:
        workpiece_id: The workpiece's file-safe id.
        round: The build round.
        model: The model steps call.
        sampling: The sampling policy steps pass to ``model``.
        services: Machines, images, web tools and the ledger.
        step: The name of the running step.
    """

    workpiece_id: str
    round: int
    model: ModelClient
    sampling: LLMPolicy
    services: RLEnvBuildServices
    step: str


def call_ledger(context: RLEnvBuildContext) -> CallLedger:
    """Where the running step records its model and tool calls: its workpiece, round and step name."""
    return CallLedger(
        ledger=context.services.ledger, workpiece_id=context.workpiece_id, round=context.round, step=context.step
    )
