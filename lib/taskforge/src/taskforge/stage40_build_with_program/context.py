# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The context a build step receives, and the exceptions a build raises when it stops.

A step raises ``BuildFailure`` when the task cannot be built as written. The code driving the build
host raises ``BuildInfrastructureFailure`` when the host fails while it runs the program.
"""

from dataclasses import dataclass

from taskforge.llm.client import ModelClient
from taskforge.llm.policy import LLMPolicy


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

    Attributes:
        cause: A short label for the failure, such as ``image_pull``.
        message: The host's account of the failure.
    """

    def __init__(self, cause: str, message: str):
        super().__init__(f"{cause}: {message}")
        self.cause = cause
        self.message = message


@dataclass(frozen=True)
class RLEnvBuildContext:
    """The first argument of every step: the workpiece being built, and the model with its sampling policy.

    A step calls ``context.model.complete_structured(messages, Output, context.sampling)``.
    """

    workpiece_id: str
    model: ModelClient
    sampling: LLMPolicy
