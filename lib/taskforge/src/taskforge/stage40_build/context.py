# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The context a build step receives, and the exception a step raises when the task cannot be built."""

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


@dataclass(frozen=True)
class RLEnvBuildContext:
    """The first argument of every step: the item being built, and the model with its sampling policy.

    A step calls ``context.model.complete_structured(messages, Output, context.sampling)``.
    """

    item_id: str
    model: ModelClient
    sampling: LLMPolicy
