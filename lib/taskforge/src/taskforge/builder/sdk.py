# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""What a builder step receives, and how it says the task cannot be built.

Every step is called as ``await fn(b, **inputs)``: ``b`` is a ``BuildContext`` and ``inputs`` are the
outputs of the steps it declares (``builder.step``), plus the ``TaskProposal`` when it declares
``"proposal"``. The context is immutable and holds no step state: ``b.model`` already records its
calls under the running step, because the runner hands each step its own copy.
"""

from dataclasses import dataclass

from taskforge.llm.recording import RecordedModel


class BuildFailure(Exception):
    """The program decided the task cannot be built as written; ``step`` names where.

    A step raises ``BuildFailure(message)``; the runner re-raises it with the step's name.
    """

    def __init__(self, message: str, step: str | None = None):
        super().__init__(message if step is None else f"{step}: {message}")
        self.message = message
        self.step = step


@dataclass(frozen=True)
class BuildContext:
    """A step's ``b``: the item it builds for and the model it calls (``b.model.structured``)."""

    item_id: str
    model: RecordedModel
