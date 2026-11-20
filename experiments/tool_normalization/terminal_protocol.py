# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Identify the pinned Terminus JSON envelope separately from the user task."""

from dataclasses import dataclass
from pathlib import Path

PREFIX = Path(__file__).with_name("terminus-json-prefix.txt").read_text()
SCREEN_SEPARATOR = "\n\nCurrent terminal state:\n"
CONFIRMATION = (
    "Are you sure you want to mark the task as complete? This will trigger your solution to be graded and you won't "
    'be able to make any further corrections. If so, include "task_complete": true in your JSON response again.'
)


@dataclass(frozen=True)
class TerminalProtocol:
    task_index: int
    task: str
    observation_indices: frozenset[int]


def terminal_protocol(original: list[dict]) -> TerminalProtocol | None:
    """Recognize only the complete known prompt; preserve other user messages."""
    first_user = next((i for i, message in enumerate(original) if message["role"] == "user"), None)
    if first_user is None:
        return None
    content = original[first_user]["content"].lstrip()
    if not content.startswith(PREFIX):
        return None
    task, separator, screen = content[len(PREFIX) :].rpartition(SCREEN_SEPARATOR)
    if not separator or not task.strip() or not screen.startswith("Current Terminal Screen:\n"):
        raise ValueError("Recognized Terminus prompt lacks its task or initial screen boundary")
    observations = set()
    for index in range(first_user + 1, len(original)):
        message = original[index]
        if message["role"] != "user" or original[index - 1]["role"] != "assistant":
            continue
        text = message["content"].strip()
        if (
            text.startswith("New Terminal Output:\n")
            or (text.startswith("Current terminal state:\nNew Terminal Output:") and text.endswith(CONFIRMATION))
            or (text.startswith("Previous response had warnings:\nWARNINGS:") and "\nNew Terminal Output:" in text)
            or (
                text.startswith("Previous response had parsing errors:\nERROR:")
                and text.endswith("Please fix these issues and provide a proper JSON response.")
            )
        ):
            observations.add(index)
    return TerminalProtocol(first_user, task, frozenset(observations))
