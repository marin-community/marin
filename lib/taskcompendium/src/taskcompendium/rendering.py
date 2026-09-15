# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render one TaskSpec into a lowering's model-visible instructions."""

from typing import Protocol

from taskcompendium.extraction import rendering_instruction
from taskcompendium.models import (
    AssistantFinal,
    FileSubmission,
    FinalActionSubmission,
    FinalState,
    JsonPath,
    Rejected,
    Rendering,
    TaskSpec,
    XmlPath,
)


class TaskFamily(Protocol):
    """Source/family adapter that resolves a stable source key into a TaskSpec."""

    def instantiate(self, key: str) -> TaskSpec | Rejected:
        """Construct one semantic instance, including private verification data."""
        ...


def render_instruction(specification: TaskSpec, rendering: Rendering, step_index: int = 0) -> str:
    """Render one model-visible instruction without evaluation machinery."""
    submission = rendering.submission
    if isinstance(submission, FinalState | FinalActionSubmission):
        return specification.steps[step_index].instructions
    suffix = rendering_instruction(submission.extractor)
    if isinstance(submission, FileSubmission):
        suffix += f" Write your submission to {submission.path}."
    return f"{specification.steps[step_index].instructions.rstrip()}\n\n{suffix}\n"


def result_tags(renderings: tuple[Rendering, ...]) -> tuple[str, ...]:
    """Return output-encoding tags contributed by a lowering's renderings."""
    tags: set[str] = set()
    for rendering in renderings:
        submission = rendering.submission
        if isinstance(submission, FileSubmission):
            tags.add("result:file")
        if isinstance(submission, (FileSubmission, AssistantFinal)):
            if isinstance(submission.extractor, JsonPath):
                tags.add("result:json")
            elif isinstance(submission.extractor, XmlPath):
                tags.add("result:xml")
    return tuple(sorted(tags))
