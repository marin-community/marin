# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render a fixed semantic instance without selecting a harness or environment."""

from typing import Protocol

import msgspec

from taskcompendium.extraction import rendering_instruction, validate_extractor
from taskcompendium.grading_paths import submission_relative
from taskcompendium.models import (
    AssistantFinal,
    Capability,
    FileSubmission,
    FinalActionSubmission,
    FinalState,
    JsonPath,
    PlainText,
    PublicResource,
    Rejected,
    Rendering,
    Resource,
    ResourceRole,
    Task,
    TaskSpecification,
    TaskStep,
    XmlPath,
)
from taskcompendium.serialization import specification_hash


class TaskSpec(Protocol):
    """Source/family adapter that resolves a stable source key into an instance."""

    def instantiate(self, key: str) -> TaskSpecification | Rejected:
        """Construct the semantic instance, including private verification data."""
        ...


def public_resources(resources: tuple[Resource, ...]) -> tuple[PublicResource, ...]:
    return tuple(PublicResource(r.path, r.content, r.executable) for r in resources if ResourceRole.AGENT in r.roles)


def render_instruction(specification: TaskSpecification, protocol: Rendering, step_index: int = 0) -> str:
    """Render task and output requirements without exposing evaluation machinery."""
    submission = protocol.submission
    if isinstance(submission, FinalState):
        return specification.steps[step_index].instructions
    if isinstance(submission, FinalActionSubmission):
        return specification.steps[step_index].instructions
    suffix = rendering_instruction(submission.extractor)
    if isinstance(submission, FileSubmission):
        suffix += f" Write your submission to {submission.path}."
    return f"{specification.steps[step_index].instructions.rstrip()}\n\n{suffix}\n"


def result_tags(renderings: tuple[Rendering, ...]) -> tuple[str, ...]:
    """Return output-encoding tags contributed by rendered submission contracts."""
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


def render_task(specification: TaskSpecification, renderings: tuple[Rendering, ...]) -> Task:
    """Project public input from one fixed instance, adding submission capabilities."""
    if len(renderings) != len(specification.steps):
        raise ValueError("Exactly one rendering per semantic step is required")
    for step, rendering in zip(specification.steps, renderings, strict=True):
        submission = rendering.submission
        state = specification.requirements.state
        paths = (
            submission.paths
            if isinstance(submission, FinalState)
            else ((submission.path,) if isinstance(submission, FileSubmission) else ())
        )
        for path in paths:
            submission_relative(path, state.workdir, state.additional_directories)
        if isinstance(submission, FinalState):
            if step.answer_requirements.kind != "final_state" or not submission.paths:
                raise ValueError("Final-state submission requires explicit state paths and task requirements")
        elif not isinstance(submission, FinalActionSubmission):
            if step.answer_requirements.kind == "final_state":
                raise ValueError("State-modification tasks require a final-state submission")
            validate_extractor(submission.extractor)
            if step.answer_requirements.kind != "text" and not isinstance(submission.extractor, PlainText):
                raise ValueError("Intrinsic literal/format requirements cannot be replaced by answer wrappers")
        elif step.answer_requirements.kind == "final_state":
            raise ValueError("State-modification tasks require a final-state submission")
    capabilities = set(specification.requirements.capabilities)
    if any(isinstance(r.submission, (FileSubmission, FinalState)) for r in renderings):
        capabilities.add(Capability.FILESYSTEM)
    if any(ResourceRole.AGENT in r.roles for r in specification.resources) or any(
        ResourceRole.AGENT in r.roles for step in specification.steps for r in step.resources
    ):
        capabilities.add(Capability.FILESYSTEM)
    return Task(
        id=specification.id,
        specification_sha256=specification_hash(specification),
        steps=tuple(
            TaskStep(
                render_instruction(specification, rendering, index),
                public_resources(step.resources),
                rendering.submission,
                step.context_requirement,
            )
            for index, (step, rendering) in enumerate(zip(specification.steps, renderings, strict=True))
        ),
        requirements=msgspec.structs.replace(specification.requirements, capabilities=tuple(sorted(capabilities))),
        resources=public_resources(specification.resources),
        metadata=specification.metadata,
        coverage_tags=tuple(sorted((*specification.coverage_tags, *result_tags(renderings)))),
    )
