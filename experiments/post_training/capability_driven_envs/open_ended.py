# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fully hydrated inputs for proposing open-ended tasks.

An ``OpenEndedTaskBundle`` is an idea type for Taskforge's generic
``ProposalSource[IdeaT]`` boundary. It contains every byte needed to author a
free-form task whose reward will come from model judgment rather than an
objective verifier. Provenance may identify external sources, but they are not
runtime dependencies: proposal generation and building must not join or fetch
additional task data.
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rigging.filesystem.path_validation import validate_relative_file_paths
from taskcompendium.models import Source, TaskResource
from taskforge.content_hash import digest


class OpenEndedTaskBrief(BaseModel):
    """The human-authored intent that proposal generation may vary into task slots."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    goal: str = Field(min_length=1)
    expected_deliverable: str = Field(min_length=1)
    question_family: str = Field(min_length=1)
    avoid: tuple[str, ...] = ()


class OpenEndedTaskBundle(BaseModel):
    """A self-contained idea for one family of open-ended task proposals.

    ``solver_resources`` are candidate inputs for the eventual TaskSpec.
    ``author_evidence`` is private construction and calibration evidence; it is
    not solver-visible and is not automatically copied into verifier resources.
    Both groups contain inline ``TaskResource`` bytes, so loading this model
    completes hydration before Taskforge proposal generation begins.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    bundle_id: str = Field(min_length=1)
    split_group: str = Field(min_length=1)
    brief: OpenEndedTaskBrief
    solver_resources: tuple[TaskResource, ...] = Field(min_length=1)
    author_evidence: tuple[TaskResource, ...] = Field(min_length=1)
    source: Source
    tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_resource_paths(self) -> "OpenEndedTaskBundle":
        solver_paths = [resource.path for resource in self.solver_resources]
        evidence_paths = [resource.path for resource in self.author_evidence]
        validate_relative_file_paths(solver_paths)
        validate_relative_file_paths(evidence_paths)
        all_paths = solver_paths + evidence_paths
        duplicates = sorted(path for path in set(all_paths) if all_paths.count(path) > 1)
        if duplicates:
            raise ValueError(f"resource paths repeat across the bundle: {duplicates}")
        return self

    @property
    def content_digest(self) -> str:
        """Return the canonical content identity, including private evidence bytes."""
        return digest(self)

    def build_input(self) -> "OpenEndedBuildInput":
        """Return the public immutable input available to a Taskforge build program."""
        return OpenEndedBuildInput(
            bundle_id=self.bundle_id,
            bundle_digest=self.content_digest,
            split_group=self.split_group,
            solver_resources=self.solver_resources,
            source=self.source,
            tags=self.tags,
        )


class OpenEndedBuildInput(BaseModel):
    """The public part of a bundle that a build program may place in a TaskSpec."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bundle_id: str = Field(min_length=1)
    bundle_digest: str = Field(min_length=1)
    split_group: str = Field(min_length=1)
    solver_resources: tuple[TaskResource, ...] = Field(min_length=1)
    source: Source
    tags: tuple[str, ...] = ()
