# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The values of the analysis: the replies of its model calls, and the ideas between the merges."""

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from experiments.post_training.capability_driven_envs.generate_tasks import FILE_SAFE_ID

IDEAS_PER_ROLLOUT = 3


class Capability(BaseModel):
    """A general skill that failed rollouts lack, with a catalog ID."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, description="The skill in a few words, with no detail of a failed task.")
    outcome: str = Field(min_length=1, description="What an agent with the skill can do, in any domain.")
    includes: tuple[str, ...] = Field(min_length=1, description="The parts of the skill.")
    excludes: tuple[str, ...] = Field(default=(), description="Related skills that the capability does not cover.")
    id: str = Field(description="A file-safe ID, for example failed-runs.tool-use.stop-repeating-calls.")

    @field_validator("id")
    @classmethod
    def file_safe(cls, value: str) -> str:
        if FILE_SAFE_ID.match(value) is None:
            raise ValueError(f"capability id {value!r} is not file-safe")
        return value


class RolloutIdeas(BaseModel):
    """The general skills whose absence explains one failed rollout."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ideas: tuple[Capability, ...] = Field(min_length=1, max_length=IDEAS_PER_ROLLOUT)


class Group(Capability):
    """One merged capability, with the IDs of the lines that it holds."""

    members: tuple[str, ...] = Field(min_length=1, description="The IDs of the lines that the capability holds.")

    @model_validator(mode="after")
    def includes_behaviors(self) -> "Group":
        # GLM sometimes copies the member IDs into includes, and then generate_tasks gets no behaviors to practice.
        if set(self.includes) & set(self.members):
            raise ValueError("includes must name behaviors that the capability covers, not the IDs of its members")
        return self


class Groups(BaseModel):
    """The merged capabilities."""

    capabilities: tuple[Group, ...] = Field(min_length=1)


@dataclass(frozen=True)
class MaybeIdea:
    """A capability idea with the failed rollouts that show its gap, before the merges end."""

    idea: Capability
    evidence: frozenset[str]
    """The run IDs. A merged idea holds the union of the evidence of its ideas, so a rollout counts once."""
    tasks: frozenset[str]
    """The IDs of the tasks of the evidence, joined in the same way."""

    @property
    def num_rollouts(self) -> int:
        return len(self.evidence)

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)
