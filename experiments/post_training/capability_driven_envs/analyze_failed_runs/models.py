# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The values of the analysis: the replies of its model calls, and the ideas between the merges."""

from collections import Counter
from collections.abc import Set
from dataclasses import dataclass
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from experiments.post_training.capability_driven_envs.catalog import FILE_SAFE_ID

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
    """The general skills whose absence explains one failed rollout, or none if the rollout shows no gap."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ideas: tuple[Capability, ...] = Field(max_length=IDEAS_PER_ROLLOUT)


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


def groups_of(lines: Set[str]) -> type[Groups]:
    """The reply type of a merge of ``lines``: each group holds only these lines, and the groups have different IDs.

    The structured call sends an invalid reply back to the model with the error. A reply can leave out the lines of the
    capabilities that the merge does not keep, and a line can be in two groups.
    """

    class LineGroups(Groups):
        """The merged capabilities."""

        @model_validator(mode="after")
        def hold_known_lines(self) -> Self:
            ids = Counter(group.id for group in self.capabilities)
            problems = {
                "These IDs are not lines": {key for group in self.capabilities for key in group.members} - lines,
                "These capability IDs occur more than once": {key for key, count in ids.items() if count > 1},
            }
            errors = [f"{problem}: {', '.join(sorted(keys))}." for problem, keys in problems.items() if keys]
            if errors:
                raise ValueError(" ".join(errors))
            return self

    return LineGroups


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
