# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed proposal JSON with a Markdown body.

The JSON object holds ``header`` and ``body``; ``TASK_PROPOSAL`` decodes it. Canonical rendering sorts
keys and normalizes body whitespace, so formatting alone does not change a proposal's content digest.
"""

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Annotated

from pydantic import ConfigDict, Field, StringConstraints, TypeAdapter, model_validator, with_config

from taskforge.model.content_hash import digest, pretty_json

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class SourceKind(StrEnum):
    CAPABILITY = "capability"
    REPO = "repo"


class Environment(StrEnum):
    """What the solver works in: a prompt only, a simulated shell, or a real container."""

    REASONING = "reasoning"
    SHELLSIM = "shellsim"
    CONTAINER = "container"


class Verification(StrEnum):
    """How the private evaluator grades.

    ``answer_in_box``: a generic match of a final answer whose format is unambiguous (exact, numeric, math,
    choice). ``code``: the usual kind; the task states an exact answer structure (named fields, a units field
    per quantity, enumerated labels) that the grader parses strictly against build-computed ground truth, or
    the work is what only execution can judge (behavioural tests, running systems). ``decision_model``: a
    decision model compares the answer with the reference answer against a rubric, only where fields cannot
    capture the answer, and its fooling cases must earn no credit. ``llm_as_judge``: a model judge
    for deliverables whose prose is the product. ``composite``: a weighted list of the other kinds
    (``CompositePart``). Composite documents decode, but lowering requires a single supported grader.
    """

    ANSWER_IN_BOX = "answer_in_box"
    DECISION_MODEL = "decision_model"
    CODE = "code"
    LLM_AS_JUDGE = "llm_as_judge"
    COMPOSITE = "composite"


class Grounding(StrEnum):
    """Whether the proposal's factual claims were checked against a real source."""

    UNVERIFIED = "unverified"
    GROUNDED = "grounded"


class ResearchKind(StrEnum):
    WEB = "web"
    GITHUB = "github"


class ResearchNeed(StrEnum):
    """Whether the build depends on external facts or artifacts that must be looked up."""

    NEEDED = "needed"
    NOT_NEEDED = "not_needed"


@with_config(ConfigDict(extra="forbid"))
@dataclass(frozen=True, slots=True)
class SourceRef:
    """Where the proposal came from: a catalog capability id or a repository, and its content hash."""

    kind: SourceKind
    ref: Text
    hash: Text


@with_config(ConfigDict(extra="forbid"))
@dataclass(frozen=True, slots=True)
class ResearchItem:
    kind: ResearchKind
    purpose: Text


@with_config(ConfigDict(extra="forbid"))
@dataclass(frozen=True, slots=True)
class BuildItem:
    """One named build artifact and a one-line description; the body explains it."""

    name: Text
    description: Text


@with_config(ConfigDict(extra="forbid"))
@dataclass(frozen=True, slots=True)
class CompositePart:
    """One part of a composite verification: a grader of ``kind`` and its share of the reward.

    For now a composite is a weighted list of the other kinds, repeats allowed. It is meant to become an
    arbitrary program over graders.
    """

    kind: Verification
    weight: Annotated[float, Field(strict=True, gt=0, le=1)]


@with_config(ConfigDict(extra="forbid"))
@dataclass(frozen=True, slots=True)
class ProposalHeader:
    id: Text
    source: SourceRef
    environment: Environment
    verification: Verification
    composite: tuple[CompositePart, ...]
    grounding: Grounding
    research_need: ResearchNeed
    research_reason: Text
    research: tuple[ResearchItem, ...]
    build: tuple[BuildItem, ...]
    resources: tuple[Text, ...]
    null_reason: Text | None

    @model_validator(mode="before")
    @classmethod
    def legacy_header(cls, value: object) -> object:
        """Input catalogs may carry solver_machine, which environment replaces; the header drops it."""
        return {k: v for k, v in value.items() if k != "solver_machine"} if isinstance(value, dict) else value


@with_config(ConfigDict(extra="forbid"))
@dataclass(frozen=True, slots=True)
class TaskProposal:
    header: ProposalHeader
    body: str

    @property
    def digest(self) -> str:
        return digest(replace(self, body=_canonical_body(self.body)))


def render(p: TaskProposal) -> str:
    """Return the canonical JSON of ``p``."""
    return pretty_json(replace(p, body=_canonical_body(p.body))) + "\n"


def _canonical_body(body: str) -> str:
    lines = [line.rstrip() for line in body.replace("\r\n", "\n").strip().split("\n")]
    return "\n".join(lines) + "\n"


TASK_PROPOSAL = TypeAdapter(TaskProposal)
