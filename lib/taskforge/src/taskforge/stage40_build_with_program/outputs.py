# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The outputs of the standard template's steps, which authored programs return too.

The step store writes each output as canonical JSON and decodes it with a pydantic ``TypeAdapter``
of the step's return annotation, so every type here survives that round trip. Two step names carry
a meaning outside the program:

* ``stage30_author.loader.GRADER_STEP``: ``load_program_source`` requires a step of this name
  returning ``Grader``, and validation reads the reference answer and the prototyped candidates
  from its stored output.
* ``SOURCES_STEP``: under ``ResearchRequirement.ENFORCED``, ``run_build`` reads the research
  findings from the ``Sources`` this step returns.
"""

from dataclasses import dataclass

from taskcompendium.grader import GraderPackage
from taskcompendium.grading_result import Outcome
from taskcompendium.models import TaskResource

from taskforge.stage10_proposal.model import ResearchKind

SOURCES_STEP = "sources"


@dataclass(frozen=True)
class WebCall:
    """One executed web tool call of a research run: the tool, its arguments, and its output size."""

    tool: str
    arguments: str
    output_chars: int


@dataclass(frozen=True)
class ResearchFinding:
    """What ``sdk.research`` found for the proposal's ``research[index]``.

    Attributes:
        index: The research item's position in the proposal header.
        kind: The research item's kind.
        purpose: The research item's purpose.
        notes: The research agent's final answer.
        web_calls: The web calls that executed and returned content rather than an ``error:`` result.
        turns: Agent turns the run took.
    """

    index: int
    kind: ResearchKind
    purpose: str
    notes: str
    web_calls: tuple[WebCall, ...]
    turns: int


@dataclass(frozen=True)
class Sources:
    """The research findings of a build and the notes later steps read, one section per finding."""

    notes: str
    findings: tuple[ResearchFinding, ...]


@dataclass(frozen=True)
class Fixtures:
    """Solver-visible files, private reference data, the facts they encode, and the packages of a package machine.

    ``agent_files`` paths are relative to the machine root, under ``workspace/``. ``private_files``
    paths are relative to the grader's private directory. ``packages`` are the PyPI requirements of
    a proposal that declares ``solver_machine: packages``, and empty for any other.
    """

    agent_files: tuple[TaskResource, ...]
    private_files: tuple[TaskResource, ...]
    facts: str
    packages: tuple[str, ...]


@dataclass(frozen=True)
class GradedCandidate:
    """A candidate ``sandbox_effects.grade_candidate`` graded during the build, with its grade.

    ``files`` are the workspace files the candidate left, sorted by path.
    """

    reply: str
    files: tuple[TaskResource, ...]
    status: Outcome
    reward: float | None


@dataclass(frozen=True)
class Grader:
    """The task's grader, the answer it was written for, and the candidates it was prototyped on.

    Attributes:
        package: The grader (``spec.draft.answer_grader``, ``python_grader`` or ``script_grader``),
            passed to ``spec.draft.assemble`` as ``grader``.
        answer_contract: The exact output format the solver must follow, for the instruction.
        reference_reply: A correct final reply. The grader step grades it before returning, and
            validation replays it as the reference control.
        reference_files: Files a correct solver leaves in the task machine, paths relative to the
            machine root, when the grader reads any.
        secret_values: Answer strings the instruction must not contain.
        rejection_ceiling: The highest reward the grader gives a submission that is wrong in
            substance: 0 for an all-or-nothing grader, else the most partial credit a wrong answer can
            collect under its scheme. The empty answer must score at most this, and validation's
            negative controls demand at most this (``spec.controls.check_expectations``).
        prototyped: Every candidate the grader step graded, in order. Validation refuses a control
            that repeats one of them, other than the reference answer and the empty answer.
    """

    package: GraderPackage
    answer_contract: str
    reference_reply: str
    reference_files: tuple[TaskResource, ...]
    secret_values: tuple[str, ...]
    rejection_ceiling: float
    prototyped: tuple[GradedCandidate, ...]


@dataclass(frozen=True)
class Instructions:
    """The solver-facing system message, if any, and the user instruction."""

    system: str | None
    instruction: str
