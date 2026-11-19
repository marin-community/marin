# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The semantic grader type derived from a TaskSpec's actual grader configuration.

The type describes how a task is graded — by an LLM judge, a deterministic comparison, a math
comparator, executed tests, a checker script, a reward model, or nothing — independently of where
the grader runs. The execution locus (in process, package, image, remote judge) is a separate
axis; ``experiments/.../inventory.py`` already records it. A source whose tasks disagree carries
the aggregate ``MIXED`` value with a per-value breakdown.
"""

from dataclasses import dataclass
from enum import StrEnum

from verifyit.spec import JudgeSpec, Spec

from taskcompendium.models import Grader, NoGrader, ScriptGrader, VerifyitGrader, judge_spec, verifyit_spec


class GraderType(StrEnum):
    """How a task's grader decides its reward."""

    LLM_JUDGE = "llm-judge"
    EXACT_MATCH = "exact-match"
    MATH_VERIFY = "math-verify"
    UNIT_TESTS = "unit-tests"
    SCRIPT = "script"
    REWARD_MODEL = "reward-model"
    NONE = "none"
    MIXED = "mixed"
    """Aggregate over a population with more than one grader type; never derived from one task."""


EXACT_MATCH_MODES = frozenset(
    {
        "exact",
        "structured_exact",
        "mcq",
        "predicted_action",
        "structure_closure",
        "reasoning_markup",
        "json-schema",
        "xml-elements",
        "csv-columns",
        "ifeval",
    }
)
"""Deterministic answer and format comparators: they compare or validate the response against a
fixed spec without executing code or consulting a model."""

MATH_VERIFY_MODES = frozenset({"math", "numeric"})
"""Symbolic math-verify comparators and numeric tolerance comparisons."""

UNIT_TEST_MODES = frozenset({"pytest", "junit", "gotest", "stdio"})
"""Executed test graders: unit-test frameworks and stdin/stdout case runners."""

SCRIPT_MODES = frozenset({"script", "reasoning-gym"})
"""Checker-script graders, including reasoning-gym's vendored regeneration script."""


@dataclass(frozen=True)
class JudgeDetails:
    """The judge contract behind an ``LLM_JUDGE`` grader."""

    rubric: str
    """The judge rubric kind: ``reference``, ``checklist``, or ``labels``."""

    model: str
    """The pinned judge model, empty when the runtime supplies the endpoint's model."""

    reference_based: bool
    """Whether the judge sees reference answers (or gates them exactly)."""

    exact_gate: bool
    """Whether exact reference matches are decided without a model call."""


def judge_details(spec: JudgeSpec) -> JudgeDetails:
    """Summarize a judge contract for the Atlas detail pane."""
    return JudgeDetails(
        rubric=spec.rubric,
        model=spec.model,
        reference_based=bool(spec.references) or bool(spec.exact_gate_answers),
        exact_gate=spec.exact_gate,
    )


def grader_mode_type(mode: str) -> GraderType:
    """Classify one verifyit mode; an unmapped mode is an implementation gap and fails loudly."""
    if mode == "judge":
        return GraderType.LLM_JUDGE
    if mode in EXACT_MATCH_MODES:
        return GraderType.EXACT_MATCH
    if mode in MATH_VERIFY_MODES:
        return GraderType.MATH_VERIFY
    if mode in UNIT_TEST_MODES:
        return GraderType.UNIT_TESTS
    if mode in SCRIPT_MODES:
        return GraderType.SCRIPT
    raise ValueError(f"Verifyit mode {mode!r} has no GraderType mapping; extend grader_type.py")


def derive_grader_type(grader: Grader) -> GraderType:
    """Classify a TaskSpec's grader; an unclassifiable grader is an error, not an unknown."""
    if isinstance(grader, VerifyitGrader):
        return grader_mode_type(grader.mode)
    if isinstance(grader, ScriptGrader):
        return GraderType.LLM_JUDGE if grader.judge is not None else GraderType.SCRIPT
    if isinstance(grader, NoGrader):
        if grader.contract.get("reward_model") is not None:
            return GraderType.REWARD_MODEL
        return GraderType.NONE
    raise ValueError(f"Unclassifiable grader: {grader!r}")


def task_judge_spec(grader: Grader) -> JudgeSpec | None:
    """The task's judge contract, whatever grader kind carries it."""
    return judge_spec(grader)


def verifyit_grader_spec(grader: Grader) -> Spec | None:
    """The task's verifyit spec, or ``None`` for non-verifyit graders."""
    if isinstance(grader, VerifyitGrader):
        return verifyit_spec(grader)
    return None
