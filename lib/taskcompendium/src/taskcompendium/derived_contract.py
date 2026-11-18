# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive a task's contract shape from its final TaskSpec, independently of declared tags.

The derived shape — agentic, turn count, reply-only delivery, workspace grading and grader
type — is computed late in ingestion from the finished contract. Declared tags may keep only
what the contract cannot tell (license, family, upstream provenance); a declared shape tag
that contradicts the derived shape fails ingestion instead of silently disagreeing.
"""

from collections import Counter
from dataclasses import dataclass, field

from verifyit.spec import JudgeSpec

from taskcompendium.execution_applicability import is_nonagentic_task
from taskcompendium.grader_type import GraderType, JudgeDetails, derive_grader_type
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    Grader,
    ScriptGrader,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
    judge_spec,
)

REPLY_ANSWER_TYPES = frozenset({AnswerType.TEXT, AnswerType.NUMBER, AnswerType.JSON})
"""Answer types delivered as text in the assistant reply, not in a workspace file or state."""

SHAPE_TAGS = frozenset({"agentic", "single-turn", "multi-turn", "reply-only"})
"""Tags ingestion derives from the contract; declaring one is only allowed to agree."""


@dataclass(frozen=True)
class DerivedContract:
    """One task's derived contract shape."""

    agentic: bool
    single_turn: bool
    reply_only: bool
    workspace_graded: bool
    grader_type: GraderType


def is_workspace_graded(grader: Grader) -> bool:
    """Whether the grader runs in a fresh machine the agent never shares.

    In-process verifyit comparators and host-side judges grade the extracted answer without a
    grading machine; script graders and env-backed verifyit modes provision one.
    """
    if isinstance(grader, ScriptGrader):
        return True
    return isinstance(grader, VerifyitGrader) and grader.environment is not None


def derive_contract(task: TaskSpec) -> DerivedContract:
    """Derive one task's contract shape from its final TaskSpec.

    ``reply_only`` records that the finished contract asks for the answer as an assistant reply;
    for rewritten sources the converter's delivery rewrite is what produced that shape, so the
    rewrite need not be re-recorded.
    """
    events = task.context.events
    single_turn = sum(1 for event in events if isinstance(event, TextMessage) and event.role == "user") == 1
    single_turn = single_turn and not any(isinstance(event, AssistantToolCalls) for event in events)
    return DerivedContract(
        agentic=not is_nonagentic_task(task),
        single_turn=single_turn,
        reply_only=task.answer_type in REPLY_ANSWER_TYPES,
        workspace_graded=is_workspace_graded(task.grader),
        grader_type=derive_grader_type(task.grader),
    )


def shape_contradictions(declared_tags: frozenset[str] | set[str], derived: DerivedContract) -> tuple[str, ...]:
    """Declared shape tags that disagree with one task's derived shape."""
    contradictions = []
    if "agentic" in declared_tags and not derived.agentic:
        contradictions.append("agentic")
    if "single-turn" in declared_tags and not derived.single_turn:
        contradictions.append("single-turn")
    if "multi-turn" in declared_tags and derived.single_turn:
        contradictions.append("multi-turn")
    if "reply-only" in declared_tags and not derived.reply_only:
        contradictions.append("reply-only")
    return tuple(contradictions)


@dataclass(frozen=True)
class DerivedContractSummary:
    """A population's derived contract shape, with per-value grader-type counts."""

    task_count: int
    grader_type_counts: dict[str, int] = field(default_factory=dict)
    agentic_count: int = 0
    single_turn_count: int = 0
    reply_only_count: int = 0
    workspace_graded_count: int = 0
    judge_rubric_counts: dict[str, int] = field(default_factory=dict)
    judge_reference_count: int = 0
    judge_no_reference_count: int = 0

    @property
    def grader_type(self) -> GraderType:
        """The source-level value: the single derived type, or ``MIXED`` with a breakdown."""
        types = [GraderType(value) for value in self.grader_type_counts]
        if len(types) > 1:
            return GraderType.MIXED
        if not types:
            raise ValueError("Cannot name the grader type of an empty population")
        return types[0]

    @property
    def mixed(self) -> bool:
        return len(self.grader_type_counts) > 1

    def judge_details(self) -> JudgeDetails | None:
        """The population's judge profile, only when every judge shares one rubric."""
        if not self.judge_rubric_counts or len(self.judge_rubric_counts) > 1:
            return None
        rubric = next(iter(self.judge_rubric_counts))
        return JudgeDetails(
            rubric=rubric,
            model="",
            reference_based=self.judge_reference_count > 0 and self.judge_no_reference_count == 0,
            exact_gate=False,
        )


def summarize_contracts(
    derived: tuple[DerivedContract, ...], judges: tuple[JudgeSpec, ...] = ()
) -> DerivedContractSummary:
    """Aggregate derived task contracts into a source-level summary.

    ``judges`` carries one judge contract per ``LLM_JUDGE`` task, so the summary can record the
    rubric and reference profile the population's judges actually use.
    """
    if not derived:
        raise ValueError("A contract summary requires at least one task")
    return DerivedContractSummary(
        task_count=len(derived),
        grader_type_counts=dict(sorted(Counter(contract.grader_type.value for contract in derived).items())),
        agentic_count=sum(contract.agentic for contract in derived),
        single_turn_count=sum(contract.single_turn for contract in derived),
        reply_only_count=sum(contract.reply_only for contract in derived),
        workspace_graded_count=sum(contract.workspace_graded for contract in derived),
        judge_rubric_counts=dict(sorted(Counter(judge.rubric for judge in judges).items())),
        judge_reference_count=sum(bool(judge.references) or bool(judge.exact_gate_answers) for judge in judges),
        judge_no_reference_count=sum(not (bool(judge.references) or bool(judge.exact_gate_answers)) for judge in judges),
    )


def summary_contradictions(declared_tags: frozenset[str] | set[str], summary: DerivedContractSummary) -> tuple[str, ...]:
    """Declared shape tags that disagree with the population's derived shape.

    Declared shape tags are population-wide claims: every task must agree with the tag, so a
    single contradicting task fails ingestion.
    """
    contradictions = []
    if "agentic" in declared_tags and summary.agentic_count == 0:
        contradictions.append("agentic")
    if "single-turn" in declared_tags and summary.single_turn_count != summary.task_count:
        contradictions.append("single-turn")
    if "multi-turn" in declared_tags and summary.single_turn_count > 0:
        contradictions.append("multi-turn")
    if "reply-only" in declared_tags and summary.reply_only_count != summary.task_count:
        contradictions.append("reply-only")
    return tuple(contradictions)


def task_contracts(tasks: tuple[TaskSpec, ...]) -> DerivedContractSummary:
    """Derive and summarize a source population in one pass."""
    contracts = tuple(derive_contract(task) for task in tasks)
    judges = tuple(
        spec
        for task, contract in zip(tasks, contracts, strict=True)
        if contract.grader_type is GraderType.LLM_JUDGE
        for spec in (judge_spec(task.grader),)
        if spec is not None
    )
    return summarize_contracts(contracts, judges)
