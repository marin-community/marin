# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The per-task generation loop, run on the taskforge effect stack.

Each selected task gets its own effect ``Context`` and event log. Generation emits
``InferenceRequest`` effects (one per attempt, disambiguated by ``sample`` so relaunches
replay recorded calls instead of regenerating); verification emits ``GradeCandidate``
effects that replay the candidate through the real engine and the task's own grader.
Outcomes are written as ``attempt.json`` plus ``status.json`` per task, so the store
builder can assemble rows without re-reading conversations.
"""

import dataclasses
import json
import logging
from dataclasses import dataclass, replace

import fsspec
from rigging.filesystem.factory import open_url

from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import TaskSpec, TextMessage
from taskforge.effects.context import Context
from taskforge.effects.log import EventLog, JsonlSink
from taskforge.llm.policy import LLMPolicy
from taskforge.lowering import InvalidTaskSpec, lower_with_settings, lowering_settings
from taskforge.model.content_hash import digest
from taskforge.model.controls import Candidate
from taskforge.model.llm import Completion, FinishReason, Message, Role
from taskforge.policy import Policy
from taskforge.policy import sampling as policy_sampling
from taskforge.resources.bundle import Resources
from taskforge.resources.grading import GradeCandidate
from taskforge.resources.inference import InferenceRequest
from taskforge.wiring import install

from experiments.post_training.task_curation.balanced_reasoning.difficulty import url_protocol
from experiments.post_training.task_curation.balanced_reasoning.graceful import (
    CONVERGENCE_PROMPT,
    check_graceful_trace,
)
from experiments.post_training.task_curation.balanced_reasoning.selection import SelectedTask
from experiments.post_training.task_curation.balanced_reasoning.style import style_report

logger = logging.getLogger(__name__)

RESOLUTION_SOLVED_SHORTEST = "solved_shortest"
RESOLUTION_UNSOLVED = "unsolved"
KIND_SOLVE = "solve"
KIND_GRACEFUL = "graceful"

START_THINK = "<|start_think|>"
END_THINK = "<|end_think|>"
"""The reasoning delimiters the Marin chat template's analysis channel renders."""


@dataclass(frozen=True, slots=True)
class AttemptTrace:
    """One generated attempt, its grade, and its style verdict."""

    kind: str
    content: str
    reasoning: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    budget_hit: bool
    reward: float | None
    grade_status: str | None
    grade_passed: bool | None
    style_degenerate: bool
    style_reason: str | None
    graceful_resolution: str | None = None

    @property
    def eligible(self) -> bool:
        """Whether this attempt may ship: graded pass, committed, not degenerate."""
        return (
            not self.budget_hit
            and not self.style_degenerate
            and self.grade_passed is True
            and self.finish_reason == FinishReason.STOP.value
        )


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    """What the lane decided for one selected task."""

    task_id: str
    source_id: str
    task_sha256: str
    domain: str
    band: str
    solve_rate: float
    difficulty_basis: str
    budget_cap_tokens: int
    attempts_requested: int
    attempts: tuple[AttemptTrace, ...]
    selected_index: int | None
    resolution_kind: str
    grader_kind: str
    model: str
    temperature: float
    policy_digest: str
    task_messages_document: tuple[dict[str, str], ...]
    """The task's own turns as role/content pairs; the store adds the selected trace."""
    unsolved_reason: str | None = None

    def to_document(self) -> dict:
        return dataclasses.asdict(self)

    def task_prompt_messages(self, attempt: AttemptTrace) -> list[dict[str, str]]:
        """The row conversation: the task's turns plus the selected trace."""
        return [*self.task_messages_document, {"role": "assistant", "content": assistant_trace_text(attempt)}]


def assistant_trace_text(attempt: AttemptTrace) -> str:
    """The assistant turn as one string, reasoning marked for the analysis channel."""
    if attempt.reasoning.strip():
        return f"{START_THINK}{attempt.reasoning}{END_THINK}{attempt.content}"
    return attempt.content


def outcome_from_document(document: dict) -> TaskOutcome:
    """Rebuild a :class:`TaskOutcome` from its ``attempt.json`` document."""
    attempts = tuple(AttemptTrace(**attempt) for attempt in document["attempts"])
    return TaskOutcome(
        task_id=document["task_id"],
        source_id=document["source_id"],
        task_sha256=document["task_sha256"],
        domain=document["domain"],
        band=document["band"],
        solve_rate=document["solve_rate"],
        difficulty_basis=document["difficulty_basis"],
        budget_cap_tokens=document["budget_cap_tokens"],
        attempts_requested=document["attempts_requested"],
        attempts=attempts,
        selected_index=document["selected_index"],
        resolution_kind=document["resolution_kind"],
        grader_kind=document["grader_kind"],
        model=document["model"],
        temperature=document["temperature"],
        policy_digest=document["policy_digest"],
        task_messages_document=tuple(
            {"role": turn["role"], "content": turn["content"]} for turn in document["task_messages_document"]
        ),
        unsolved_reason=document.get("unsolved_reason"),
    )


def _grade_passed(grade: GradeResult) -> bool | None:
    if grade.status != Outcome.GRADED:
        return False
    if grade.passed is not None:
        return grade.passed
    if grade.reward is None:
        return None
    return grade.reward >= grade.score_max


def task_messages(task: TaskSpec) -> tuple[Message, ...]:
    """The conversation the generator receives: the task's own turns, text only."""
    messages: list[Message] = []
    for event in task.context.events:
        if isinstance(event, TextMessage):
            messages.append(Message(role=Role(event.role), content=event.content))
        else:
            raise InvalidTaskSpec(
                "The reasoning lane generates single-turn traces; a task context with tool calls "
                f"({type(event).__name__}) belongs to a later, agentic round"
            )
    if not any(message.role == Role.USER for message in messages):
        raise InvalidTaskSpec("The task context carries no user turn")
    return tuple(messages)


def _attempt_trace(kind: str, completion: Completion, grade: GradeResult | None) -> AttemptTrace:
    style = style_report(f"{completion.reasoning}\n{completion.content}")
    return AttemptTrace(
        kind=kind,
        content=completion.content,
        reasoning=completion.reasoning,
        finish_reason=completion.finish_reason.value,
        prompt_tokens=completion.usage.prompt_tokens,
        completion_tokens=completion.usage.completion_tokens,
        reasoning_tokens=completion.usage.reasoning_tokens,
        budget_hit=completion.finish_reason == FinishReason.LENGTH,
        reward=grade.reward if grade is not None else None,
        grade_status=grade.status.value if grade is not None else None,
        grade_passed=_grade_passed(grade) if grade is not None else None,
        style_degenerate=style.degenerate,
        style_reason=style.reason(),
    )


def _best_failed_attempt(attempts: tuple[AttemptTrace, ...]) -> AttemptTrace:
    """The failed attempt to continue from: highest reward, then shortest."""
    return max(attempts, key=lambda attempt: ((attempt.reward or 0.0), -attempt.completion_tokens))


def _write_status(workdir: str, resolution: str) -> None:
    status = {"resolution": resolution}
    write_workdir_text(workdir, "status.json", json.dumps(status, sort_keys=True))


def write_workdir_text(workdir: str, name: str, text: str) -> None:
    """Write one text file into the task's workdir (a local path or an object-store URL)."""
    with fsspec.open(f"{workdir.rstrip('/')}/{name}", "wt") as sink:
        sink.write(text)


async def run_selected_task(
    entry: SelectedTask,
    task: TaskSpec,
    policy: Policy,
    resources: Resources,
    workdir: str,
) -> TaskOutcome:
    """Generate, verify and select traces for one selected task.

    The task's effect log lives at ``<workdir>/events.jsonl``; a relaunch replays recorded
    model calls and grades instead of regenerating them. Raises on infrastructure failure;
    unsolved tasks return a normal ``TaskOutcome`` with the reason recorded.
    """
    workdir = str(workdir)
    budget = entry.budget
    sampling = replace(policy_sampling(policy), max_tokens=budget.cap_tokens)
    messages = task_messages(task)
    task_turns = tuple({"role": message.role.value, "content": message.content} for message in messages)
    sink = JsonlSink(f"{workdir.rstrip('/')}/events.jsonl")
    events_path = sink.path
    previous = await sink.read() if fsspec.filesystem(url_protocol(events_path)).exists(events_path) else ()
    log = EventLog(sink=sink, replay_from=previous)
    ctx = Context(log=log)
    outcome: TaskOutcome
    try:
        async with ctx.scope(f"task:{entry.task.task_id}"):
            install(ctx, resources)
            spec = lower_with_settings(task, lowering_settings(policy), resources.factories)
            traces: list[AttemptTrace] = []
            for index in range(budget.attempts):
                completion = await ctx.emit(
                    InferenceRequest(messages=messages, tools=(), sampling=sampling, sample=index + 1)
                )
                grade = await ctx.emit(GradeCandidate(spec, Candidate(reply=completion.content)))
                traces.append(_attempt_trace(KIND_SOLVE, completion, grade))
            selected_index = _shortest_eligible(traces)
            unsolved_reason: str | None = None
            if selected_index is None:
                resolution = RESOLUTION_UNSOLVED
                graceful = await _graceful_pass(ctx, entry.task.task_id, messages, sampling, spec, traces)
                if graceful is not None:
                    trace, resolution = graceful
                    traces.append(trace)
                    selected_index = len(traces) - 1
                else:
                    unsolved_reason = "the generator produced no passing attempt within the band budget"
            else:
                resolution = RESOLUTION_SOLVED_SHORTEST
            outcome = TaskOutcome(
                task_id=entry.task.task_id,
                source_id=entry.task.source_id,
                task_sha256=entry.task.task_sha256,
                domain=entry.task.domain,
                band=entry.band.value,
                solve_rate=entry.difficulty.solve_rate,
                difficulty_basis=entry.difficulty.basis,
                budget_cap_tokens=budget.cap_tokens,
                attempts_requested=budget.attempts,
                attempts=tuple(traces),
                selected_index=selected_index,
                resolution_kind=resolution,
                grader_kind=task.grader.kind,
                model=policy.setting("author.model", str),
                temperature=sampling.temperature,
                policy_digest=digest(dict(policy.entries)),
                task_messages_document=task_turns,
                unsolved_reason=unsolved_reason,
            )
    finally:
        await ctx.drain()
        await resources.close_round()
    fsspec.filesystem(url_protocol(workdir)).makedirs(workdir, exist_ok=True)
    write_workdir_text(workdir, "attempt.json", json.dumps(outcome.to_document(), indent=2, sort_keys=True))
    _write_status(workdir, outcome.resolution_kind)
    logger.info(
        "TASK %s band=%s resolution=%s attempts=%d selected=%s",
        entry.task.task_id,
        entry.band.value,
        outcome.resolution_kind,
        len(outcome.attempts),
        outcome.selected_index,
    )
    return outcome


def _shortest_eligible(traces: list[AttemptTrace]) -> int | None:
    """The index of the shortest eligible trace, or ``None`` when none is eligible."""
    eligible = [(index, trace) for index, trace in enumerate(traces) if trace.eligible]
    if not eligible:
        return None
    index, _ = min(eligible, key=lambda pair: (pair[1].completion_tokens, pair[0]))
    return index


async def _graceful_pass(
    ctx: Context,
    task_id: str,
    messages: tuple[Message, ...],
    sampling: LLMPolicy,
    spec,
    traces: list[AttemptTrace],
) -> tuple[AttemptTrace, str] | None:
    """Continue the best failed attempt under the graceful form contract.

    A graceful trace is form-verified, not correctness-verified: its grade is recorded for
    metadata only and never gates shipping.
    """
    failed = tuple(trace for trace in traces if trace.kind == KIND_SOLVE)
    if not failed:
        return None
    best = _best_failed_attempt(failed)
    conversation = (
        *messages,
        Message(role=Role.ASSISTANT, content=best.content, reasoning=best.reasoning),
        Message(role=Role.USER, content=CONVERGENCE_PROMPT),
    )
    completion = await ctx.emit(
        InferenceRequest(messages=conversation, tools=(), sampling=sampling, sample=len(traces) + 1)
    )
    check = check_graceful_trace(completion.reasoning, completion.content, completion.finish_reason.value)
    grade = await ctx.emit(GradeCandidate(spec, Candidate(reply=completion.content)))
    trace = _attempt_trace(KIND_GRACEFUL, completion, grade)
    trace = dataclasses.replace(trace, graceful_resolution=check.resolution, style_reason=check.reason)
    if check.passed:
        return trace, check.resolution
    logger.info("GRACEFUL %s rejected: %s", task_id, check.reason)
    return None
