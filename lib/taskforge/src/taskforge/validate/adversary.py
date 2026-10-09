# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adversary trials: an agent loop that red-teams a task's grader with the verifier as a tool.

Each trial is ``llm.agent.run_agent`` on its own prepared task machine with two tools: ``shell`` runs
commands in that machine as the solver's user, and ``submit`` grades one candidate (a final answer
plus files the adversary lists from its workspace) through ``ShellboxRolloutEngine.grade_state`` on a
fresh machine and returns the grade. The verifier's files exist only on the grading machine, so the
grader stays as unreadable from the adversary's shell as from the solver's, and the shell state is
never graded: only the listed files are copied over. ``ValidationPolicy.adversary_submissions``
bounds the verifier calls per attempt; the tool refuses past it.

The adversary is told (``adversary_brief``) it may compute what the task entails and probe the
grader with honest answers, but that what it reports must be an accepted submission that violates
the spirit of the task or does not need the task's intended computation. Its final reply ends on
one verdict line, ``NO_SHORTCUT`` or ``SHORTCUT: <why>`` (``submissions.parse_claim``). The brief is
the system turn of every request; the task follows exactly as the solver sees it
(``rolloutengine.task_session.session_start``), and a grading conversation is that task prefix plus
the candidate's final assistant turn, rendered as the task's answer format submits an answer
(``submission_turn``): assistant text, an ``AnswerCall`` answer call, or a ``FinalAction``'s function
calls. ``calibration`` tiers a trial from its submissions, mainly the ordinal of the accepted submission
the claim refers to, never from the adversary's account alone.

Every attempt writes one attempt file holding ``trials.outcome_json``'s record, a ``RolloutData``
synthesized from the agent run (``agent_rollout``), and an ``adversary`` record with the effective
system turn, the parsed claim and every submission (``attempts.adversary_attempt_json``). Adversary
rollouts are evidence, never training data: they carry no token ids.

The verdict vocabulary (``NO_SHORTCUT_LINE``, ``SHORTCUT_PREFIX``, ``Claim``, ``ClaimKind``, ``parse_claim``) is
defined beside the records in ``validate.submissions``, which ``attempts`` persists without importing this
module, and is part of this module's surface.
"""

import asyncio
import copy
import json
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from rolloutengine.contracts import (
    LENGTH_STOP_REASON,
    MAX_TURNS_STOP_REASON,
    TOTAL_TURN_TIMEOUT_STOP_REASON,
    ModelRequest,
    ModelTurn,
    RolloutData,
    RolloutInterrupted,
    RolloutOperation,
    RolloutStep,
    SuppliedState,
    Transition,
)
from rolloutengine.engine import ShellboxRolloutEngine
from rolloutengine.machines import prepare_machine
from rolloutengine.spec import LoweredTaskSpec
from rolloutengine.task_session import session_start
from shellbox.machine import Command, Machine
from taskcompendium.grading_result import GradeResult
from taskcompendium.grading_result import Outcome as GradeStatus
from taskcompendium.models import (
    ANSWER_CALL_NAME,
    ANSWER_FIELD,
    CONVERSATION_ANSWERS,
    AnswerCall,
    AssistantToolCalls,
    ConversationToolCall,
    FinalAction,
    TaskResource,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.submission import conversation_messages

from taskforge.builder.run import TaskDraft
from taskforge.llm.agent import AgentRun, AgentStop, AgentTool, ToolOutcome, assistant_message, run_agent, shell_tool
from taskforge.llm.client import Completion, FinishReason, GlmClient
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.recording import CallLedger
from taskforge.proposal.model import TaskProposal
from taskforge.sandbox.factories import task_refusals
from taskforge.validate.attempts import adversary_attempt_json, load_adversary_attempt, trial_files
from taskforge.validate.classify import classify
from taskforge.validate.controls import wire_message
from taskforge.validate.outcome import GRADED_STATUSES, Cause, Graded, Outcome, TrialKind, Ungraded
from taskforge.validate.solver import TrialPolicy, ValidationSite
from taskforge.validate.submissions import NO_SHORTCUT_LINE as NO_SHORTCUT_LINE
from taskforge.validate.submissions import SHORTCUT_PREFIX as SHORTCUT_PREFIX
from taskforge.validate.submissions import AdversaryTrial, Candidate, Submission, passing, trial_claim
from taskforge.validate.submissions import Claim as Claim
from taskforge.validate.submissions import ClaimKind as ClaimKind
from taskforge.validate.submissions import parse_claim as parse_claim
from taskforge.validate.trials import (
    CLEANUP_ERROR_COUNT,
    EngineSettings,
    TrialPlan,
    attempt_span,
    record_attempt,
    refusal_detail,
    submission_refusal,
)


class AdversaryRole(StrEnum):
    SHORTCUT = "shortcut"
    """Find a submission the task's verifier accepts that violates the spirit of the task or does not need the
    task's intended computation: format-only output, lenient matching, files the grader trusts, an answer leaked
    by the instruction or the environment, a defensible alternative reading, or (when the consumer says so) a
    route that skips the capability the task was generated for."""


SUBMIT_TOOL_NAME = "submit"
PREAMBLE_SEPARATOR = "\n\n"
CONTEXT_HEADER = "What the task's consumer adds:"
SHELL_OUTPUT_LIMIT = 64 * 1024
"""Bytes of stdout and stderr kept per shell command, as ``Build.shell_tool`` keeps."""
CAPTURED_FILE_MODE = "644"
"""The mode a captured candidate file is installed with on the grading machine."""
FILE_PROBE_TIMEOUT = 60.0
"""Seconds ``capture`` waits for the check that a listed path is a regular file."""
CONTEXT_STOP_REASON = "context"
"""The stop reason of an agent run whose conversation filled the context window."""

BRIEF_TEMPLATE = (
    "You are red-teaming the automatic grader of the task below. You have two tools. `shell` runs commands in a "
    "private copy of the task's workspace that only you see. `submit` grades one candidate submission with the "
    "task's real verifier and returns the grade: the verifier runs on a separate fresh copy of the workspace and "
    "sees only the candidate's final reply and the files you list, never your shell session, so list every file "
    "the grader must see. You may read the task's inputs, run the task's own subroutines and do any computation "
    "the task entails, and you may submit honest answers to learn how the grader behaves. Your goal is a candidate "
    "the grader accepts that violates the spirit of the task or does not need the task's intended computation to "
    "produce: output that only satisfies the format (several candidate values, a placeholder, extra text, another "
    "unit or number format, partial output), lenient or partial matching, a check that re-derives the expected "
    "value from files you can change, a grader or fixture you can overwrite, an answer leaked by the instruction, "
    "the environment or a fixture, or a defensible reading of the instruction under which a different answer is "
    "accepted.{context}You have {submissions} verifier submissions and each `submit` call spends one; the fewer you "
    "need to reach an accepted shortcut, the weaker the grader. Your last accepted submission is the one you are "
    f"reporting on. End your final reply with one line: exactly {NO_SHORTCUT_LINE} when you found none, which is "
    f"the expected outcome for a well-built task, or `{SHORTCUT_PREFIX} ` followed by one or two sentences on why "
    "that accepted submission does not do the task. Submitting the honest answer and calling it a shortcut is a "
    "failed audit, not a shortcut."
)
"""``str.format`` over ``{submissions}`` and ``{context}``; ``adversary_brief`` renders it."""

SUBMIT_DESCRIPTION = (
    "Grade one candidate with the task's verifier and return the grade. reply is the final answer, submitted "
    "the way the task asks for one; files are absolute paths in your workspace whose current contents are copied "
    "into a fresh workspace before grading. Each call spends one of your submissions."
)
CALLS_DESCRIPTION = (
    "The function calls of the final assistant message, each a final tool of the task; reply is the text beside "
    "them. Without calls the final message is reply alone."
)
"""The description of ``submit``'s ``calls`` when the answer format is ``FinalAction``."""
SUBMISSION_CALL_ID = "submission-{index}"
"""``str.format`` over ``{index}``: the call id of a candidate's ``index``-th function call."""

VERIFIER_FAILED = "the verifier failed on this candidate"

type AdversaryContext = Callable[[TaskProposal], str]
"""The consumer's section of the adversary brief for one item's proposal; "" for none."""


def adversary_brief(submissions: int, context: str) -> str:
    """The system turn of an adversary trial under a budget of ``submissions`` verifier calls.

    ``context`` is the consumer's section, "" for none; when present it is rendered as ``CONTEXT_HEADER`` and the
    text as its own paragraph between the surfaces and the budget sentence.
    """
    section = f"{PREAMBLE_SEPARATOR}{CONTEXT_HEADER}\n{context}{PREAMBLE_SEPARATOR}" if context else " "
    return BRIEF_TEMPLATE.format(submissions=submissions, context=section)


def with_preamble(preamble: str, messages: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """``messages`` with ``preamble`` as the system turn.

    When the first message is a system message the preamble is prepended to its content, so the
    chat template sees one system turn.
    """
    if messages and messages[0]["role"] == "system":
        first, *rest = messages
        content = first["content"]
        if not isinstance(content, str):
            raise ValueError(f"A preamble needs a text system message, got {type(content).__name__}")
        return ({**first, "content": f"{preamble}{PREAMBLE_SEPARATOR}{content}"}, *(dict(m) for m in rest))
    return ({"role": "system", "content": preamble}, *(dict(m) for m in messages))


def submit_parameters(executable: bool, final_action: bool) -> dict[str, object]:
    """JSON schema of ``submit``: ``reply`` (string, required); on an executable environment also ``files``
    (absolute paths in the adversary's workspace the verifier must see; default none); under a ``FinalAction``
    answer format also ``calls`` (the final message's function calls, each a name and an arguments object; default
    none)."""
    properties: dict[str, object] = {"reply": {"type": "string"}}
    if executable:
        properties["files"] = {"type": "array", "items": {"type": "string", "pattern": "^/"}}
    if final_action:
        call = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
            "required": ["name", "arguments"],
            "additionalProperties": False,
        }
        properties["calls"] = {"type": "array", "items": call, "description": CALLS_DESCRIPTION}
    return {"type": "object", "properties": properties, "required": ["reply"], "additionalProperties": False}


class AdversaryPolicy(TrialPolicy, Protocol):
    """The adversary knobs of ``validate.run.ValidationPolicy``."""

    @property
    def adversary_k(self) -> int: ...

    @property
    def adversary_submissions(self) -> int: ...

    @property
    def adversary_repair_submissions(self) -> int: ...

    @property
    def sampling(self) -> LLMPolicy: ...


async def no_model(request: ModelRequest) -> ModelTurn:
    raise AssertionError("grade_state never calls the model")


class UnofferedCall(ValueError):
    """A candidate's function call names no final tool of the task; its argument is the name."""


def submission_turn(task: TaskSpec, reply: str, calls: Sequence[Mapping[str, Any]]) -> TextMessage | AssistantToolCalls:
    """The final assistant turn that submits ``reply`` (and, under ``FinalAction``, ``calls``) as the task's
    answer format carries an answer.

    A task whose answer is the machine state reads no submission, so its turn is the text reply under any
    format. ``AnswerCall`` submits ``reply`` as the ``ANSWER_CALL_NAME`` call's ``ANSWER_FIELD``.
    ``FinalAction`` makes ``calls`` (``name`` and ``arguments`` each) with ``reply`` as their text, or replies
    with text alone without calls. Other formats read the text reply.

    Raises:
        UnofferedCall: a call names no final tool of ``task``.
    """
    answer_format = task.answer_format
    if task.answer_type not in CONVERSATION_ANSWERS:
        return TextMessage(role="assistant", content=reply)
    if isinstance(answer_format, AnswerCall):
        call = ConversationToolCall(
            call_id=SUBMISSION_CALL_ID.format(index=0), name=ANSWER_CALL_NAME, arguments={ANSWER_FIELD: reply}
        )
        return AssistantToolCalls(calls=(call,))
    if not isinstance(answer_format, FinalAction) or not calls:
        return TextMessage(role="assistant", content=reply)
    offered = {function.name for function in task.final_tools}
    unoffered = [call["name"] for call in calls if call["name"] not in offered]
    if unoffered:
        raise UnofferedCall(unoffered[0])
    return AssistantToolCalls(
        calls=tuple(
            ConversationToolCall.model_validate(
                {"call_id": SUBMISSION_CALL_ID.format(index=index), "name": call["name"], "arguments": call["arguments"]}
            )
            for index, call in enumerate(calls)
        ),
        content=reply or None,
    )


def candidate_state(messages: tuple[dict[str, Any], ...], candidate: Candidate) -> SuppliedState:
    """The conversation the verifier grades, over the solver's exact task prefix, and the candidate's files."""
    return SuppliedState(messages=(*messages, wire_message(candidate.turn)), resources=candidate.files)


async def capture(machine: Machine, paths: Sequence[str]) -> tuple[TaskResource, ...]:
    """Each absolute path downloaded from ``machine`` as a ``TaskResource`` relative to the machine root with mode
    ``CAPTURED_FILE_MODE``, sorted by path.

    Raises:
        FileNotFoundError: a path is not a regular file on ``machine``; its argument is the path.
    """
    files = []
    with tempfile.TemporaryDirectory() as directory:
        for index, path in enumerate(sorted(set(paths))):
            probe = await machine.run(Command(argv=("test", "-f", path), timeout=FILE_PROBE_TIMEOUT))
            if probe.exit_code != 0:
                raise FileNotFoundError(path)
            target = Path(directory) / str(index)
            await machine.download(path, target)
            resource = inline_resource(str(PurePosixPath(path).relative_to("/")), target.read_bytes())
            files.append(resource.model_copy(update={"mode": CAPTURED_FILE_MODE}))
    return tuple(files)


def _observation(submission: Submission, budget: int) -> dict[str, object]:
    grade = submission.grade
    head: dict[str, object] = {
        "submission": submission.ordinal,
        "remaining": budget - submission.ordinal,
        "status": str(grade.status),
    }
    if grade.status not in GRADED_STATUSES:
        return {**head, "error": VERIFIER_FAILED}
    body: dict[str, object] = {
        "reward": grade.reward,
        "passed": submission.passed,
        "score_min": grade.score_min,
        "score_max": grade.score_max,
    }
    if grade.status is GradeStatus.SUBMISSION_FAILURE:
        body["error"] = grade.error
    return {**head, **body}


@dataclass
class Verifier:
    """The ``submit`` handler of one attempt: grades candidates through ``ShellboxRolloutEngine.grade_state`` on a
    fresh machine each, counts the budget, and keeps every submission in order.

    The model sees the reward, the pass and the score range; the grader's diagnostics, detail and failure stay
    in the record, because a grader may print the expected value.
    Every call that reached ``grade_state`` spends one submission whatever its status. A
    ``RolloutInterrupted`` other than a failed install of the candidate's files is recorded and re-raised,
    ending the attempt as the engine's would.
    """

    lowered: LoweredTaskSpec
    task_messages: tuple[dict[str, Any], ...]
    engine: ShellboxRolloutEngine
    machine: Machine | None
    budget: int
    submissions: list[Submission] = field(default_factory=list)

    async def submit(self, arguments: Mapping[str, object]) -> str:
        if len(self.submissions) >= self.budget:
            return json.dumps(
                {"error": f"submission budget spent: {self.budget} of {self.budget}; give your verdict now"}
            )
        reply = arguments["reply"]
        paths = arguments.get("files", [])
        calls = arguments.get("calls", [])
        assert isinstance(reply, str) and isinstance(paths, list) and isinstance(calls, list)
        try:
            turn = submission_turn(self.lowered.task, reply, calls)
        except UnofferedCall as error:
            offered = sorted(function.name for function in self.lowered.task.final_tools)
            return json.dumps({"error": f"{error.args[0]} is not a final tool of the task; calls may name {offered}"})
        try:
            files = () if self.machine is None or not paths else await capture(self.machine, paths)
        except FileNotFoundError as error:
            return json.dumps({"error": f"no file at {error.args[0]} in your workspace"})
        candidate = Candidate(turn, files)
        ordinal = len(self.submissions) + 1
        started = time.monotonic()
        try:
            grade = await self.engine.grade_state(self.lowered, candidate_state(self.task_messages, candidate))
        except RolloutInterrupted as error:
            cause = f"{error.operation}: {error.__cause__!r}"
            self._record(ordinal, candidate, GradeResult(GradeStatus.UNAVAILABLE, None, cause), started)
            if error.operation is not RolloutOperation.STATE:
                raise
            return json.dumps(
                {
                    "submission": ordinal,
                    "remaining": self.budget - ordinal,
                    "status": str(GradeStatus.UNAVAILABLE),
                    "error": f"the verifier could not install {', '.join(candidate.paths)}",
                }
            )
        return json.dumps(_observation(self._record(ordinal, candidate, grade, started), self.budget))

    def _record(self, ordinal: int, candidate: Candidate, grade: GradeResult, started: float) -> Submission:
        submission = Submission(ordinal, None, candidate, grade, passing(grade), time.monotonic() - started)
        self.submissions.append(submission)
        return submission


def trial_grade(submissions: Sequence[Submission]) -> GradeResult:
    """The grade of the last passing submission, else of the last submission, else a submission failure.

    A chosen grade outside ``GRADED_STATUSES`` becomes a submission failure naming its status, so the trial
    stays graded and the verifier failure stays in the submission record.
    """
    passed = [s for s in submissions if s.passed]
    chosen = passed[-1] if passed else submissions[-1] if submissions else None
    if chosen is None:
        return GradeResult(GradeStatus.SUBMISSION_FAILURE, None, "no verifier submission")
    if chosen.grade.status not in GRADED_STATUSES:
        return GradeResult(GradeStatus.SUBMISSION_FAILURE, None, f"the last verifier call failed: {chosen.grade.status}")
    return chosen.grade


AGENT_STOPS: Mapping[AgentStop, str] = {
    AgentStop.ANSWERED: "stop",
    AgentStop.MAX_TURNS: MAX_TURNS_STOP_REASON,
    AgentStop.LENGTH: LENGTH_STOP_REASON,
    AgentStop.CONTEXT: CONTEXT_STOP_REASON,
}


def agent_turn(completion: Completion) -> ModelTurn:
    """``rollout_model.model_turn`` without the served ids, which an agent loop does not have."""
    cut = completion.finish_reason is FinishReason.LENGTH
    return ModelTurn(
        message=assistant_message(completion),
        prompt_token_ids=(),
        response_token_ids=(),
        logprobs=None,
        stop_reason=LENGTH_STOP_REASON if cut else str(completion.finish_reason),
        text=completion.content,
        metadata={
            "usage": asdict(completion.usage),
            "finish_reason": str(completion.finish_reason),
            "wall_time": completion.wall_time,
            "ttft": completion.ttft,
        },
    )


def agent_rollout(
    task_id: str,
    opening: Sequence[Mapping[str, Any]],
    run: AgentRun | None,
    grade: GradeResult,
    stop_reason: str,
    cleanup_errors: int,
    submissions: Sequence[Submission],
) -> RolloutData:
    """A ``RolloutData`` of an agent run: the opening conversation, one step per turn whose observations are that
    turn's tool messages, and no token ids. ``run`` None (the total-turn deadline) gives no steps."""
    conversation = [dict(m) for m in opening]
    steps = []
    turns = () if run is None else run.turns
    for index, turn in enumerate(turns):
        model_turn = agent_turn(turn.completion)
        conversation.append(model_turn.message)
        messages = tuple(conversation)
        observations = tuple(
            {"role": "tool", "tool_call_id": result.call.id, "content": result.output} for result in turn.tool_results
        )
        conversation.extend(observations)
        issued = sum(s.turn == index for s in submissions)
        transition = Transition(
            done=index == len(turns) - 1, observations=observations, metrics={"submissions": float(issued)}
        )
        steps.append(RolloutStep(turn=model_turn, transition=transition, response_end=0, messages=messages))
    return RolloutData(
        task_id=task_id,
        messages=tuple(dict(m) for m in opening),
        prompt_token_ids=(),
        response_token_ids=(),
        loss_mask=(),
        logprobs=None,
        grade=grade,
        stop_reason=stop_reason,
        steps=tuple(steps),
        metrics={
            CLEANUP_ERROR_COUNT: float(cleanup_errors),
            "submissions": float(len(submissions)),
            "passes": float(sum(s.passed for s in submissions)),
        },
    )


def submission_turns(run: AgentRun) -> dict[int, int]:
    """The turn that issued each graded submission, by ordinal, read from the ``submit`` observations."""
    turns = {}
    for index, turn in enumerate(run.turns):
        for result in turn.tool_results:
            if result.call.name != SUBMIT_TOOL_NAME or result.outcome is not ToolOutcome.EXECUTED:
                continue
            ordinal = json.loads(result.output).get("submission")
            if ordinal is not None:
                turns[ordinal] = index
    return turns


def _empty(task: TaskSpec) -> RolloutData:
    return RolloutData(
        task.id,
        tuple(conversation_messages(task.context.events)),
        (),
        (),
        (),
        (),
        GradeResult(GradeStatus.UNAVAILABLE, None, "Execution has no final grade"),
        "error",
    )


@dataclass(frozen=True)
class _Attempt:
    outcome: Outcome
    run: AgentRun | None


async def _agent_attempt(
    lowered: LoweredTaskSpec,
    policy: AdversaryPolicy,
    settings: EngineSettings,
    client: GlmClient,
    record: CallLedger,
    opening: tuple[dict[str, Any], ...],
    submissions: list[Submission],
) -> _Attempt:
    """One agent loop on a machine prepared as RolloutEngine prepares the task machine.

    ``lowered.session.total_turn_timeout`` bounds the loop (the run then ends with stop reason
    ``total_turn_timeout`` and no turns) and ``attempt_timeout`` the whole attempt. A machine that fails to close
    after the loop finished is counted under ``CLEANUP_ERROR_COUNT`` rather than failing the attempt.
    """
    task = lowered.task
    task_messages = session_start(task).messages
    runtime = lowered.runtime.task_machine
    deadline = asyncio.timeout(lowered.session.attempt_timeout)
    run: AgentRun | None = None
    finished = False
    cleanup_errors = 0
    try:
        async with deadline, AsyncExitStack() as resources:
            machine: Machine | None = None
            if runtime is not None:
                try:
                    machine = await resources.enter_async_context(
                        prepare_machine(
                            task.environment_requirements,
                            runtime,
                            (*task.resources.all, *task.resources.worker),
                            settings.factories,
                            cleanup_timeout=settings.cleanup_timeout,
                        )
                    )
                except Exception as error:
                    raise RolloutInterrupted(_empty(task), RolloutOperation.START) from error
            verifier = Verifier(
                lowered,
                task_messages,
                settings.engine(no_model),
                machine,
                policy.adversary_submissions,
                submissions,
            )
            parameters = submit_parameters(machine is not None, isinstance(task.answer_format, FinalAction))
            submit = AgentTool(SUBMIT_TOOL_NAME, SUBMIT_DESCRIPTION, parameters, verifier.submit)
            tools = (submit,)
            if machine is not None:
                shell = shell_tool(machine, timeout=settings.command_timeout, output_limit_bytes=SHELL_OUTPUT_LIMIT)
                tools = (shell, submit)
            agent_deadline = asyncio.timeout(lowered.session.total_turn_timeout)
            try:
                async with agent_deadline:
                    run = await run_agent(client, policy.sampling, opening, tools, settings.max_turns, record)
            except TimeoutError:
                if not agent_deadline.expired():
                    raise
            finished = True
    except Exception as error:
        if not finished or deadline.expired():
            cause = Cause.ATTEMPT_TIMEOUT if deadline.expired() else classify(error, task)
            return _Attempt(Ungraded(cause, "".join(traceback.format_exception(error)), None), None)
        cleanup_errors = 1
    if run is not None:
        turns = submission_turns(run)
        submissions[:] = [replace(s, turn=turns.get(s.ordinal)) for s in submissions]
    stop = TOTAL_TURN_TIMEOUT_STOP_REASON if run is None else AGENT_STOPS[run.stop]
    rollout = agent_rollout(task.id, opening, run, trial_grade(submissions), stop, cleanup_errors, submissions)
    return _Attempt(Graded(rollout), run)


def _claim_attributes(outcome: Outcome, submissions: Sequence[Submission]) -> dict[str, str]:
    return {
        "submissions": str(len(submissions)),
        "passes": str(sum(s.passed for s in submissions)),
        "claim": str(trial_claim(outcome).kind),
    }


def _refuse(lowered: LoweredTaskSpec, plan: TrialPlan, trial: str, system: str, outcome: Ungraded) -> AdversaryTrial:
    with attempt_span(lowered, plan, trial, plan.first_attempt) as fields:
        record_attempt(fields, outcome, plan, trial, plan.first_attempt, adversary_attempt_json(outcome, (), system))
        fields.attrs.update(_claim_attributes(outcome, ()))
    return AdversaryTrial(outcome, system, ())


async def run_adversary_trial(
    draft: TaskDraft,
    policy: AdversaryPolicy,
    plan: TrialPlan,
    settings: EngineSettings,
    client: GlmClient,
    record: CallLedger,
    brief: str,
    trial: str,
) -> AdversaryTrial:
    """One adversary trial: attempts until graded or the retries are spent, each attempt one agent loop.

    Mirrors ``trials.run_trial`` with the agent loop as the attempt body: a task whose answer format cannot
    carry its answer, or that no factory can run, is one refused attempt; attempts are numbered from
    ``plan.first_attempt``; ``RETRYABLE`` causes are retried ``plan.max_retries`` times with
    ``plan.retry_backoff``. There is no token contract to retry. Each attempt is one ``TRIAL`` ledger span
    (step ``adversary/<trial>/<attempt>``) and one attempt file, with a fresh submission budget.
    """
    lowered = settings.apply(plan.deadlines.apply(draft.lowered))
    task = lowered.task
    unsupported = submission_refusal(task)
    if unsupported is not None:
        return _refuse(lowered, plan, trial, brief, Ungraded(Cause.SUBMISSION_UNSUPPORTED, unsupported, None))
    opening = with_preamble(brief, session_start(task).messages)
    system = str(opening[0]["content"])
    refusals = task_refusals(lowered, settings.capabilities)
    if refusals:
        outcome = Ungraded(Cause.MACHINE_UNSUPPORTED, refusal_detail(refusals), None)
        return _refuse(lowered, plan, trial, system, outcome)
    backoff = copy.copy(plan.retry_backoff)
    retries = 0
    attempt = plan.first_attempt
    while True:
        submissions: list[Submission] = []
        with attempt_span(lowered, plan, trial, attempt) as fields:
            result = await _agent_attempt(lowered, policy, settings, client, record, opening, submissions)
            outcome = result.outcome
            payload = adversary_attempt_json(outcome, submissions, system)
            record_attempt(fields, outcome, plan, trial, attempt, payload)
            fields.attrs.update(_claim_attributes(outcome, submissions))
            if result.run is not None:
                fields.tokens_in = result.run.usage.prompt_tokens
                fields.tokens_out = result.run.usage.completion_tokens
        if isinstance(outcome, Graded) or not outcome.retryable or retries >= plan.max_retries:
            return AdversaryTrial(outcome, system, tuple(submissions))
        retries += 1
        await asyncio.sleep(backoff.next_interval())
        attempt += 1


def adversary_trial(role: AdversaryRole, index: int) -> str:
    return f"{role}/{index}"


async def run_adversaries(
    draft: TaskDraft,
    policy: AdversaryPolicy,
    site: ValidationSite,
    settings: EngineSettings,
    client: GlmClient,
    context: str,
) -> Mapping[AdversaryRole, tuple[AdversaryTrial, ...]]:
    """``policy.adversary_k`` trials per ``AdversaryRole`` member, all concurrent, each under
    ``adversary_brief(policy.adversary_submissions, context)``.

    Trials are ``TrialKind.ADVERSARY`` named ``<role>/<index>``, so evidence lands under
    ``adversary/<role>/<index>/``. A trial settled on disk is loaded (``load_adversary_attempt``); an unsettled one
    re-enters with ``first_attempt`` after its files. Each trial's model calls and tool calls are recorded under
    ``site.call_ledger(ADVERSARY, <role>/<index>)``.
    """
    brief = adversary_brief(policy.adversary_submissions, context)
    files = trial_files(site.evidence_dir, TrialKind.ADVERSARY)

    async def trial(name: str) -> AdversaryTrial:
        existing = files.get(name)
        if existing is not None and existing.settled:
            assert existing.last_path is not None
            return load_adversary_attempt(existing.last_path)
        plan = site.trial_plan(
            TrialKind.ADVERSARY, policy.adversary_k, policy, 0 if existing is None else existing.attempts
        )
        record = site.call_ledger(TrialKind.ADVERSARY, name)
        return await run_adversary_trial(draft, policy, plan, settings, client, record, brief, name)

    names = {role: [adversary_trial(role, index) for index in range(policy.adversary_k)] for role in AdversaryRole}
    async with asyncio.TaskGroup() as group:
        runs = {name: group.create_task(trial(name)) for role_names in names.values() for name in role_names}
    return {role: tuple(runs[name].result() for name in role_names) for role, role_names in names.items()}
