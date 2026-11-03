# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adversary trials: an agent that searches a task's grader for defects with the verifier as a tool.

An ``AdversarySpec`` names one adversary and states its goal; a round runs a roster of them, each under its own
``AdversaryPolicy`` and evidence path, and nothing downstream reads the name except as the trial's id. The default
specs are ``FALSE_ACCEPT`` (wrong answers that earn full credit), ``FALSE_REJECT`` (correct answers that lose credit)
and ``SHORTCUT`` (accepted answers that skip the task's computation).

Each attempt is one ``llm.agent.run_agent`` loop with three tools. ``shell`` runs commands on a private task
machine, started as the engine starts the solver's (``rolloutengine.machines.task_machine``: the task's resources,
worker files and setup commands). ``submit`` grades one candidate, a final reply and the files it names on that
machine, exactly as validation replays a control: ``trials.run_trial`` with a ``controls.ScriptedModel`` that
replays ``spec.controls.candidate_turns`` on a fresh machine, so the verifier's files exist only on the grading
machine. ``finish_search`` hands in a ``SearchReport``: the defects found, each naming the graded submissions that
show it and whether it needs addressing, and a summary. It refuses a report before ``min_probes`` distinct
candidates were graded, or whose defects cite no graded submission; an accepted report ends the loop. A task
without a task machine offers ``submit`` and ``finish_search`` alone. ``AdversaryPolicy.submissions`` bounds the
submissions of an attempt; a ``submit`` past it returns an error and grades nothing.

The two modes differ only in the brief (``adversary_brief``). ``BLACKBOX`` gives the task as the solver sees it
(the shell harness's ``task_prompt``), the tools and the budget. ``WHITEBOX`` adds the grader's specification and
verifier files, every control of the round with its expected reward (the reference answer among them) and the
task's example answers, unclipped.

Each attempt is one ``TRIAL`` ledger span, step ``adversary/<trial>/<attempt>``, and the agent's model calls and
tool calls are recorded under the same step. Its attempt file, ``<evidence_dir>/adversary/<trial>/attempt-<n>.json``,
holds ``trials.outcome_json``'s record and an ``adversary`` key, an ``AdversaryRecord``: the spec's name, the model,
the policy's digest, the mode, the brief, the system turn, the report, the grader's digest and every submission with
its grade and exact files (base64 when not UTF-8). The rollout is one conversation whose steps are the agent's
turns; it carries no token ids. Each submission is graded as the ``CONTROL`` trial ``s<ordinal>`` under
``<evidence_dir>/adversary/<trial>/submissions/<attempt>/``, whose attempt file keeps the grader's full result.

An attempt is ``Ungraded`` when any submission's grading or the agent loop itself failed: unavailable execution
takes precedence over an exhausted turn budget, which takes precedence over an authored task defect. An
interrupted attempt keeps its partial rollout and submissions. An attempt that ended without an accepted report is
``SEARCH_INCOMPLETE``. Otherwise it reports its last submission's grade, or ``SUBMISSION_FAILURE`` without one.
Whether a defect is real is review's judgement (``second_review`` is a stub), not the round's.
"""

import asyncio
import base64
import json
import posixpath
import re
import tempfile
import time
import traceback
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import TypeAdapter
from rolloutengine.harness.shell import DEFAULT_SHELL_TOOL, task_prompt
from rolloutengine.machines import task_machine
from rolloutengine.models import (
    LENGTH_STOP_REASON,
    MAX_TURNS_STOP_REASON,
    TOTAL_TURN_TIMEOUT_STOP_REASON,
    ConversationRecord,
    ModelTurn,
    Rollout,
    RolloutInterrupted,
    RolloutOperation,
    RolloutStep,
    TaskRolloutSpec,
    TokenUsage,
    Transition,
)
from shellbox.machine import Command, Machine
from taskcompendium.grading_result import GradeResult
from taskcompendium.grading_result import Outcome as GradeStatus
from taskcompendium.models import TaskResource, TaskSpec
from taskcompendium.runtime.resources import inline_resource, resource_bytes
from taskcompendium.runtime.shell import resolve_shell_tool
from taskcompendium.submission import conversation_messages

from taskforge.content_hash import canonical_json, digest
from taskforge.llm.agent import (
    AgentInterrupted,
    AgentRun,
    AgentStop,
    AgentTool,
    AgentTurn,
    ReasoningReplay,
    ToolOutcome,
    assistant_message,
    run_agent,
    shell_tool,
)
from taskforge.llm.client import Completion, FinishReason, ModelClient
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.recording import CallLedger
from taskforge.sandbox.host_failures import machine_lost
from taskforge.spec.controls import (
    Control,
    ScriptedModel,
    Transcript,
    Workspace,
    candidate_turns,
    context_assistant_turns,
)
from taskforge.spec.draft import GRADER_RESOURCE_ROOT
from taskforge.stage50_validate.adversary_prompts import (
    BRIEF_TEMPLATE,
    FALSE_ACCEPT_GOAL,
    FALSE_REJECT_GOAL,
    NO_CONTROLS,
    NO_EXAMPLES,
    NO_VERIFIER_FILES,
    SHORTCUT_GOAL,
    SUBMIT_DESCRIPTION,
    TOOLS_SUBMIT_ONLY,
    TOOLS_WITH_SHELL,
    VERIFIER_FAILED,
    WHITEBOX_HEADER,
)
from taskforge.stage50_validate.classify import classify
from taskforge.stage50_validate.evidence import passed
from taskforge.stage50_validate.outcome import GRADED_STATUSES, TASK_DEFECTS, Cause, Graded, Outcome, TrialKind, Ungraded
from taskforge.stage50_validate.trials import (
    CLEANUP_ERROR_COUNT,
    EngineSettings,
    TrialPlan,
    attempt_span,
    outcome_json,
    record_attempt,
    refusal,
    run_attempts,
    run_trial,
    task_digest,
)

SUBMIT_TOOL_NAME = "submit"
SUBMISSIONS_DIR = "submissions"
SUBMISSION_PREFIX = "s"
ADVERSARY_KEY = "adversary"
PREAMBLE_SEPARATOR = "\n\n"
SHELL_OUTPUT_LIMIT = 64 * 1024
CAPTURED_FILE_MODE = "644"
FILE_PROBE_TIMEOUT = 60.0
CONTEXT_STOP_REASON = "context"
MACHINE_LOST_STOP_REASON = "machine_lost"
ADVERSARY_NAME = re.compile(r"[a-z0-9_]+")
"""An adversary's name, which names its trials' directories (``adversary/<name>-<i>``)."""


class AdversaryMode(StrEnum):
    BLACKBOX = "blackbox"
    """The task as the solver sees it, ``shell`` and ``submit``."""
    WHITEBOX = "whitebox"
    """Black box plus the grader, the round's controls with their expected rewards and the example answers."""


@dataclass(frozen=True)
class AdversarySpec:
    """One adversary of a round's roster: ``name``, unique in the roster, and ``brief``, the goal its brief states.

    The roster is data: a new adversary is a new spec, and neither the round nor review branches on a name.
    """

    name: str
    brief: str

    def __post_init__(self) -> None:
        if not ADVERSARY_NAME.fullmatch(self.name) or not self.brief.strip():
            raise ValueError(f"AdversarySpec needs a name matching {ADVERSARY_NAME.pattern} and a non-empty brief")


FALSE_ACCEPT = AdversarySpec("false_accept", FALSE_ACCEPT_GOAL)
FALSE_REJECT = AdversarySpec("false_reject", FALSE_REJECT_GOAL)
SHORTCUT = AdversarySpec("shortcut", SHORTCUT_GOAL)


@dataclass(frozen=True)
class AdversaryPolicy:
    """How a validation round runs one adversary's trials; the round's deadlines, retries and backoff apply to them.

    Attributes:
        spec: The adversary these trials run.
        k: Adversary trials per round; 0 runs none.
        submissions: Verifier submissions per attempt; ``None`` is unbounded.
        max_turns: Agent turns per attempt.
        sampling: The adversary's sampling policy; an agent loop has no continuation.
        mode: What the adversary knows of the grader.
        min_probes: Distinct candidates an attempt grades before ``finish_search`` accepts its report.
    """

    spec: AdversarySpec
    k: int
    submissions: int | None
    max_turns: int | None
    sampling: LLMPolicy
    mode: AdversaryMode
    min_probes: int

    def __post_init__(self) -> None:
        if (
            self.k < 0
            or (self.submissions is not None and self.submissions < 1)
            or (self.max_turns is not None and self.max_turns < 1)
            or self.min_probes < 1
        ):
            raise ValueError("Search counts must be positive, except k may disable the adversary with zero")
        if self.submissions is not None and self.min_probes > self.submissions:
            raise ValueError("min_probes cannot exceed the submission budget")


@dataclass(frozen=True)
class Defect:
    """A submitted defect, backed by tested candidates, with the finder's requested disposition."""

    submission_ids: tuple[int, ...]
    explanation: str
    needs_addressing: bool


@dataclass(frozen=True)
class SearchReport:
    """An explicit completed search; probe candidates alone are not defect submissions."""

    defects: tuple[Defect, ...]
    summary: str


@dataclass(frozen=True)
class SubmittedFile:
    """An exact candidate file, preserved as text or base64."""

    path: str
    content: str
    encoding: str = "utf-8"


@dataclass(frozen=True)
class SubmissionRecord:
    """One graded submission: ``trial`` is the ``CONTROL`` trial whose attempt file holds its full grade, and
    ``turn`` the agent turn that submitted it."""

    ordinal: int
    turn: int | None
    trial: str
    reply: str
    files: tuple[SubmittedFile, ...]
    grade: GradeResult
    passed: bool
    wall_time: float


@dataclass(frozen=True)
class AdversaryRecord:
    """The ``adversary`` key of an adversary attempt file. ``system`` is empty for a trial that was not started."""

    name: str
    model: str
    policy_digest: str
    mode: AdversaryMode
    brief: str
    system: str
    report: SearchReport | None
    grader_digest: str
    submissions: tuple[SubmissionRecord, ...]


def adversary_record(payload: bytes) -> AdversaryRecord:
    """The ``AdversaryRecord`` of an adversary attempt file; ``trials.attempt_outcome`` reads its outcome."""
    return _RECORD.validate_python(json.loads(payload)[ADVERSARY_KEY])


def _text(resource: TaskResource) -> str:
    data = resource_bytes(resource)
    try:
        return data.decode()
    except UnicodeDecodeError:
        return "base64:" + base64.b64encode(data).decode("ascii")


def submitted_file(resource: TaskResource) -> SubmittedFile:
    data = resource_bytes(resource)
    try:
        return SubmittedFile(f"/{resource.path}", data.decode())
    except UnicodeDecodeError:
        return SubmittedFile(f"/{resource.path}", base64.b64encode(data).decode("ascii"), "base64")


def grader_source(task: TaskSpec) -> dict[str, Any]:
    """The grader as a white-box adversary sees it: its specification and its verifier files at their grading paths."""
    return {
        "grader_spec": task.grader.model_dump(mode="json"),
        "grader_source": [
            {"path": str(PurePosixPath(GRADER_RESOURCE_ROOT, r.path)), "content": _text(r)}
            for r in task.resources.verifier
        ],
    }


def control_view(control: Control) -> dict[str, Any]:
    """A control as a white-box adversary sees it: its labels, its expectation and its payload."""
    payload = control.payload
    if isinstance(payload, Transcript):
        body: dict[str, Any] = {"transcript": [turn.model_dump(mode="json") for turn in payload.turns]}
    else:
        assert isinstance(payload, Workspace)
        body = {"workspace_files": [{"path": f"/{file.path}", "content": _text(file)} for file in payload.files]}
    expect = control.expect
    return {
        "id": control.id,
        "kind": str(control.kind),
        "category": str(control.category),
        "concern": str(control.concern),
        "expect": {"status": str(expect.status), "reward_min": expect.reward_min, "reward_max": expect.reward_max},
        **body,
    }


def whitebox_context(task: TaskSpec, controls: Sequence[Control]) -> str:
    """The white-box section of the brief: the grader, the controls with their expected rewards, the example answers."""
    source = grader_source(task)
    spec = json.dumps(source["grader_spec"], indent=1, ensure_ascii=False)
    files = (
        "\n\n".join(f"### {file['path']}\n```\n{file['content']}\n```" for file in source["grader_source"])
        or NO_VERIFIER_FILES
    )
    views = [control_view(control) for control in controls]
    control_text = (
        "\n\n".join(
            f"### control {view['id']} ({view['kind']}, {view['category']}, concern {view['concern']}; "
            f"expected {json.dumps(view['expect'])})\n"
            + json.dumps(
                {key: value for key, value in view.items() if key in ("transcript", "workspace_files")},
                ensure_ascii=False,
                indent=1,
            )
            for view in views
        )
        or NO_CONTROLS
    )
    examples = [message for message in conversation_messages(task.context.events) if message.get("role") == "assistant"]
    example_text = "\n\n".join(json.dumps(m, ensure_ascii=False) for m in examples) if examples else NO_EXAMPLES
    return (
        f"{WHITEBOX_HEADER}\n\n## Grader specification\n```json\n{spec}\n```\n\n## Grader source\n\n{files}\n\n"
        f"## Controls with expected rewards\n\n{control_text}\n\n"
        f"## Example answers in the task's conversation\n\n{example_text}"
    )


def adversary_brief(spec: TaskRolloutSpec, controls: Sequence[Control], policy: AdversaryPolicy) -> str:
    """The brief of an adversary on ``spec``: the white-box section of ``controls`` only in ``WHITEBOX`` mode."""
    whitebox = policy.mode is AdversaryMode.WHITEBOX
    context = whitebox_context(spec.task, controls) if whitebox else ""
    return BRIEF_TEMPLATE.format(
        goal=policy.spec.brief,
        tools=TOOLS_WITH_SHELL if spec.task_machine is not None else TOOLS_SUBMIT_ONLY,
        submissions="unlimited" if policy.submissions is None else policy.submissions,
        probes=policy.min_probes,
        context=f"{PREAMBLE_SEPARATOR}{context}{PREAMBLE_SEPARATOR}" if context else " ",
    )


def opening_messages(spec: TaskRolloutSpec, brief: str) -> tuple[dict[str, Any], ...]:
    """The brief as the system turn over the task as the solver sees it: the shell harness's ``task_prompt`` with
    its default shell tool when the spec has a task machine."""
    task = spec.task
    shell = resolve_shell_tool(task, DEFAULT_SHELL_TOOL) if spec.task_machine is not None else None
    messages = task_prompt(task, shell, ()).messages
    if messages and messages[0]["role"] == "system":
        first, *rest = messages
        content = first["content"]
        if not isinstance(content, str):
            raise ValueError("A brief needs a text system message to precede")
        return ({**first, "content": f"{brief}{PREAMBLE_SEPARATOR}{content}"}, *(dict(m) for m in rest))
    return ({"role": "system", "content": brief}, *(dict(m) for m in messages))


def submit_parameters(executable: bool) -> dict[str, object]:
    properties: dict[str, object] = {"reply": {"type": "string"}}
    if executable:
        properties["files"] = {"type": "array", "items": {"type": "string", "pattern": "^/"}}
    return {"type": "object", "properties": properties, "required": ["reply"], "additionalProperties": False}


class BadPath(ValueError):
    """A listed file path is not absolute and normalized; its argument is the path."""


def normalized_paths(paths: Sequence[str]) -> list[str]:
    """``paths`` normalized (``/workspace/../tests/x`` is ``/tests/x``); each must be absolute and stay so.

    Raises:
        BadPath: a path is relative or is the root.
    """
    out = []
    for path in paths:
        if not path.startswith("/"):
            raise BadPath(path)
        clean = posixpath.normpath(path)
        if not clean.startswith("/") or clean == "/":
            raise BadPath(path)
        out.append(clean)
    return out


async def capture(machine: Machine, paths: Sequence[str]) -> tuple[TaskResource, ...]:
    """The files at ``paths`` on ``machine``, as resources at the same paths.

    Raises:
        FileNotFoundError: a path is not a regular file; its argument is the path.
        BadPath: a path is not absolute.
    """
    files = []
    with tempfile.TemporaryDirectory() as directory:
        for index, path in enumerate(sorted(set(normalized_paths(paths)))):
            probe = await machine.run(Command(argv=("test", "-f", path), timeout=FILE_PROBE_TIMEOUT))
            if probe.exit_code != 0:
                raise FileNotFoundError(path)
            target = Path(directory) / str(index)
            await machine.download(path, target)
            resource = inline_resource(str(PurePosixPath(path).relative_to("/")), target.read_bytes())
            files.append(resource.model_copy(update={"mode": CAPTURED_FILE_MODE}))
    return tuple(files)


def _submission_grade(outcome: Outcome) -> GradeResult:
    if isinstance(outcome, Graded):
        return outcome.grade
    return GradeResult(GradeStatus.UNAVAILABLE, None, f"{outcome.cause}: {outcome.detail[:500]}")


def _observation(submission: SubmissionRecord, budget: int | None) -> dict[str, object]:
    """What ``submit`` returns to the adversary: the grade, or that the verifier failed."""
    grade = submission.grade
    head: dict[str, object] = {
        "submission": submission.ordinal,
        "remaining": None if budget is None else budget - submission.ordinal,
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


@dataclass(frozen=True)
class Submission:
    """A recorded candidate and the exact verifier outcome it produced."""

    record: SubmissionRecord
    outcome: Outcome


@dataclass
class Verifier:
    """The ``submit`` handler of one attempt: grades each candidate as a control is graded, on a fresh machine."""

    spec: TaskRolloutSpec
    settings: EngineSettings
    plan: TrialPlan
    machine: Machine | None
    budget: int | None
    min_probes: int
    submissions: list[Submission] = field(default_factory=list)
    report: SearchReport | None = None

    async def finish(self, arguments: Mapping[str, object]) -> str:
        report = TypeAdapter(SearchReport).validate_python(arguments)
        known = {submission.record.ordinal: submission for submission in self.submissions}
        distinct = {digest((item.record.reply, item.record.files)) for item in self.submissions}
        if len(distinct) < self.min_probes:
            return json.dumps({"error": f"submit at least {self.min_probes} candidates before finishing"})
        for defect in report.defects:
            if not defect.explanation.strip() or not defect.submission_ids:
                return json.dumps({"error": "each defect needs an explanation and candidate evidence"})
            if len(set(defect.submission_ids)) != len(defect.submission_ids):
                return json.dumps({"error": "submission_ids must be unique within each defect"})
            for ordinal in defect.submission_ids:
                submission = known.get(ordinal)
                if submission is None or not isinstance(submission.outcome, Graded):
                    return json.dumps({"error": f"submission {ordinal} has no completed grade"})
        self.report = report
        return json.dumps({"completed": True})

    async def submit(self, arguments: Mapping[str, object]) -> str:
        if self.budget is not None and len(self.submissions) >= self.budget:
            return json.dumps(
                {"error": f"submission budget spent: {self.budget} of {self.budget}; call finish_search now"}
            )
        reply = arguments["reply"]
        paths = arguments.get("files", [])
        assert isinstance(reply, str) and isinstance(paths, list)
        try:
            files = () if self.machine is None or not paths else await capture(self.machine, [str(p) for p in paths])
        except FileNotFoundError as error:
            return json.dumps({"error": f"no file at {error.args[0]} in your workspace"})
        except BadPath as error:
            return json.dumps({"error": f"{error.args[0]} is not an absolute, normalized path in your workspace"})
        ordinal = len(self.submissions) + 1
        name = f"{SUBMISSION_PREFIX}{ordinal}"
        started = time.monotonic()
        model = ScriptedModel(candidate_turns(reply, files), context_assistant_turns(self.spec.task))
        outcome = await run_trial(self.spec, self.plan, self.settings, model, name)
        grade = _submission_grade(outcome)
        submission = SubmissionRecord(
            ordinal=ordinal,
            turn=None,
            trial=name,
            reply=reply,
            files=tuple(submitted_file(file) for file in files),
            grade=grade,
            passed=passed(grade),
            wall_time=time.monotonic() - started,
        )
        self.submissions.append(Submission(submission, outcome))
        return json.dumps(_observation(submission, self.budget))


def _failure(rollout: Rollout, failures: Sequence[Ungraded]) -> Ungraded:
    # Unavailable execution takes precedence over exhausted budgets, then authored defects.
    chosen = min(failures, key=lambda failure: (failure.cause in TASK_DEFECTS, failure.cause is Cause.BUDGET_EXHAUSTED))
    return replace(chosen, rollout=rollout)


def trial_outcome(rollout: Rollout, submissions: Sequence[Submission]) -> Outcome:
    """Retain failures across every submission and the search's own completion status."""
    failures = [submission.outcome for submission in submissions if isinstance(submission.outcome, Ungraded)]
    stop_causes = {
        MACHINE_LOST_STOP_REASON: Cause.MACHINE_TERMINATED,
        TOTAL_TURN_TIMEOUT_STOP_REASON: Cause.AGENT_TIMEOUT,
        MAX_TURNS_STOP_REASON: Cause.BUDGET_EXHAUSTED,
        LENGTH_STOP_REASON: Cause.GENERATION_LIMIT,
        CONTEXT_STOP_REASON: Cause.GENERATION_LIMIT,
    }
    if cause := stop_causes.get(rollout.stop_reason):
        failures.append(Ungraded(cause, f"adversary stopped at {rollout.stop_reason}", rollout, None))
    if failures:
        return _failure(rollout, failures)
    chosen = submissions[-1] if submissions else None
    if chosen is None:
        return Graded(rollout, GradeResult(GradeStatus.SUBMISSION_FAILURE, None, "no verifier submission"), False)
    return replace(chosen.outcome, rollout=rollout)


AGENT_STOPS: Mapping[AgentStop, str] = {
    AgentStop.ANSWERED: "stop",
    AgentStop.MAX_TURNS: MAX_TURNS_STOP_REASON,
    AgentStop.LENGTH: LENGTH_STOP_REASON,
    AgentStop.CONTEXT: CONTEXT_STOP_REASON,
    AgentStop.STOPPED: "stop",
    AgentStop.INTERRUPTED: "interrupted",
}


def agent_turn(completion: Completion) -> ModelTurn:
    cut = completion.finish_reason is FinishReason.LENGTH
    return ModelTurn(
        message=assistant_message(completion, ReasoningReplay.KEEP),
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
        usage=TokenUsage(completion.usage.prompt_tokens, completion.usage.completion_tokens),
    )


def agent_rollout(
    run_id: str,
    opening: Sequence[Mapping[str, Any]],
    run: AgentRun | None,
    stop_reason: str,
    cleanup_errors: int,
    submissions: Sequence[SubmissionRecord],
) -> Rollout:
    """The agent loop as a one-conversation ``Rollout``: a step per model turn, its tool results as observations."""
    conversation = [dict(message) for message in opening]
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
        issued = sum(submission.turn == index for submission in submissions)
        transition = Transition(
            done=index == len(turns) - 1, observations=observations, metrics={"submissions": float(issued)}
        )
        steps.append(RolloutStep(turn=model_turn, transition=transition, response_end=0, messages=messages))
    record = ConversationRecord(
        messages=tuple(conversation),
        prompt_token_ids=(),
        response_token_ids=(),
        loss_mask=(),
        logprobs=None,
        steps=tuple(steps),
    )
    return Rollout(
        run_id=run_id,
        conversations=(record,),
        stop_reason=stop_reason,
        metrics={
            CLEANUP_ERROR_COUNT: float(cleanup_errors),
            "submissions": float(len(submissions)),
            "passes": float(sum(submission.passed for submission in submissions)),
        },
    )


def submission_turns(run: AgentRun) -> dict[int, int]:
    """The agent turn of each submission, by ordinal, read from the ``submit`` results."""
    turns = {}
    for index, turn in enumerate(run.turns):
        for result in turn.tool_results:
            if result.call.name != SUBMIT_TOOL_NAME or result.outcome is not ToolOutcome.EXECUTED:
                continue
            ordinal = json.loads(result.output).get("submission")
            if ordinal is not None:
                turns[ordinal] = index
    return turns


@asynccontextmanager
async def attempt_machine(spec: TaskRolloutSpec, settings: EngineSettings, run_id: str) -> AsyncIterator[Machine | None]:
    """The attempt's private task machine, or ``None`` when the spec has none. A start failure is raised as the
    engine's ``RolloutInterrupted`` during ``START`` with the failure as its cause, so ``classify`` names it as it
    names a solver's."""
    runtime = spec.task_machine
    if runtime is None:
        yield None
        return
    async with AsyncExitStack() as stack:
        try:
            machine = await stack.enter_async_context(
                task_machine(
                    spec.task, settings.factories[runtime.backend], runtime, cleanup_timeout=spec.limits.cleanup_timeout
                )
            )
        except Exception as error:
            raise RolloutInterrupted(Rollout(run_id, (), "error"), RolloutOperation.START) from error
        yield machine


async def _agent_attempt(
    spec: TaskRolloutSpec,
    policy: AdversaryPolicy,
    settings: EngineSettings,
    model: ModelClient,
    record: CallLedger,
    opening: tuple[dict[str, Any], ...],
    submission_plan: TrialPlan,
    run_id: str,
) -> tuple[Outcome, tuple[Submission, ...], SearchReport | None]:
    """One agent loop on a private task machine under the attempt deadline, its turns under the total-turn deadline.

    A failure before the loop finished is ``Ungraded`` with ``classify``'s cause; a machine close that fails after
    it counts as a cleanup error of the rollout.
    """
    deadline = asyncio.timeout(spec.limits.attempt_timeout)
    run: AgentRun | None = None
    finished = False
    lost = False
    cleanup_errors = 0
    verifier: Verifier | None = None

    def finished_turn(_: AgentTurn) -> bool:
        return verifier is not None and verifier.report is not None

    try:
        async with deadline, attempt_machine(spec, settings, run_id) as machine:
            verifier = Verifier(spec, settings, submission_plan, machine, policy.submissions, policy.min_probes)
            submit = AgentTool(
                SUBMIT_TOOL_NAME, SUBMIT_DESCRIPTION, submit_parameters(machine is not None), verifier.submit
            )
            finish = AgentTool(
                "finish_search",
                "Report defects with candidate IDs, explanation and needs_addressing; use no defects if none found.",
                TypeAdapter(SearchReport).json_schema(),
                verifier.finish,
            )
            tools: tuple[AgentTool, ...] = (submit, finish)
            if machine is not None:
                shell = shell_tool(machine, timeout=settings.command_timeout, output_limit_bytes=SHELL_OUTPUT_LIMIT)
                tools = (shell, submit, finish)
            agent_deadline = asyncio.timeout(spec.limits.total_turn_timeout)
            try:
                async with agent_deadline:
                    run = await run_agent(
                        model,
                        policy.sampling,
                        opening,
                        tools,
                        policy.max_turns,
                        record,
                        stop=finished_turn,
                    )
            except AgentInterrupted as error:
                run = error.run
                raise
            except TimeoutError:
                if not agent_deadline.expired():
                    raise
            except RuntimeError as error:
                if not machine_lost(error):
                    raise
                lost = True
            finished = True
    except Exception as error:
        if not finished or deadline.expired():
            if deadline.expired():
                interrupted = RolloutInterrupted(Rollout(run_id, (), "error"), RolloutOperation.ATTEMPT)
                interrupted.__cause__ = error
                error = interrupted
            submissions = () if verifier is None else tuple(verifier.submissions)
            rollout = agent_rollout(run_id, opening, run, "interrupted", 0, tuple(s.record for s in submissions))
            outcome = Ungraded(classify(error, spec.task), "".join(traceback.format_exception(error)), rollout, None)
            failures = [s.outcome for s in submissions if isinstance(s.outcome, Ungraded)]
            return _failure(rollout, [*failures, outcome]), submissions, None
        cleanup_errors = 1
    submissions = () if verifier is None else tuple(verifier.submissions)
    if run is not None:
        turns = submission_turns(run)
        submissions = tuple(
            replace(submission, record=replace(submission.record, turn=turns.get(submission.record.ordinal)))
            for submission in submissions
        )
    if lost:
        stop = MACHINE_LOST_STOP_REASON
    else:
        stop = TOTAL_TURN_TIMEOUT_STOP_REASON if run is None else AGENT_STOPS[run.stop]
    rollout = agent_rollout(
        run_id, opening, run, stop, cleanup_errors, tuple(submission.record for submission in submissions)
    )
    outcome = trial_outcome(rollout, submissions)
    report = None if verifier is None else verifier.report
    if isinstance(outcome, Graded) and report is None:
        outcome = Ungraded(Cause.SEARCH_INCOMPLETE, "search did not finish with a valid report", rollout, outcome.grade)
    return outcome, submissions, report


def _attempt_json(outcome: Outcome, record: AdversaryRecord) -> bytes:
    payload = json.loads(outcome_json(outcome))
    payload[ADVERSARY_KEY] = json.loads(canonical_json(record))
    return json.dumps(payload, ensure_ascii=False, indent=1).encode()


async def run_adversary_trial(
    spec: TaskRolloutSpec,
    brief: str,
    policy: AdversaryPolicy,
    plan: TrialPlan,
    settings: EngineSettings,
    model: ModelClient,
    trial: str,
    submission_retries: int,
) -> Outcome:
    """Run a search, retrying retryable failures up to ``plan.max_retries`` times.

    Each retry starts a new attempt; prior partial rollouts and submissions remain archived.

    Args:
        spec: The lowered spec; its limits already include the validation policy.
        brief: The adversary's brief.
        policy: The adversary policy.
        plan: An ``ADVERSARY`` plan; ``plan.first_attempt`` numbers the first attempt.
        settings: The engine settings of the adversary's machine and of every submission's grading.
        model: The adversary's model.
        trial: The trial's name.
        submission_retries: The retries of a submission's grading.
    """
    trial_digest = task_digest(spec, settings.command_timeout)
    grader_digest = digest(grader_source(spec.task))
    refused = refusal(spec)
    if refused is not None:
        record = AdversaryRecord(
            policy.spec.name, model.model, digest(policy), policy.mode, brief, "", None, grader_digest, ()
        )
        with attempt_span(trial_digest, plan, trial, plan.first_attempt) as fields:
            record_attempt(fields, refused, plan, trial, plan.first_attempt, _attempt_json(refused, record))
        return refused
    opening = opening_messages(spec, brief)
    system = str(opening[0]["content"])

    async def attempt_once(attempt: int, step: str) -> Outcome:
        submission_plan = replace(
            plan,
            kind=TrialKind.CONTROL,
            k=1,
            max_retries=submission_retries,
            evidence_dir=plan.evidence_dir / str(TrialKind.ADVERSARY) / trial / SUBMISSIONS_DIR / str(attempt),
            first_attempt=0,
        )
        calls = CallLedger(plan.ledger, plan.workpiece_id, plan.round, step)
        with attempt_span(trial_digest, plan, trial, attempt) as fields:
            outcome, submissions, report = await _agent_attempt(
                spec, policy, settings, model, calls, opening, submission_plan, step
            )
            record = AdversaryRecord(
                policy.spec.name,
                model.model,
                digest(policy),
                policy.mode,
                brief,
                system,
                report,
                grader_digest,
                tuple(submission.record for submission in submissions),
            )
            record_attempt(fields, outcome, plan, trial, attempt, _attempt_json(outcome, record))
            fields.attrs.update(
                {
                    "mode": str(policy.mode),
                    "submissions": str(len(submissions)),
                    "passes": str(sum(submission.record.passed for submission in submissions)),
                    "defects": json.dumps([asdict(defect) for defect in report.defects] if report else None),
                    "adversary": policy.spec.name,
                }
            )
        return outcome

    return await run_attempts(plan, trial, attempt_once)


_RECORD: TypeAdapter[AdversaryRecord] = TypeAdapter(AdversaryRecord)


def second_review(spec: TaskRolloutSpec, search: AdversaryRecord, defect: Defect) -> None:
    """Optional future review with the full task and candidate evidence; not performed."""
    return None
