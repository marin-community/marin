# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adversary trials: an agent that red-teams a task's grader with the verifier as a tool.

Each attempt is one ``llm.agent.run_agent`` loop with two tools. ``shell`` runs commands on a private task
machine, started as the engine starts the solver's (``rolloutengine.machines.task_machine``: the task's resources,
worker files and setup commands). ``submit`` grades one candidate, a final reply and the files it names on that
machine, exactly as validation replays a control: ``trials.run_trial`` with a ``controls.ScriptedModel`` that
replays ``spec.controls.candidate_turns`` on a fresh machine, so the verifier's files exist only on the grading
machine. A decision-model task is graded there on the host with ``EngineSettings.decision``, as its solver trials are
(``spec.decision.run_decision_trial``), and a host without a decision client refuses it before the first attempt as
``run_trial`` does. A task without a task machine offers ``submit`` alone. ``AdversaryPolicy.submissions`` bounds the
submissions of an attempt; a ``submit`` past it returns an error and grades nothing.

The two modes differ only in the brief (``adversary_brief``). ``BLACKBOX`` gives the task as the solver sees it
(the shell harness's ``task_prompt``), the tools, the budget and the verdict line. ``WHITEBOX`` adds the grader's
specification and verifier files, every control of the round with its expected reward (the reference answer among
them) and the task's example answers, and asks the adversary to name the grader line its submission defeats. A
decision-model grader has no program: its specification is ``decision_view``, what its decision model reads (the
rubric and the reference among it), the question and levels it answers and the bar that sets the reward.

Each attempt is one ``TRIAL`` ledger span, step ``adversary/<trial>/<attempt>``, and the agent's model calls and
tool calls are recorded under the same step. Its attempt file, ``<evidence_dir>/adversary/<trial>/attempt-<n>.json``,
holds ``trials.outcome_json``'s record and an ``adversary`` key, an ``AdversaryRecord``: the mode, the brief, the
system turn, the adversary's claim, the grader's digest and every submission with its grade. The rollout is one
conversation whose steps are the agent's turns; it carries no token ids. Each submission is graded as the
``CONTROL`` trial ``s<ordinal>`` under ``<evidence_dir>/adversary/<trial>/submissions/<attempt>/``, whose attempt
file keeps the grader's full result.

An attempt that ran its loop is ``Graded`` with the grade of its last passing submission, else of its last
submission, and ``SUBMISSION_FAILURE`` without one. An attempt whose machine did not start or whose deadline
expired is ``Ungraded`` with the cause ``classify`` gives it, and is retried like a solver attempt. The final
reply's verdict line is recorded as the adversary's ``Claim``; whether a passing submission is a shortcut is not
judged here.
"""

import asyncio
import copy
import json
import posixpath
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
    AgentRun,
    AgentStop,
    AgentTool,
    ReasoningReplay,
    ToolOutcome,
    assistant_message,
    run_agent,
    shell_tool,
)
from taskforge.llm.client import Completion, FinishReason, ModelClient
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.recording import CallLedger
from taskforge.sandbox.factories import task_refusals
from taskforge.spec.controls import Control, Transcript, Workspace, candidate_turns
from taskforge.spec.decision import (
    LEVELS,
    QUESTION,
    REJECTED_REWARD_MAX,
    DecisionGrader,
    StateSource,
    decision_state,
    task_decision_grader,
)
from taskforge.stage50_validate.adversary_prompts import (
    BRIEF_TEMPLATE,
    DECISION_LINE_CLAUSE,
    DECISION_WHITEBOX_HEADER,
    GRADER_LINE_MARKER,
    NO_CONTROLS,
    NO_EXAMPLES,
    NO_SHORTCUT_LINE,
    NO_VERIFIER_FILES,
    SHORTCUT_PREFIX,
    SUBMIT_DESCRIPTION,
    TOOLS_SUBMIT_ONLY,
    TOOLS_WITH_SHELL,
    VERIFIER_FAILED,
    WHITEBOX_HEADER,
    WHITEBOX_LINE_CLAUSE,
)
from taskforge.stage50_validate.classify import classify
from taskforge.stage50_validate.controls import ScriptedModel, context_assistant_turns
from taskforge.stage50_validate.outcome import GRADED_STATUSES, Cause, Graded, Outcome, TrialKind, Ungraded
from taskforge.stage50_validate.trials import (
    CLEANUP_ERROR_COUNT,
    EngineSettings,
    TrialPlan,
    attempt_span,
    attempt_step,
    outcome_json,
    record_attempt,
    refusal_detail,
    require_decision_client,
    rollout_steps,
    run_trial,
    submission_refusal,
    task_digest,
)

SUBMIT_TOOL_NAME = "submit"
GRADER_RESOURCE_ROOT = "/tests"
"""Where the verifier's resources are installed on the grading machine."""
SUBMISSIONS_DIR = "submissions"
SUBMISSION_PREFIX = "s"
ADVERSARY_KEY = "adversary"
PREAMBLE_SEPARATOR = "\n\n"
SHELL_OUTPUT_LIMIT = 64 * 1024
CAPTURED_FILE_MODE = "644"
FILE_PROBE_TIMEOUT = 60.0
CONTEXT_STOP_REASON = "context"
MACHINE_LOST_STOP_REASON = "machine_lost"
MACHINE_LOST_MESSAGES = ("Machine is closed", "ShellSim terminated")
GRADER_FILE_CHARS = 60_000
"""Characters of each verifier file a white-box brief shows; a longer file is cut with a marker."""
CONTROL_CHARS = 4_000
"""Characters of each workspace file of a control a white-box brief shows."""
VERDICT_DECORATION = "*_`#> "
"""Markdown a model wraps a verdict line in (``**SHORTCUT:**``, a quote or heading marker), stripped before matching."""


class AdversaryMode(StrEnum):
    BLACKBOX = "blackbox"
    """The task as the solver sees it, ``shell`` and ``submit``."""
    WHITEBOX = "whitebox"
    """Black box plus the grader, the round's controls with their expected rewards and the example answers."""


@dataclass(frozen=True)
class AdversaryPolicy:
    """How a validation round runs its adversary trials; the round's deadlines, retries and backoff apply to them.

    Attributes:
        k: Adversary trials per round; 0 runs none.
        submissions: Verifier submissions per attempt.
        repair_submissions: The submissions within which a first accepted shortcut makes a grader repair finding.
            The round records each submission's ordinal and grade; review applies this bound.
        max_turns: Agent turns per attempt.
        sampling: The adversary's sampling policy; an agent loop has no continuation.
        mode: What the adversary knows of the grader.
        min_probes: Shortcut candidates the brief asks for before a verdict, capped below the budget.
    """

    k: int
    submissions: int
    repair_submissions: int
    max_turns: int
    sampling: LLMPolicy
    mode: AdversaryMode
    min_probes: int

    def __post_init__(self) -> None:
        if self.k < 0 or self.submissions < 1 or self.max_turns < 1 or self.min_probes < 0:
            raise ValueError("AdversaryPolicy needs k >= 0, submissions >= 1, max_turns >= 1 and min_probes >= 0")
        if not 0 <= self.repair_submissions <= self.submissions:
            raise ValueError("AdversaryPolicy needs 0 <= repair_submissions <= submissions")
        if self.sampling.max_continuations != 0:
            raise ValueError("AdversaryPolicy needs sampling.max_continuations == 0: an agent loop has no continuation")


class ClaimKind(StrEnum):
    SHORTCUT = "shortcut"
    NO_SHORTCUT = "no_shortcut"
    NONE = "none"
    """The final reply has no verdict line, was cut off, or the attempt ended without a reply."""


@dataclass(frozen=True)
class Claim:
    """The adversary's verdict: for a shortcut, why the accepted submission does not do the task and, in a
    white-box reply, the grader line it defeats."""

    kind: ClaimKind
    why: str
    grader_line: str


NO_CLAIM = Claim(ClaimKind.NONE, "", "")


@dataclass(frozen=True)
class SubmittedFile:
    """A file of a submission; ``content`` is its text, or a note of its size when it is not UTF-8."""

    path: str
    content: str


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

    mode: AdversaryMode
    brief: str
    system: str
    claim: Claim
    grader_digest: str
    submissions: tuple[SubmissionRecord, ...]


def adversary_record(payload: bytes) -> AdversaryRecord:
    """The ``AdversaryRecord`` of an adversary attempt file; ``trials.attempt_outcome`` reads its outcome."""
    return _RECORD.validate_python(json.loads(payload)[ADVERSARY_KEY])


def _verdict_line(line: str) -> str:
    stripped = line.strip().lstrip(VERDICT_DECORATION)
    if stripped.startswith(SHORTCUT_PREFIX):
        return f"{SHORTCUT_PREFIX} {stripped.removeprefix(SHORTCUT_PREFIX).lstrip(VERDICT_DECORATION).lstrip()}".strip()
    if stripped.startswith(NO_SHORTCUT_LINE):
        return NO_SHORTCUT_LINE + stripped.removeprefix(NO_SHORTCUT_LINE).strip().rstrip(VERDICT_DECORATION)
    return stripped


def parse_claim(reply: str) -> Claim:
    """The verdict of a final reply: its last line opening with ``NO_SHORTCUT`` or ``SHORTCUT:``, markdown emphasis
    stripped. A shortcut's text runs from there to the end of the reply, and a ``grader line:`` in it starts the
    grader line, on the same line or a later one."""
    lines = [line.strip() for line in reply.splitlines() if line.strip()]
    for index in range(len(lines) - 1, -1, -1):
        line = _verdict_line(lines[index])
        if line == NO_SHORTCUT_LINE:
            return Claim(ClaimKind.NO_SHORTCUT, "", "")
        if line.startswith(SHORTCUT_PREFIX):
            text = " ".join([line.removeprefix(SHORTCUT_PREFIX).strip(), *lines[index + 1 :]])
            why, _, grader_line = text.partition(GRADER_LINE_MARKER)
            return Claim(ClaimKind.SHORTCUT, why.strip(), grader_line.strip())
    return NO_CLAIM


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit]} [... {len(text) - limit} chars cut]"


def _text(resource: TaskResource) -> str:
    data = resource_bytes(resource)
    try:
        return data.decode()
    except UnicodeDecodeError:
        return f"<{len(data)} bytes, not text>"


def decision_view(grader: DecisionGrader) -> dict[str, Any]:
    """A decision-model grader as a white-box adversary sees it: what decides a submission's reward.

    That is the state its decision model reads, key by key as ``grader.state`` lays it out, without the candidate's
    keys (``candidate_keys``), which hold the visible text of the submission's final reply; the question and the
    levels it answers; and the acceptance level and the reward rule. The levels the build's known-correct and fooling
    answers read (``calibration``, ``fooling``) decide no submission's reward and are not shown.
    """
    state = decision_state(grader, "")
    return {
        "kind": grader.kind,
        "state": {key: value for key, value in state.items() if grader.state[key] is not StateSource.SOLVER_TURN_OUTPUT},
        "candidate_keys": [key for key, source in grader.state.items() if source is StateSource.SOLVER_TURN_OUTPUT],
        "question": QUESTION,
        "levels": list(LEVELS),
        "acceptance_level": grader.bar,
        "reward_rule": (
            "an expected level (levels numbered from 0) at or above acceptance_level earns 1.0; below it the reward "
            f"is at most {REJECTED_REWARD_MAX}"
        ),
    }


def grader_source(task: TaskSpec) -> dict[str, Any]:
    """The grader as a white-box adversary sees it: its specification and its verifier files at their grading paths.

    A decision-model grader has no verifier files, and its specification is ``decision_view``.
    """
    decision = task_decision_grader(task)
    if decision is not None:
        return {"grader_spec": decision_view(decision), "grader_source": []}
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
        body = {
            "workspace_files": [
                {"path": f"/{file.path}", "content": _clip(_text(file), CONTROL_CHARS)} for file in payload.files
            ]
        }
    expect = control.expect
    return {
        "id": control.id,
        "kind": str(control.kind),
        "category": str(control.category),
        "concern": str(control.concern),
        "expect": {"status": str(expect.status), "reward_min": expect.reward_min, "reward_max": expect.reward_max},
        **body,
    }


def _grader_section(task: TaskSpec) -> str:
    """The white-box header and the grader: a program's specification and verifier files, or ``decision_view``."""
    source = grader_source(task)
    spec = json.dumps(source["grader_spec"], indent=1, ensure_ascii=False)
    if task_decision_grader(task) is not None:
        return f"{DECISION_WHITEBOX_HEADER}\n\n## Decision-model grader\n```json\n{spec}\n```"
    files = (
        "\n\n".join(
            f"### {file['path']}\n```\n{_clip(file['content'], GRADER_FILE_CHARS)}\n```"
            for file in source["grader_source"]
        )
        or NO_VERIFIER_FILES
    )
    return f"{WHITEBOX_HEADER}\n\n## Grader specification\n```json\n{spec}\n```\n\n## Grader source\n\n{files}"


def whitebox_context(task: TaskSpec, controls: Sequence[Control]) -> str:
    """The white-box section of the brief: the grader, the controls with their expected rewards, the example answers."""
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
        f"{_grader_section(task)}\n\n## Controls with expected rewards\n\n{control_text}\n\n"
        f"## Example answers in the task's conversation\n\n{example_text}"
    )


def adversary_brief(spec: TaskRolloutSpec, controls: Sequence[Control], policy: AdversaryPolicy) -> str:
    """The brief of an adversary on ``spec``: the white-box section of ``controls`` only in ``WHITEBOX`` mode."""
    whitebox = policy.mode is AdversaryMode.WHITEBOX
    context = whitebox_context(spec.task, controls) if whitebox else ""
    line_clause = DECISION_LINE_CLAUSE if task_decision_grader(spec.task) is not None else WHITEBOX_LINE_CLAUSE
    return BRIEF_TEMPLATE.format(
        tools=TOOLS_WITH_SHELL if spec.task_machine is not None else TOOLS_SUBMIT_ONLY,
        submissions=policy.submissions,
        probes=min(policy.min_probes, max(policy.submissions - 1, 1)),
        context=f"{PREAMBLE_SEPARATOR}{context}{PREAMBLE_SEPARATOR}" if context else " ",
        line_clause=line_clause if whitebox else "",
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


def passing(grade: GradeResult) -> bool:
    """Whether a submission's grade accepts it: the grader's pass flag, else a reward at the scale's maximum."""
    if grade.status not in GRADED_STATUSES or grade.reward is None:
        return False
    return grade.reward >= grade.score_max if grade.passed is None else grade.passed


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


def _observation(submission: SubmissionRecord, budget: int) -> dict[str, object]:
    """What ``submit`` returns to the adversary: the grade, or that the verifier failed."""
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
    """The ``submit`` handler of one attempt: grades each candidate as a control is graded, on a fresh machine."""

    spec: TaskRolloutSpec
    settings: EngineSettings
    plan: TrialPlan
    machine: Machine | None
    budget: int
    submissions: list[SubmissionRecord] = field(default_factory=list)

    async def submit(self, arguments: Mapping[str, object]) -> str:
        if len(self.submissions) >= self.budget:
            return json.dumps(
                {"error": f"submission budget spent: {self.budget} of {self.budget}; give your verdict now"}
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
        grade = _submission_grade(await run_trial(self.spec, self.plan, self.settings, model, name))
        submission = SubmissionRecord(
            ordinal=ordinal,
            turn=None,
            trial=name,
            reply=reply,
            files=tuple(SubmittedFile(f"/{file.path}", _text(file)) for file in files),
            grade=grade,
            passed=passing(grade),
            wall_time=time.monotonic() - started,
        )
        self.submissions.append(submission)
        return json.dumps(_observation(submission, self.budget))


def trial_grade(submissions: Sequence[SubmissionRecord]) -> GradeResult:
    """The grade of an attempt: its last passing submission's, else its last submission's when the verifier graded
    it, else a submission failure."""
    passed = [submission for submission in submissions if submission.passed]
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
            try:
                ordinal = json.loads(result.output).get("submission")
            except ValueError:
                continue
            if ordinal is not None:
                turns[ordinal] = index
    return turns


def claim_of(outcome: Outcome) -> Claim:
    """The verdict of the attempt's final text reply; ``NONE`` without one, or when the reply was cut off."""
    rollout = outcome.rollout
    steps = () if rollout is None else rollout_steps(rollout)
    if not steps or steps[-1].turn.stop_reason == LENGTH_STOP_REASON:
        return NO_CLAIM
    last = steps[-1].turn.message
    return NO_CLAIM if last.get("tool_calls") else parse_claim(str(last.get("content") or ""))


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
                    spec.task, settings.factories[runtime.backend], runtime, cleanup_timeout=settings.cleanup_timeout
                )
            )
        except Exception as error:
            raise RolloutInterrupted(Rollout(run_id, (), "error"), RolloutOperation.START) from error
        yield machine


def _machine_lost(error: RuntimeError) -> bool:
    return str(error).startswith(MACHINE_LOST_MESSAGES)


async def _agent_attempt(
    spec: TaskRolloutSpec,
    policy: AdversaryPolicy,
    settings: EngineSettings,
    model: ModelClient,
    record: CallLedger,
    opening: tuple[dict[str, Any], ...],
    submissions: list[SubmissionRecord],
    submission_plan: TrialPlan,
    run_id: str,
) -> Outcome:
    """One agent loop on a private task machine under the attempt deadline, its turns under the total-turn deadline.

    A failure before the loop finished is ``Ungraded`` with ``classify``'s cause; a machine close that fails after
    it counts as a cleanup error of the rollout.
    """
    deadline = asyncio.timeout(spec.limits.attempt_timeout)
    run: AgentRun | None = None
    finished = False
    lost = False
    cleanup_errors = 0
    try:
        async with deadline, attempt_machine(spec, settings, run_id) as machine:
            verifier = Verifier(spec, settings, submission_plan, machine, policy.submissions, submissions)
            submit = AgentTool(
                SUBMIT_TOOL_NAME, SUBMIT_DESCRIPTION, submit_parameters(machine is not None), verifier.submit
            )
            tools: tuple[AgentTool, ...] = (submit,)
            if machine is not None:
                shell = shell_tool(machine, timeout=settings.command_timeout, output_limit_bytes=SHELL_OUTPUT_LIMIT)
                tools = (shell, submit)
            agent_deadline = asyncio.timeout(spec.limits.total_turn_timeout)
            try:
                async with agent_deadline:
                    run = await run_agent(model, policy.sampling, opening, tools, policy.max_turns, record)
            except TimeoutError:
                if not agent_deadline.expired():
                    raise
            except RuntimeError as error:
                if not _machine_lost(error):
                    raise
                lost = True
            finished = True
    except Exception as error:
        if not finished or deadline.expired():
            cause = Cause.ATTEMPT_TIMEOUT if deadline.expired() else classify(error, spec.task)
            return Ungraded(cause, "".join(traceback.format_exception(error)), None, None)
        cleanup_errors = 1
    if run is not None:
        turns = submission_turns(run)
        submissions[:] = [replace(submission, turn=turns.get(submission.ordinal)) for submission in submissions]
    if lost:
        stop = MACHINE_LOST_STOP_REASON
    else:
        stop = TOTAL_TURN_TIMEOUT_STOP_REASON if run is None else AGENT_STOPS[run.stop]
    rollout = agent_rollout(run_id, opening, run, stop, cleanup_errors, submissions)
    return Graded(rollout, trial_grade(submissions), False)


def _refused(spec: TaskRolloutSpec, settings: EngineSettings) -> Ungraded | None:
    """The ``Ungraded`` of a trial that is not started, as ``run_trial`` refuses one."""
    unsupported = submission_refusal(spec.task)
    if unsupported is not None:
        return Ungraded(Cause.SUBMISSION_UNSUPPORTED, unsupported, None, None)
    refusals = task_refusals(spec, settings.factories)
    if refusals:
        return Ungraded(Cause.MACHINE_UNSUPPORTED, refusal_detail(refusals), None, None)
    return None


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
    """Run one adversary trial of ``spec`` with ``brief`` (``adversary_brief``), attempting it again after a backoff
    while it fails for a retryable cause, up to ``plan.max_retries`` times.

    Args:
        spec: The lowered spec; ``plan.deadlines`` and ``settings`` apply to it as to every trial.
        brief: The adversary's brief.
        policy: The adversary policy.
        plan: An ``ADVERSARY`` plan; ``plan.first_attempt`` numbers the first attempt.
        settings: The engine settings of the adversary's machine and of every submission's grading.
        model: The adversary's model.
        trial: The trial's name.
        submission_retries: The retries of a submission's grading.

    Raises:
        ValueError: the task has a decision-model grader and ``settings.decision`` is ``None``, raised before any
            attempt, as ``run_trial`` raises it.
    """
    require_decision_client(spec.task, settings)
    spec = settings.apply(plan.deadlines.apply(spec))
    trial_digest = task_digest(spec, settings.command_timeout)
    grader_digest = digest(grader_source(spec.task))
    refused = _refused(spec, settings)
    if refused is not None:
        record = AdversaryRecord(policy.mode, brief, "", NO_CLAIM, grader_digest, ())
        with attempt_span(trial_digest, plan, trial, plan.first_attempt) as fields:
            record_attempt(fields, refused, plan, trial, plan.first_attempt, _attempt_json(refused, record))
        return refused
    opening = opening_messages(spec, brief)
    system = str(opening[0]["content"])
    backoff = copy.copy(plan.retry_backoff)
    retries = 0
    attempt = plan.first_attempt
    while True:
        step = attempt_step(plan, trial, attempt)
        submissions: list[SubmissionRecord] = []
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
            outcome = await _agent_attempt(
                spec, policy, settings, model, calls, opening, submissions, submission_plan, step
            )
            claim = claim_of(outcome)
            record = AdversaryRecord(policy.mode, brief, system, claim, grader_digest, tuple(submissions))
            record_attempt(fields, outcome, plan, trial, attempt, _attempt_json(outcome, record))
            fields.attrs.update(
                {
                    "mode": str(policy.mode),
                    "submissions": str(len(submissions)),
                    "passes": str(sum(submission.passed for submission in submissions)),
                    "claim": str(claim.kind),
                }
            )
        if isinstance(outcome, Graded) or not outcome.retryable or retries >= plan.max_retries:
            return outcome
        retries += 1
        await asyncio.sleep(backoff.next_interval())
        attempt += 1


_RECORD: TypeAdapter[AdversaryRecord] = TypeAdapter(AdversaryRecord)
