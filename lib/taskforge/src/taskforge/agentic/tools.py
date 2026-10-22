# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Taskforge tools of the agent: a ``ToolProvider`` that the shell harness serves beside its shell tool.

One ``TaskforgeTools`` serves one agent session, for one idea. The agent works in its workspace with the shell; the
tools do only what the shell cannot. They read and write files in the workspace, and they run a candidate task's
grader and rollouts on new ShellSim machines that the agent cannot see.
They record the evidence of each version of a task under the digest of its ``TaskSpec``, and ``submit`` accepts a
task only when ``missing_evidence`` finds nothing missing. The session keeps the accepted tasks and the spend of the
tools for the caller.

A mistake of the agent, for example a bad path, an invalid task or an empty budget, goes back to the agent as an
error result. A failure of the host, for example an unavailable judge model, raises and ends the session.
"""

import asyncio
import csv
import io
import json
import logging
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, validate_call
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.shell import ShellHarness, ToolCallResult, ToolContext
from rolloutengine.models import (
    BUDGET_STOP_REASON,
    LENGTH_STOP_REASON,
    MAX_TURNS_STOP_REASON,
    TOTAL_TURN_TIMEOUT_STOP_REASON,
    GradedRollout,
    MachineRuntimeSpec,
    ModelRequest,
    ModelTurn,
    Rollout,
    RolloutInterrupted,
    TaskRolloutSpec,
    TokenUsage,
    TraceMode,
)
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, Command, Machine, NetworkPolicy
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    EnvironmentRequirements,
    FunctionDefinition,
    InlineFile,
    ProviderRequirement,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
    grades_with_judge,
)
from taskcompendium.runtime.resources import resource_bytes
from verifyit.spec import IfevalSpec, spec_from_table

from taskforge.agentic.config import AgenticConfig
from taskforge.agentic.gate import Evidence, Gate, missing_evidence
from taskforge.agentic.prompts import ADVERSARY_PROMPT, BLIND_SOLVE_PROMPT, GUESS_PROMPT, REVIEW_TASK_PROMPT
from taskforge.capability import FILE_SAFE_ID, CapabilityIdea, capability_idea_record
from taskforge.content_hash import digest, pretty_json
from taskforge.llm.client import ContextExhausted
from taskforge.llm.policy import Message
from taskforge.llm.structured import StructuredOutputError

logger = logging.getLogger(__name__)

AGENT_TOOLS = "taskforge-agentic:v1"
"""The action interface of the tools, in the agent task's ``tool_providers``."""
SOLVER_HARNESS = "solver"
WORKDIR = PurePosixPath("/workspace")
ROLLOUTS_DIR = "rollouts"
MAX_ID_CHARS = 128
"""The most characters of a task id. The tools and the driver name directories after it, and a file name has at most
255 characters."""
WEB_DIR = "web"


def machine_runtime(backend: str) -> MachineRuntimeSpec:
    """A machine of ``backend`` with no network and the backend's own defaults."""
    return MachineRuntimeSpec(
        backend=backend,
        network=NetworkPolicy.DENY,
        cpus=None,
        memory_mb=None,
        storage_mb=None,
        gpus=0,
        user=None,
        startup_timeout=None,
        cleanup_timeout=None,
    )


SHELLSIM_MACHINE = machine_runtime(Backend.SHELLSIM)
ANSWER_TYPES = frozenset({AnswerType.TEXT, AnswerType.NUMBER, AnswerType.JSON})
"""The answer types of a final message: native actions need tools that the solver does not have."""
PLAIN_DECIMAL = re.compile(r"[+-]?\d+(\.\d+)?")
"""An answer whose last digit gives its precision, unlike a fraction or scientific notation."""
OUTPUT_CHARS = 2000
"""The most characters of a program output, an answer, a page or a file that one tool result shows."""
REVIEW_ANSWERS = 20
JUDGE_CALLS_PER_REVIEW = 2
"""A review is a blind solve and the review itself."""
REVIEW_FILE_CHARS = 20_000
"""The most characters of each solver file that the judge reads: it must see a synopsis or a table to judge it."""
MAX_EMPTY_BUDGET_CALLS = 3
"""Calls on an empty budget after which the tools stop the session, so that an agent cannot loop on the refusal."""
EARLY_FINISH_BUDGET_SHARE = 0.5
"""``finish`` refuses once when fewer than n tasks are accepted and more than this share of the solver rollouts are
left: an agent can stop after a few too-easy versions, when a harder version would get into the band."""
RAN_OUT_STOP_REASONS = frozenset(
    {MAX_TURNS_STOP_REASON, BUDGET_STOP_REASON, LENGTH_STOP_REASON, TOTAL_TURN_TIMEOUT_STOP_REASON}
)
"""Stop reasons of a solver that ran out of a limit before it answered: a failed run, not a format failure."""
OPTION_PROBLEM = (
    "The answer selects an option from a list, and reviews do not find the flaws of such tasks well. Make a task whose "
    "answer the solver computes or writes."
)
"""On 313 labeled tasks, 27% of the option tasks that a review accepted had a blocking flaw, for example a correct
option that a guess finds, against 8% of the other tasks."""
GUESS_RUNS = 4
"""Rollouts of ``submit`` in which the solver model guesses the answer of a task with files from its prompt alone."""
MAX_GUESS_PASSES = 2
"""The most guesses that can pass: more show an answer that the prompt gives away, for example by a cue."""
MAX_FILE_BYTES = 64 * 1024 * 1024
"""The largest workspace file that a tool reads, for example a TaskSpec with large inline files."""


class ToolError(Exception):
    """A mistake of the agent, which the tool gives back to it as an error."""


class BudgetExhausted(ToolError):
    """A call that needs a budget with nothing left."""


class BlindSolution(BaseModel):
    """A judge's own solution of a task, made without its reference answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    readings: tuple[str, ...] = Field(
        description="Each place where the prompt or the files allow more than one reasonable reading, with the "
        "answer of each reading, or none."
    )
    answer: str = Field(description="Your answer to the task, in the task's answer format.")


class OptionList(BaseModel):
    """A list of options in a task, from which the answer selects one."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    options: tuple[str, ...] = Field(
        min_length=2,
        description=(
            "The options, quoted in full in their order. For an option that is a code, quote the text that defines it."
        ),
    )
    correct: int = Field(ge=1, description="The place of the correct option in the list, from 1.")


class TaskReview(BaseModel):
    """The review of one task. The decision comes last, so that the judge writes its analysis first."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    option_lists: tuple[OptionList, ...] = Field(
        description=(
            "Each list of options in the prompt or the files from which the answer selects one option, with the "
            "place of the correct option. Empty when the answer selects no option."
        )
    )
    failure_causes: tuple[str, ...] = Field(
        description=(
            "For each distinct failed answer: the answer, the reading or the mistake that gives it, and the "
            "statement of the prompt or the files that this reading breaks, quoted. Write 'no statement' when "
            "no statement excludes the reading."
        )
    )
    problems: tuple[str, ...] = Field(description="Each problem that you found, or none.")
    summary: str = Field(min_length=1)
    accept: bool


@dataclass(frozen=True)
class AcceptedTask:
    """A task that ``submit`` accepted, with the evidence that the gate accepted and the review."""

    task: TaskSpec
    evidence: Evidence
    review: TaskReview


@dataclass
class ToolSpend:
    """What the tools used for one idea, beside the agent's own tokens."""

    solver_rollouts: int = 0
    judge_calls: int = 0
    web_calls: int = 0
    solver_tokens: int = 0
    judge_tokens: int = 0


def _tool(name: str, description: str, properties: dict[str, Any] | None = None) -> FunctionDefinition:
    properties = properties or {}
    parameters = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    return FunctionDefinition(name=name, description=description, parameters=parameters)


PATH = {"type": "string", "description": "The path of a TaskSpec JSON file in the workspace."}
TOOL_DEFINITIONS = (
    _tool(
        "web_search",
        "Search the web.",
        {"objective": {"type": "string"}, "queries": {"type": "array", "items": {"type": "string"}}},
    ),
    _tool(
        "web_fetch",
        "Read the parts of web pages that are relevant to the objective.",
        {"urls": {"type": "array", "items": {"type": "string"}}, "objective": {"type": "string"}},
    ),
    _tool(
        "grade",
        "Grade a control answer with the task's grader, with no model.",
        {"path": PATH, "answer": {"type": "string"}, "expect": {"type": "string", "enum": ["pass", "fail"]}},
    ),
    _tool(
        "rollout",
        "Run a solver model n times on each task, all at the same time.",
        {
            "paths": {"type": "array", "items": PATH, "minItems": 1},
            "n": {"type": "integer", "minimum": 1},
            "role": {"type": "string", "enum": ["solver", "adversary"]},
        },
    ),
    _tool(
        "submit",
        "Submit a task. The gate checks the task and its evidence, then an independent reviewer reads it.",
        {"path": PATH},
    ),
    _tool("finish", "Stop, with a report on the tasks and the problems.", {"report": {"type": "string"}}),
)


class TaskforgeTools:
    """The tool provider of one agent session, and the results of the session.

    ``accepted`` holds the accepted tasks in order, ``report`` the agent's report from ``finish``, and ``spend`` what
    the tools used.
    """

    def __init__(self, idea: CapabilityIdea, n: int, config: AgenticConfig):
        self.idea = idea
        self.n = n
        self.config = config
        self.solver_engine = RolloutEngine(
            {Backend.SHELLSIM: ShellSimMachineFactory()},
            {SOLVER_HARNESS: ShellHarness(command_timeout=config.command_timeout)},
        )
        self.evidence: defaultdict[str, Evidence] = defaultdict(Evidence)
        self.failed_answers: defaultdict[str, list[str]] = defaultdict(list)
        # The answers of the controls that came out as the agent expected, from which submit makes wrong answers.
        self.right_answers: defaultdict[str, list[str]] = defaultdict(list)
        self.wrong_answers: defaultdict[str, list[str]] = defaultdict(list)
        self.adversary_runs: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        self.transcripts: Counter[str] = Counter()
        self.accepted: list[AcceptedTask] = []
        self.report: str | None = None
        self.spend = ToolSpend()
        self.empty_budget_calls = 0
        self.early_finish_refused = False
        self.slots = asyncio.Semaphore(config.max_concurrent_rollouts)

    def tools(self, requirement: ProviderRequirement) -> tuple[FunctionDefinition, ...]:
        return TOOL_DEFINITIONS

    async def open(self, context: ToolContext) -> "ToolSession":
        if context.machine is None:
            raise ValueError("The Taskforge tools need the agent's workspace machine")
        return ToolSession(self, context.machine)


class ToolSession:
    """The tools of one attempt, on the agent's workspace machine."""

    def __init__(self, tools: TaskforgeTools, machine: Machine):
        self.tools = tools
        self.config = tools.config
        self.machine = machine
        handlers: dict[str, Callable[..., Awaitable[ToolCallResult]]] = {
            "web_search": self.web_search,
            "web_fetch": self.web_fetch,
            "grade": self.grade,
            "rollout": self.rollout,
            "submit": self.submit,
            "finish": self.finish,
        }
        assert set(handlers) == {tool.name for tool in TOOL_DEFINITIONS}
        self.handlers = {name: validate_call(handler) for name, handler in handlers.items()}

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        handler = self.handlers.get(name)
        if handler is None:
            return _error(f"Unknown tool: {name}")
        try:
            result = await handler(**arguments)
        except ValidationError as error:
            result = _error(f"Invalid arguments of {name}: {error}")
        except BudgetExhausted as error:
            self.tools.empty_budget_calls += 1
            stop = self.tools.empty_budget_calls >= MAX_EMPTY_BUDGET_CALLS
            if stop and self.tools.report is None:
                self.tools.report = (
                    f"The tools stopped the session after {MAX_EMPTY_BUDGET_CALLS} calls on an empty budget: {error}"
                )
            result = _result({"error": str(error), "session_stopped": stop}, done=stop)
        except ToolError as error:
            result = _error(str(error))
        logger.info(
            "%s %s(%s): %s",
            self.tools.idea.capability_id,
            name,
            _clip(json.dumps(arguments), limit=200),
            _clip(result.content, limit=300),
        )
        return result

    async def close(self) -> None:
        pass

    async def web_search(self, objective: str, queries: list[str]) -> ToolCallResult:
        self._spend("web_calls", self.config.budget.web_calls, 1)
        response = await _web(self.config.web.search(objective, queries))
        path = await self._save_web("search", response)
        results = [
            {"url": item.get("url"), "title": item.get("title"), "excerpts": _clip(" ".join(item.get("excerpts") or ()))}
            for item in response.get("results", ())
        ]
        return _result({"saved": path, "results": results})

    async def web_fetch(self, urls: list[str], objective: str) -> ToolCallResult:
        self._spend("web_calls", self.config.budget.web_calls, 1)
        response = await _web(self.config.web.extract(urls, objective))
        path = await self._save_web("fetch", response)
        results = [
            {"url": item.get("url"), "excerpts": _clip(" ".join(item.get("excerpts") or ()))}
            for item in response.get("results", ())
        ]
        return _result({"saved": path, "results": results, "errors": response.get("errors", [])})

    async def grade(self, path: str, answer: str, expect: Literal["pass", "fail"]) -> ToolCallResult:
        task = await self._load(path)
        grade = await self._grade(task, answer)
        status = grade.status
        if status == Outcome.SUBMISSION_FAILURE:
            # A wrong control must show that the grader rejects a wrong value, not that the format rejects the answer.
            return _result(
                {
                    "outcome": status,
                    "control": False,
                    "error": (
                        "The answer format rejects this answer, so it is no control. Grade answers in the "
                        "answer format."
                    ),
                }
            )
        if status != Outcome.GRADED:
            raise ToolError(f"The grade failed ({status}); check the task: {grade.error}")
        passed = _passed(grade)
        key = _task_digest(task)
        self.tools.evidence[key].controls.append((expect == "pass", passed))
        if passed and expect == "pass":
            self.tools.right_answers[key].append(answer)
        elif not passed and expect == "fail":
            self.tools.wrong_answers[key].append(answer)
        return _result(
            {
                "outcome": grade.status,
                "reward": grade.reward,
                "passed": passed,
                "as_expected": passed == (expect == "pass"),
                "error": grade.error,
            }
        )

    async def _grade(self, task: TaskSpec, answer: str) -> GradeResult:
        """The grade of ``answer`` by the task's grader, from a rollout of a model that gives only that answer."""
        try:
            result = await self.tools.solver_engine.run_task(_solver_spec(task, self.config), _ReplayModel(answer))
        except RolloutInterrupted as error:
            raise ToolError(f"The grade failed; check the task: {_failure(error.rollout)}") from error
        except (ValueError, NotImplementedError) as error:
            raise ToolError(f"The grade failed; check the task: {error}") from error
        return result.grade

    async def rollout(self, paths: list[str], n: int, role: Literal["solver", "adversary"]) -> ToolCallResult:
        if n < 1:
            raise ToolError("n must be at least 1")
        if not paths or len(set(paths)) < len(paths):
            raise ToolError("Give one or more different paths")
        tasks = [await self._load(path) for path in paths]
        # The same checks as at submit, so that no rollout goes to a task that submit cannot accept.
        for path, task in zip(paths, tasks, strict=True):
            problems = _unsupported(task, self.tools.solver_engine, self.config)
            if problems:
                raise ToolError(f"{path} cannot run: {'; '.join(problems)}")
        specs = [_solver_spec(_adversary_task(task) if role == "adversary" else task, self.config) for task in tasks]
        self._spend("solver_rollouts", self.config.budget.solver_rollouts, n * len(paths))
        results = await asyncio.gather(
            *(self._rollouts(task, spec, n, role) for task, spec in zip(tasks, specs, strict=True))
        )
        left = self.config.budget.solver_rollouts - self.tools.spend.solver_rollouts
        return _result({"results": dict(zip(paths, results, strict=True)), "solver_rollouts_left": left})

    async def _rollouts(
        self, task: TaskSpec, spec: TaskRolloutSpec, n: int, role: Literal["solver", "adversary"]
    ) -> dict[str, Any]:
        """Run ``n`` rollouts of ``spec``, record their evidence for ``task``, and save their transcripts."""
        key = _task_digest(task)
        directory = _absolute(f"{ROLLOUTS_DIR}/{task.id}/{key[:12]}")
        # Take the transcript numbers before the rollouts, so that concurrent calls for one task do not share them.
        first = self.tools.transcripts[f"{directory}/{role}"]
        self.tools.transcripts[f"{directory}/{role}"] += n
        outcomes = await asyncio.gather(*(self._solve(spec) for _ in range(n)))
        answers, ungraded = [], Counter()
        evidence = self.tools.evidence[key]
        for index, outcome in enumerate(outcomes):
            rollout = outcome.rollout
            grade = outcome.grade if isinstance(outcome, GradedRollout) else None
            await self._write(f"{directory}/{role}-{first + index}.txt", _transcript(rollout, grade).encode())
            if grade is None or grade.status not in (Outcome.GRADED, Outcome.SUBMISSION_FAILURE):
                reason = f"{grade.status}: {grade.error}" if grade is not None else f"interrupted: {_failure(rollout)}"
                ungraded[_clip(reason, limit=300)] += 1
                continue
            passed = _passed(grade)
            answer = _final_answer(rollout)
            format_failure = (
                grade.status == Outcome.SUBMISSION_FAILURE and rollout.stop_reason not in RAN_OUT_STOP_REASONS
            )
            answers.append({"passed": passed, "format_failure": format_failure, "answer": _clip(answer, limit=200)})
            if role == "solver" and format_failure:
                evidence.solver_format_failures += 1
            elif role == "solver":
                evidence.solver_runs += 1
                evidence.solver_passes += passed
                if not passed:
                    self.tools.failed_answers[key].append(answer)
            elif grade.status == Outcome.GRADED:
                # An adversary without a graded answer shows nothing that the review can judge, so it does not count.
                evidence.adversary_runs += 1
                evidence.adversary_passes += passed
                record = {"passed": passed, "answer": _clip(answer, limit=300), "commands": _commands(rollout)}
                self.tools.adversary_runs[key].append(record)
        graded = [answer for answer in answers if not answer["format_failure"]]
        passes = sum(answer["passed"] for answer in graded)
        return {
            "graded": len(graded),
            "format_failures": len(answers) - len(graded),
            "passes": passes,
            "pass_rate": passes / len(graded) if graded else None,
            "not_graded": dict(ungraded),
            "answers": answers,
            "transcripts": directory,
            "evidence": asdict(evidence),
            **_next_step(evidence, self.config.gate),
        }

    async def submit(self, path: str) -> ToolCallResult:
        task = await self._load(path)
        key = _task_digest(task)
        if any(_task_digest(accepted.task) == key for accepted in self.tools.accepted):
            raise ToolError("This task is already accepted")
        problems = _unsupported(task, self.tools.solver_engine, self.config)
        problems += _malformed_files(task)
        if problems:
            return _result({"accepted": False, "problems": problems})
        evidence = self.tools.evidence[key]
        missing = missing_evidence(evidence, self.config.gate)
        if missing:
            return _result({"accepted": False, "missing": missing, "evidence": asdict(evidence)})
        problems = await self._accepted_wrong_answers(task, key)
        if problems:
            return _result({"accepted": False, "problems": problems})
        guessed = await self._guesses(task)
        if guessed > MAX_GUESS_PASSES:
            problem = (
                f"A solver with the prompt alone and no files guessed the answer in {guessed} of {GUESS_RUNS} attempts: "
                "the prompt gives the answer away, for example by a cue."
            )
            return _result({"accepted": False, "problems": [problem]})
        # The review comes last, so it sees all the evidence of this version of the task.
        review = await self._review(task, key, evidence)
        if not review.accept:
            return _result({"accepted": False, "review": review.model_dump(mode="json")})
        accepted = AcceptedTask(task, replace(evidence, controls=list(evidence.controls)), review)
        self.tools.accepted.append(accepted)
        count = len(self.tools.accepted)
        reply = {"accepted": True, "accepted_tasks": count, "remaining": self.tools.n - count, "review": review.summary}
        return _result(reply, done=count >= self.tools.n)

    async def _accepted_wrong_answers(self, task: TaskSpec, key: str) -> list[str]:
        """The problems of a grader that accepts wrong answers made from the controls of this version: the right
        answer with one unit more or less in its last digit, the right and the wrong answer as a list in the two
        orders, and an empty answer. A grader that accepts the list in one order only reads one position of it."""
        # An agent can also grade a more precise right answer, so the one with the fewest decimals gives the precision.
        right = min(
            (answer.strip() for answer in self.tools.right_answers[key]),
            key=lambda answer: (PLAIN_DECIMAL.fullmatch(answer) is None, len(answer.partition(".")[2])),
        )
        wrong = self.tools.wrong_answers[key][-1].strip()
        neighbors = _neighbors(right)
        lists = [f"{wrong}, {right}", f"{right}, {wrong}"]
        answers = [*neighbors, *lists, ""]
        grades = await asyncio.gather(*(self._grade(task, answer) for answer in answers))
        accepted = {
            answer
            for answer, grade in zip(answers, grades, strict=True)
            if grade.status == Outcome.GRADED and _passed(grade)
        }
        problems = [
            f"The grader accepts {answer}, one unit from your right control {right} in its last digit. Write the "
            "reference with the precision that the prompt asks for, and a tolerance below one unit of it"
            for answer in neighbors
            if answer in accepted
        ]
        if set(lists) <= accepted:
            problems.append(
                f"The grader accepts {lists[0]!r} and {lists[1]!r}, so an answer that lists candidates passes. Grade "
                "the whole answer, for example with an ExactSpec without substring matching"
            )
        if "" in accepted:
            problems.append("The grader accepts an empty answer")
        return problems

    async def _guesses(self, task: TaskSpec) -> int:
        """How many of ``GUESS_RUNS`` solvers with the prompt alone guess the answer of a task with files.

        The guesses are a part of ``submit``, so they do not use the rollout budget of the agent.
        """
        if not task.resources.worker:
            return 0
        spec = _solver_spec(_guess_task(task), self.config)
        outcomes = await asyncio.gather(*(self._solve(spec) for _ in range(GUESS_RUNS)))
        return sum(isinstance(outcome, GradedRollout) and _passed(outcome.grade) for outcome in outcomes)

    async def _review(self, task: TaskSpec, key: str, evidence: Evidence) -> TaskReview:
        # The budget must hold the whole review, so that a review never stops after its blind solve.
        self._spend("judge_calls", self.config.budget.judge_calls, JUDGE_CALLS_PER_REVIEW)
        # The judge first solves the task without its reference answer, as a check that is not anchored on it.
        blind = await self._judge(BLIND_SOLVE_PROMPT, pretty_json(blind_content(self.tools.idea, task)), BlindSolution)
        content = review_content(
            self.tools.idea,
            task,
            evidence,
            self.tools.failed_answers[key],
            self.tools.adversary_runs[key],
            [accepted.task for accepted in self.tools.accepted],
            blind,
        )
        review = await self._judge(REVIEW_TASK_PROMPT, pretty_json(content), TaskReview)
        if not review.option_lists:
            return review
        return review.model_copy(update={"accept": False, "problems": (*review.problems, OPTION_PROBLEM)})

    async def finish(self, report: str) -> ToolCallResult:
        budget = self.config.budget.solver_rollouts
        left = budget - self.tools.spend.solver_rollouts
        accepted = len(self.tools.accepted)
        if accepted < self.tools.n and left > EARLY_FINISH_BUDGET_SHARE * budget and not self.tools.early_finish_refused:
            self.tools.early_finish_refused = True
            return _error(
                f"{accepted} of {self.tools.n} tasks are accepted and {left} of {budget} solver rollouts are left. "
                "Use them: make the task harder with more steps or more data, or make a different task. "
                "Call finish again only if the capability cannot give tasks in the band."
            )
        self.tools.report = report
        return _result({"finished": True, "accepted_tasks": len(self.tools.accepted)}, done=True)

    async def _solve(self, spec: TaskRolloutSpec) -> GradedRollout | RolloutInterrupted:
        async with self.tools.slots:
            try:
                outcome: GradedRollout | RolloutInterrupted = await self.tools.solver_engine.run_task(
                    spec, self.config.solver_model
                )
            except RolloutInterrupted as error:
                # A failed solver attempt, for example a model failure, is not evidence about the task.
                outcome = error
        self.tools.spend.solver_tokens += rollout_tokens(outcome.rollout)
        return outcome

    async def _judge[T: BaseModel](self, system: str, user: str, output_type: type[T]) -> T:
        messages: list[Message] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            result = await self.config.judge.complete_structured(messages, output_type, self.config.judge_policy)
        except StructuredOutputError as error:
            raise ToolError(f"The reviewer gave no valid reply; try again: {error}") from error
        except ContextExhausted as error:
            raise ToolError(
                "The reviewer cannot read this task: its prompt and files are too long. Make them shorter."
            ) from error
        self.tools.spend.judge_tokens += sum(
            completion.usage.prompt_tokens + completion.usage.completion_tokens for completion in result.completions
        )
        return result.value

    async def _save_web(self, kind: str, response: Mapping[str, Any]) -> str:
        path = _absolute(f"{WEB_DIR}/{kind}-{self.tools.spend.web_calls}.json")
        await self._write(path, json.dumps(response, indent=1).encode())
        return path

    def _spend(self, name: str, limit: int, count: int) -> None:
        used = getattr(self.tools.spend, name)
        if used >= limit:
            raise BudgetExhausted(f"The budget has no {name.replace('_', ' ')} left")
        if used + count > limit:
            raise ToolError(f"The budget has {limit - used} {name.replace('_', ' ')} left")
        setattr(self.tools.spend, name, used + count)

    async def _read(self, path: str) -> bytes:
        path = _absolute(path)
        command = Command(("cat", path), timeout=self.config.command_timeout, output_limit_bytes=MAX_FILE_BYTES)
        result = await self.machine.run(command)
        if result.exit_code != 0:
            raise ToolError(f"Cannot read {path}: {_clip(result.stderr.decode(errors='replace'))}")
        if result.stdout_truncated:
            raise ToolError(f"{path} is larger than {MAX_FILE_BYTES} bytes")
        return result.stdout

    async def _write(self, path: str, data: bytes) -> None:
        with tempfile.TemporaryDirectory(prefix="taskforge-agentic-") as directory:
            source = Path(directory) / "file"
            source.write_bytes(data)
            await self.machine.upload(source, _absolute(path))

    async def _load(self, path: str) -> TaskSpec:
        """The TaskSpec at ``path``, refused at once when no grade or rollout may run on it: its id cannot name its
        output directory, or its machine would get what the review cannot see."""
        try:
            task = TaskSpec.model_validate_json(await self._read(path))
        except ValidationError as error:
            raise ToolError(f"{path} is not a valid TaskSpec: {_clip(str(error))}") from error
        if FILE_SAFE_ID.match(task.id) is None or len(task.id) > MAX_ID_CHARS:
            raise ToolError(
                f"The task id {_clip(task.id, limit=200)!r} must be at most {MAX_ID_CHARS} letters, digits, dots, "
                "dashes and underscores, starting with a letter or a digit"
            )
        key = _task_digest(task)
        if any(accepted.task.id == task.id and _task_digest(accepted.task) != key for accepted in self.tools.accepted):
            raise ToolError(f"An accepted task already has the id {task.id!r}: give this task another id")
        # The engine resolves ${NAME} in an environment variable from the host, which holds the driver's keys, and
        # setup commands can rewrite the files after they are installed. The review sees neither.
        requirements = task.environment_requirements
        if requirements.environment_variables:
            raise ToolError("Tasks cannot set environment_variables: put what the solver needs in the prompt or files")
        if requirements.setup_commands:
            raise ToolError("Tasks cannot have setup_commands: put the solver's files in resources.worker")
        return task


@dataclass
class _ReplayModel:
    """A model that answers with ``answer`` and no tool call, so that the grader grades a given answer."""

    answer: str

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        message = {"role": "assistant", "content": self.answer}
        return ModelTurn(message, (), (), None, "stop", text=self.answer, usage=TokenUsage(0, 0))


def _next_step(evidence: Evidence, gate: Gate) -> dict[str, str]:
    """What a version whose pass rate is in the band so far still needs before ``submit``, as a nudge to finish it."""
    low, high = gate.solver_band
    if evidence.solver_runs == 0 or not low <= evidence.solver_pass_rate <= high:
        return {}
    needs = {
        "controls": "a correct control that passes and a wrong control that fails",
        "solver band": f"{gate.min_solver_runs} graded solver rollouts in the band",
        "answer format": "fewer format failures",
        "adversary": f"{gate.min_adversary_runs} adversary rollouts with a graded answer",
    }
    missing = [needs[name] for name in missing_evidence(evidence, gate)]
    if not missing:
        return {"next": "This version has its evidence: submit it now."}
    return {"next": f"This version is in the band so far. Get {'; '.join(missing)}, then submit it."}


def blind_content(idea: CapabilityIdea, task: TaskSpec) -> dict[str, Any]:
    """What the blind solve of a review reads: the capability and the public task, with no reference answer."""
    return {"capability": capability_idea_record(idea), "task": _public_task(task, file_chars=REVIEW_FILE_CHARS)}


def review_content(
    idea: CapabilityIdea,
    task: TaskSpec,
    evidence: Evidence,
    failed_answers: list[str],
    adversary_runs: list[dict[str, Any]],
    accepted: list[TaskSpec],
    blind: BlindSolution,
) -> dict[str, Any]:
    """What the reviewer of ``submit`` reads: the public task, the grader, the evidence, the blind solution of the
    task, and the accepted tasks with the start of each file, where a copied story or table shows.

    ``shared_failed_answers`` counts the failed answers that repeat, because a repeated wrong answer usually comes
    from one other reading of the task.
    """
    shared = Counter(_normal_answer(answer) for answer in failed_answers)
    return {
        "capability": capability_idea_record(idea),
        "task": _public_task(task, file_chars=REVIEW_FILE_CHARS),
        "blind_solution": blind.model_dump(mode="json"),
        "grader": task.grader.model_dump(mode="json"),
        "solver_evidence": {
            "runs": evidence.solver_runs,
            "passes": evidence.solver_passes,
            "format_failures": evidence.solver_format_failures,
            "failed_answers": [_clip(answer, limit=300) for answer in failed_answers][:REVIEW_ANSWERS],
            "shared_failed_answers": {answer: count for answer, count in shared.most_common() if count > 1},
        },
        # Passed runs first, so that the limit never hides a shortcut from the review.
        "adversary_runs": sorted(adversary_runs, key=lambda run: not run["passed"])[:REVIEW_ANSWERS],
        "accepted_tasks": [_public_task(other) for other in accepted],
    }


def _normal_answer(answer: str) -> str:
    """``answer`` without emphasis, case and a final period, and as a number when it is one."""
    text = answer.strip().strip("*`").strip().rstrip(".").lower()
    try:
        return str(float(text.replace(",", "")))
    except ValueError:
        return _clip(text, limit=100)


def rollout_tokens(rollout: Rollout) -> int:
    """The tokens of all the model requests of ``rollout``, from the usage that the transport reports."""
    return sum(step.turn.request_tokens for conversation in rollout.conversations for step in conversation.steps)


async def _web(request: Awaitable[dict[str, Any]]) -> dict[str, Any]:
    try:
        return await request
    except httpx.HTTPError as error:
        raise ToolError(f"The web request failed: {error}") from error


def _task_digest(task: TaskSpec) -> str:
    return digest(task.model_dump(mode="json"))


def _absolute(path: str) -> str:
    return str(WORKDIR / path)


def _needs_machine(task: TaskSpec) -> bool:
    requirements = task.environment_requirements
    return bool(requirements.capabilities or requirements.setup_commands or task.resources.worker or task.output_paths)


def _solver_spec(task: TaskSpec, config: AgenticConfig) -> TaskRolloutSpec:
    return TaskRolloutSpec(
        task=task,
        harness=SOLVER_HARNESS,
        task_machine=SHELLSIM_MACHINE if _needs_machine(task) else None,
        verifier_machine=None,
        limits=config.solver_limits,
        trace=TraceMode.MESSAGES,
    )


def _unsupported(task: TaskSpec, engine: RolloutEngine, config: AgenticConfig) -> list[str]:
    """What keeps the agent's tools from running ``task``, which ``rollout`` and ``submit`` refuse."""
    problems = []
    if task.answer_type not in ANSWER_TYPES:
        problems.append(f"The answer type must be one of {sorted(ANSWER_TYPES)}, not {task.answer_type}")
    grader = task.grader
    if not isinstance(grader, VerifyitGrader) or grader.environment is not None:
        problems.append("The grader must be a verifyit grader without an environment")
    else:
        try:
            if grades_with_judge(grader):
                problems.append("Judge graders are not supported")
            elif _grades_keywords(grader):
                problems.append(
                    "Keyword graders are not supported: a correct paraphrase fails and an answer that only repeats "
                    "the words passes. Ask for a value that a NumericSpec or an ExactSpec can grade"
                )
        except (TypeError, ValueError) as error:
            problems.append(f"The grader is invalid: {error}")
    requirements = task.environment_requirements
    if _needs_machine(task) and requirements.command_semantics != CommandSemantics.SHELL_SIMULATOR:
        problems.append("A task with a shell or files must use command_semantics shell_simulator")
    if task.resources.all:
        problems.append(
            "Put the solver's files in resources.worker and the grader's files in resources.verifier, not in "
            "resources.all: the review reads only the worker files"
        )
    if requirements.tool_providers or task.interaction_tools or task.final_tools:
        problems.append("Tools other than the shell are not supported")
    if not problems:
        try:
            engine.check(_solver_spec(task, config))
        except (ValueError, NotImplementedError) as error:
            problems.append(f"The rollout engine rejects the task: {error}")
    return problems


def _grades_keywords(grader: VerifyitGrader) -> bool:
    spec = spec_from_table({"mode": grader.mode, **grader.parameters})
    return isinstance(spec, IfevalSpec) and any(
        constraint.name.startswith("keywords:") for constraint in spec.constraints
    )


def _neighbors(answer: str) -> list[str]:
    """``answer`` with one unit more and one unit less in its last digit, when it is a plain decimal."""
    if PLAIN_DECIMAL.fullmatch(answer) is None:
        return []
    value = Decimal(answer)
    unit = Decimal(1).scaleb(-len(answer.partition(".")[2]))
    return [str(value + unit), str(value - unit)]


def _malformed_files(task: TaskSpec) -> list[str]:
    """The solver's CSV and JSON files that do not parse, for example a CSV field with an unquoted comma."""
    problems = []
    for resource in task.resources.worker:
        if not isinstance(resource.source, InlineFile):
            continue
        text = resource_bytes(resource).decode(errors="replace")
        if resource.path.endswith(".csv"):
            rows = [row for row in csv.reader(io.StringIO(text)) if row]
            widths = {len(row) for row in rows}
            if len(widths) > 1:
                problems.append(
                    f"{resource.path} is not a valid CSV: its rows have {sorted(widths)} fields. Quote the fields "
                    "that contain commas"
                )
        elif resource.path.endswith(".json"):
            try:
                json.loads(text)
            except json.JSONDecodeError as error:
                problems.append(f"{resource.path} is not valid JSON: {error}")
    return problems


def _public_task(task: TaskSpec, file_chars: int = OUTPUT_CHARS) -> dict[str, Any]:
    """What the solver sees: the prompt messages, the answer format, and the start of each of its files."""
    files = {
        resource.path: _clip(resource_bytes(resource).decode(errors="replace"), limit=file_chars)
        for resource in task.resources.worker
        if isinstance(resource.source, InlineFile)
    }
    return {
        "prompt": [event.content for event in task.context.events if isinstance(event, TextMessage)],
        "answer_type": task.answer_type,
        "answer_format": task.answer_format.model_dump(mode="json"),
        "capabilities": list(task.environment_requirements.capabilities),
        "files": files,
    }


def _adversary_task(task: TaskSpec) -> TaskSpec:
    return _with_system_prompt(task, ADVERSARY_PROMPT)


def _guess_task(task: TaskSpec) -> TaskSpec:
    """``task`` with no files and no shell, and the guess prompt as its system message."""
    resources = task.resources.model_copy(update={"worker": ()})
    task = task.model_copy(update={"environment_requirements": EnvironmentRequirements(), "resources": resources})
    return _with_system_prompt(task, GUESS_PROMPT)


def _with_system_prompt(task: TaskSpec, prompt: str) -> TaskSpec:
    """``task`` with ``prompt`` as its system message, before any system message of its own."""
    events = list(task.context.events)
    first = events[0] if events else None
    if isinstance(first, TextMessage) and first.role == "system":
        events[0] = TextMessage(role="system", content=f"{prompt}\n\n{first.content}")
    else:
        events.insert(0, TextMessage(role="system", content=prompt))
    return task.model_copy(update={"context": task.context.model_copy(update={"events": tuple(events)})})


def _passed(grade: GradeResult) -> bool:
    if grade.status != Outcome.GRADED or grade.reward is None:
        return False
    return grade.passed if grade.passed is not None else grade.reward >= grade.score_max


def _final_answer(rollout: Rollout) -> str:
    messages = rollout.conversations[-1].messages if rollout.conversations else ()
    replies = [message for message in messages if message.get("role") == "assistant"]
    return str(replies[-1].get("content") or "") if replies else ""


def _failure(rollout: Rollout) -> str:
    """The exception type and the diagnostics of an interrupted attempt."""
    failure = rollout.failure
    if failure is None:
        return "no failure record"
    return _clip(f"{failure.exception_type}: {json.dumps(failure.diagnostics, default=str)}", limit=1000)


def _commands(rollout: Rollout) -> list[str]:
    """The arguments of each tool call in ``rollout``, for example the shell commands of an adversary."""
    return [
        _clip(str((call.get("function") or {}).get("arguments")), limit=200)
        for conversation in rollout.conversations
        for message in conversation.messages
        for call in message.get("tool_calls") or ()
    ][:REVIEW_ANSWERS]


def _transcript(rollout: Rollout, grade: GradeResult | None) -> str:
    header = {
        "run_id": rollout.run_id,
        "stop_reason": rollout.stop_reason,
        "failure": asdict(rollout.failure) if rollout.failure is not None else None,
        "grade": {"status": grade.status, "reward": grade.reward, "error": grade.error} if grade is not None else None,
    }
    lines = [json.dumps(header, default=str)]
    for conversation in rollout.conversations:
        for number, message in enumerate(conversation.messages):
            functions = [call.get("function") or {} for call in message.get("tool_calls") or ()]
            calls = [f"<tool call> {function.get('name')}({function.get('arguments')})" for function in functions]
            text = " ".join([str(message.get("content") or ""), *calls]).strip()
            lines.append(f"[{number}] {message.get('role')}: {text}")
    return "\n".join(lines) + "\n"


def _clip(text: str, *, limit: int = OUTPUT_CHARS) -> str:
    return text if len(text) <= limit else f"{text[:limit]}..."


def _result(value: Mapping[str, Any], *, done: bool = False) -> ToolCallResult:
    return ToolCallResult(json.dumps(value, default=str), done=done)


def _error(message: str) -> ToolCallResult:
    return _result({"error": message})
