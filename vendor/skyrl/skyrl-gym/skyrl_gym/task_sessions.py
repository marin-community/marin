"""Direct task sessions for answers, search, and code execution."""

import asyncio
import json
import logging
import re
from collections.abc import Callable
from concurrent.futures import Executor
from functools import partial
from typing import Any

from rolloutengine.models import ModelTurn, Prompt
from shellbox.machine import ExitReason, Machine
from skyrl_gym.task_records import TaskTransition, episode_prompt
from skyrl_gym.source_task import session_parameters
from taskcompendium.grading_result import GradeResult, Outcome
from rolloutengine.models import TaskRolloutSpec

from skyrl_gym.answer_tasks import ground_truth
from skyrl_gym.code_execution import execute_code
from skyrl_gym.envs.gsm8k.utils import compute_score as math_score
from skyrl_gym.envs.lcb.livecodebench import (
    BINARY_REWARD_MODE,
    LCB_REWARD_MODES,
    extract_code_from_model,
    normalize_lcb_ground_truth,
)
from skyrl_gym.envs.search.utils import compute_score as search_score
from skyrl_gym.python_execution import PythonKernel
from skyrl_gym.task_records import terminal_grade
from skyrl_gym.tools.search import SearchClient

AnswerGrader = Callable[[ModelTurn, dict[str, Any], dict[str, Any]], TaskTransition]
logger = logging.getLogger(__name__)
SEARCH_CODE_TIMEOUT = 10.0


class BlockingOperations:
    """Keep cancelled thread operations until session cleanup can release their resources."""

    def __init__(self, executor: Executor | None):
        self.executor = executor
        self.pending: set[asyncio.Future[Any]] = set()

    async def run[T](self, operation: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        pending = asyncio.get_running_loop().run_in_executor(self.executor, partial(operation, *args, **kwargs))
        self.pending.add(pending)
        try:
            return await asyncio.shield(pending)
        finally:
            if pending.done():
                self.pending.discard(pending)

    async def close(self) -> None:
        cancellation = None
        failure = None
        for pending in tuple(self.pending):
            while not pending.done():
                try:
                    await asyncio.shield(pending)
                except asyncio.CancelledError as error:
                    cancellation = error
                except Exception as error:
                    failure = error
            if not pending.cancelled() and (error := pending.exception()) is not None:
                failure = error
            self.pending.discard(pending)
        if cancellation is not None:
            raise cancellation from failure
        if failure is not None:
            raise failure


class AnswerTaskSession:
    """Grade one model turn without a separate episode controller."""

    def __init__(
        self,
        lowered: TaskRolloutSpec,
        machine: Machine | None,
        *,
        grader: AnswerGrader,
        executor: Executor | None = None,
    ):
        task = lowered.task
        self.task = task
        parameters = session_parameters(task)
        self.config = parameters.config
        self.extras = parameters.extras
        self.grader = grader
        self.blocking = BlockingOperations(executor)
        self.result = GradeResult(Outcome.UNAVAILABLE, None, "The task has no completed turn")

    async def prepare(self) -> Prompt:
        return episode_prompt(self.task)

    async def advance(self, turn: ModelTurn) -> TaskTransition:
        transition = await self.blocking.run(self.grader, turn, self.config, self.extras)
        assert transition.done and transition.grade is not None
        self.result = transition.grade
        return transition

    async def grade(
        self, messages: tuple[dict[str, Any], ...], *, stop_reason: str, outputs: dict[str, bytes]
    ) -> GradeResult:
        return self.result

    async def close(self) -> None:
        await self.blocking.close()


class MathTaskSession:
    """Return per-turn math rewards and correction prompts until success or the turn limit."""

    def __init__(self, lowered: TaskRolloutSpec, machine: Machine | None):
        task = lowered.task
        parameters = session_parameters(task)
        self.task = task
        self.expected = ground_truth(parameters.extras)
        self.max_turns = lowered.limits.max_turns
        self.grades: list[GradeResult] = []

    async def prepare(self) -> Prompt:
        return episode_prompt(self.task)

    async def advance(self, turn: ModelTurn) -> TaskTransition:
        reward = math_score(turn.text, self.expected, method="strict", format_score=0.2 / self.max_turns)
        correct = reward == 1.0
        grade = GradeResult(Outcome.GRADED, float(correct), passed=correct)
        self.grades.append(grade)
        remaining = self.max_turns - len(self.grades)
        done = remaining == 0 or reward == 1.0
        observations = ()
        if not done:
            prompt = (
                "Please provide your step-by-step reasoning, "
                "and also include a tentative numeric answer at the end in the exact format: '#### ANSWER'."
                if remaining > 1
                else "Now provide only the final numeric answer in the exact format: '#### ANSWER'."
            )
            observations = ({"role": "user", "content": prompt},)
        return TaskTransition(
            done=done, observations=observations, reward=reward, grade=grade, metrics={"steps": len(self.grades)}
        )

    async def grade(
        self, messages: tuple[dict[str, Any], ...], *, stop_reason: str, outputs: dict[str, bytes]
    ) -> GradeResult:
        return terminal_grade(self.grades)

    async def close(self) -> None:
        pass


class SearchTaskSession:
    """Execute search actions and grade the task transcript on completion."""

    def __init__(self, lowered: TaskRolloutSpec, machine: Machine | None, *, executor: Executor | None = None):
        task = lowered.task
        parameters = session_parameters(task)
        config = parameters.config
        self.task = task
        self.expected = ground_truth(parameters.extras)
        self.max_turns = lowered.limits.max_turns
        self.blocking = BlockingOperations(executor)
        self.tool = SearchClient(**{key: config[key] for key in ("search_url", "topk", "timeout", "log_requests")})
        self.transcript: list[str] = []
        self.grades: list[GradeResult] = []

    async def prepare(self) -> Prompt:
        return episode_prompt(self.task)

    async def advance(self, turn: ModelTurn) -> TaskTransition:
        for tag in ("</search>", "</answer>"):
            if tag in turn.text and turn.text.split(tag, 1)[1]:
                raise ValueError(f"The model response must end at {tag}")
        self.transcript.append(turn.text)
        done = len(self.grades) + 1 >= self.max_turns or ("<answer>" in turn.text and "</answer>" in turn.text)
        reward = search_score("".join(self.transcript), self.expected) if done else 0.0
        grade = (
            GradeResult(Outcome.GRADED, reward, passed=reward == 1.0)
            if done
            else GradeResult(Outcome.UNAVAILABLE, None, "Tool execution is not a terminal verdict")
        )
        self.grades.append(grade)
        if done:
            return TaskTransition(done=True, reward=reward, grade=grade)
        match = re.search(r"<search>(.*?)</search>", turn.text, re.DOTALL)
        query = match.group(1) if match else None
        output = await self.blocking.run(self.tool.search, query)
        observation = "\n<information>" + output + "</information>\n"
        self.transcript.append(observation)
        return TaskTransition(
            done=False,
            observations=({"role": "user", "content": observation},),
            reward=0.0,
            grade=grade,
            metrics={"tool_name": "search", "tool_input": [query]},
        )

    async def grade(
        self, messages: tuple[dict[str, Any], ...], *, stop_reason: str, outputs: dict[str, bytes]
    ) -> GradeResult:
        return terminal_grade(self.grades)

    async def close(self) -> None:
        try:
            await self.blocking.close()
        finally:
            self.tool.close()


class CodeTaskSession:
    """Grade one candidate program in Shellbox, with hidden test outputs on the worker."""

    def __init__(self, lowered: TaskRolloutSpec, machine: Machine | None):
        task = lowered.task
        parameters = session_parameters(task)
        self.task = task
        self.machine = machine
        reward_mode = parameters.config.get("reward_mode", BINARY_REWARD_MODE)
        if reward_mode not in LCB_REWARD_MODES:
            raise ValueError(f"Unsupported LCB reward_mode: {reward_mode!r}")
        self.reward_mode = reward_mode
        try:
            self.tests = json.loads(normalize_lcb_ground_truth(ground_truth(parameters.extras)))
        except (ValueError, TypeError):
            logger.exception("Invalid LCB ground truth")
            self.tests = None
        if self.tests:
            assert machine is not None
        self.result = GradeResult(Outcome.UNAVAILABLE, None, "The task has no completed turn")

    async def prepare(self) -> Prompt:
        return episode_prompt(self.task)

    async def advance(self, turn: ModelTurn) -> TaskTransition:
        code = extract_code_from_model(turn.text)
        if not self.tests:
            self.result = GradeResult(Outcome.INFRA_ERROR, None, "Invalid LCB task")
            return TaskTransition(
                done=True,
                reward=0.0,
                grade=self.result,
                metrics={"parsed_code": None, "verifier_error": "invalid reward_model.ground_truth"},
            )
        assert self.machine is not None
        try:
            reward, details = await execute_code(self.machine, self.tests, code or "", reward_mode=self.reward_mode)
        except (OSError, RuntimeError, ValueError) as error:
            logger.exception("LCB verification failed for task %s", self.task.id)
            self.result = GradeResult(
                Outcome.INFRA_ERROR,
                None,
                "Code verification unavailable",
                diagnostics={"error_type": type(error).__name__, "error_message": str(error)},
            )
            return TaskTransition(done=True, reward=0.0, grade=self.result)
        self.result = GradeResult(Outcome.GRADED, reward)
        return TaskTransition(done=True, reward=reward, grade=self.result, metrics={"parsed_code": code, **details})

    async def grade(
        self, messages: tuple[dict[str, Any], ...], *, stop_reason: str, outputs: dict[str, bytes]
    ) -> GradeResult:
        return self.result

    async def close(self) -> None:
        pass


class SearchCodeTaskSession:
    """Execute search and Python actions, then grade the completed task transcript.

    Each Python action receives a fresh namespace.
    """

    def __init__(self, lowered: TaskRolloutSpec, machine: Machine | None, *, executor: Executor | None = None):
        task = lowered.task
        assert machine is not None
        parameters = session_parameters(task)
        self.task = task
        self.expected = ground_truth(parameters.extras)
        self.max_turns = lowered.limits.max_turns
        self.blocking = BlockingOperations(executor)
        self.search = SearchClient(**parameters.config.get("search", {}))
        self.python = PythonKernel(machine)
        self.transcript: list[str] = []
        self.grades: list[GradeResult] = []

    async def prepare(self) -> Prompt:
        await self.python.start()
        return episode_prompt(self.task)

    async def advance(self, turn: ModelTurn) -> TaskTransition:
        self.transcript.append(turn.text)
        done = len(self.grades) + 1 >= self.max_turns or ("<solution>" in turn.text and "</solution>" in turn.text)
        reward = math_score("\n".join(self.transcript), self.expected) if done else 0.0
        grade = (
            GradeResult(Outcome.GRADED, reward, passed=reward == 1.0)
            if done
            else GradeResult(Outcome.UNAVAILABLE, None, "Tool execution is not a terminal verdict")
        )
        self.grades.append(grade)
        if done:
            return TaskTransition(done=True, reward=reward, grade=grade)
        match = re.search(r"<tool>(.*?)</tool>", turn.text, re.DOTALL)
        tool = re.search(r"<(\w+)>(.*?)</\1>", match.group(1), re.DOTALL) if match else None
        if tool is None:
            output = "No valid tool block found in action string."
        elif tool.group(1) == "search":
            output = await self.blocking.run(self.search.search, tool.group(2).strip())
        elif tool.group(1) == "python":
            result = await self.python.execute(
                "exec(" + repr(tool.group(2).strip()) + ", {})", timeout=SEARCH_CODE_TIMEOUT
            )
            if result.reason == ExitReason.TIMED_OUT:
                output = f"Error executing Python code: Execution timed out after {SEARCH_CODE_TIMEOUT} seconds."
            elif result.exit_code != 0 or result.stderr:
                output = "Error executing Python code: " + (result.stdout + result.stderr).decode().strip()
            else:
                output = result.stdout.decode().strip()
        else:
            output = f"Unknown tool: {tool.group(1)}"
        self.transcript.append(output)
        return TaskTransition(
            done=False,
            observations=({"role": "user", "content": output},) if output else (),
            reward=0.0,
            grade=grade,
        )

    async def grade(
        self, messages: tuple[dict[str, Any], ...], *, stop_reason: str, outputs: dict[str, bytes]
    ) -> GradeResult:
        return terminal_grade(self.grades)

    async def close(self) -> None:
        try:
            await self.blocking.close()
        finally:
            try:
                await self.python.close()
            finally:
                self.search.close()
