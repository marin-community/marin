# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``generate_tasks`` with a scripted agent, solver and reviewer on real ShellSim machines and the real gate."""

import dataclasses
import itertools
import json
import shlex
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel
from rolloutengine.models import ModelRequest, ModelTurn, RolloutLimits, TokenUsage
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from taskcompendium.models import VerifyitGrader

from taskforge.agentic import example_build
from taskforge.agentic.agent import generate_tasks
from taskforge.agentic.config import AgenticConfig, ToolBudget
from taskforge.agentic.gate import Gate
from taskforge.agentic.parallel import ParallelWeb
from taskforge.agentic.prompts import ADVERSARY_PROMPT
from taskforge.agentic.tools import MAX_EMPTY_BUDGET_CALLS, BlindSolution, OptionList, TaskReview
from taskforge.capability import CapabilityIdea
from taskforge.llm.client import Completion, StructuredResult
from taskforge.llm.policy import LLMPolicy, Message

TASK_ID = "ledger--00"
TASK = f"tasks/{TASK_ID}.json"
TASK_JSON = example_build.task_spec(TASK_ID).model_dump_json()
CHANGED = "tasks/changed.json"
GRADER = example_build.task_spec(TASK_ID).grader
assert isinstance(GRADER, VerifyitGrader)
EXPECTED = str(GRADER.parameters["expected"])
IDEA = CapabilityIdea(
    capability_id="example.stock-ledger",
    subject_id="example",
    subject_name="Example",
    catalog_version="test",
    capability={"kind": "capability", "id": "example.stock-ledger", "name": "Read a ledger"},
    prerequisite_edges=(),
    capability_hash="0",
)
LIMITS = RolloutLimits(
    max_turns=40,
    max_total_tokens=None,
    model_turn_timeout=None,
    tool_turn_timeout=None,
    total_turn_timeout=None,
    attempt_timeout=None,
    verifier_timeout=None,
    cleanup_timeout=30,
)


def call(name: str, **arguments: object) -> tuple[str, dict[str, object]]:
    return name, arguments


SCRIPT = (
    # The test workspace is ShellSim, which has no Python, so the agent writes the TaskSpec with the shell.
    call("shell", command=f"mkdir -p tasks; printf '%s' {shlex.quote(TASK_JSON)} > {TASK}"),
    call("submit", path=TASK),
    call("grade", path=TASK, answer="1", expect="pass"),  # A wrong control of the agent's own does not block.
    call("grade", path=TASK, answer=EXPECTED, expect="pass"),
    call("grade", path=TASK, answer="999999", expect="fail"),
    call("rollout", paths=[TASK], n=8, role="solver"),
    call("rollout", paths=[TASK], n=4, role="adversary"),
    call("shell", command=f"sed 's/Reply with the number alone/Reply with a number/' {TASK} > {CHANGED}"),
    call("submit", path=CHANGED),
    call("submit", path=TASK),
)


@dataclass
class ScriptedAgent:
    """Calls the tools of ``script`` in order, one call in each turn."""

    script: Sequence[tuple[str, dict[str, object]]] = SCRIPT
    turns: int = 0

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        name, arguments = self.script[self.turns]
        self.turns += 1
        tool_call = {
            "id": f"call-{self.turns}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }
        message = {"role": "assistant", "content": "", "tool_calls": [tool_call]}
        return ModelTurn(message, (), (), None, "tool_calls", usage=TokenUsage(10, 1))


@dataclass
class Solver:
    """Solves every second task attempt. The adversary role always answers 0."""

    attempts: itertools.count = field(default_factory=itertools.count)

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        adversary = any(ADVERSARY_PROMPT in str(message.get("content")) for message in request.messages)
        answer = "0" if adversary else (EXPECTED if next(self.attempts) % 2 == 0 else "1")
        return ModelTurn({"role": "assistant", "content": answer}, (), (), None, "stop", usage=TokenUsage(5, 1))


@dataclass
class AcceptingJudge:
    """Accepts each task, and quotes ``option_lists`` as the option lists of the task."""

    option_lists: tuple[OptionList, ...] = ()
    model: str = "judge"

    async def complete(self, messages: Sequence[Message], policy: LLMPolicy) -> Completion:
        raise AssertionError("The tools ask the judge only for structured replies")

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], policy: LLMPolicy
    ) -> StructuredResult[T]:
        replies = {
            BlindSolution: {"readings": [], "answer": EXPECTED},
            TaskReview: {
                "option_lists": [options.model_dump() for options in self.option_lists],
                "failure_causes": [],
                "problems": [],
                "summary": "ok",
                "accept": True,
            },
        }
        return StructuredResult(output_type.model_validate(replies[output_type]), ())


def agentic_config(agent_model: Callable[[ModelRequest], Awaitable[ModelTurn]], budget: ToolBudget, web: ParallelWeb):
    return AgenticConfig(
        agent_model=agent_model,
        workspace_factory=ShellSimMachineFactory(),
        solver_model=Solver(),
        judge=AcceptingJudge(),
        judge_policy=LLMPolicy(),
        web=web,
        agent_limits=LIMITS,
        solver_limits=LIMITS,
        gate=Gate(min_solver_runs=8, solver_band=(0.1, 0.8), max_format_failure_rate=0.25, min_adversary_runs=4),
        budget=budget,
        command_timeout=10,
        max_concurrent_rollouts=4,
    )


async def test_submit_accepts_only_the_task_version_with_complete_evidence():
    async with ParallelWeb("unused") as web:
        config = agentic_config(ScriptedAgent(), ToolBudget(solver_rollouts=20, judge_calls=2, web_calls=0), web)
        result = await generate_tasks(IDEA, 1, config)

    assert result.failure is None
    replies = {
        message["tool_call_id"]: json.loads(message["content"])
        for message in result.rollout.conversations[-1].messages
        if message["role"] == "tool"
    }
    assert replies["call-2"]["missing"] == ["controls", "solver band", "adversary"]
    # The rollout results tell the agent what an in-band version still needs.
    assert "adversary rollouts" in replies["call-6"]["results"][TASK]["next"]
    assert replies["call-7"]["results"][TASK]["next"] == "This version has its evidence: submit it now."
    assert replies["call-9"]["missing"] == ["controls", "solver band", "adversary"]
    [accepted] = result.tasks
    assert accepted.task == example_build.task_spec(TASK_ID)
    assert (accepted.evidence.solver_runs, accepted.evidence.solver_passes) == (8, 4)
    assert (accepted.evidence.adversary_runs, accepted.evidence.adversary_passes) == (4, 0)
    assert accepted.evidence.controls == [(True, False), (True, True), (False, False)]
    assert accepted.review.accept
    # A review is a blind solve and the review itself.
    assert (result.spend.solver_rollouts, result.spend.judge_calls) == (12, 2)


async def test_submit_refuses_a_task_whose_correct_option_is_the_longest():
    wrong = ("Lower the cooling water flow by one tenth.", "Raise the feed rate of the reactor by one tenth.")
    correct = "Raise the cooling water flow by one tenth, because the hot steady state then disappears below 440 K."
    judge = AcceptingJudge(option_lists=(OptionList(options=(*wrong, correct), correct=3),))
    script = (*SCRIPT, call("finish", report="done"))
    async with ParallelWeb("unused") as web:
        config = agentic_config(ScriptedAgent(script), ToolBudget(solver_rollouts=20, judge_calls=2, web_calls=0), web)
        result = await generate_tasks(IDEA, 1, dataclasses.replace(config, judge=judge))

    [submit] = [
        json.loads(m["content"]) for m in result.rollout.conversations[-1].messages if m.get("tool_call_id") == "call-10"
    ]
    assert not submit["accepted"]
    assert any("longer than any other option" in problem for problem in submit["review"]["problems"])
    assert result.tasks == ()


@dataclass
class LoopingAgent:
    """Asks for a web search on every turn, as an agent that ignores its empty budget would."""

    turns: int = 0

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        self.turns += 1
        arguments = json.dumps({"objective": "anything", "queries": ["anything"]})
        tool_call = {
            "id": f"call-{self.turns}",
            "type": "function",
            "function": {"name": "web_search", "arguments": arguments},
        }
        message = {"role": "assistant", "content": "", "tool_calls": [tool_call]}
        return ModelTurn(message, (), (), None, "tool_calls", usage=TokenUsage(10, 1))


async def test_the_tools_stop_a_session_that_loops_on_an_empty_budget():
    agent = LoopingAgent()
    async with ParallelWeb("unused") as web:
        config = agentic_config(agent, ToolBudget(solver_rollouts=0, judge_calls=0, web_calls=0), web)
        result = await generate_tasks(IDEA, 1, config)

    assert agent.turns == MAX_EMPTY_BUDGET_CALLS
    assert result.tasks == ()
    assert result.report is not None and "empty budget" in result.report


async def test_finish_refuses_once_while_most_of_the_budget_is_left():
    script = (call("finish", report="too easy"), call("finish", report="too easy"))
    async with ParallelWeb("unused") as web:
        config = agentic_config(ScriptedAgent(script), ToolBudget(solver_rollouts=20, judge_calls=0, web_calls=0), web)
        result = await generate_tasks(IDEA, 1, config)

    [refusal] = [
        json.loads(m["content"]) for m in result.rollout.conversations[-1].messages if m.get("tool_call_id") == "call-1"
    ]
    assert "20 of 20 solver rollouts are left" in refusal["error"]
    assert result.report == "too easy"


async def test_submit_refuses_a_numeric_tolerance_that_accepts_the_next_values():
    # The reference is an integer, so a tolerance of 1 also accepts the reference plus or minus 1.
    loose = GRADER.model_copy(update={"parameters": {**GRADER.parameters, "tolerance_abs": 1.0}})
    task = example_build.task_spec(TASK_ID).model_copy(update={"grader": loose})
    script = (
        call("shell", command=f"mkdir -p tasks; printf '%s' {shlex.quote(task.model_dump_json())} > {TASK}"),
        call("submit", path=TASK),
        call("finish", report="done"),
    )
    async with ParallelWeb("unused") as web:
        config = agentic_config(ScriptedAgent(script), ToolBudget(solver_rollouts=0, judge_calls=0, web_calls=0), web)
        result = await generate_tasks(IDEA, 1, config)

    [submit] = [
        json.loads(m["content"]) for m in result.rollout.conversations[-1].messages if m.get("tool_call_id") == "call-2"
    ]
    assert not submit["accepted"]
    assert any("also accepts" in problem for problem in submit["problems"])
