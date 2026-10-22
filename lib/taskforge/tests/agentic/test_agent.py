# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``generate_tasks`` with a scripted agent, solver and reviewer on real ShellSim machines and the real gate."""

import dataclasses
import itertools
import json
import shlex
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from rolloutengine.models import ModelRequest, ModelTurn, RolloutLimits, TokenUsage
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from taskcompendium.models import VerifyitGrader

from taskforge.agentic import example_build
from taskforge.agentic.agent import AgentResult, generate_tasks
from taskforge.agentic.config import AgenticConfig, ToolBudget
from taskforge.agentic.gate import Gate
from taskforge.agentic.parallel import ParallelWeb
from taskforge.agentic.prompts import ADVERSARY_PROMPT, GUESS_PROMPT
from taskforge.agentic.tools import MAX_EMPTY_BUDGET_CALLS, OPTION_PROBLEM, BlindSolution, OptionList, TaskReview
from taskforge.capability import CapabilityIdea
from taskforge.llm.client import Completion, ContextExhausted, StructuredResult
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


def tool_replies(result: AgentResult, call_id: str) -> list[dict[str, Any]]:
    """The tool replies to the agent's call ``call_id``, as JSON."""
    messages = result.rollout.conversations[-1].messages
    return [json.loads(m["content"]) for m in messages if m.get("tool_call_id") == call_id]


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
    """Solves every second task attempt and answers ``wrong`` in the others. The adversary role answers
    ``adversary_answer``, and a guess answers 0 or the reference."""

    guesses_right: bool = False
    wrong: str = "1"
    adversary_answer: str = "0"
    attempts: itertools.count = field(default_factory=itertools.count)

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        def told(prompt: str) -> bool:
            return any(prompt in str(message.get("content")) for message in request.messages)

        if told(GUESS_PROMPT):
            answer = EXPECTED if self.guesses_right else "0"
        elif told(ADVERSARY_PROMPT):
            answer = self.adversary_answer
        else:
            answer = EXPECTED if next(self.attempts) % 2 == 0 else self.wrong
        return ModelTurn({"role": "assistant", "content": answer}, (), (), None, "stop", usage=TokenUsage(5, 1))


@dataclass
class PythonSolver:
    """Runs python3 in its shell, then answers with what the shell returned."""

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        last = request.messages[-1]
        if last.get("role") == "tool":
            message = {"role": "assistant", "content": str(last.get("content"))}
            return ModelTurn(message, (), (), None, "stop", usage=TokenUsage(5, 1))
        arguments = json.dumps({"command": "python3 -c 'print(6 * 7)'"})
        tool_call = {"id": "call-1", "type": "function", "function": {"name": "shell", "arguments": arguments}}
        message = {"role": "assistant", "content": "", "tool_calls": [tool_call]}
        return ModelTurn(message, (), (), None, "tool_calls", usage=TokenUsage(5, 1))


@dataclass
class LoopingSolver:
    """Runs a shell command on every turn and never answers."""

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        arguments = json.dumps({"command": "echo still working"})
        tool_call = {
            "id": f"call-{len(request.messages)}",
            "type": "function",
            "function": {"name": "shell", "arguments": arguments},
        }
        message = {"role": "assistant", "content": "", "tool_calls": [tool_call]}
        return ModelTurn(message, (), (), None, "tool_calls", usage=TokenUsage(5, 1))


@dataclass
class NullCallSolver:
    """Makes a tool call whose function is null, then answers 0."""

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        if request.messages[-1].get("role") == "tool":
            return ModelTurn({"role": "assistant", "content": "0"}, (), (), None, "stop", usage=TokenUsage(5, 1))
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function", "function": None}],
        }
        return ModelTurn(message, (), (), None, "tool_calls", usage=TokenUsage(5, 1))


@dataclass
class AcceptingJudge:
    """Accepts each task, and quotes ``option_lists`` as the option lists of the task."""

    option_lists: tuple[OptionList, ...] = ()
    model: str = "judge"

    async def complete(
        self, messages: Sequence[Message], policy: LLMPolicy, request_fields: Mapping[str, object] | None = None
    ) -> Completion:
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


async def test_submit_refuses_a_task_whose_answer_selects_an_option():
    options = ("Lower the cooling water flow.", "Raise the cooling water flow.", "Raise the feed rate.")
    judge = AcceptingJudge(option_lists=(OptionList(options=options),))
    script = (*SCRIPT, call("finish", report="done"))
    async with ParallelWeb("unused") as web:
        config = agentic_config(ScriptedAgent(script), ToolBudget(solver_rollouts=20, judge_calls=2, web_calls=0), web)
        result = await generate_tasks(IDEA, 1, dataclasses.replace(config, judge=judge))

    [submit] = tool_replies(result, "call-10")
    assert not submit["accepted"]
    assert submit["review"]["problems"] == [OPTION_PROBLEM]
    assert result.tasks == ()


async def test_submit_refuses_a_task_whose_answer_the_prompt_alone_gives():
    script = (*SCRIPT, call("finish", report="done"))
    async with ParallelWeb("unused") as web:
        config = agentic_config(ScriptedAgent(script), ToolBudget(solver_rollouts=20, judge_calls=2, web_calls=0), web)
        result = await generate_tasks(IDEA, 1, dataclasses.replace(config, solver_model=Solver(guesses_right=True)))

    [submit] = tool_replies(result, "call-10")
    assert any("guessed the answer in 4 of 4" in problem for problem in submit["problems"])
    assert result.tasks == ()


class OverflowJudge(AcceptingJudge):
    """Rejects every request as too long for its context."""

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], policy: LLMPolicy
    ) -> StructuredResult[T]:
        raise ContextExhausted("prompt too long", ())


async def test_a_task_too_long_for_the_reviewer_is_refused_and_the_session_continues():
    script = (*SCRIPT, call("finish", report="done"))
    async with ParallelWeb("unused") as web:
        config = agentic_config(ScriptedAgent(script), ToolBudget(solver_rollouts=20, judge_calls=2, web_calls=0), web)
        result = await generate_tasks(IDEA, 1, dataclasses.replace(config, judge=OverflowJudge()))

    [submit] = tool_replies(result, "call-10")
    assert "too long" in submit["error"]
    assert (result.failure, result.report) == (None, "done")


async def test_the_tools_stop_a_session_that_loops_on_an_empty_budget():
    # An agent that ignores its empty budget and asks for a web search on every turn.
    agent = ScriptedAgent(tuple(call("web_search", objective="any", queries=["any"]) for _ in range(LIMITS.max_turns)))
    async with ParallelWeb("unused") as web:
        config = agentic_config(agent, ToolBudget(solver_rollouts=0, judge_calls=0, web_calls=0), web)
        result = await generate_tasks(IDEA, 1, config)

    assert agent.turns == MAX_EMPTY_BUDGET_CALLS
    assert result.tasks == ()
    assert result.report is not None and "empty budget" in result.report


async def test_a_session_at_its_turn_limit_gets_a_report_of_the_tools():
    script = tuple(call("shell", command="echo still working") for _ in range(LIMITS.max_turns))
    async with ParallelWeb("unused") as web:
        config = agentic_config(ScriptedAgent(script), ToolBudget(solver_rollouts=20, judge_calls=0, web_calls=0), web)
        result = await generate_tasks(IDEA, 1, config)

    assert result.rollout.stop_reason == "max_turns"
    assert result.report is not None and "max_turns" in result.report


async def test_the_tools_refuse_a_task_whose_id_cannot_name_a_directory():
    task = example_build.task_spec("../" + "x" * 300)
    script = (
        call("shell", command=f"mkdir -p tasks; printf '%s' {shlex.quote(task.model_dump_json())} > {TASK}"),
        call("rollout", paths=[TASK], n=1, role="solver"),
        call("finish", report="done"),
    )
    async with ParallelWeb("unused") as web:
        config = agentic_config(ScriptedAgent(script), ToolBudget(solver_rollouts=1, judge_calls=0, web_calls=0), web)
        result = await generate_tasks(IDEA, 1, config)

    [refusal] = tool_replies(result, "call-2")
    assert "must be at most 128 letters, digits" in refusal["error"]
    assert result.spend.solver_rollouts == 0


async def test_a_task_cannot_take_the_id_of_an_accepted_task():
    # CHANGED is another version of TASK with the same id, and TASK is accepted first.
    script = (*SCRIPT, call("rollout", paths=[CHANGED], n=1, role="solver"), call("finish", report="done"))
    async with ParallelWeb("unused") as web:
        config = agentic_config(ScriptedAgent(script), ToolBudget(solver_rollouts=20, judge_calls=2, web_calls=0), web)
        result = await generate_tasks(IDEA, 2, config)

    [refusal] = tool_replies(result, "call-11")
    assert f"An accepted task already has the id {TASK_ID!r}" in refusal["error"]
    assert len(result.tasks) == 1


async def test_a_control_that_the_answer_format_rejects_is_no_control():
    script = (
        call("shell", command=f"mkdir -p tasks; printf '%s' {shlex.quote(TASK_JSON)} > {TASK}"),
        call("grade", path=TASK, answer="", expect="fail"),
        call("grade", path=TASK, answer=EXPECTED, expect="pass"),
        call("submit", path=TASK),
        call("finish", report="done"),
    )
    async with ParallelWeb("unused") as web:
        config = agentic_config(ScriptedAgent(script), ToolBudget(solver_rollouts=0, judge_calls=0, web_calls=0), web)
        result = await generate_tasks(IDEA, 1, config)

    [empty], [submit] = tool_replies(result, "call-2"), tool_replies(result, "call-4")
    assert empty["control"] is False
    assert "controls" in submit["missing"]


async def test_the_solver_of_the_example_task_has_python():
    script = (
        call("shell", command=f"mkdir -p tasks; printf '%s' {shlex.quote(TASK_JSON)} > {TASK}"),
        call("rollout", paths=[TASK], n=1, role="solver"),
        call("finish", report="done"),
    )
    async with ParallelWeb("unused") as web:
        config = agentic_config(ScriptedAgent(script), ToolBudget(solver_rollouts=1, judge_calls=0, web_calls=0), web)
        result = await generate_tasks(IDEA, 1, dataclasses.replace(config, solver_model=PythonSolver()))

    [rollout] = tool_replies(result, "call-2")
    [answer] = rollout["results"][TASK]["answers"]
    assert "42" in answer["answer"] and "not found" not in answer["answer"]


async def test_the_tools_refuse_environment_variables_and_setup_commands():
    requirements = example_build.task_spec(TASK_ID).environment_requirements
    tasks = {
        "tasks/env.json": requirements.model_copy(update={"environment_variables": {"K": "${PARALLEL_API_KEY}"}}),
        "tasks/setup.json": requirements.model_copy(update={"setup_commands": ("echo 0 > /workspace/ledger.csv",)}),
    }
    writes = [
        call("shell", command=f"mkdir -p tasks; printf '%s' {shlex.quote(task.model_dump_json())} > {path}")
        for path, environment in tasks.items()
        for task in [example_build.task_spec(TASK_ID).model_copy(update={"environment_requirements": environment})]
    ]
    script = (
        *writes,
        *(call("rollout", paths=[path], n=1, role="solver") for path in tasks),
        call("finish", report="x"),
    )
    async with ParallelWeb("unused") as web:
        config = agentic_config(ScriptedAgent(script), ToolBudget(solver_rollouts=2, judge_calls=0, web_calls=0), web)
        result = await generate_tasks(IDEA, 1, config)

    [env], [setup] = tool_replies(result, "call-3"), tool_replies(result, "call-4")
    assert "cannot set environment_variables" in env["error"]
    assert "cannot have setup_commands" in setup["error"]
    assert result.spend.solver_rollouts == 0


async def test_a_solver_that_runs_out_of_turns_fails_its_run():
    script = (
        call("shell", command=f"mkdir -p tasks; printf '%s' {shlex.quote(TASK_JSON)} > {TASK}"),
        call("rollout", paths=[TASK], n=1, role="solver"),
        call("finish", report="done"),
    )
    async with ParallelWeb("unused") as web:
        config = agentic_config(ScriptedAgent(script), ToolBudget(solver_rollouts=1, judge_calls=0, web_calls=0), web)
        solver_limits = LIMITS.model_copy(update={"max_turns": 3})
        result = await generate_tasks(
            IDEA, 1, dataclasses.replace(config, solver_model=LoopingSolver(), solver_limits=solver_limits)
        )

    [rollout] = tool_replies(result, "call-2")
    evidence = rollout["results"][TASK]["evidence"]
    assert (evidence["solver_runs"], evidence["solver_passes"], evidence["solver_format_failures"]) == (1, 0, 0)


async def test_the_pass_rate_of_a_rollout_counts_only_graded_answers():
    script = (
        call("shell", command=f"mkdir -p tasks; printf '%s' {shlex.quote(TASK_JSON)} > {TASK}"),
        call("rollout", paths=[TASK], n=4, role="solver"),
        call("finish", report="done"),
    )
    async with ParallelWeb("unused") as web:
        config = agentic_config(ScriptedAgent(script), ToolBudget(solver_rollouts=4, judge_calls=0, web_calls=0), web)
        result = await generate_tasks(IDEA, 1, dataclasses.replace(config, solver_model=Solver(wrong="")))

    [rollout] = tool_replies(result, "call-2")
    reply = rollout["results"][TASK]
    assert (reply["graded"], reply["format_failures"], reply["pass_rate"]) == (2, 2, 1.0)


async def test_an_adversary_rollout_without_a_graded_answer_does_not_count():
    script = (
        call("shell", command=f"mkdir -p tasks; printf '%s' {shlex.quote(TASK_JSON)} > {TASK}"),
        call("rollout", paths=[TASK], n=4, role="adversary"),
        call("finish", report="done"),
    )
    async with ParallelWeb("unused") as web:
        config = agentic_config(ScriptedAgent(script), ToolBudget(solver_rollouts=4, judge_calls=0, web_calls=0), web)
        result = await generate_tasks(IDEA, 1, dataclasses.replace(config, solver_model=Solver(adversary_answer="")))

    [rollout] = tool_replies(result, "call-2")
    assert rollout["results"][TASK]["evidence"]["adversary_runs"] == 0


async def test_a_review_needs_the_judge_budget_for_all_of_its_calls():
    script = (*SCRIPT, call("finish", report="done"))
    async with ParallelWeb("unused") as web:
        config = agentic_config(ScriptedAgent(script), ToolBudget(solver_rollouts=20, judge_calls=1, web_calls=0), web)
        result = await generate_tasks(IDEA, 1, config)

    [submit] = tool_replies(result, "call-10")
    assert "1 judge calls left" in submit["error"]
    assert result.spend.judge_calls == 0


async def test_rollout_refuses_a_task_that_submit_cannot_accept():
    example = example_build.task_spec(TASK_ID)
    shared = example.resources.model_copy(update={"all": example.resources.worker, "worker": ()})
    task = example.model_copy(update={"resources": shared})
    script = (
        call("shell", command=f"mkdir -p tasks; printf '%s' {shlex.quote(task.model_dump_json())} > {TASK}"),
        call("rollout", paths=[TASK], n=1, role="solver"),
        call("finish", report="done"),
    )
    async with ParallelWeb("unused") as web:
        config = agentic_config(ScriptedAgent(script), ToolBudget(solver_rollouts=1, judge_calls=0, web_calls=0), web)
        result = await generate_tasks(IDEA, 1, config)

    [refusal] = tool_replies(result, "call-2")
    assert "not in resources.all" in refusal["error"]
    assert result.spend.solver_rollouts == 0


async def test_a_solver_tool_call_with_no_function_is_saved():
    script = (
        call("shell", command=f"mkdir -p tasks; printf '%s' {shlex.quote(TASK_JSON)} > {TASK}"),
        call("rollout", paths=[TASK], n=1, role="solver"),
        call("finish", report="done"),
    )
    async with ParallelWeb("unused") as web:
        config = agentic_config(ScriptedAgent(script), ToolBudget(solver_rollouts=1, judge_calls=0, web_calls=0), web)
        result = await generate_tasks(IDEA, 1, dataclasses.replace(config, solver_model=NullCallSolver()))

    [rollout] = tool_replies(result, "call-2")
    assert result.failure is None
    assert "results" in rollout


async def test_finish_refuses_once_while_most_of_the_budget_is_left():
    script = (call("finish", report="too easy"), call("finish", report="too easy"))
    async with ParallelWeb("unused") as web:
        config = agentic_config(ScriptedAgent(script), ToolBudget(solver_rollouts=20, judge_calls=0, web_calls=0), web)
        result = await generate_tasks(IDEA, 1, config)

    [refusal] = tool_replies(result, "call-1")
    assert "20 of 20 solver rollouts are left" in refusal["error"]
    assert result.report == "too easy"


async def test_submit_refuses_a_grader_that_accepts_a_wrong_answer_made_from_the_controls():
    # The reference is an integer, so a tolerance of 1 also accepts the right control plus or minus 1.
    loose = GRADER.model_copy(update={"parameters": {**GRADER.parameters, "tolerance_abs": 1.0}})
    task = example_build.task_spec(TASK_ID).model_copy(update={"grader": loose})
    script = (
        call("shell", command=f"mkdir -p tasks; printf '%s' {shlex.quote(task.model_dump_json())} > {TASK}"),
        call("grade", path=TASK, answer=EXPECTED, expect="pass"),
        call("grade", path=TASK, answer="999999", expect="fail"),
        call("rollout", paths=[TASK], n=8, role="solver"),
        call("rollout", paths=[TASK], n=4, role="adversary"),
        call("submit", path=TASK),
        call("finish", report="done"),
    )
    async with ParallelWeb("unused") as web:
        config = agentic_config(ScriptedAgent(script), ToolBudget(solver_rollouts=12, judge_calls=2, web_calls=0), web)
        result = await generate_tasks(IDEA, 1, config)

    [submit] = tool_replies(result, "call-6")
    assert not submit["accepted"]
    assert [problem.split(",")[0] for problem in submit["problems"]] == [
        f"The grader accepts {int(EXPECTED) + 1}",
        f"The grader accepts {int(EXPECTED) - 1}",
    ]
