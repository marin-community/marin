# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import errno
import json
import os
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from functools import partial
from pathlib import Path, PurePosixPath
from typing import override

import httpx
import pytest
from lib.taskforge.tests.effect_fakes import local_records
from lib.taskforge.tests.resource_doubles import FakeCommand, FakeEnv, FakeInference, FakeWeb
from lib.taskforge.tests.resources.iris_fakes import LocalController, local_iris_machine
from lib.taskforge.tests.stage_fakes import OVER_CREDITED, over_credited
from rolloutengine.models import GradedRollout, Model, TaskRolloutSpec
from rolloutengine.test_support import arithmetic_task, machine_runtime, rollout_limits
from taskcompendium.grader import verifyit_package
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import CommandSemantics, EnvironmentRequirements
from taskcompendium.runtime.resources import inline_resource, resource_bytes
from verifyit.spec import NumericSpec

from taskforge.effects.context import Context
from taskforge.effects.events import Resolved
from taskforge.effects.issues import ContextExceeded, Hold, OutputBudgetExhausted, Transient
from taskforge.effects.log import BufferSink, EventLog, Opened
from taskforge.harness.loop import AgenticHarness
from taskforge.harness.protocol import (
    Returned,
    ReturnRequest,
    Stopped,
    StopReason,
    Submitted,
    Tool,
    ToolResult,
)
from taskforge.harness.role import install_env
from taskforge.harness.tools.adversary import AdversaryReport, ReportTool
from taskforge.harness.tools.files import ReadFileTool, WriteFileTool
from taskforge.harness.tools.finish import FinishTool
from taskforge.harness.tools.grade import GradeTool
from taskforge.harness.tools.shell import ShellTool
from taskforge.harness.tools.submit import SubmitTool
from taskforge.harness.tools.web import WebFetchTool, WebSearchTool
from taskforge.llm.policy import LLMPolicy, replay_arguments
from taskforge.lowering import GRADING_MACHINE, LoweringSettings, requirements, rollout_engine
from taskforge.model.controls import CONTROLS, Candidate, Control, ControlKind, RewardBand
from taskforge.model.llm import Completion, FinishReason, Message, Role, ToolCall, Usage
from taskforge.model.workpiece import Conversation, Submission
from taskforge.policy import GRADER_BASE_IMAGE
from taskforge.resources.env import (
    CommandResult,
    Env,
    ReadFile,
    RunCommand,
    ShellboxEnv,
    ShellboxEnvFactory,
    WriteFile,
    read_file,
    run_command,
    write_file,
)
from taskforge.resources.grading import GradeCandidate, GradingResource, grade
from taskforge.resources.rollouts import Engine
from taskforge.resources.sandbox.factories import shellsim_factory
from taskforge.resources.web import ParallelWeb, WebFetch, WebSearch, fetch, search

pytestmark = pytest.mark.asyncio


def accept(submission: Submission) -> tuple[str, ...]:
    return ()


def environment(ctx: Context) -> FakeEnv:
    task = arithmetic_task().model_dump_json().encode()
    scripts = {
        "task": FakeCommand(
            CommandResult(0, "task ready", "", 0.1, False), {PurePosixPath("/workspace/task.json"): task}
        ),
        "controls": FakeCommand(
            CommandResult(0, "controls ready", "", 0.1, False), {PurePosixPath("/workspace/controls.json"): b"[]"}
        ),
    }
    env = FakeEnv("image", scripts)
    ctx.handle(RunCommand, partial(run_command, notify=ctx.notify), env)
    ctx.handle(ReadFile, partial(read_file, notify=ctx.notify), env)
    ctx.handle(WriteFile, partial(write_file, notify=ctx.notify), env)
    web = FakeWeb({"query": "s" * 10000}, {"https://example.org": "f" * 10000})
    ctx.handle(WebSearch, partial(search, notify=ctx.notify), web)
    ctx.handle(WebFetch, partial(fetch, notify=ctx.notify), web)
    return env


@asynccontextmanager
async def opened(ctx: Context, env: Env) -> AsyncIterator[None]:
    """One author attempt on ``env``, the scope its re-executed effects need."""
    async with ctx.scope("author", Opened("author/0", env.fingerprint().digest)):
        yield


def call(name: str, args: object, identity: str) -> Completion:
    return Completion("", (ToolCall(identity, name, json.dumps(args)),), reasoning="plan")


async def test_malformed_report_and_schema_errors_recover_with_replayable_history() -> None:
    malformed = '{"findings": '
    script = (
        Completion(
            "",
            (
                ToolCall("1", "report", malformed),
                ToolCall("2", "missing", "{}"),
                ToolCall("3", "report", '{"findings": "wrong"}'),
                ToolCall("4", "shell", '{"command": "must not execute", "timeout": "30"}'),
            ),
        ),
        call("report", {"findings": []}, "5"),
    )
    log = EventLog()
    ctx = Context(log=log)
    client = FakeInference(script)
    env = environment(ctx)
    async with opened(ctx, env):
        result = await AgenticHarness(262144, LLMPolicy(), "continue", 20000, replay_arguments).run(
            ctx, Conversation((), ()), client, {"shell": ShellTool(30, 120), "report": ReportTool(arithmetic_task(), {})}
        )
    await ctx.drain()
    assert isinstance(result, Returned) and result.payload == AdversaryReport(())
    history = client.requests[1].conversation
    assert [m.tool_call_id for m in history if m.role == Role.TOOL] == ["1", "2", "3", "4"]
    assert all(isinstance(json.loads(call.arguments), dict) for message in history for call in message.tool_calls)
    assert malformed in json.loads(history[0].tool_calls[0].arguments).values()
    assert "call report again" in history[1].content and "failed validation" in history[3].content
    assert any(isinstance(record.event, Resolved) and record.event.value == script[0] for record in log.records)
    assert not any(isinstance(record.event, RunCommand) for record in log.records)


async def test_web_transport_and_replay_cache_http_results() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"results": ["evidence"]})

    log = EventLog()
    async with httpx.AsyncClient(
        base_url="https://api.parallel.ai/v1", transport=httpx.MockTransport(handler)
    ) as client:
        web = ParallelWeb(client)
        ctx = Context(log=log)
        ctx.handle(WebSearch, partial(search, notify=ctx.notify), web)
        ctx.handle(WebFetch, partial(fetch, notify=ctx.notify), web)
        assert json.loads(await ctx.emit(WebSearch("query"))) == {"results": ["evidence"]}
        await ctx.emit(WebFetch("https://example.org"))
        await ctx.drain()
        replay_web = FakeWeb({}, {}, identity=web.fingerprint().digest)
        replay = Context(log=EventLog(replay_from=log.records))
        replay.handle(WebSearch, partial(search, notify=replay.notify), replay_web)
        assert json.loads(await replay.emit(WebSearch("query"))) == {"results": ["evidence"]}
        await replay.drain()
    assert [str(request.url) for request in requests] == [
        "https://api.parallel.ai/v1/search",
        "https://api.parallel.ai/v1/extract",
    ]
    assert json.loads(requests[0].content)["search_queries"] == ["query"]
    assert replay_web.requests == []


async def test_grade_uses_current_draft_and_report_snapshots_candidate_files() -> None:
    ctx = Context(log=EventLog())
    env = environment(ctx)
    async with opened(ctx, env):
        settings = LoweringSettings(GRADING_MACHINE, GRADING_MACHINE, rollout_limits())
        grading = GradingResource(rollout_engine({}, command_timeout=1), "fixture")
        ctx.handle(GradeCandidate, partial(grade, notify=ctx.notify), grading)
        tool = GradeTool("/workspace/task.json", {}, settings)
        await ctx.emit(RunCommand("task", 30))
        first = await tool(ctx, {"reply": "12"})
        changed = arithmetic_task(grader=verifyit_package(NumericSpec("7", tolerance_abs=0, tolerance_rel=0)).grader)
        await ctx.emit(WriteFile(PurePosixPath("/workspace/task.json"), changed.model_dump_json().encode()))
        second = await tool(ctx, {"reply": "12"})
        assert isinstance(first, ToolResult) and json.loads(first.content)["reward"] == 1
        assert isinstance(second, ToolResult) and json.loads(second.content)["reward"] == 0
        await ctx.emit(WriteFile(PurePosixPath("/workspace/answer"), b"candidate"))
        report = await ReportTool(arithmetic_task(environment_requirements=requirements(image=None)), {})(
            ctx,
            {
                "findings": [
                    {
                        "defect": "false_reject",
                        "why_the_grade_is_wrong": "a correct answer earned nothing",
                        "candidate": {"reply": "12", "files": ["/workspace/answer"]},
                        "grader_observed_status": "graded",
                        "grader_observed_reward": 0.0,
                        "deserved_reward": 1.0,
                    }
                ]
            },
        )
        await ctx.emit(WriteFile(PurePosixPath("/workspace/answer"), b"changed"))
    await ctx.drain()
    assert isinstance(report, ReturnRequest) and isinstance(report.payload, AdversaryReport)
    assert resource_bytes(report.payload.findings[0].candidate.files[0]) == b"candidate"


@pytest.mark.parametrize(
    "name,args,original",
    [
        ("web_search", {"query": "query"}, "s" * 10000),
        ("web_fetch", {"url": "https://example.org"}, "f" * 10000),
        ("unknown" * 100, {}, "Unknown tool: " + "unknown" * 100),
        ("shell", {"command": "task"}, None),
    ],
    ids=["search", "fetch", "unknown", "shell"],
)
async def test_tool_output_spill_slices_and_persisted_replay(
    name: str, args: dict[str, object], original: str | None
) -> None:
    sink = BufferSink()
    ctx = Context(log=EventLog(sink=sink))
    env = environment(ctx)
    async with opened(ctx, env):
        tools = {
            "shell": ShellTool(30, 120),
            "web_search": WebSearchTool(),
            "web_fetch": WebFetchTool(),
            "finish": FinishTool(),
        }
        harness, initial = AgenticHarness(1000, LLMPolicy(), "continue", 8, replay_arguments), Conversation((), ())
        client = FakeInference((call(name, args, "../../escape"), call("finish", {"reason": "done"}, "done")))
        result = await harness.run(ctx, initial, client, tools)
        [(path, data)] = [
            (path, data) for path, data in env.files.items() if path.parent == PurePosixPath(".tool-output")
        ]
        if original is None:
            assert json.loads(data)["stdout"] == "task ready"
            original = data.decode()
        assert data.decode() == original and ".." not in path.parts
        content = client.requests[1].conversation[-1].content
        assert content.startswith(original[:8] + "\n[Full output saved to ") and str(path) in content
        assert await ReadFileTool()(ctx, {"path": str(path), "offset": 8, "limit": 20}) == ToolResult(original[8:28])
    await ctx.drain()
    replay = Context(log=EventLog(replay_from=local_records(sink)))
    replay_env, replay_client = environment(replay), FakeInference(())
    async with opened(replay, replay_env):
        assert await harness.run(replay, initial, replay_client, tools) == result
        # A replayed attempt issues its recorded sandbox requests in order, to the end.
        assert await ReadFileTool()(replay, {"path": str(path), "offset": 8, "limit": 20}) == ToolResult(original[8:28])
    await replay.drain()
    assert not replay_client.requests and replay_env.files == env.files


@pytest.mark.parametrize("write", [False, True])
@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError(2, "missing", "/bad"),
        IsADirectoryError(21, "directory", "/bad"),
        NotADirectoryError(20, "parent", "/bad"),
    ],
)
async def test_file_path_errors_return_feedback(write: bool, failure: Exception) -> None:
    ctx = Context(log=EventLog())

    async def fail(event: ReadFile | WriteFile) -> bytes:
        raise failure

    ctx.handle(ReadFile, fail)
    ctx.handle(WriteFile, fail)
    tool = WriteFileTool() if write else ReadFileTool()
    result = await tool(ctx, {"path": "/bad", "content": "data"} if write else {"path": "/bad"})
    await ctx.drain()
    assert isinstance(result, ToolResult) and "/bad" in result.content


async def test_grade_reports_a_missing_draft_but_propagates_grading_infrastructure_errors() -> None:
    class Unreachable(Engine):
        @override
        async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
            raise ConnectionError("iris controller unreachable")

    ctx = Context(log=EventLog())
    env = environment(ctx)
    async with opened(ctx, env):
        grading = GradingResource(Unreachable(), "fixture")
        ctx.handle(GradeCandidate, partial(grade, notify=ctx.notify), grading)
        tool = GradeTool(
            "/workspace/task.json", {}, LoweringSettings(GRADING_MACHINE, GRADING_MACHINE, rollout_limits())
        )
        missing = await tool(ctx, {"reply": "12"})
        assert isinstance(missing, ToolResult) and "/workspace/task.json" in missing.content
        await ctx.emit(RunCommand("task", 30))
        with pytest.raises(ConnectionError, match="unreachable"):
            await tool(ctx, {"reply": "12"})


# Regression: the grading machine's output limit escaped the grade tool, and validation
# credited the adversary with an exhausted search that reported nothing.
async def test_grading_machine_exhaustion_is_grade_tool_feedback_and_the_session_continues() -> None:
    ctx = Context(log=EventLog())
    env = environment(ctx)

    async def output_limited(event: GradeCandidate) -> GradeResult:
        raise OutputBudgetExhausted("output budget exhausted")

    ctx.handle(GradeCandidate, output_limited)
    tools = {
        "grade": GradeTool(
            "/workspace/task.json", {}, LoweringSettings(GRADING_MACHINE, GRADING_MACHINE, rollout_limits())
        ),
        "shell": ShellTool(30, 120),
        "finish": FinishTool(),
    }
    script = (
        call("shell", {"command": "task"}, "1"),
        call("grade", {"reply": "12"}, "2"),
        call("finish", {"reason": "done"}, "3"),
    )
    client = FakeInference(script)
    async with opened(ctx, env):
        result = await AgenticHarness(1000, LLMPolicy(), "continue", 20000, replay_arguments).run(
            ctx, Conversation((), ()), client, tools
        )
    await ctx.drain()
    assert isinstance(result, Stopped) and result.reason is StopReason.FINISHED
    assert client.requests[2].conversation[-1].content.startswith("Grading this candidate failed and produced no grade")


# Regression: a live grade showed "cut": 3.0, "level": 4.0 and the same grade
# reloaded from the journal showed 3 and 4, so the author's next request changed and a cold resume paid for a recorded
# turn. The grade tool renders the numbers one way either side of a resume.
@pytest.mark.parametrize("level", [4.0, 4, 3.5])
async def test_a_decision_model_grade_renders_identically_after_a_cold_resume(level: float) -> None:
    async def graded(event: GradeCandidate) -> GradeResult:
        read = {"model": "clef-test", "level": level, "cut": 3.0}
        return GradeResult(Outcome.GRADED, 1.0, detail={"decision_model": read})

    async def unpaid(event: GradeCandidate) -> GradeResult:
        raise AssertionError("a recorded grade is read again")

    settings = LoweringSettings(GRADING_MACHINE, GRADING_MACHINE, rollout_limits())
    tools = {
        "grade": GradeTool("/workspace/task.json", {}, settings),
        "shell": ShellTool(30, 120),
        "finish": FinishTool(),
    }
    script = (call("shell", {"command": "task"}, "1"), call("grade", {"reply": "12"}, "2"))
    harness = AgenticHarness(1000, LLMPolicy(), "continue", 20000, replay_arguments)
    sink = BufferSink()
    live = FakeInference((*script, call("finish", {"reason": "done"}, "3")))
    results, cold = [], FakeInference(())
    for ctx, handler, client in ((Context(log=EventLog(sink=sink)), graded, live), (None, unpaid, cold)):
        ctx = ctx or Context(log=EventLog(replay_from=local_records(sink)))  # a cold process: the journal alone
        env = environment(ctx)
        ctx.handle(GradeCandidate, handler)
        async with opened(ctx, env):
            results.append(await harness.run(ctx, Conversation((), ()), client, tools))
        await ctx.drain()
    assert json.loads(live.requests[2].conversation[-1].content)["decision_model"]["cut"] == 3
    assert (cold.requests, results[0]) == ([], results[1])  # every author turn replays: the same request bytes


async def test_grade_holds_when_the_host_cannot_lower_a_valid_draft() -> None:
    ctx = Context(log=EventLog())
    env = environment(ctx)
    async with opened(ctx, env):
        container = arithmetic_task().model_copy(
            update={
                "environment_requirements": EnvironmentRequirements(
                    docker_image=GRADER_BASE_IMAGE,
                    command_semantics=CommandSemantics.LINUX_PROCESS,
                    capabilities=("shell",),
                )
            }
        )
        await ctx.emit(WriteFile(PurePosixPath("/workspace/task.json"), container.model_dump_json().encode()))
        tool = GradeTool(
            "/workspace/task.json", {}, LoweringSettings(GRADING_MACHINE, GRADING_MACHINE, rollout_limits())
        )
        with pytest.raises(Hold, match="Unknown Shellbox factory"):
            await tool(ctx, {"reply": "12"})
    await ctx.drain()


@pytest.mark.parametrize("fails", [False, True])
async def test_empty_turns_nudge_until_finish_or_three_consecutive_failures(fails: bool) -> None:
    script = (Completion("first"), Completion(""), Completion("second"), Completion(""), Completion(""))
    client = FakeInference((*script, Completion("") if fails else call("finish", {"reason": "done"}, "finish")))
    ctx = Context(log=EventLog())
    harness = AgenticHarness(262144, LLMPolicy(), "use tools", 20000, replay_arguments)
    if fails:
        with pytest.raises(Hold, match="provider returned 3 empty completions"):
            await harness.run(ctx, Conversation((), ()), client, {"finish": FinishTool()})
    else:
        result = await harness.run(ctx, Conversation((), ()), client, {"finish": FinishTool()})
        assert isinstance(result, Stopped) and result.reason == StopReason.FINISHED
    await ctx.drain()
    assert len(client.requests) == 6
    assert all(request.messages[-1] == Message(Role.USER, "use tools") for request in client.requests[1:])


@pytest.mark.parametrize("recovers", [True, False])
async def test_reasoning_only_length_retries_unchanged_conversation_and_replays(recovers: bool) -> None:
    failed = Completion(
        "",
        finish_reason=FinishReason.LENGTH,
        usage=Usage(prompt_tokens=48461, completion_tokens=565939, reasoning_tokens=565939),
        reasoning="repeated reasoning",
    )
    client = FakeInference((failed, call("finish", {"reason": "done"}, "finish") if recovers else failed))
    sink = BufferSink()
    ctx = Context(log=EventLog(sink=sink))
    initial = Conversation((Message(Role.USER, "build"),), ())
    harness = AgenticHarness(262144, LLMPolicy(), "use tools", 20000, replay_arguments)
    tools = {"finish": FinishTool()}
    result = await harness.run(ctx, initial, client, tools)
    await ctx.drain()
    assert isinstance(result, Stopped)
    assert result.reason == (StopReason.FINISHED if recovers else StopReason.DEGENERATE_OUTPUT)
    assert len(client.requests) == 2
    assert client.requests[0].conversation == client.requests[1].conversation == initial.messages
    assert all(message.reasoning != failed.reasoning for message in result.session.messages)
    if not recovers:
        assert result.detail == "565939 reasoning tokens, finish=length"
        assert result.session.messages == initial.messages
    records = local_records(sink)
    assert any(isinstance(record.event, Resolved) and record.event.value == failed for record in records)
    replay = Context(log=EventLog(replay_from=records))
    replay_client = FakeInference(())
    assert await harness.run(replay, initial, replay_client, tools) == result
    await replay.drain()
    assert replay_client.requests == []


@pytest.mark.parametrize("name", ["grade", "submit"])
async def test_candidate_files_without_task_machine_return_feedback(name: str) -> None:
    ctx = Context(log=EventLog())
    env = environment(ctx)
    async with opened(ctx, env):
        await ctx.emit(RunCommand("task", 30))
        candidate = Candidate("12", (inline_resource("workspace/answer", b"12"),))
        controls = (Control(candidate, RewardBand(Outcome.GRADED, reward_min=1), ControlKind.POSITIVE),)
        await ctx.emit(WriteFile(PurePosixPath("/workspace/controls.json"), CONTROLS.dump_json(controls)))
        await ctx.emit(WriteFile(PurePosixPath("/workspace/answer"), b"12"))
        settings = LoweringSettings(GRADING_MACHINE, GRADING_MACHINE, rollout_limits())
        tools: dict[str, Tool[object]] = {
            "grade": GradeTool("/workspace/task.json", {}, settings),
            "submit": SubmitTool(accept, "/workspace/task.json", "/workspace/controls.json"),
        }
        args: dict[str, dict[str, object]] = {
            "grade": {"reply": "12", "files": ["/workspace/answer"]},
            "submit": {"task_path": "/workspace/task.json", "controls_path": "/workspace/controls.json"},
        }
        # No grading executor: malformed author data must never reach the executor backstop.
        result = await tools[name](ctx, args[name])
    await ctx.drain()
    assert isinstance(result, ToolResult) and "Candidate files require a task machine" in result.content


@pytest.mark.parametrize("overflow", ["provider", "prompt", "reply", "tool"])
async def test_context_exit_counts_retained_turns_and_only_appended_tokens(overflow: str) -> None:
    initial = Conversation((Message(Role.ASSISTANT, "earlier"),), ())
    first = Completion(
        "", call("shell", {"command": "task"}, "1").tool_calls, usage=Usage(prompt_tokens=200000, cached_tokens=199000)
    )
    second = (
        ContextExceeded("requested 383788 tokens")
        if overflow == "provider"
        else Completion(
            "answer" if overflow == "reply" else "",
            () if overflow == "reply" else call("shell", {"command": "task"}, "2").tool_calls,
            finish_reason=FinishReason.LENGTH,
            usage=Usage(
                prompt_tokens=262144 if overflow == "prompt" else 262143,
                completion_tokens=1 if overflow == "reply" else 0,
            ),
        )
    )
    ctx = Context(log=EventLog())
    env = environment(ctx)
    async with opened(ctx, env):
        client = FakeInference((first, second))
        result = await AgenticHarness(262144, LLMPolicy(), "continue", 20000, replay_arguments).run(
            ctx, initial, client, {"shell": ShellTool(30, 120)}
        )
    await ctx.drain()
    assert isinstance(result, Stopped) and result.reason == StopReason.CONTEXT_LIMIT
    assert result.turns == 3 and len(client.requests) == 2
    assert result.session.messages[-1].tool_call_id == {"provider": "1", "reply": None}.get(overflow, "2")
    if overflow == "provider":
        assert "requested 383788 tokens" in result.summary
        assert result.session.messages == client.requests[-1].conversation


async def test_adversary_report_requires_claim_evidence_and_uses_bound_task() -> None:
    ctx = Context(log=EventLog())
    env = environment(ctx)
    async with opened(ctx, env):
        task = arithmetic_task()
        tool = ReportTool(task, {})
        commentary = await tool(ctx, {"findings": ["No grading defect found"]})
        assert isinstance(commentary, ToolResult)
        claim = over_credited("wrong answer accepted", "9")
        report = await tool(ctx, {"findings": [claim]})
        assert isinstance(report, ReturnRequest) and isinstance(report.payload, AdversaryReport)
        finding = report.payload.findings[0]
        assert finding.claim == "wrong answer accepted" and finding.candidate.reply == "9"
        assert finding.defect == OVER_CREDITED
        # The adversary can edit its scratch task file, but cannot change the bound task's machine requirement.
        edited = arithmetic_task(environment_requirements=requirements(image=None))
        await ctx.emit(WriteFile(PurePosixPath("/workspace/task.json"), edited.model_dump_json().encode()))
        await ctx.emit(WriteFile(PurePosixPath("/workspace/answer"), b"9"))
        rejected = await tool(
            ctx, {"findings": [{**claim, "candidate": {"reply": "9", "files": ["/workspace/answer"]}}]}
        )
        empty = await tool(ctx, {"findings": []})
    await ctx.drain()
    assert isinstance(rejected, ToolResult) and "Candidate files require a task machine" in rejected.content
    assert isinstance(empty, ReturnRequest) and isinstance(empty.payload, AdversaryReport)
    assert empty.payload.findings == ()


async def test_malformed_grader_checks_return_feedback_and_corrected_checks_are_reported() -> None:
    ctx = Context(log=EventLog())
    env = environment(ctx)
    async with opened(ctx, env):
        await ctx.emit(RunCommand("task", 30))

        async def invalid_checks(event: GradeCandidate) -> GradeResult:
            return GradeResult(Outcome.GRADED, 1, detail={"checks": {"A": False}})

        ctx.handle(GradeCandidate, invalid_checks)
        settings = LoweringSettings(GRADING_MACHINE, GRADING_MACHINE, rollout_limits())
        tool = GradeTool("/workspace/task.json", {}, settings)
        result = await tool(ctx, {"reply": "12"})
        assert isinstance(result, ToolResult) and "checks must be a list of" in result.content
        assert "name" in result.content and "passed" in result.content and "weight" in result.content

        async def valid_checks(event: GradeCandidate) -> GradeResult:
            return GradeResult(Outcome.GRADED, 0, detail={"checks": [{"name": "A", "passed": False, "weight": 1}]})

        ctx.handle(GradeCandidate, valid_checks)
        corrected = await tool(ctx, {"reply": "9"})
    await ctx.drain()
    assert isinstance(corrected, ToolResult)
    assert json.loads(corrected.content)["grader_checks"] == [{"name": "A", "passed": False, "weight": 1}]


def iris_env(ctx: Context, workdir: Path, controller: LocalController) -> ShellboxEnv:
    env = ShellboxEnv(local_iris_machine(workdir, controller), "image", AsyncExitStack(), str(workdir))
    install_env(ctx, env)
    return env


# Regression: Iris reported ENOENT below a regular file and below a directory it cannot search, and
# then archived a directory read and held the workpiece.
@pytest.mark.parametrize(
    "path,code",
    [
        ("grade.py", errno.ENOENT),
        ("present.txt/child", errno.ENOTDIR),
        ("locked/grade.py", errno.EACCES),
        ("tree", errno.EISDIR),
    ],
)
async def test_reading_a_refused_path_on_iris_is_tool_feedback_with_its_errno(
    tmp_path: Path, path: str, code: int
) -> None:
    if code == errno.EACCES and os.geteuid() == 0:
        pytest.skip("root searches every directory")
    (tmp_path / "present.txt").write_text("present")
    (tmp_path / "locked").mkdir(mode=0o600)
    (tmp_path / "tree" / "nested").mkdir(parents=True)
    ctx = Context(log=EventLog())
    env = iris_env(ctx, tmp_path, LocalController())
    async with opened(ctx, env):
        try:
            result = await ReadFileTool()(ctx, {"path": path})
        finally:
            (tmp_path / "locked").chmod(0o700)
    await ctx.drain()
    assert result == ToolResult(f"File operation failed: [Errno {code}] {os.strerror(code)}: '{path}'")


# Regression: a missing transfer helper and a full host disk reached the model as path errors, and
# then the helper failure was a raw error the driver stored as a bug.
@pytest.mark.parametrize("failure", ["guest helper", "host disk"])
async def test_file_tool_infrastructure_failures_are_not_author_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    (tmp_path / "present.txt").write_text("present")
    ctx = Context(log=EventLog())
    env = iris_env(ctx, tmp_path, LocalController(prelude="wc() { echo 'wc: not found' >&2; return 127; }; "))
    async with opened(ctx, env):
        if failure == "guest helper":
            with pytest.raises(Hold, match="wc: not found"):
                await ReadFileTool()(ctx, {"path": "present.txt"})
        else:

            def full_disk(self: Path, data: bytes) -> int:
                raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), str(self))

            monkeypatch.setattr(Path, "write_bytes", full_disk)
            with pytest.raises(Hold, match="No space left on device"):
                await WriteFileTool()(ctx, {"path": "notes.txt", "content": "valid"})
    await ctx.drain()


# Regression: a ShellSim symlink loop crashed the workpiece with a raw SimulationError.
async def test_reading_a_shellsim_symlink_loop_is_tool_feedback() -> None:
    ctx = Context(log=EventLog())
    requirements = EnvironmentRequirements(command_semantics=CommandSemantics.SHELL_SIMULATOR)
    env = await ShellboxEnvFactory(shellsim_factory(), machine_runtime(backend="shellsim"), 5).create(requirements)
    install_env(ctx, env)
    async with opened(ctx, env):
        try:
            await ctx.emit(RunCommand("ln -s /workspace/loop /workspace/loop", 5))
            result = await ReadFileTool()(ctx, {"path": "/workspace/loop"})
        finally:
            await env.close()
    await ctx.drain()
    loop = os.strerror(errno.ELOOP)
    assert result == ToolResult(f"File operation failed: [Errno {errno.ELOOP}] {loop}: '/workspace/loop'")


async def test_read_file_ranges_are_bytes_before_text_or_base64_decoding() -> None:
    ctx = Context(log=EventLog())
    env = environment(ctx)
    async with opened(ctx, env):
        await ctx.emit(WriteFile(PurePosixPath("unicode"), "éXYZ".encode()))
        text = await ReadFileTool()(ctx, {"path": "unicode", "offset": 2, "limit": 2})
        binary = await ReadFileTool()(ctx, {"path": "unicode", "offset": 0, "limit": 2, "base64": True})
    await ctx.drain()
    assert text == ToolResult("XY")
    assert binary == ToolResult("w6k=")


@pytest.mark.parametrize("status", [401, 403, 404, 422, 429, 503])
async def test_web_rejections_distinguish_author_errors_from_service_errors(status: int) -> None:
    async with httpx.AsyncClient(
        base_url="https://api.parallel.ai/v1",
        transport=httpx.MockTransport(lambda _: httpx.Response(status, text="bad URL")),
    ) as http:
        web = ParallelWeb(http)
        if status in (401, 403):
            with pytest.raises(Hold):
                await web.fetch("bad")
        elif status in (429, 503):
            with pytest.raises(Transient):
                await web.fetch("bad")
        else:
            assert str(status) in await web.fetch("bad")


DRAFT = ("/workspace/task.json", "/workspace/controls.json")


async def test_draft_gate_and_mixed_return_turn_preserve_all_replies_and_replay() -> None:
    sink = BufferSink()
    log = EventLog(sink=sink)
    ctx = Context(log=log)
    env = environment(ctx)
    async with opened(ctx, env):
        await ctx.emit(RunCommand("task", 30))
        await ctx.emit(RunCommand("controls", 30))
        rejected = await SubmitTool(lambda _: ("Missing negative controls",), *DRAFT)(ctx, {})
        assert rejected == ToolResult("Missing negative controls")
        mixed = Completion(
            "",
            (*call("submit", {}, "a").tool_calls, *call("shell", {"command": "task", "timeout": 1000}, "b").tool_calls),
        )
        client = FakeInference((mixed, call("submit", {}, "c")))
        tools = {
            "submit": SubmitTool(accept, *DRAFT),
            "shell": ShellTool(30, 120),
        }
        harness, initial = AgenticHarness(262144, LLMPolicy(), "continue", 20000, replay_arguments), Conversation((), ())
        result = await harness.run(ctx, initial, client, tools)
    await ctx.drain()
    assert any(isinstance(record.event, RunCommand) and record.event.timeout == 120 for record in log.records)
    assert isinstance(result, Submitted) and result.submission.task == arithmetic_task()
    replies = [message for message in client.requests[1].conversation if message.role == "tool"]
    assert [message.tool_call_id for message in replies] == ["a", "b"] and "called alone" in replies[0].content
    assert result.session.messages[-1].tool_calls[0].id == "c" and result.session.messages[-1].reasoning == "plan"
    replay = Context(log=EventLog(replay_from=local_records(sink)))
    replay_env, replay_client = environment(replay), FakeInference(())
    async with opened(replay, replay_env):
        await replay.emit(RunCommand("task", 30))
        await replay.emit(RunCommand("controls", 30))
        assert await SubmitTool(lambda _: ("Missing negative controls",), *DRAFT)(replay, {}) == rejected
        assert await harness.run(replay, initial, replay_client, tools) == result
    await replay.drain()
    assert not replay_client.requests and replay_env.files == env.files
