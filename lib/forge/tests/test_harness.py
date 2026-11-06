# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from functools import partial
from pathlib import Path, PurePosixPath

import httpx
import pytest
from rolloutengine.test_support import arithmetic_task, rollout_limits
from taskcompendium.grader import verifyit_package
from taskcompendium.runtime.resources import resource_bytes
from verifyit.spec import NumericSpec

from forge.effects.context import Context
from forge.effects.issues import Hold
from forge.effects.log import EventLog, JsonlSink
from forge.harness.classify import host_failure
from forge.harness.events import ToolReturned
from forge.harness.loop import AgenticHarness, Limits
from forge.harness.protocol import Returned, ReturnRequest, Stopped, StopReason, Submitted, Tool, ToolResult
from forge.harness.tools.adversary import AdversaryReport, ReportTool
from forge.harness.tools.files import ReadFileTool, WriteFileTool
from forge.harness.tools.finish import FinishTool
from forge.harness.tools.grade import GradeTool
from forge.harness.tools.shell import ShellTool
from forge.harness.tools.submit import SubmitTool
from forge.harness.tools.validate import ValidateRequest, ValidateTool
from forge.harness.tools.web import WebFetchTool, WebSearchTool
from forge.harness.web import FakeWeb, ParallelWeb, WebFetch, WebSearch, fetch, search
from forge.lowering import GRADING_MACHINE, LoweringSettings, rollout_engine
from forge.model.llm import Completion, Message, Sampling, ToolCall, Usage
from forge.model.serialization import TASK_SPEC
from forge.model.workpiece import Conversation, Submission
from forge.policy.defaults import DEFAULT_POLICY
from forge.resources.env import (
    CommandResult,
    FakeCommand,
    FakeEnv,
    ReadFile,
    RunCommand,
    WriteFile,
    read_file,
    run_command,
    write_file,
)
from forge.resources.grading import GradeCandidate, GradingResource, grade
from forge.resources.inference import FakeInference
from forge.resources.sandbox.factories import MachineHost
from forge.wiring import build_config

pytestmark = pytest.mark.asyncio


def accept(submission: Submission) -> tuple[str, ...]:
    return ()


def environment(ctx: Context) -> FakeEnv:
    task = TASK_SPEC.dump_json(arithmetic_task())
    scripts = {
        "task": FakeCommand(
            CommandResult(0, "task ready", "", 0.1, False), {PurePosixPath("/workspace/task.json"): task}
        ),
        "controls": FakeCommand(
            CommandResult(0, "controls ready", "", 0.1, False), {PurePosixPath("/workspace/controls.json"): b"[]"}
        ),
    }
    env = FakeEnv("image", scripts, ctx.emit)
    ctx.resource(env, env.fingerprint)
    ctx.handle(RunCommand, partial(run_command, notify=ctx.emit), uses=(env,))
    ctx.handle(ReadFile, partial(read_file, notify=ctx.emit), uses=(env,))
    ctx.handle(WriteFile, partial(write_file, notify=ctx.emit), uses=(env,))
    return env


def call(name: str, args: object, identity: str) -> Completion:
    return Completion("", (ToolCall(identity, name, json.dumps(args)),), reasoning="plan")


async def test_shell_submit_session_and_persisted_replay_spend_no_inference(tmp_path: Path) -> None:
    script = (
        call("shell", {"command": "task"}, "1"),
        call("shell", {"command": "controls"}, "2"),
        call(
            "submit",
            {"task_path": "/workspace/task.json", "controls_path": "/workspace/controls.json", "notes": "done"},
            "3",
        ),
    )
    sink = JsonlSink(str(tmp_path / "session.jsonl"))
    ctx = Context(log=EventLog(sink=sink))
    env = environment(ctx)
    harness = AgenticHarness(Limits(5, None), Sampling(), "continue", 20000)
    tools = {"shell": ShellTool(), "submit": SubmitTool(accept)}
    initial = Conversation((Message("user", "build"),), ())
    result = await harness.run(ctx, initial, env, FakeInference(script), tools)
    await ctx.drain()
    assert isinstance(result, Submitted)
    assert result.submission.task.id == arithmetic_task().id
    assert [m.role for m in result.session.messages] == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert result.session.messages[-1].tool_calls[0].id == "3"
    assert result.session.messages[-1].reasoning == "plan"
    replay = Context(log=EventLog(replay_from=await sink.read()))
    replay_env = environment(replay)
    client = FakeInference(())
    replay_result = await harness.run(replay, initial, replay_env, client, tools)
    await replay.drain()
    assert replay_result == result
    assert client.requests == []
    assert replay_env.files == env.files


@pytest.mark.parametrize(
    "limits,script,reason,turns",
    [
        (Limits(None, None), (Completion("first"), Completion("second")), StopReason.FINISHED, 2),
        (Limits(1, None), (Completion("first"),), StopReason.TURN_LIMIT, 1),
        (
            Limits(None, 10),
            (Completion("first", usage=Usage(prompt_tokens=10, cached_tokens=4)),),
            StopReason.CONTEXT_LIMIT,
            1,
        ),
    ],
)
async def test_nudge_and_limits_preserve_completed_conversation(
    limits: Limits, script: tuple[Completion, ...], reason: StopReason, turns: int
) -> None:
    ctx = Context(log=EventLog())
    client = FakeInference(script)
    result = await AgenticHarness(limits, Sampling(), "use tools", 20000).run(
        ctx, Conversation((), ()), environment(ctx), client, {}
    )
    await ctx.drain()
    assert isinstance(result, Stopped) and result.reason == reason
    assert len(client.requests) == turns
    assert Message("user", "use tools") in result.session.messages
    if turns == 2:
        assert client.requests[1].messages[-1] == Message("user", "use tools")


async def test_argument_and_unknown_tool_errors_return_to_model_before_finish() -> None:
    script = (
        Completion("", (ToolCall("1", "shell", "{"), ToolCall("2", "missing", "{}"), ToolCall("3", "shell", "{}"))),
        call("finish", {"reason": "done"}, "4"),
    )
    ctx = Context(log=EventLog())
    client = FakeInference(script)
    result = await AgenticHarness(Limits(3, None), Sampling(), "continue", 20000).run(
        ctx, Conversation((), ()), environment(ctx), client, {"shell": ShellTool(), "finish": FinishTool()}
    )
    await ctx.drain()
    assert isinstance(result, Stopped) and result.reason == StopReason.FINISHED
    assert [m.tool_call_id for m in client.requests[1].messages if m.role == "tool"] == ["1", "2", "3"]
    assert result.session.messages[-1].tool_calls[0].id == "4"


async def test_classifier_raises_through_scope_check_but_ordinary_exit_is_feedback() -> None:
    ctx = Context(log=EventLog())
    async with ctx.scope("ordinary"):
        ctx.observe(ToolReturned, host_failure)
        await ctx.emit(ToolReturned("1", "shell", '{"exit_code": 1, "stderr": "AssertionError"}'))
        await ctx.check(Hold)
    with pytest.raises(ExceptionGroup) as raised:
        async with ctx.scope("lost"):
            ctx.observe(ToolReturned, host_failure)
            await ctx.emit(ToolReturned("2", "shell", "No space left on device"))
            await ctx.check(Hold)
    await ctx.drain()
    assert isinstance(raised.value.exceptions[0], Hold)


async def test_submit_gate_feedback_and_validate_capture_current_draft() -> None:
    ctx = Context(log=EventLog())
    env = environment(ctx)
    await ctx.emit(RunCommand("task", 30))
    await ctx.emit(RunCommand("controls", 30))
    tool = SubmitTool(lambda submission: ("Missing negative controls",))
    result = await tool(ctx, {"task_path": "/workspace/task.json", "controls_path": "/workspace/controls.json"}, env)
    assert isinstance(result, ToolResult) and result.content == "Missing negative controls"
    client = FakeInference((call("validate", {}, "validate-1"),))
    outcome = await AgenticHarness(Limits(2, None), Sampling(), "continue", 20000).run(
        ctx, Conversation((), ()), env, client, {"validate": ValidateTool(accept)}
    )
    await ctx.drain()
    assert isinstance(outcome, Returned) and isinstance(outcome.payload, ValidateRequest)
    assert outcome.payload.submission.task == arithmetic_task()
    assert outcome.session.messages[-1].tool_calls[0].id == "validate-1"


async def test_binary_file_tools_round_trip_bytes_and_utf8_replacement() -> None:
    ctx = Context(log=EventLog())
    env = environment(ctx)
    await WriteFileTool()(ctx, {"path": "/binfile", "content": "/wA=", "base64": True}, env)
    encoded = await ReadFileTool()(ctx, {"path": "/binfile", "base64": True}, env)
    text = await ReadFileTool()(ctx, {"path": "/binfile"}, env)
    await ctx.drain()
    assert isinstance(encoded, ToolResult) and encoded.content == "/wA="
    assert isinstance(text, ToolResult) and text.content == "\ufffd\x00"


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
        ctx.resource(web, web.fingerprint)
        ctx.handle(WebSearch, partial(search, notify=ctx.emit), uses=(web,))
        ctx.handle(WebFetch, partial(fetch, notify=ctx.emit), uses=(web,))
        assert json.loads(await ctx.emit(WebSearch("query"))) == {"results": ["evidence"]}
        await ctx.emit(WebFetch("https://example.org"))
        await ctx.drain()
        replay_web = FakeWeb({}, {}, identity=web.fingerprint().digest)
        replay = Context(log=EventLog(replay_from=log.records))
        replay.resource(replay_web, replay_web.fingerprint)
        replay.handle(WebSearch, partial(search, notify=replay.emit), uses=(replay_web,))
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
    settings = LoweringSettings(MachineHost.LAPTOP, GRADING_MACHINE, GRADING_MACHINE, rollout_limits())
    grading = GradingResource(rollout_engine({}, command_timeout=1), "fixture")
    ctx.resource(grading, grading.fingerprint)
    ctx.handle(GradeCandidate, partial(grade, notify=ctx.emit), uses=(grading,))
    tool = GradeTool("/workspace/task.json", {}, settings)
    await ctx.emit(RunCommand("task", 30))
    first = await tool(ctx, {"reply": "12"}, env)
    changed = arithmetic_task(grader=verifyit_package(NumericSpec("7", tolerance_abs=0, tolerance_rel=0)).grader)
    await ctx.emit(WriteFile(PurePosixPath("/workspace/task.json"), TASK_SPEC.dump_json(changed)))
    second = await tool(ctx, {"reply": "12"}, env)
    assert isinstance(first, ToolResult) and json.loads(first.content)["reward"] == 1
    assert isinstance(second, ToolResult) and json.loads(second.content)["reward"] == 0
    await ctx.emit(WriteFile(PurePosixPath("/workspace/answer"), b"candidate"))
    report = await ReportTool()(
        ctx, {"findings": ["false reject"], "candidates": [{"reply": "12", "files": ["/workspace/answer"]}]}, env
    )
    await ctx.emit(WriteFile(PurePosixPath("/workspace/answer"), b"changed"))
    await ctx.drain()
    assert isinstance(report, ReturnRequest) and isinstance(report.payload, AdversaryReport)
    assert resource_bytes(report.payload.candidates[0].files[0]) == b"candidate"


async def test_web_and_error_results_are_capped_before_the_next_inference() -> None:
    log = EventLog()
    ctx = Context(log=log)
    web = FakeWeb({"query": "s" * 10000}, {"https://example.org": "f" * 10000})
    ctx.resource(web, web.fingerprint)
    ctx.handle(WebSearch, partial(search, notify=ctx.emit), uses=(web,))
    ctx.handle(WebFetch, partial(fetch, notify=ctx.emit), uses=(web,))
    calls = (
        call("web_search", {"query": "query"}, "1").tool_calls[0],
        call("web_fetch", {"url": "https://example.org"}, "2").tool_calls[0],
        call("unknown" * 100, {}, "3").tool_calls[0],
    )
    client = FakeInference((Completion("", calls), call("finish", {"reason": "done"}, "4")))
    result = await AgenticHarness(Limits(3, None), Sampling(), "continue", 80).run(
        ctx,
        Conversation((), ()),
        environment(ctx),
        client,
        {"web_search": WebSearchTool(), "web_fetch": WebFetchTool(), "finish": FinishTool()},
    )
    await ctx.drain()
    assert isinstance(result, Stopped)
    outputs = [message for message in client.requests[1].messages if message.role == "tool"]
    assert [message.tool_call_id for message in outputs] == ["1", "2", "3"]
    assert all(len(message.content) == 80 and message.content.endswith("[output truncated]") for message in outputs)
    assert all(len(record.event.content) <= 80 for record in log.records if isinstance(record.event, ToolReturned))


@pytest.mark.parametrize("kind", ["read", "grade-task", "grade-candidate", "submit", "validate"])
async def test_missing_paths_are_tool_feedback(kind: str) -> None:
    ctx = Context(log=EventLog())
    env = environment(ctx)
    await ctx.emit(RunCommand("task", 30))

    async def read_missing(event: ReadFile, resource: FakeEnv) -> bytes:
        try:
            return await resource.read(event.path)
        except KeyError as error:
            raise FileNotFoundError(str(event.path)) from error

    ctx.handle(ReadFile, read_missing, uses=(env,))
    settings = LoweringSettings(MachineHost.LAPTOP, GRADING_MACHINE, GRADING_MACHINE, rollout_limits())
    missing = "cand/release_notes.py" if kind == "grade-candidate" else "/workspace/controls.json"
    tools: dict[str, Tool[object]] = {
        "read": ReadFileTool(),
        "grade-task": GradeTool(missing, {}, settings),
        "grade-candidate": GradeTool("/workspace/task.json", {}, settings),
        "submit": SubmitTool(accept),
        "validate": ValidateTool(accept),
    }
    arguments: dict[str, dict[str, object]] = {
        "read": {"path": missing},
        "grade-task": {"reply": "12"},
        "grade-candidate": {"reply": "12", "files": [missing]},
        "submit": {"task_path": "/workspace/task.json", "controls_path": missing},
        "validate": {},
    }
    result = await tools[kind](ctx, arguments[kind], env)
    await ctx.drain()
    assert isinstance(result, ToolResult) and missing in result.content


@pytest.mark.parametrize("output_tokens", [131072, 8192])
async def test_model_window_and_output_allowance_determine_context_exit(output_tokens: int) -> None:
    policy = DEFAULT_POLICY.overlaid({"sampling.max_tokens": output_tokens})
    harness = build_config(policy, "S", {}).author
    budget = 262144 - output_tokens
    client = FakeInference(
        tuple(
            Completion(
                "",
                call("shell", {"command": "task"}, str(tokens)).tool_calls,
                usage=Usage(prompt_tokens=tokens, cached_tokens=tokens - 1000),
            )
            for tokens in (budget - 1, budget)
        )
    )
    ctx = Context(log=EventLog())
    result = await harness.run(ctx, Conversation((), ()), environment(ctx), client, {"shell": ShellTool()})
    await ctx.drain()
    assert isinstance(result, Stopped) and result.reason == StopReason.CONTEXT_LIMIT
    assert len(client.requests) == 2
    assert not any(message.role == "user" for message in result.session.messages)
