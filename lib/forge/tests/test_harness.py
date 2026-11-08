# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from functools import partial
from pathlib import Path, PurePosixPath
from tarfile import AbsoluteLinkError, TarInfo

import httpx
import pytest
from rolloutengine.test_support import arithmetic_task, rollout_limits
from taskcompendium.grader import verifyit_package
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.runtime.resources import inline_resource, resource_bytes
from taskforge.llm.policy import LLMPolicy
from verifyit.spec import NumericSpec

from forge.effects.context import Context
from forge.effects.events import Resolved
from forge.effects.issues import Hold, Transient
from forge.effects.log import EventLog, JsonlSink
from forge.harness.loop import AgenticHarness
from forge.harness.protocol import (
    Returned,
    ReturnRequest,
    Stopped,
    StopReason,
    Submitted,
    Tool,
    ToolResult,
)
from forge.harness.tools.adversary import AdversaryReport, ReportTool
from forge.harness.tools.files import ReadFileTool, WriteFileTool
from forge.harness.tools.finish import FinishTool
from forge.harness.tools.grade import GradeTool
from forge.harness.tools.shell import ShellTool
from forge.harness.tools.submit import SubmitTool
from forge.harness.tools.validate import ValidateRequest, ValidateTool
from forge.harness.tools.web import WebFetchTool, WebSearchTool
from forge.lowering import GRADING_MACHINE, LoweringSettings, requirements, rollout_engine
from forge.model.controls import CONTROLS, Candidate, Control, ControlKind, RewardBand
from forge.model.llm import Completion, Message, Role, ToolCall, Usage
from forge.model.workpiece import Conversation, Submission
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
from forge.resources.inference import ContextExceeded, FakeInference
from forge.resources.web import FakeWeb, ParallelWeb, WebFetch, WebSearch, fetch, search

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
    harness = AgenticHarness(262144, LLMPolicy(), "continue", 20000)
    tools = {
        "shell": ShellTool(30, 120),
        "submit": SubmitTool(accept, "/workspace/task.json", "/workspace/controls.json"),
    }
    initial = Conversation((Message(Role.USER, "build"),), ())
    result = await harness.run(ctx, initial, FakeInference(script), tools)
    await ctx.drain()
    assert isinstance(result, Submitted)
    assert result.submission.task.id == arithmetic_task().id
    assert [m.role for m in result.session.messages] == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert result.session.messages[-1].tool_calls[0].id == "3"
    assert result.session.messages[-1].reasoning == "plan"
    replay = Context(log=EventLog(replay_from=await sink.read()))
    replay_env = environment(replay)
    client = FakeInference(())
    replay_result = await harness.run(replay, initial, client, tools)
    await replay.drain()
    assert replay_result == result
    assert client.requests == []
    assert replay_env.files == env.files


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
    environment(ctx)
    result = await AgenticHarness(262144, LLMPolicy(), "continue", 20000).run(
        ctx, Conversation((), ()), client, {"shell": ShellTool(30, 120), "report": ReportTool(arithmetic_task())}
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


async def test_submit_gate_feedback_and_validate_capture_current_draft() -> None:
    ctx = Context(log=EventLog())
    environment(ctx)
    await ctx.emit(RunCommand("task", 30))
    await ctx.emit(RunCommand("controls", 30))
    tool = SubmitTool(
        lambda submission: ("Missing negative controls",), "/workspace/task.json", "/workspace/controls.json"
    )
    result = await tool(ctx, {"task_path": "/workspace/task.json", "controls_path": "/workspace/controls.json"})
    assert isinstance(result, ToolResult) and result.content == "Missing negative controls"
    client = FakeInference((call("validate", {}, "validate-1"),))
    outcome = await AgenticHarness(262144, LLMPolicy(), "continue", 20000).run(
        ctx,
        Conversation((), ()),
        client,
        {"validate": ValidateTool(accept, "/workspace/task.json", "/workspace/controls.json")},
    )
    await ctx.drain()
    assert isinstance(outcome, Returned) and isinstance(outcome.payload, ValidateRequest)
    assert outcome.payload.submission.task == arithmetic_task()
    assert outcome.session.messages[-1].tool_calls[0].id == "validate-1"


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
    environment(ctx)
    settings = LoweringSettings(GRADING_MACHINE, GRADING_MACHINE, rollout_limits())
    grading = GradingResource(rollout_engine({}, command_timeout=1), "fixture")
    ctx.resource(grading, grading.fingerprint)
    ctx.handle(GradeCandidate, partial(grade, notify=ctx.emit), uses=(grading,))
    tool = GradeTool("/workspace/task.json", {}, settings)
    await ctx.emit(RunCommand("task", 30))
    first = await tool(ctx, {"reply": "12"})
    changed = arithmetic_task(grader=verifyit_package(NumericSpec("7", tolerance_abs=0, tolerance_rel=0)).grader)
    await ctx.emit(WriteFile(PurePosixPath("/workspace/task.json"), changed.model_dump_json().encode()))
    second = await tool(ctx, {"reply": "12"})
    assert isinstance(first, ToolResult) and json.loads(first.content)["reward"] == 1
    assert isinstance(second, ToolResult) and json.loads(second.content)["reward"] == 0
    await ctx.emit(WriteFile(PurePosixPath("/workspace/answer"), b"candidate"))
    report = await ReportTool(arithmetic_task(environment_requirements=requirements(image=None)))(
        ctx,
        {
            "findings": [
                {
                    "claim": "false reject",
                    "candidate": {"reply": "12", "files": ["/workspace/answer"]},
                    "expected": {"status": "graded", "reward_max": 0},
                }
            ]
        },
    )
    await ctx.emit(WriteFile(PurePosixPath("/workspace/answer"), b"changed"))
    await ctx.drain()
    assert isinstance(report, ReturnRequest) and isinstance(report.payload, AdversaryReport)
    assert resource_bytes(report.payload.findings[0].candidate.files[0]) == b"candidate"


async def test_web_and_error_results_are_saved_before_the_next_inference() -> None:
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
    env = environment(ctx)
    result = await AgenticHarness(262144, LLMPolicy(), "continue", 80).run(
        ctx,
        Conversation((), ()),
        client,
        {"web_search": WebSearchTool(), "web_fetch": WebFetchTool(), "finish": FinishTool()},
    )
    await ctx.drain()
    assert isinstance(result, Stopped)
    outputs = [message for message in client.requests[1].conversation if message.role == "tool"]
    assert [message.tool_call_id for message in outputs] == ["1", "2", "3"]
    originals = ("s" * 10000, "f" * 10000, "Unknown tool: " + "unknown" * 100)
    assert len(env.files) == 3
    for message, original in zip(outputs, originals, strict=True):
        assert message.content.startswith(original[:80] + "\n[Full output saved to ")
        saved = next(path for path, data in env.files.items() if data.decode() == original)
        assert str(saved) in message.content
        assert f"{len(original)} characters" in message.content
        sliced = await ReadFileTool()(ctx, {"path": str(saved), "offset": 80, "limit": 60})
        assert sliced == ToolResult(original[80:140])
    await ctx.drain()


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError(2, "Missing", "/bad"),
        IsADirectoryError(21, "Directory", "/bad"),
        AbsoluteLinkError(TarInfo("/bad")),
    ],
)
async def test_file_errors_are_feedback_at_the_shared_tool_boundary(failure: Exception) -> None:
    ctx = Context(log=EventLog())

    async def failed_read(event: ReadFile) -> bytes:
        raise failure

    ctx.handle(ReadFile, failed_read)
    result = await ReadFileTool()(ctx, {"path": "/bad"})
    await ctx.drain()
    assert isinstance(result, ToolResult) and "/bad" in result.content


async def test_cached_tokens_are_not_counted_twice_at_context_exit() -> None:
    context_tokens = 262144
    harness = AgenticHarness(context_tokens, LLMPolicy(), "continue", 20000)
    budget = context_tokens
    client = FakeInference(
        tuple(
            Completion(
                "",
                call("shell", {"command": "task"}, str(tokens)).tool_calls,
                usage=Usage(prompt_tokens=tokens, cached_tokens=tokens - 1000),
            )
            for tokens in (budget - 1000, budget)
        )
    )
    ctx = Context(log=EventLog())
    environment(ctx)
    result = await harness.run(ctx, Conversation((), ()), client, {"shell": ShellTool(30, 120)})
    await ctx.drain()
    assert isinstance(result, Stopped) and result.reason == StopReason.CONTEXT_LIMIT
    assert len(client.requests) == 2
    assert not any(message.role == "user" for message in result.session.messages)


async def test_text_and_isolated_empty_stops_keep_nudging_until_explicit_finish() -> None:
    script = (
        Completion("first"),
        Completion(""),
        Completion("second"),
        Completion(""),
        Completion(""),
        call("finish", {"reason": "done"}, "finish"),
    )
    client = FakeInference(script)
    ctx = Context(log=EventLog())
    environment(ctx)
    result = await AgenticHarness(262144, LLMPolicy(), "use tools", 20000).run(
        ctx, Conversation((), ()), client, {"finish": FinishTool()}
    )
    await ctx.drain()
    assert isinstance(result, Stopped) and result.reason == StopReason.FINISHED
    assert len(client.requests) == len(script)
    assert all(request.messages[-1] == Message(Role.USER, "use tools") for request in client.requests[1:])
    assert result.session.messages[-1].tool_calls[0].id == "finish"


async def test_three_consecutive_empty_completions_hold_provider_failure() -> None:
    client = FakeInference((Completion(""),) * 3)
    ctx = Context(log=EventLog())
    with pytest.raises(Hold, match="provider returned 3 empty completions"):
        environment(ctx)
        await AgenticHarness(262144, LLMPolicy(), "use tools", 20000).run(ctx, Conversation((), ()), client, {})
    await ctx.drain()
    assert len(client.requests) == 3


@pytest.mark.parametrize("name", ["grade", "submit", "validate"])
async def test_candidate_files_without_task_machine_return_feedback(name: str) -> None:
    ctx = Context(log=EventLog())
    environment(ctx)
    await ctx.emit(RunCommand("task", 30))
    candidate = Candidate("12", (inline_resource("workspace/answer", b"12"),))
    controls = (Control(candidate, RewardBand(Outcome.GRADED, reward_min=1), ControlKind.POSITIVE),)
    await ctx.emit(WriteFile(PurePosixPath("/workspace/controls.json"), CONTROLS.dump_json(controls)))
    await ctx.emit(WriteFile(PurePosixPath("/workspace/answer"), b"12"))
    settings = LoweringSettings(GRADING_MACHINE, GRADING_MACHINE, rollout_limits())
    tools: dict[str, Tool[object]] = {
        "grade": GradeTool("/workspace/task.json", {}, settings),
        "submit": SubmitTool(accept, "/workspace/task.json", "/workspace/controls.json"),
        "validate": ValidateTool(accept, "/workspace/task.json", "/workspace/controls.json"),
    }
    args: dict[str, dict[str, object]] = {
        "grade": {"reply": "12", "files": ["/workspace/answer"]},
        "submit": {"task_path": "/workspace/task.json", "controls_path": "/workspace/controls.json"},
        "validate": {},
    }
    # No grading executor: malformed author data must never reach the executor backstop.
    result = await tools[name](ctx, args[name])
    await ctx.drain()
    assert isinstance(result, ToolResult) and "Candidate files require a task machine" in result.content


async def test_provider_context_overflow_preserves_the_last_completed_conversation() -> None:
    client = FakeInference(
        (call("shell", {"command": "task"}, "1"), ContextExceeded("maximum context length: requested 383788 tokens"))
    )
    ctx = Context(log=EventLog())
    environment(ctx)
    resumed = Conversation(
        (
            Message(Role.USER, "build"),
            Message(Role.ASSISTANT, "", (ToolCall("prior", "validate", "{}"),)),
            Message(Role.TOOL, "validated", tool_call_id="prior"),
        ),
        (),
    )
    result = await AgenticHarness(262144, LLMPolicy(), "use tools", 20000).run(
        ctx, resumed, client, {"shell": ShellTool(30, 120)}
    )
    await ctx.drain()
    assert isinstance(result, Stopped) and result.reason == StopReason.CONTEXT_LIMIT
    assert result.turns == 3
    assert "requested 383788 tokens" in result.summary
    assert result.session.messages == client.requests[1].conversation
    assert result.session.messages[-1].role == "tool"
    assert result.session.messages[-1].tool_call_id == "1"


@pytest.mark.parametrize("completion_tokens,output_chars", [(10, 20), (0, 40)])
async def test_next_prompt_counts_reply_tokens_and_appended_tool_text(completion_tokens: int, output_chars: int) -> None:
    ctx = Context(log=EventLog())
    web = FakeWeb({"query": "x" * 1000}, {})
    ctx.resource(web, web.fingerprint)
    ctx.handle(WebSearch, partial(search, notify=ctx.emit), uses=(web,))
    client = FakeInference(
        (
            Completion(
                "",
                call("web_search", {"query": "query"}, "1").tool_calls,
                usage=Usage(prompt_tokens=90, completion_tokens=completion_tokens, cached_tokens=80),
            ),
        )
    )
    environment(ctx)
    result = await AgenticHarness(100, LLMPolicy(), "use tools", output_chars).run(
        ctx, Conversation((), ()), client, {"web_search": WebSearchTool()}
    )
    await ctx.drain()
    assert isinstance(result, Stopped) and result.reason == StopReason.CONTEXT_LIMIT
    assert len(client.requests) == 1
    assert result.session.messages[-1].content.startswith("x" * output_chars + "\n[Full output saved to ")


async def test_adversary_report_requires_claim_evidence_and_uses_bound_task() -> None:
    ctx = Context(log=EventLog())
    environment(ctx)
    task = arithmetic_task()
    tool = ReportTool(task)
    commentary = await tool(ctx, {"findings": ["No grading defect found"]})
    assert isinstance(commentary, ToolResult)
    claim = {
        "claim": "wrong answer accepted",
        "candidate": {"reply": "9"},
        "expected": {"status": "graded", "reward_min": 1},
    }
    report = await tool(ctx, {"findings": [claim]})
    assert isinstance(report, ReturnRequest) and isinstance(report.payload, AdversaryReport)
    finding = report.payload.findings[0]
    assert finding.claim == "wrong answer accepted" and finding.candidate.reply == "9"
    assert finding.expected.met_by(GradeResult(Outcome.GRADED, 1))
    assert not finding.expected.met_by(GradeResult(Outcome.GRADED, 0))
    # The adversary can edit its scratch task file, but cannot change the bound task's machine requirement.
    edited = arithmetic_task(environment_requirements=requirements(image=None))
    await ctx.emit(WriteFile(PurePosixPath("/workspace/task.json"), edited.model_dump_json().encode()))
    await ctx.emit(WriteFile(PurePosixPath("/workspace/answer"), b"9"))
    rejected = await tool(ctx, {"findings": [{**claim, "candidate": {"reply": "9", "files": ["/workspace/answer"]}}]})
    empty = await tool(ctx, {"findings": []})
    await ctx.drain()
    assert isinstance(rejected, ToolResult) and "Candidate files require a task machine" in rejected.content
    assert isinstance(empty, ReturnRequest) and isinstance(empty.payload, AdversaryReport)
    assert empty.payload.findings == ()


async def test_malformed_grader_checks_return_feedback_and_corrected_checks_are_reported() -> None:
    ctx = Context(log=EventLog())
    environment(ctx)
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


async def test_large_shell_output_is_saved_and_replayed_without_inference(tmp_path: Path) -> None:
    sink = JsonlSink(str(tmp_path / "outputs.jsonl"))
    ctx = Context(log=EventLog(sink=sink))
    env = environment(ctx)
    harness = AgenticHarness(262144, LLMPolicy(), "continue", 8)
    initial = Conversation((), ())
    tools = {"shell": ShellTool(30, 120), "finish": FinishTool()}
    client = FakeInference(
        (call("shell", {"command": "task"}, "../../escape"), call("finish", {"reason": "done"}, "done"))
    )
    result = await harness.run(ctx, initial, client, tools)
    await ctx.drain()
    saved = [(path, data) for path, data in env.files.items() if path.parent == PurePosixPath(".tool-output")]
    assert len(saved) == 1
    path, data = saved[0]
    assert json.loads(data)["stdout"] == "task ready"
    assert str(path) in client.requests[1].conversation[-1].content
    assert ".." not in path.parts
    replay = Context(log=EventLog(replay_from=await sink.read()))
    replay_env = environment(replay)
    replay_client = FakeInference(())
    assert await harness.run(replay, initial, replay_client, tools) == result
    await replay.drain()
    assert not replay_client.requests
    assert replay_env.files == env.files


@pytest.mark.parametrize(
    "failure",
    [
        IsADirectoryError("directory"),
        FileNotFoundError("missing"),
        NotADirectoryError("parent"),
    ],
)
async def test_write_path_errors_return_feedback(failure: OSError) -> None:
    ctx = Context(log=EventLog())
    environment(ctx)

    async def fail_write(event: WriteFile) -> None:
        raise type(failure)(1, str(failure), str(event.path))

    ctx.handle(WriteFile, fail_write)
    result = await WriteFileTool()(ctx, {"path": "/bad/path", "content": "data"})
    await ctx.drain()
    assert isinstance(result, ToolResult) and "/bad/path" in result.content


async def test_read_file_ranges_are_bytes_before_text_or_base64_decoding() -> None:
    ctx = Context(log=EventLog())
    environment(ctx)
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


async def test_mixed_validate_turn_answers_every_call_before_retry() -> None:
    log = EventLog()
    ctx = Context(log=log)
    environment(ctx)
    await ctx.emit(RunCommand("task", 30))
    await ctx.emit(RunCommand("controls", 30))
    mixed = Completion(
        "", (*call("validate", {}, "a").tool_calls, *call("shell", {"command": "task", "timeout": 1000}, "b").tool_calls)
    )
    client = FakeInference((mixed, call("validate", {}, "c")))
    tools = {
        "validate": ValidateTool(accept, "/workspace/task.json", "/workspace/controls.json"),
        "shell": ShellTool(30, 120),
    }
    result = await AgenticHarness(262144, LLMPolicy(), "continue", 20000).run(ctx, Conversation((), ()), client, tools)
    await ctx.drain()
    assert any(isinstance(record.event, RunCommand) and record.event.timeout == 120 for record in log.records)
    assert isinstance(result, Returned)
    replies = [message for message in client.requests[1].conversation if message.role == "tool"]
    assert [message.tool_call_id for message in replies] == ["a", "b"]
    assert "called alone" in replies[0].content
    assert result.session.messages[-1].tool_calls[0].id == "c"
