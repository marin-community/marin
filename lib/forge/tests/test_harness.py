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
from forge.model.llm import Completion, FinishReason, Message, Role, ToolCall, Usage
from forge.model.proposal import TASK_PROPOSAL, TaskProposal, Verification
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


pytest_plugins = ("lib.forge.tests.test_proposal",)


def accept(submission: Submission, verification: Verification) -> tuple[str, ...]:
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
    web = FakeWeb({"query": "s" * 10000}, {"https://example.org": "f" * 10000})
    ctx.resource(web, web.fingerprint)
    ctx.handle(WebSearch, partial(search, notify=ctx.emit), uses=(web,))
    ctx.handle(WebFetch, partial(fetch, notify=ctx.emit), uses=(web,))
    return env


def call(name: str, args: object, identity: str) -> Completion:
    return Completion("", (ToolCall(identity, name, json.dumps(args)),), reasoning="plan")


async def test_malformed_report_and_schema_errors_recover_with_replayable_history() -> None:
    # Pilot j: S/d05.contaminant_transport_fate/09 and T/d15.biomechanics.prosthetic_orthotic_design/10.
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
    tmp_path: Path, name: str, args: dict[str, object], original: str | None
) -> None:
    sink = JsonlSink(str(tmp_path / "outputs.jsonl"))
    ctx = Context(log=EventLog(sink=sink))
    env = environment(ctx)
    tools = {
        "shell": ShellTool(30, 120),
        "web_search": WebSearchTool(),
        "web_fetch": WebFetchTool(),
        "finish": FinishTool(),
    }
    harness, initial = AgenticHarness(1000, LLMPolicy(), "continue", 8), Conversation((), ())
    client = FakeInference((call(name, args, "../../escape"), call("finish", {"reason": "done"}, "done")))
    result = await harness.run(ctx, initial, client, tools)
    await ctx.drain()
    [(path, data)] = [(path, data) for path, data in env.files.items() if path.parent == PurePosixPath(".tool-output")]
    if original is None:
        assert json.loads(data)["stdout"] == "task ready"
        original = data.decode()
    assert data.decode() == original and ".." not in path.parts
    content = client.requests[1].conversation[-1].content
    assert content.startswith(original[:8] + "\n[Full output saved to ") and str(path) in content
    assert await ReadFileTool()(ctx, {"path": str(path), "offset": 8, "limit": 20}) == ToolResult(original[8:28])
    await ctx.drain()
    replay = Context(log=EventLog(replay_from=await sink.read()))
    replay_env, replay_client = environment(replay), FakeInference(())
    assert await harness.run(replay, initial, replay_client, tools) == result
    await replay.drain()
    assert not replay_client.requests and replay_env.files == env.files


@pytest.mark.parametrize("write", [False, True])
@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError(2, "missing", "/bad"),
        IsADirectoryError(21, "directory", "/bad"),
        NotADirectoryError(20, "parent", "/bad"),
        AbsoluteLinkError(TarInfo("/bad")),
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


@pytest.mark.parametrize("fails", [False, True])
async def test_empty_turns_nudge_until_finish_or_three_consecutive_failures(fails: bool) -> None:
    script = (Completion("first"), Completion(""), Completion("second"), Completion(""), Completion(""))
    client = FakeInference((*script, Completion("") if fails else call("finish", {"reason": "done"}, "finish")))
    ctx = Context(log=EventLog())
    harness = AgenticHarness(262144, LLMPolicy(), "use tools", 20000)
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
async def test_reasoning_only_length_retries_unchanged_conversation_and_replays(tmp_path: Path, recovers: bool) -> None:
    # Pilot o: S/d34.insights.quant_research/06 exhausted the endpoint in reasoning only.
    failed = Completion(
        "",
        finish_reason=FinishReason.LENGTH,
        usage=Usage(prompt_tokens=48461, completion_tokens=565939, reasoning_tokens=565939),
        reasoning="repeated reasoning",
    )
    client = FakeInference((failed, call("finish", {"reason": "done"}, "finish") if recovers else failed))
    sink = JsonlSink(str(tmp_path / "session.jsonl"))
    ctx = Context(log=EventLog(sink=sink))
    initial = Conversation((Message(Role.USER, "build"),), ())
    harness = AgenticHarness(262144, LLMPolicy(), "use tools", 20000)
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
    records = await sink.read()
    assert any(isinstance(record.event, Resolved) and record.event.value == failed for record in records)
    replay = Context(log=EventLog(replay_from=records))
    replay_client = FakeInference(())
    assert await harness.run(replay, initial, replay_client, tools) == result
    await replay.drain()
    assert replay_client.requests == []


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


@pytest.mark.parametrize("overflow", ["provider", "prompt", "reply", "tool"])
async def test_context_exit_counts_retained_turns_and_only_appended_tokens(overflow: str) -> None:
    # Pilots i/j: T/d08.design.drawings_tolerances/04 and T/d17.procedures.preoperative/25.
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
    environment(ctx)
    client = FakeInference((first, second))
    result = await AgenticHarness(262144, LLMPolicy(), "continue", 20000).run(
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


@pytest.mark.parametrize("name", ["submit", "validate"])
async def test_draft_gate_and_mixed_return_turn_preserve_all_replies_and_replay(
    tmp_path: Path, name: str, proposal: TaskProposal
) -> None:
    sink = JsonlSink(str(tmp_path / "draft.jsonl"))
    log = EventLog(sink=sink)
    ctx = Context(log=log)
    env = environment(ctx)
    await ctx.emit(RunCommand("task", 30))
    await ctx.emit(RunCommand("controls", 30))
    proposal_file = WriteFile(PurePosixPath("/workspace/proposal.json"), TASK_PROPOSAL.dump_json(proposal))
    await ctx.emit(proposal_file)
    tool_type = SubmitTool if name == "submit" else ValidateTool
    rejected = await tool_type(
        lambda *_: ("Missing negative controls",), "/workspace/task.json", "/workspace/controls.json"
    )(ctx, {})
    assert rejected == ToolResult("Missing negative controls")
    mixed = Completion(
        "", (*call(name, {}, "a").tool_calls, *call("shell", {"command": "task", "timeout": 1000}, "b").tool_calls)
    )
    client = FakeInference((mixed, call(name, {}, "c")))
    tools = {name: tool_type(accept, "/workspace/task.json", "/workspace/controls.json"), "shell": ShellTool(30, 120)}
    harness, initial = AgenticHarness(262144, LLMPolicy(), "continue", 20000), Conversation((), ())
    result = await harness.run(ctx, initial, client, tools)
    await ctx.drain()
    assert any(isinstance(record.event, RunCommand) and record.event.timeout == 120 for record in log.records)
    if name == "submit":
        assert isinstance(result, Submitted) and result.submission.task == arithmetic_task()
    else:
        assert isinstance(result, Returned) and isinstance(result.payload, ValidateRequest)
        assert result.payload.submission.task == arithmetic_task()
    replies = [message for message in client.requests[1].conversation if message.role == "tool"]
    assert [message.tool_call_id for message in replies] == ["a", "b"] and "called alone" in replies[0].content
    assert result.session.messages[-1].tool_calls[0].id == "c" and result.session.messages[-1].reasoning == "plan"
    replay = Context(log=EventLog(replay_from=await sink.read()))
    replay_env, replay_client = environment(replay), FakeInference(())
    await replay.emit(RunCommand("task", 30))
    await replay.emit(RunCommand("controls", 30))
    await replay.emit(proposal_file)
    assert await harness.run(replay, initial, replay_client, tools) == result
    await replay.drain()
    assert not replay_client.requests and replay_env.files == env.files
