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
from taskcompendium.models import StdoutReward
from taskcompendium.runtime.resources import inline_resource, resource_bytes
from verifyit.spec import NumericSpec

from forge.effects.context import Context
from forge.effects.issues import Hold
from forge.effects.log import EventLog, JsonlSink
from forge.harness.classify import host_failure
from forge.harness.events import ToolReturned
from forge.harness.loop import AgenticHarness
from forge.harness.protocol import (
    Returned,
    ReturnRequest,
    Stopped,
    StopReason,
    SubmitRequest,
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
from forge.harness.web import FakeWeb, ParallelWeb, WebFetch, WebSearch, fetch, search
from forge.lowering import GRADING_MACHINE, LoweringSettings, requirements, rollout_engine, script_grader
from forge.model.controls import Candidate, Control, ControlKind, RewardBand
from forge.model.llm import Completion, Message, Sampling, ToolCall, Usage
from forge.model.serialization import CONTROLS, TASK_SPEC
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
from forge.resources.inference import ContextExceeded, FakeInference
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
    harness = AgenticHarness(262144, Sampling(), "continue", 20000)
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


async def test_argument_and_unknown_tool_errors_return_to_model_before_finish() -> None:
    script = (
        Completion("", (ToolCall("1", "shell", "{"), ToolCall("2", "missing", "{}"), ToolCall("3", "shell", "{}"))),
        call("finish", {"reason": "done"}, "4"),
    )
    ctx = Context(log=EventLog())
    client = FakeInference(script)
    result = await AgenticHarness(262144, Sampling(), "continue", 20000).run(
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
    outcome = await AgenticHarness(262144, Sampling(), "continue", 20000).run(
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
        env,
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
    result = await AgenticHarness(262144, Sampling(), "continue", 80).run(
        ctx,
        Conversation((), ()),
        env,
        client,
        {"web_search": WebSearchTool(), "web_fetch": WebFetchTool(), "finish": FinishTool()},
    )
    await ctx.drain()
    assert isinstance(result, Stopped)
    outputs = [message for message in client.requests[1].messages if message.role == "tool"]
    assert [message.tool_call_id for message in outputs] == ["1", "2", "3"]
    originals = ("s" * 10000, "f" * 10000, "Unknown tool: " + "unknown" * 100)
    assert len(env.files) == 3
    for message, original in zip(outputs, originals, strict=True):
        assert message.content.startswith(original[:80] + "\n[Full output saved to ")
        saved = next(path for path, data in env.files.items() if data.decode() == original)
        assert str(saved) in message.content
        assert f"{len(original)} characters" in message.content
        sliced = await ReadFileTool()(ctx, {"path": str(saved), "offset": 80, "limit": 60}, env)
        assert sliced == ToolResult(original[80:140])
    await ctx.drain()


@pytest.mark.parametrize("failure", ["missing", "directory", "archive"])
@pytest.mark.parametrize("kind", ["read", "grade-task", "grade-candidate", "submit", "validate", "report"])
async def test_invalid_read_paths_are_tool_feedback(kind: str, failure: str) -> None:
    ctx = Context(log=EventLog())
    env = environment(ctx)
    await ctx.emit(RunCommand("task", 30))

    async def read_missing(event: ReadFile, resource: FakeEnv) -> bytes:
        try:
            return await resource.read(event.path)
        except KeyError as error:
            if failure == "directory":
                raise IsADirectoryError(str(event.path)) from error
            if failure == "archive":
                info = TarInfo("./dev/fd")
                info.linkname = "/proc/self/fd"
                raise AbsoluteLinkError(info) from error
            raise FileNotFoundError(str(event.path)) from error

    ctx.handle(ReadFile, read_missing, uses=(env,))
    settings = LoweringSettings(MachineHost.LAPTOP, GRADING_MACHINE, GRADING_MACHINE, rollout_limits())
    missing = "cand/release_notes.py" if kind in {"grade-candidate", "report"} else "/workspace/controls.json"
    tools: dict[str, Tool[object]] = {
        "read": ReadFileTool(),
        "report": ReportTool(arithmetic_task()),
        "grade-task": GradeTool(missing, {}, settings),
        "grade-candidate": GradeTool("/workspace/task.json", {}, settings),
        "submit": SubmitTool(accept),
        "validate": ValidateTool(accept),
    }
    arguments: dict[str, dict[str, object]] = {
        "read": {"path": missing},
        "report": {
            "findings": [
                {
                    "claim": "wrong answer",
                    "candidate": {"reply": "9", "files": [missing]},
                    "expected": {"status": "graded", "reward_min": 1.0, "reward_max": 1.0},
                }
            ]
        },
        "grade-task": {"reply": "12"},
        "grade-candidate": {"reply": "12", "files": [missing]},
        "submit": {"task_path": "/workspace/task.json", "controls_path": missing},
        "validate": {},
    }
    result = await tools[kind](ctx, arguments[kind], env)
    await ctx.drain()
    assert isinstance(result, ToolResult) and missing in result.content
    if failure == "directory":
        assert "is a directory" in result.content
    if failure == "archive":
        assert "archive extraction failed" in result.content


@pytest.mark.parametrize("context_tokens", [262144, 131072])
async def test_model_window_determines_context_exit(context_tokens: int) -> None:
    policy = DEFAULT_POLICY.overlaid({"provider.context_tokens": context_tokens})
    harness = build_config(policy, "S", {}).author
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
    result = await harness.run(ctx, Conversation((), ()), environment(ctx), client, {"shell": ShellTool()})
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
    result = await AgenticHarness(262144, Sampling(), "use tools", 20000).run(
        ctx, Conversation((), ()), environment(ctx), client, {"finish": FinishTool()}
    )
    await ctx.drain()
    assert isinstance(result, Stopped) and result.reason == StopReason.FINISHED
    assert len(client.requests) == len(script)
    assert all(request.messages[-1] == Message("user", "use tools") for request in client.requests[1:])
    assert result.session.messages[-1].tool_calls[0].id == "finish"


async def test_three_consecutive_empty_completions_hold_provider_failure() -> None:
    client = FakeInference((Completion(""),) * 3)
    ctx = Context(log=EventLog())
    with pytest.raises(Hold, match="provider returned 3 empty completions"):
        await AgenticHarness(262144, Sampling(), "use tools", 20000).run(
            ctx, Conversation((), ()), environment(ctx), client, {}
        )
    await ctx.drain()
    assert len(client.requests) == 3


async def test_invalid_grader_image_is_feedback_before_any_grading_effect() -> None:
    ctx = Context(log=EventLog())
    env = environment(ctx)
    image = "registry.example.com/python@sha256:" + "a" * 64
    task = arithmetic_task(
        grader=script_grader(
            ("sh", "-c", "echo 1"), StdoutReward(), environment=requirements(image=image), answer_path=None, timeout=1
        ).grader
    )
    await ctx.emit(WriteFile(PurePosixPath("/workspace/task.json"), TASK_SPEC.dump_json(task)))
    settings = LoweringSettings(MachineHost.IRIS, GRADING_MACHINE, GRADING_MACHINE, rollout_limits())
    # No GradeCandidate executor: the invalid image must be rejected before scheduling grading.
    result = await GradeTool("/workspace/task.json", {}, settings)(ctx, {"reply": "12"}, env)
    await ctx.drain()
    assert isinstance(result, ToolResult) and "grader image must be" in result.content and image in result.content


@pytest.mark.parametrize("name", ["grade", "submit", "validate"])
async def test_candidate_files_without_task_machine_return_feedback(name: str) -> None:
    ctx = Context(log=EventLog())
    env = environment(ctx)
    await ctx.emit(RunCommand("task", 30))
    candidate = Candidate("12", (inline_resource("workspace/answer", b"12"),))
    controls = (Control(candidate, RewardBand(Outcome.GRADED, reward_min=1), ControlKind.POSITIVE),)
    await ctx.emit(WriteFile(PurePosixPath("/workspace/controls.json"), CONTROLS.dump_json(controls)))
    await ctx.emit(WriteFile(PurePosixPath("/workspace/answer"), b"12"))
    settings = LoweringSettings(MachineHost.IRIS, GRADING_MACHINE, GRADING_MACHINE, rollout_limits())
    tools: dict[str, Tool[object]] = {
        "grade": GradeTool("/workspace/task.json", {}, settings),
        "submit": SubmitTool(accept),
        "validate": ValidateTool(accept),
    }
    args: dict[str, dict[str, object]] = {
        "grade": {"reply": "12", "files": ["/workspace/answer"]},
        "submit": {"task_path": "/workspace/task.json", "controls_path": "/workspace/controls.json"},
        "validate": {},
    }
    # No grading executor: malformed author data must never reach the executor backstop.
    result = await tools[name](ctx, args[name], env)
    await ctx.drain()
    assert isinstance(result, ToolResult) and "Candidate files require a task machine" in result.content


@pytest.mark.parametrize("name", ["submit", "validate"])
async def test_repaired_task_machine_allows_file_controls_to_leave_the_tool(name: str) -> None:
    ctx = Context(log=EventLog())
    env = environment(ctx)
    task = arithmetic_task(environment_requirements=requirements(image=None))
    candidate = Candidate("12", (inline_resource("workspace/answer", b"12"),))
    controls = (Control(candidate, RewardBand(Outcome.GRADED, reward_min=1), ControlKind.POSITIVE),)
    await ctx.emit(WriteFile(PurePosixPath("/workspace/task.json"), TASK_SPEC.dump_json(task)))
    await ctx.emit(WriteFile(PurePosixPath("/workspace/controls.json"), CONTROLS.dump_json(controls)))
    tool: Tool[object] = SubmitTool(accept) if name == "submit" else ValidateTool(accept)
    args = {"task_path": "/workspace/task.json", "controls_path": "/workspace/controls.json"} if name == "submit" else {}
    result = await tool(ctx, args, env)
    await ctx.drain()
    if name == "submit":
        assert isinstance(result, SubmitRequest) and result.submission.controls == controls
    else:
        assert isinstance(result, ReturnRequest) and isinstance(result.payload, ValidateRequest)
        assert result.payload.submission.controls == controls


async def test_provider_context_overflow_preserves_the_last_completed_conversation() -> None:
    client = FakeInference((call("shell", {"command": "task"}, "1"), ContextExceeded("maximum context length")))
    ctx = Context(log=EventLog())
    result = await AgenticHarness(262144, Sampling(), "use tools", 20000).run(
        ctx, Conversation((Message("user", "build"),), ()), environment(ctx), client, {"shell": ShellTool()}
    )
    await ctx.drain()
    assert isinstance(result, Stopped) and result.reason == StopReason.CONTEXT_LIMIT
    assert result.session.messages == client.requests[1].messages
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
    result = await AgenticHarness(100, Sampling(), "use tools", output_chars).run(
        ctx, Conversation((), ()), environment(ctx), client, {"web_search": WebSearchTool()}
    )
    await ctx.drain()
    assert isinstance(result, Stopped) and result.reason == StopReason.CONTEXT_LIMIT
    assert len(client.requests) == 1
    assert result.session.messages[-1].content.startswith("x" * output_chars + "\n[Full output saved to ")


async def test_adversary_report_requires_claim_evidence_and_uses_bound_task() -> None:
    ctx = Context(log=EventLog())
    env = environment(ctx)
    task = arithmetic_task()
    tool = ReportTool(task)
    commentary = await tool(ctx, {"findings": ["No grading defect found"]}, env)
    assert isinstance(commentary, ToolResult)
    claim = {
        "claim": "wrong answer accepted",
        "candidate": {"reply": "9"},
        "expected": {"status": "graded", "reward_min": 1},
    }
    report = await tool(ctx, {"findings": [claim]}, env)
    assert isinstance(report, ReturnRequest) and isinstance(report.payload, AdversaryReport)
    finding = report.payload.findings[0]
    assert finding.claim == "wrong answer accepted" and finding.candidate.reply == "9"
    assert finding.expected.met_by(GradeResult(Outcome.GRADED, 1))
    assert not finding.expected.met_by(GradeResult(Outcome.GRADED, 0))
    # The adversary can edit its scratch task file, but cannot change the bound task's machine requirement.
    edited = arithmetic_task(environment_requirements=requirements(image=None))
    await ctx.emit(WriteFile(PurePosixPath("/workspace/task.json"), TASK_SPEC.dump_json(edited)))
    await ctx.emit(WriteFile(PurePosixPath("/workspace/answer"), b"9"))
    rejected = await tool(
        ctx, {"findings": [{**claim, "candidate": {"reply": "9", "files": ["/workspace/answer"]}}]}, env
    )
    empty = await tool(ctx, {"findings": []}, env)
    await ctx.drain()
    assert isinstance(rejected, ToolResult) and "Candidate files require a task machine" in rejected.content
    assert isinstance(empty, ReturnRequest) and isinstance(empty.payload, AdversaryReport)
    assert empty.payload.findings == ()


async def test_malformed_grader_checks_return_feedback_and_corrected_checks_are_reported() -> None:
    ctx = Context(log=EventLog())
    env = environment(ctx)
    await ctx.emit(RunCommand("task", 30))

    async def invalid_checks(event: GradeCandidate) -> GradeResult:
        return GradeResult(Outcome.GRADED, 1, detail={"checks": {"A": False}})

    ctx.handle(GradeCandidate, invalid_checks)
    settings = LoweringSettings(MachineHost.LAPTOP, GRADING_MACHINE, GRADING_MACHINE, rollout_limits())
    tool = GradeTool("/workspace/task.json", {}, settings)
    result = await tool(ctx, {"reply": "12"}, env)
    assert isinstance(result, ToolResult) and "checks must be a list of" in result.content
    assert "name" in result.content and "passed" in result.content and "weight" in result.content

    async def valid_checks(event: GradeCandidate) -> GradeResult:
        return GradeResult(Outcome.GRADED, 0, detail={"checks": [{"name": "A", "passed": False, "weight": 1}]})

    ctx.handle(GradeCandidate, valid_checks)
    corrected = await tool(ctx, {"reply": "9"}, env)
    await ctx.drain()
    assert isinstance(corrected, ToolResult)
    assert json.loads(corrected.content)["grader_checks"] == [{"name": "A", "passed": False, "weight": 1}]


async def test_large_shell_output_is_saved_and_replayed_without_inference(tmp_path: Path) -> None:
    sink = JsonlSink(str(tmp_path / "outputs.jsonl"))
    ctx = Context(log=EventLog(sink=sink))
    env = environment(ctx)
    harness = AgenticHarness(262144, Sampling(), "continue", 8)
    initial = Conversation((), ())
    tools = {"shell": ShellTool(), "finish": FinishTool()}
    client = FakeInference(
        (call("shell", {"command": "task"}, "../../escape"), call("finish", {"reason": "done"}, "done"))
    )
    result = await harness.run(ctx, initial, env, client, tools)
    await ctx.drain()
    saved = [(path, data) for path, data in env.files.items() if path.parent == PurePosixPath(".tool-output")]
    assert len(saved) == 1
    path, data = saved[0]
    assert json.loads(data)["stdout"] == "task ready"
    assert str(path) in client.requests[1].messages[-1].content
    assert ".." not in path.parts
    replay = Context(log=EventLog(replay_from=await sink.read()))
    replay_env = environment(replay)
    replay_client = FakeInference(())
    assert await harness.run(replay, initial, replay_env, replay_client, tools) == result
    await replay.drain()
    assert not replay_client.requests
    assert replay_env.files == env.files
