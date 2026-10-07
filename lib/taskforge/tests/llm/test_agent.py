# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from collections.abc import Mapping

import pytest
from rigging.timing import ExponentialBackoff
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import MachineSpec, ShellSimBuiltins

from taskforge.ledger.records import EntryKind, LedgerEntry
from taskforge.llm.agent import (
    INVALID_ARGUMENTS_KEY,
    AgentRun,
    AgentStop,
    AgentTool,
    ToolOutcome,
    run_agent,
    shell_tool,
)
from taskforge.llm.client import GlmClient, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.recording import CallLedger

MESSAGES = [{"role": "system", "content": "be useful"}, {"role": "user", "content": "do it"}]
ECHO_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


class ListLedger:
    def __init__(self) -> None:
        self.entries: list[LedgerEntry] = []

    def record(self, entry: LedgerEntry) -> None:
        self.entries.append(entry)


def echo_tool(calls: list[Mapping[str, object]]) -> AgentTool:
    async def handler(arguments: Mapping[str, object]) -> str:
        calls.append(arguments)
        return f"echo: {arguments['text']}"

    return AgentTool(name="echo", description="Echo text.", parameters=ECHO_SCHEMA, handler=handler)


def run(fake_glm, tools, ledger=None, max_turns=8, policy=LLMPolicy()) -> AgentRun:
    async def go() -> AgentRun:
        endpoint = GlmEndpoint(base_url=fake_glm.base_url, token="test-token", pool=Pool.HIGH)
        backoff = ExponentialBackoff(initial=0.001, maximum=0.001)
        record = CallLedger(ledger=ledger or ListLedger(), item_id="item-1", round=0, step="author")
        async with GlmClient(endpoint, backoff=backoff) as client:
            return await run_agent(client, policy, MESSAGES, tools, max_turns, record)

    return asyncio.run(go())


def tool_messages(fake_glm, request: int) -> list[dict]:
    return [m for m in fake_glm.requests[request]["messages"] if m["role"] == "tool"]


def test_tool_call_then_answer_replays_reasoning_and_records_turns(fake_glm):
    fake_glm.stream(
        reasoning="I should echo.", tool_calls=(("echo", '{"text": "hi"}'),), finish="tool_calls", completion_tokens=7
    )
    fake_glm.stream(content="done", reasoning="echoed", completion_tokens=3)
    calls: list[Mapping[str, object]] = []
    ledger = ListLedger()

    result = run(fake_glm, [echo_tool(calls)], ledger)

    assert result.stop is AgentStop.ANSWERED
    assert calls == [{"text": "hi"}]
    assert result.usage.completion_tokens == 10
    second = fake_glm.requests[1]
    assert second["tools"][0]["function"]["name"] == "echo"
    replayed = second["messages"][2]
    assert replayed["reasoning_content"] == "I should echo."
    assert replayed["tool_calls"][0]["function"] == {"name": "echo", "arguments": '{"text": "hi"}'}
    assert tool_messages(fake_glm, 1) == [{"role": "tool", "tool_call_id": "call-0", "content": "echo: hi"}]
    assert result.messages[-1] == {"role": "assistant", "content": "done", "reasoning_content": "echoed"}
    assert [(e.kind, e.attrs["turn"]) for e in ledger.entries] == [
        (EntryKind.LLM_CALL, "0"),
        (EntryKind.STEP, "0"),
        (EntryKind.LLM_CALL, "1"),
    ]
    assert ledger.entries[0].tokens_out == 7
    assert ledger.entries[0].finish_reason == "tool_calls"
    assert ledger.entries[1].attrs["outcome"] == ToolOutcome.EXECUTED


@pytest.mark.parametrize(
    ("name", "arguments", "outcome"),
    [
        ("echo", '{"text": "unterminated', ToolOutcome.INVALID_ARGUMENTS),
        ("echo", '["text"]', ToolOutcome.INVALID_ARGUMENTS),
        ("echo", '{"txt": "hi"}', ToolOutcome.INVALID_ARGUMENTS),
        ("missing", '{"text": "hi"}', ToolOutcome.UNKNOWN_TOOL),
    ],
    ids=["unparseable", "not-an-object", "schema-violation", "unknown-tool"],
)
def test_bad_tool_call_returns_an_error_result_and_the_run_continues(fake_glm, name, arguments, outcome):
    fake_glm.stream(tool_calls=((name, arguments),), finish="tool_calls")
    fake_glm.stream(tool_calls=(("echo", '{"text": "again"}'),), finish="tool_calls")
    fake_glm.stream(content="done")
    calls: list[Mapping[str, object]] = []

    result = run(fake_glm, [echo_tool(calls)])

    assert result.stop is AgentStop.ANSWERED
    assert calls == [{"text": "again"}]
    assert result.turns[0].tool_results[0].outcome is outcome
    assert tool_messages(fake_glm, 1)[0]["content"].startswith("error:")
    replayed = json.loads(fake_glm.requests[1]["messages"][2]["tool_calls"][0]["function"]["arguments"])
    assert isinstance(replayed, dict)


def test_unparseable_arguments_are_replayed_wrapped_in_an_object(fake_glm):
    fake_glm.stream(tool_calls=(("echo", '{"text": "unterminated'),), finish="tool_calls")
    fake_glm.stream(content="done")

    run(fake_glm, [echo_tool([])])

    replayed = fake_glm.requests[1]["messages"][2]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(replayed) == {INVALID_ARGUMENTS_KEY: '{"text": "unterminated'}


@pytest.mark.parametrize(
    ("finish", "completion_tokens"),
    [("length", 5), ("tool_calls", 64)],
    ids=["finish-length", "vllm-reports-tool-calls-at-budget"],
)
def test_length_cut_inside_a_tool_call_runs_complete_calls_and_returns_the_cut_one_as_an_error(
    fake_glm, finish, completion_tokens
):
    fake_glm.stream(
        tool_calls=(("echo", '{"text": "whole"}'), ("echo", '{"text": "cut off he')),
        finish=finish,
        completion_tokens=completion_tokens,
    )
    fake_glm.stream(tool_calls=(("echo", '{"text": "retried"}'),), finish="tool_calls")
    fake_glm.stream(content="done")
    calls: list[Mapping[str, object]] = []

    result = run(fake_glm, [echo_tool(calls)], policy=LLMPolicy(max_tokens=64))

    assert result.stop is AgentStop.ANSWERED
    assert calls == [{"text": "whole"}, {"text": "retried"}]
    assert [r.outcome for r in result.turns[0].tool_results] == [ToolOutcome.EXECUTED, ToolOutcome.TRUNCATED_CALL]
    assert "cut off by the output limit" in tool_messages(fake_glm, 1)[1]["content"]


def test_length_cut_whose_arguments_happen_to_parse_is_still_not_executed(fake_glm):
    fake_glm.stream(tool_calls=(("echo", '{"text": "rm -rf /tmp/x"}'),), finish="length")
    fake_glm.stream(content="done")
    calls: list[Mapping[str, object]] = []

    result = run(fake_glm, [echo_tool(calls)])

    assert calls == []
    assert result.turns[0].tool_results[0].outcome is ToolOutcome.TRUNCATED_CALL


def test_reply_still_cut_off_after_continuations_stops_with_length(fake_glm):
    fake_glm.stream(content="partial", finish="length")

    result = run(fake_glm, [echo_tool([])], policy=LLMPolicy(max_continuations=0))

    assert result.stop is AgentStop.LENGTH
    assert result.messages[-1]["content"] == "partial"


def test_context_filled_mid_run_returns_the_turns_so_far(fake_glm):
    context_error = json.dumps({"error": {"message": "This model's maximum context length is 262144 tokens."}})
    fake_glm.stream(tool_calls=(("echo", '{"text": "one"}'),), finish="tool_calls")
    fake_glm.status(400, context_error)
    fake_glm.status(400, context_error)
    calls: list[Mapping[str, object]] = []
    ledger = ListLedger()

    result = run(fake_glm, [echo_tool(calls)], ledger=ledger)

    assert result.stop is AgentStop.CONTEXT
    assert calls == [{"text": "one"}]
    assert len(result.turns) == 1
    assert result.messages[-1] == {"role": "tool", "tool_call_id": "call-0", "content": "echo: one"}
    last_call = [e for e in ledger.entries if e.kind is EntryKind.LLM_CALL][-1]
    assert last_call.cause == "GlmContextExhausted"


def test_max_turns_stops_after_running_the_last_turns_tools(fake_glm):
    for text in ("one", "two"):
        fake_glm.stream(tool_calls=(("echo", json.dumps({"text": text})),), finish="tool_calls")
    calls: list[Mapping[str, object]] = []

    result = run(fake_glm, [echo_tool(calls)], max_turns=2)

    assert result.stop is AgentStop.MAX_TURNS
    assert calls == [{"text": "one"}, {"text": "two"}]
    assert result.messages[-1]["role"] == "tool"


def test_tool_handler_exception_propagates_and_is_recorded(fake_glm):
    async def broken(arguments: Mapping[str, object]) -> str:
        raise ConnectionError("sandbox gone")

    tool = AgentTool(name="echo", description="Echo text.", parameters=ECHO_SCHEMA, handler=broken)
    fake_glm.stream(tool_calls=(("echo", '{"text": "hi"}'),), finish="tool_calls")
    ledger = ListLedger()

    with pytest.raises(ConnectionError):
        run(fake_glm, [tool], ledger)

    assert (ledger.entries[-1].kind, ledger.entries[-1].cause) == (EntryKind.STEP, "ConnectionError")


def test_shell_tool_runs_in_the_machine_and_reports_exit_code_and_truncation():
    async def go() -> tuple[dict, dict]:
        machine = await ShellSimMachineFactory().create(MachineSpec(source=ShellSimBuiltins()))
        tool = shell_tool(machine, timeout=30, output_limit_bytes=4)
        await tool.handler({"command": "echo persisted > note.txt"})
        first = json.loads(await tool.handler({"command": "cat note.txt"}))
        second = json.loads(await tool.handler({"command": "echo oops >&2; exit 3"}))
        await machine.close()
        return first, second

    first, second = asyncio.run(go())

    assert (first["stdout"], first["exit_code"], first["truncated"]) == ("pers", 0, True)
    assert (second["stderr"], second["exit_code"], second["truncated"]) == ("oops", 3, True)
