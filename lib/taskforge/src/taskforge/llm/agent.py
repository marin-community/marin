# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Taskforge agent loop: GLM-5.3 tool calling over ``GlmClient``.

Each turn is one ``GlmClient.complete`` call, so it streams, retries, holds, starts at the model's
full output budget and continues on ``finish_reason == "length"``. The assistant turn is replayed
as ``rollout_model.assistant_wire_message`` builds it, with ``reasoning_content`` (GLM's chat
template ignores ``reasoning``). Tool calls in one turn run in order, one at a time, because shell
calls depend on each other's effects.

Model mistakes go back to the model as tool results and the run continues:

* arguments that are not a JSON object, or that fail the tool's JSON schema;
* a call to a tool that does not exist;
* a call cut off by the output limit. ``GlmClient`` does not continue a reply that already holds a
  tool call, and reports it as ``finish_reason == "length"`` (including vLLM's ``tool_calls``
  report of a reply that spent its whole budget). The last call of a cut reply is not executed,
  even if its arguments happen to parse; the model is told it was cut off and asked to re-issue it
  in smaller pieces. Calls before it in the same reply are complete and run normally.

Arguments that do not parse to an object are replayed as ``{"invalid_arguments": <raw text>}``:
vLLM parses replayed ``tool_calls[].function.arguments`` and GLM's template iterates them as a
mapping, so replaying the raw text would make every later request fail. (RolloutEngine instead
ends a rollout on such arguments, so ``rollout_model`` replays them as served.)

Exceptions raised by a tool handler propagate; the caller classifies them. Every model turn is
recorded as an ``LLM_CALL`` ledger span and every executed or rejected tool call as a ``STEP``
span, through the caller's ledger.
"""

import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

import jsonschema
from rolloutengine.shell_tool import SHELL_TOOL_NAME, shell_observation, shell_tool_definition
from shellbox.machine import Command, Machine

from taskforge.ledger.records import EntryKind, Ledger, span
from taskforge.llm.client import Completion, FinishReason, GlmClient, GlmContextExhausted, ToolCall, Usage
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.rollout_model import assistant_wire_message

INVALID_ARGUMENTS_KEY = "invalid_arguments"
TRUNCATED_CALL_MESSAGE = (
    "error: this tool call was cut off by the output limit before its arguments were complete, so it "
    "was not executed. Re-issue it; if it carried a large payload (for example a whole file), split the "
    "work into several smaller calls."
)
NO_USAGE = Usage(prompt_tokens=0, completion_tokens=0, reasoning_tokens=0, cached_tokens=0)


@dataclass(frozen=True)
class AgentTool:
    """A function tool: its JSON-schema ``parameters`` validate arguments before ``handler`` runs."""

    name: str
    description: str
    parameters: Mapping[str, object]
    handler: Callable[[Mapping[str, object]], Awaitable[str]]

    def definition(self) -> dict[str, object]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": dict(self.parameters)},
        }


class AgentStop(StrEnum):
    ANSWERED = "answered"
    """The model replied without tool calls and was not cut off."""
    MAX_TURNS = "max_turns"
    LENGTH = "length"
    """A reply without tool calls stayed cut off after ``LLMPolicy.max_continuations``; it is kept."""
    CONTEXT = "context"
    """The conversation filled the context window before the next reply; the turns so far are kept."""


class ToolOutcome(StrEnum):
    EXECUTED = "executed"
    INVALID_ARGUMENTS = "invalid_arguments"
    UNKNOWN_TOOL = "unknown_tool"
    TRUNCATED_CALL = "truncated_call"


@dataclass(frozen=True)
class ToolResult:
    call: ToolCall
    outcome: ToolOutcome
    output: str
    """What the model sees as the tool message content."""
    wall_time: float


@dataclass(frozen=True)
class AgentTurn:
    completion: Completion
    tool_results: tuple[ToolResult, ...]


@dataclass(frozen=True)
class AgentRun:
    messages: tuple[Message, ...]
    """The conversation as sent, including replayed assistant turns and tool results."""
    turns: tuple[AgentTurn, ...]
    stop: AgentStop
    usage: Usage


@dataclass(frozen=True)
class AgentLedger:
    """Where ``run_agent`` records its spans, all under one item, round and step."""

    ledger: Ledger
    item_id: str
    round: int
    step: str


def replay_arguments(arguments: str) -> str:
    """The arguments to replay: unchanged if they parse to a JSON object, else wrapped in one."""
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return arguments
    return json.dumps({INVALID_ARGUMENTS_KEY: arguments})


def assistant_message(completion: Completion) -> Message:
    """``assistant_wire_message`` with each tool call's arguments passed through ``replay_arguments``."""
    calls = tuple(replace(call, arguments=replay_arguments(call.arguments)) for call in completion.tool_calls)
    return assistant_wire_message(replace(completion, tool_calls=calls))


class _Rejected(Exception):
    """A tool call whose handler must not run; ``output`` goes back to the model."""

    def __init__(self, outcome: ToolOutcome, output: str):
        super().__init__(output)
        self.outcome = outcome
        self.output = output


def _checked_arguments(call: ToolCall, tools: Mapping[str, AgentTool]) -> dict[str, object]:
    """``call``'s parsed arguments; raises ``_Rejected`` if its handler must not run."""
    tool = tools.get(call.name)
    if tool is None:
        raise _Rejected(ToolOutcome.UNKNOWN_TOOL, f"error: unknown tool {call.name!r}; available tools: {sorted(tools)}")
    try:
        arguments = json.loads(call.arguments)
    except json.JSONDecodeError as error:
        raise _Rejected(ToolOutcome.INVALID_ARGUMENTS, f"error: tool arguments are not valid JSON ({error})") from error
    if not isinstance(arguments, dict):
        raise _Rejected(ToolOutcome.INVALID_ARGUMENTS, "error: tool arguments must be a JSON object")
    try:
        jsonschema.validate(arguments, dict(tool.parameters))
    except jsonschema.ValidationError as error:
        raise _Rejected(
            ToolOutcome.INVALID_ARGUMENTS, f"error: tool arguments do not match the schema ({error.message})"
        ) from error
    return arguments


async def _tool_result(
    call: ToolCall, truncated: bool, tools: Mapping[str, AgentTool], record: AgentLedger, turn: int
) -> ToolResult:
    """Run or reject ``call``; ``outcome`` is recorded only once known, so a handler exception leaves just ``cause``."""
    started = time.monotonic()
    with span(record.ledger, EntryKind.STEP, item_id=record.item_id, round=record.round, step=record.step) as fields:
        fields.attrs.update({"turn": str(turn), "tool": call.name, "call_id": call.id})
        try:
            if truncated:
                raise _Rejected(ToolOutcome.TRUNCATED_CALL, TRUNCATED_CALL_MESSAGE)
            arguments = _checked_arguments(call, tools)
        except _Rejected as rejected:
            outcome, output = rejected.outcome, rejected.output
        else:
            output = await tools[call.name].handler(arguments)
            outcome = ToolOutcome.EXECUTED
        fields.attrs["outcome"] = outcome
    return ToolResult(call=call, outcome=outcome, output=output, wall_time=time.monotonic() - started)


async def _complete(
    client: GlmClient,
    policy: LLMPolicy,
    conversation: Sequence[Message],
    request_fields: Mapping[str, object],
    record: AgentLedger,
    turn: int,
) -> Completion:
    with span(record.ledger, EntryKind.LLM_CALL, item_id=record.item_id, round=record.round, step=record.step) as fields:
        fields.model = client.endpoint.model
        fields.attrs["turn"] = str(turn)
        completion = await client.complete(conversation, policy, request_fields)
        fields.tokens_in = completion.usage.prompt_tokens
        fields.tokens_out = completion.usage.completion_tokens
        fields.tokens_reasoning = completion.usage.reasoning_tokens
        fields.finish_reason = completion.finish_reason
        fields.attrs.update(
            {
                "cached_tokens": str(completion.usage.cached_tokens),
                "continuations": str(completion.continuations),
                "attempts": str(len(completion.attempts)),
                "tool_calls": str(len(completion.tool_calls)),
            }
        )
    return completion


async def run_agent(
    client: GlmClient,
    policy: LLMPolicy,
    messages: Sequence[Message],
    tools: Sequence[AgentTool],
    max_turns: int,
    record: AgentLedger,
) -> AgentRun:
    """Run the tool loop until the model answers, a reply stays cut off, the context fills, or ``max_turns`` replies.

    A turn that ``GlmClient`` rejects with ``GlmContextExhausted`` ends the run with
    ``AgentStop.CONTEXT``; its ``LLM_CALL`` span carries the exception as ``cause``.

    Args:
        client: Shared GLM client; concurrent runs share its connection pool.
        policy: Sampling and continuation policy for every turn.
        messages: The opening conversation, usually a system and a user message.
        tools: Tools offered on every turn; names must be unique.
        max_turns: Model replies allowed; tool calls in the last reply still run.
        record: Ledger and span identity for the turn and tool-call entries.
    """
    by_name = {tool.name: tool for tool in tools}
    if len(by_name) != len(tools):
        raise ValueError(f"tool names are not unique: {[tool.name for tool in tools]}")
    request_fields: dict[str, object] = {"tools": [tool.definition() for tool in tools]} if tools else {}
    conversation = list(messages)
    turns: list[AgentTurn] = []
    usage = NO_USAGE
    for turn in range(max_turns):
        try:
            completion = await _complete(client, policy, conversation, request_fields, record, turn)
        except GlmContextExhausted:
            return AgentRun(tuple(conversation), tuple(turns), AgentStop.CONTEXT, usage)
        usage = usage + completion.usage
        conversation.append(assistant_message(completion))
        if not completion.tool_calls:
            turns.append(AgentTurn(completion, ()))
            stop = AgentStop.LENGTH if completion.finish_reason is FinishReason.LENGTH else AgentStop.ANSWERED
            return AgentRun(tuple(conversation), tuple(turns), stop, usage)
        cut = completion.finish_reason is FinishReason.LENGTH
        last = len(completion.tool_calls) - 1
        results = []
        for index, call in enumerate(completion.tool_calls):
            result = await _tool_result(call, cut and index == last, by_name, record, turn)
            results.append(result)
            conversation.append({"role": "tool", "tool_call_id": call.id, "content": result.output})
        turns.append(AgentTurn(completion, tuple(results)))
    return AgentRun(tuple(conversation), tuple(turns), AgentStop.MAX_TURNS, usage)


def shell_tool(machine: Machine, *, timeout: float, output_limit_bytes: int, user: str | None = None) -> AgentTool:
    """RolloutEngine's ``shell(command)`` tool and observation format, run as ``sh -c`` through ``machine``.

    Args:
        machine: Where commands run; files persist between calls until the caller closes it.
        timeout: Seconds per command before the machine reports it timed out.
        output_limit_bytes: Bytes kept of each of stdout and stderr.
        user: User the commands run as. RolloutEngine runs a task's shell as ``TaskExecution.agent_user``;
            pass it to act as the solver does. None is the machine's default user, as for ``Command``.
    """

    async def run(arguments: Mapping[str, object]) -> str:
        command = arguments["command"]
        assert isinstance(command, str)
        result = await machine.run(
            Command(argv=("sh", "-c", command), timeout=timeout, output_limit_bytes=output_limit_bytes, user=user)
        )
        return shell_observation(result)

    function = shell_tool_definition()["function"]
    return AgentTool(
        name=SHELL_TOOL_NAME, description=function["description"], parameters=function["parameters"], handler=run
    )
