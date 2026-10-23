# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Taskforge agent loop: GLM-5.3 tool calling over a ``ModelClient``.

Each turn is one ``ModelClient.complete`` call; through ``GlmClient`` it streams, retries, holds,
sends the policy's full output budget and continues on ``finish_reason == "length"``. The assistant turn is replayed
as ``rollout_model.assistant_wire_message`` builds it. ``ReasoningReplay`` decides whether it keeps
``reasoning_content`` (GLM's chat template ignores ``reasoning``): GLM's own reasoning replayed on
every later turn is what fills long agent prompts, and the template renders an absent
``reasoning_content`` as the same empty think block as ``""``. Tool calls in one turn run in order,
one at a time, because shell calls depend on each other's effects. Every turn carries the record's
``rollout_model.routing_key`` as ``prompt_cache_key``, so the router keeps a run's turns on the node
that already holds its prefix in cache.

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
mapping, so replaying the raw text would make every later request fail. (``rollout_model``
replays a rollout's arguments as served; ``rollout_model.assistant_wire_message`` gives the reason.)

Exceptions raised by a tool handler propagate; the caller classifies them. Every model turn is
recorded as an ``LLM_CALL`` ledger span (``llm.recording.recorded_complete``) and every executed or
rejected tool call as a ``STEP`` span, through the caller's ledger.
"""

import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

import jsonschema
from pydantic import JsonValue
from shellbox.machine import Machine
from taskcompendium.models import FunctionCall, FunctionDefinition
from taskcompendium.runtime.shell import BoundShellTool, ShellToolConfig, run_shell_call

from taskforge.ledger.records import EntryKind, span
from taskforge.llm.client import (
    Completion,
    ContextExhausted,
    FinishReason,
    ModelClient,
    ModelRequestRejected,
    ModelUnavailable,
    ToolCall,
    Usage,
)
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.recording import CallLedger, recorded_complete
from taskforge.llm.rollout_model import PROMPT_CACHE_KEY, assistant_wire_message, routing_key

INVALID_ARGUMENTS_KEY = "invalid_arguments"
TRUNCATED_CALL_MESSAGE = (
    "error: this tool call was cut off by the output limit before its arguments were complete, so it "
    "was not executed. Re-issue it; if it carried a large payload (for example a whole file), split the "
    "work into several smaller calls."
)
NO_USAGE = Usage(prompt_tokens=0, completion_tokens=0, reasoning_tokens=0, cached_tokens=0)
SHELL_TOOL = ShellToolConfig()
"""RolloutEngine's default shell tool, which ``shell_tool`` offers under the same name and argument."""
SHELL_TOOL_DESCRIPTION = "Run a Bash command in the task workspace. Files persist between commands."


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


def clip_tool_output(text: str, *, limit: int = 2000, head: int | None = None, tail: int = 0) -> str:
    """Keep a short reply whole, or retain its selected head and tail with a truncation marker."""
    if len(text) <= limit:
        return text
    head = limit if head is None else head
    if tail == 0:
        return f"{text[:head]}..."
    dropped = len(text) - head - tail
    return f"{text[:head]}\n... [{dropped} characters dropped] ...\n{text[-tail:]}"


class ReasoningReplay(StrEnum):
    """Whether replayed assistant turns of a ``run_agent`` loop carry their ``reasoning_content``."""

    KEEP = "keep"
    DROP = "drop"


class AgentStop(StrEnum):
    ANSWERED = "answered"
    """The model replied without tool calls and was not cut off."""
    MAX_TURNS = "max_turns"
    LENGTH = "length"
    """A reply without tool calls stayed cut off after ``LLMPolicy.max_continuations``; it is kept."""
    CONTEXT = "context"
    """The conversation filled the context window before the next reply; the turns so far are kept."""
    INTERRUPTED = "interrupted"
    """A model request failed; AgentInterrupted carries the completed conversation for resumption."""
    STOPPED = "stopped"
    """The caller's ``stop`` callback ended the run after a turn's tool calls ran; the turns so far are kept."""


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
    """The conversation as sent, including replayed assistant turns and tool results; under
    ``ReasoningReplay.DROP`` the assistant turns lack ``reasoning_content``, which each turn's
    ``completion`` still holds."""
    turns: tuple[AgentTurn, ...]
    stop: AgentStop
    usage: Usage


class AgentInterrupted(ModelUnavailable):
    """An unavailable model request with the agent's completed turns and exact continuation messages."""

    def __init__(self, cause: ModelUnavailable | ModelRequestRejected, run: AgentRun):
        super().__init__(str(cause), cause.attempts)
        self.run = run


def replay_arguments(arguments: str) -> str:
    """The arguments to replay: unchanged if they parse to a JSON object, else wrapped in one."""
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return arguments
    return json.dumps({INVALID_ARGUMENTS_KEY: arguments})


def assistant_message(completion: Completion, replay: ReasoningReplay) -> Message:
    """``assistant_wire_message`` with replay-safe arguments (``replay_arguments``); under DROP the
    ``reasoning_content`` key is absent."""
    calls = tuple(replace(call, arguments=replay_arguments(call.arguments)) for call in completion.tool_calls)
    message = assistant_wire_message(replace(completion, tool_calls=calls))
    if replay is ReasoningReplay.DROP:
        del message["reasoning_content"]
    return message


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
        # The violation is named by its path and constraint, never by the offending value: jsonschema's
        # own message quotes the value, and a long argument would come back whole into the context.
        raise _Rejected(
            ToolOutcome.INVALID_ARGUMENTS,
            f"error: tool arguments do not match the schema at {error.json_path}: "
            f"{error.validator} {json.dumps(error.validator_value)}",
        ) from error
    return arguments


async def _tool_result(
    call: ToolCall, truncated: bool, tools: Mapping[str, AgentTool], record: CallLedger, turn: int
) -> ToolResult:
    """Run or reject ``call``; ``outcome`` is recorded only once known, so a handler exception leaves just ``cause``."""
    started = time.monotonic()
    with span(
        record.ledger, EntryKind.STEP, workpiece_id=record.workpiece_id, round=record.round, step=record.step
    ) as fields:
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


async def run_agent(
    client: ModelClient,
    policy: LLMPolicy,
    messages: Sequence[Message],
    tools: Sequence[AgentTool],
    max_turns: int | None,
    record: CallLedger,
    replay: ReasoningReplay = ReasoningReplay.KEEP,
    stop: Callable[[AgentTurn], bool] | None = None,
) -> AgentRun:
    """Run the tool loop until the model answers, a reply stays cut off, the context fills, ``stop`` asks, or
    ``max_turns`` replies.

    A turn that the client rejects with ``ContextExhausted`` ends the run with
    ``AgentStop.CONTEXT``; its ``LLM_CALL`` span carries the exception as ``cause``. Other model
    failures raise ``AgentInterrupted`` with the completed conversation and tool turns for resumption.

    Args:
        client: Shared model client; concurrent runs share its connection pool.
        policy: Sampling and continuation policy for every turn.
        messages: The opening conversation, usually a system and a user message.
        tools: Tools offered on every turn; names must be unique.
        max_turns: Model replies allowed, or None for no turn limit; last-turn tool calls still run.
        record: Ledger and span identity for the turn and tool-call entries, and the run's ``routing_key``.
        replay: Whether replayed assistant turns carry their reasoning; they do by default.
        stop: Called with each turn that made tool calls, after they ran; ``True`` ends the run with
            ``AgentStop.STOPPED``, for example once a tool has handed in the run's result.
    """
    by_name = {tool.name: tool for tool in tools}
    if len(by_name) != len(tools):
        raise ValueError(f"tool names are not unique: {[tool.name for tool in tools]}")
    request_fields: dict[str, object] = {PROMPT_CACHE_KEY: routing_key(record)}
    if tools:
        request_fields["tools"] = [tool.definition() for tool in tools]
    conversation = list(messages)
    turns: list[AgentTurn] = []
    usage = NO_USAGE
    turn = 0
    while max_turns is None or turn < max_turns:
        try:
            completion = await recorded_complete(
                client, conversation, policy, request_fields, record, {"turn": str(turn)}
            )
        except ContextExhausted:
            return AgentRun(tuple(conversation), tuple(turns), AgentStop.CONTEXT, usage)
        except (ModelUnavailable, ModelRequestRejected) as error:
            partial = AgentRun(tuple(conversation), tuple(turns), AgentStop.INTERRUPTED, usage)
            raise AgentInterrupted(error, partial) from error
        usage = usage + completion.usage
        conversation.append(assistant_message(completion, replay))
        if not completion.tool_calls:
            turns.append(AgentTurn(completion, ()))
            ended = AgentStop.LENGTH if completion.finish_reason is FinishReason.LENGTH else AgentStop.ANSWERED
            return AgentRun(tuple(conversation), tuple(turns), ended, usage)
        cut = completion.finish_reason is FinishReason.LENGTH
        last = len(completion.tool_calls) - 1
        results = []
        for index, call in enumerate(completion.tool_calls):
            result = await _tool_result(call, cut and index == last, by_name, record, turn)
            results.append(result)
            conversation.append({"role": "tool", "tool_call_id": call.id, "content": result.output})
        turns.append(AgentTurn(completion, tuple(results)))
        if stop is not None and stop(turns[-1]):
            return AgentRun(tuple(conversation), tuple(turns), AgentStop.STOPPED, usage)
        turn += 1
    return AgentRun(tuple(conversation), tuple(turns), AgentStop.MAX_TURNS, usage)


def shell_tool(machine: Machine, *, timeout: float, output_limit_bytes: int) -> AgentTool:
    """RolloutEngine's default ``shell`` tool and its observation format, run through ``machine``.

    The definition is the one ``taskcompendium.runtime.shell.resolve_shell_tool`` binds for a task
    with the shell capability, and each call runs as ``run_shell_call`` runs a solver's, so an agent
    sees the shell a solver sees.

    Args:
        machine: Where commands run; files persist between calls until the caller closes it.
        timeout: Seconds per command before the machine reports it timed out.
        output_limit_bytes: Bytes kept of each of stdout and stderr.
    """
    parameter = SHELL_TOOL.command_parameter
    parameters: dict[str, JsonValue] = {
        "type": "object",
        "properties": {parameter: {"type": "string"}},
        "required": [parameter],
        "additionalProperties": False,
    }
    definition = FunctionDefinition(name=SHELL_TOOL.name, description=SHELL_TOOL_DESCRIPTION, parameters=parameters)
    bound = BoundShellTool(definition, parameter)

    async def run(arguments: Mapping[str, object]) -> str:
        call = FunctionCall.model_validate({"name": SHELL_TOOL.name, "arguments": dict(arguments)})
        return await run_shell_call(machine, call, bound, timeout=timeout, output_limit_bytes=output_limit_bytes)

    return AgentTool(name=SHELL_TOOL.name, description=SHELL_TOOL_DESCRIPTION, parameters=parameters, handler=run)
