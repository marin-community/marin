# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The mini-swe-agent loop, and its ``AgentHarness``.

Adapted from mini-swe-agent's ``DefaultAgent``, ``LocalEnvironment`` and tool-call actions (MIT) and Harbor's
``ContextLimitedModel`` (Apache-2.0); see ``rolloutengine.agents.mini_swe``. Harbor runs mini-swe-agent with
``--yolo --exit-immediately -c mini.yaml --agent-class DefaultAgent`` and no cost or step limit. The agent sends
the system and task prompts of ``mini.yaml``, then one request per step with a single ``bash`` tool. It runs each
tool call's command in a fresh ``/bin/sh`` subshell with a 30-second timeout and answers with the rendered
observation. A reply without a valid ``bash`` call gets the format-error prompt instead, and the reply itself
leaves the history. A command whose first output line is ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``, with exit
status 0, ends the episode; the engine grades the task machine, so the submitted text is unused.

Before each request, the agent counts the prompt with the policy server's ``/tokenize`` and limits the reply to
the context left: the smaller of ``max_input_tokens`` and the served ``max_model_len``, less the prompt. When no
reply fits, or the server rejects the request as too long, the episode ends normally, as Harbor's
``LimitsExceeded`` ends mini-swe-agent.
"""

import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import jinja2
import yaml
from shellbox.machine import Command, ExitReason, Machine
from taskcompendium.models import TextMessage

from rolloutengine.agents.environment import EXEC_OUTPUT_LIMIT_BYTES
from rolloutengine.agents.model_relay import PromptTokenizer
from rolloutengine.harness.core import HarnessContext
from rolloutengine.harness.external import AgentCompletion, AgentHarness, LengthStopPolicy, ReplyMatch
from rolloutengine.models import ContextWindowExceeded, ModelTurn, Prompt

logger = logging.getLogger(__name__)

CONFIG = yaml.safe_load(Path(__file__).with_name("mini.yaml").read_text())
TEMPLATES = jinja2.Environment(undefined=jinja2.StrictUndefined)
SYSTEM_TEMPLATE = TEMPLATES.from_string(CONFIG["agent"]["system_template"])
INSTANCE_TEMPLATE = TEMPLATES.from_string(CONFIG["agent"]["instance_template"])
OBSERVATION_TEMPLATE = TEMPLATES.from_string(CONFIG["model"]["observation_template"])
FORMAT_ERROR_TEMPLATE = TEMPLATES.from_string(CONFIG["model"]["format_error_template"])
COMMAND_ENV: dict[str, str] = CONFIG["environment"]["env"]
COMMAND_TIMEOUT = 30
"""mini-swe-agent's default ``LocalEnvironment`` timeout, in seconds."""
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}
# mini-swe-agent runs each command with ``subprocess.run(command, shell=True, stderr=STDOUT)``: ``/bin/sh -c`` with
# both streams in one pipe. The wrapper's shell redirects stderr and execs that shell, so the command sees the same
# ``$0``. The command is passed once, as "$1".
SHELL_WRAPPER = 'exec /bin/sh -c "$1" 2>&1'
SHELL_WRAPPER_NAME = "mini-swe-command"
# LocalEnvironment renders the task prompt's system information from Python's platform.uname() in the machine.
UNAME_COMMAND = "uname -s && uname -r && uname -v && uname -m"


class MiniSweModelFormat(StrEnum):
    """How the agent formats the history it sends: Harbor's ``model_format`` of mini-swe-agent."""

    OPENAI = "openai"
    """Each assistant message as the server returned it."""
    K2_HORIZON = "k2_horizon"
    """Each assistant message also has a ``reasoning`` field, empty when the reply had no reasoning, as K2's chat
    template requires after tool calls."""


@dataclass(frozen=True)
class MiniSweOptions:
    """mini-swe-agent's settings: Harbor's ``model_info`` limits and ``model_format``.

    ``max_input_tokens`` is the context window and ``max_output_tokens`` the largest reply that a request asks
    for.
    """

    max_input_tokens: int
    max_output_tokens: int
    model_format: MiniSweModelFormat

    def __post_init__(self) -> None:
        if self.max_input_tokens < 1 or self.max_output_tokens < 1:
            raise ValueError("mini-swe-agent token limits must be positive")


@dataclass(frozen=True)
class CommandOutput:
    """A command's result as mini-swe-agent's observation template reads it."""

    output: str
    returncode: int
    exception_info: str


class FormatError(Exception):
    """The reply has no valid ``bash`` tool call; ``message`` is the prompt that tells the model."""

    def __init__(self, error: str):
        super().__init__(error)
        self.message = {"role": "user", "content": FORMAT_ERROR_TEMPLATE.render(error=error)}


def mini_swe_harness(options: MiniSweOptions, tokenize: PromptTokenizer) -> AgentHarness:
    """An ``AgentHarness`` that runs mini-swe-agent with ``options`` and counts its prompts with ``tokenize``.

    mini-swe-agent does not read the stop reason, so a reply cut by its output limit is parsed like any other,
    and the next request starts a new conversation.
    """

    async def run(context: HarnessContext, complete: AgentCompletion) -> None:
        await run_mini_swe(options, tokenize, context, complete)

    return AgentHarness(run, length_stop=LengthStopPolicy.NEW_CONVERSATION, reply_match=ReplyMatch.EXACT)


async def run_mini_swe(
    options: MiniSweOptions, tokenize: PromptTokenizer, context: HarnessContext, complete: AgentCompletion
) -> None:
    """Run mini-swe-agent on the task's instruction until it submits or its context is full.

    Raises:
        ValueError: The task is not one user instruction, or it has no machine.
    """
    events = context.task.context.events
    if len(events) != 1 or not isinstance(events[0], TextMessage) or events[0].role != "user":
        raise ValueError("mini-swe-agent requires one user instruction as the task context")
    machine = context.machine
    if machine is None:
        raise ValueError("mini-swe-agent requires a task machine")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_TEMPLATE.render()},
        {
            "role": "user",
            "content": INSTANCE_TEMPLATE.render(task=events[0].content, **await system_information(machine)),
        },
    ]
    while True:
        turn = await query(options, tokenize, complete, messages)
        if turn is None:
            logger.warning("mini-swe-agent ran out of context in attempt %s; grading its work", context.rollout_id)
            return
        message = served_message(turn)
        try:
            actions = toolcall_actions(message.get("tool_calls") or [])
        except FormatError as error:
            messages.append(error.message)
            continue
        messages.append(message)
        for tool_call_id, command in actions:
            output = await execute(machine, command)
            if submitted(output):
                return
            messages.append(
                {
                    "content": OBSERVATION_TEMPLATE.render(output=output),
                    "tool_call_id": tool_call_id,
                    "role": "tool",
                }
            )


async def system_information(machine: Machine) -> dict[str, str]:
    """The ``platform.uname()`` fields of the task prompt, read on ``machine``."""
    result = await machine.run(Command(("/bin/sh", "-c", UNAME_COMMAND)))
    if result.exit_code != 0:
        raise RuntimeError(f"uname failed on the task machine: {result.stdout.decode(errors='replace')}")
    system, release, version, uname_machine = result.stdout.decode().splitlines()
    return {"system": system, "release": release, "version": version, "machine": uname_machine}


async def query(
    options: MiniSweOptions, tokenize: PromptTokenizer, complete: AgentCompletion, messages: list[dict[str, Any]]
) -> ModelTurn | None:
    """Ask the model for the next step within the context left, or return ``None`` when no reply fits."""
    if options.model_format is MiniSweModelFormat.K2_HORIZON:
        messages = [
            (
                {**message, "reasoning": message.get("reasoning") or message.get("reasoning_content") or ""}
                if message["role"] == "assistant"
                else message
            )
            for message in messages
        ]
    tokenization = await tokenize(
        {"messages": messages, "tools": [BASH_TOOL], "add_generation_prompt": True, "chat_template_kwargs": {}}
    )
    remaining = min(options.max_input_tokens, tokenization["max_model_len"]) - tokenization["count"]
    if remaining < 1:
        return None
    try:
        return await complete(
            Prompt(tuple(messages), {"tools": [BASH_TOOL], "max_tokens": min(options.max_output_tokens, remaining)})
        )
    except ContextWindowExceeded:
        return None


def served_message(turn: ModelTurn) -> dict[str, Any]:
    """The assistant message as the server sent it, which mini-swe-agent keeps in its history.

    ``ChatModel`` keeps the reply's ``reasoning`` and ``refusal`` as metadata.
    """
    restored = {key: turn.metadata[key] for key in ("reasoning", "refusal") if key in turn.metadata}
    return {**restored, **turn.message}


def toolcall_actions(tool_calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """The ID and command of each ``bash`` tool call.

    Raises:
        FormatError: There is no tool call, or one is not a ``bash`` call with a string ``command``.
    """
    if not tool_calls:
        raise FormatError("No tool calls found in the response. Every response MUST include at least one tool call.")
    actions = []
    for tool_call in tool_calls:
        error = ""
        arguments: Any = {}
        try:
            arguments = json.loads(tool_call["function"]["arguments"])
        except json.JSONDecodeError as decode_error:
            error = f"Error parsing tool call arguments: {decode_error}. "
        if tool_call["function"]["name"] != "bash":
            error += f"Unknown tool '{tool_call['function']['name']}'."
        # mini-swe-agent fails on arguments that are not an object and on a command that is not a string; the
        # port answers both with the format-error prompt.
        if not isinstance(arguments, dict) or "command" not in arguments:
            error += "Missing 'command' argument in bash tool call."
        elif not isinstance(arguments["command"], str):
            error += "The 'command' argument of the bash tool call must be a string."
        if error:
            raise FormatError(error.strip())
        actions.append((tool_call["id"], arguments["command"]))
    return actions


async def execute(machine: Machine, command: str) -> CommandOutput:
    """Run ``command`` as mini-swe-agent's ``LocalEnvironment`` runs it, with ``mini.yaml``'s environment."""
    result = await machine.run(
        Command(
            ("/bin/sh", "-c", SHELL_WRAPPER, SHELL_WRAPPER_NAME, command),
            env=dict(COMMAND_ENV),
            timeout=COMMAND_TIMEOUT,
            output_limit_bytes=EXEC_OUTPUT_LIMIT_BYTES,
        )
    )
    if result.reason is ExitReason.TIMED_OUT:
        # mini-swe-agent decodes the partial output of a timed-out command without newline translation, and
        # reports Python's TimeoutExpired.
        return CommandOutput(
            result.stdout.decode(errors="replace"),
            -1,
            "An error occurred while executing the command: "
            f"Command '{command}' timed out after {COMMAND_TIMEOUT} seconds",
        )
    assert result.exit_code is not None
    # Text-mode subprocess output has universal newlines.
    output = result.stdout.decode(errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    return CommandOutput(output, result.exit_code, "")


def submitted(output: CommandOutput) -> bool:
    lines = output.output.lstrip().splitlines(keepends=True)
    return bool(lines) and lines[0].strip() == SUBMIT_MARKER and output.returncode == 0
