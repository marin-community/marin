# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small external Harbor agent that routes Bash tool calls to the environment."""

import json

import httpx
from harbor.agents.base import BaseAgent, TurnCapExhaustedError
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from shellbox.machine import BashSessionProvider, ShellSession, ShellUpdate

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "Bash",
        "description": (
            "Run a command in the persistent Bash shell. Directory, variables, functions, and jobs persist. "
            "A completed command returns output and exit_code; a long command returns status=running. "
            "To continue a running command, omit command to read, pass input to send text, or set signal=interrupt. "
            "Output is capped at 128 KiB; redirect larger output to a file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "input": {
                    "type": "string",
                    "description": "Text to send to a running command; include a newline to submit",
                },
                "signal": {"type": "string", "enum": ["interrupt"], "description": "Interrupt the foreground process"},
                "wait_ms": {"type": "integer", "description": "Wait up to this many milliseconds; maximum 30000"},
            },
        },
    },
}
SHELLSIM_BASH_TOOL = {
    **BASH_TOOL,
    "function": {
        **BASH_TOOL["function"],
        "description": (
            "Run one complete command in ShellSim's persistent Bash-like shell. "
            "ShellSim uses built-in commands and does not execute arbitrary native programs. "
            "Output is capped at 128 KiB; redirect larger output to a file."
        ),
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "Shell command to run"}},
            "required": ["command"],
        },
    },
}
BASH_OUTPUT_LIMIT_BYTES = 128 * 1024


def _wait_seconds(arguments: dict) -> float:
    wait_ms = arguments.get("wait_ms", 10_000)
    if not isinstance(wait_ms, int) or wait_ms < 0 or wait_ms > 30_000:
        raise ValueError("wait_ms must be an integer between 0 and 30000")
    return wait_ms / 1000


def _shell_result(update: ShellUpdate) -> str:
    return json.dumps(
        {
            "output": update.output.decode(errors="replace").replace("\r\n", "\n"),
            "status": update.status.value,
            "exit_code": update.exit_code,
            "truncated": update.truncated,
        }
    )


class BashAgent(BaseAgent):
    """Call an OpenAI-compatible local endpoint from the Harbor process."""

    def __init__(self, *args, base_url: str, api_key: str = "unused", max_turns: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_turns = max_turns

    @staticmethod
    def name() -> str:
        return "marin-shellbox-bash"

    def version(self) -> str:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        if not isinstance(environment, BashSessionProvider):
            raise TypeError("BashAgent requires an environment with persistent Bash sessions")
        await environment.open_bash_session()

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        model = self.model_alias or self.model_name
        if model is None:
            raise ValueError("BashAgent requires model_name or model_alias")
        if not isinstance(environment, BashSessionProvider):
            raise TypeError("BashAgent requires an environment with persistent Bash sessions")
        shell = await environment.open_bash_session()
        tools = [BASH_TOOL if shell.interactive else SHELLSIM_BASH_TOOL]
        messages: list[dict] = [
            {"role": "system", "content": "Solve the task using the Bash tool. Say when finished."},
            {"role": "user", "content": instruction},
        ]
        async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
            for _ in range(self.max_turns):
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": model, "messages": messages, "tools": tools},
                )
                response.raise_for_status()
                body = response.json()
                usage = body.get("usage", {})
                context.n_input_tokens = (context.n_input_tokens or 0) + usage.get("prompt_tokens", 0)
                context.n_output_tokens = (context.n_output_tokens or 0) + usage.get("completion_tokens", 0)
                message = body["choices"][0]["message"]
                messages.append(message)
                calls = message.get("tool_calls") or []
                if not calls:
                    context.metadata = {
                        "final_message": message.get("content", ""),
                        "tool_calls": sum(m["role"] == "tool" for m in messages),
                    }
                    return
                for call in calls:
                    name = call["function"]["name"]
                    arguments = json.loads(call["function"]["arguments"])
                    if name == "Bash":
                        result = _shell_result(await _bash_action(shell, arguments))
                    else:
                        raise ValueError(f"Unsupported tool: {name}")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": result,
                        }
                    )
        raise TurnCapExhaustedError(f"BashAgent reached {self.max_turns} turns")


async def _bash_action(shell: ShellSession, arguments: dict) -> ShellUpdate:
    actions = {name for name in ("command", "input", "signal") if name in arguments}
    if len(actions) > 1:
        raise ValueError("Bash accepts one of command, input, or signal per call")
    wait = _wait_seconds(arguments)
    if "command" in arguments:
        return await shell.execute(arguments["command"], wait=wait, output_limit_bytes=BASH_OUTPUT_LIMIT_BYTES)
    if not shell.interactive:
        raise ValueError("ShellSim Bash requires command")
    if "input" in arguments:
        await shell.write(arguments["input"].encode())
    elif "signal" in arguments:
        if arguments["signal"] != "interrupt":
            raise ValueError(f"Unsupported Bash signal: {arguments['signal']}")
        await shell.interrupt()
    return await shell.read(wait=wait, output_limit_bytes=BASH_OUTPUT_LIMIT_BYTES)
