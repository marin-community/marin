# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small external Harbor agent that routes Bash tool calls to the environment."""

import json

import httpx
from harbor.agents.base import BaseAgent, TurnCapExhaustedError
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from harbor_qemu.machine import BashSessionProvider, ShellUpdate

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "Bash",
        "description": (
            "Run a command in the persistent Bash shell. Directory, variables, functions, and jobs persist. "
            "Output is capped at 128 KiB; redirect larger output to a file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "wait_ms": {"type": "integer", "description": "Wait up to this many milliseconds; maximum 30000"},
            },
            "required": ["command"],
        },
    },
}

BASH_READ_TOOL = {
    "type": "function",
    "function": {
        "name": "BashRead",
        "description": "Read output and status from the current Bash command.",
        "parameters": {
            "type": "object",
            "properties": {
                "wait_ms": {"type": "integer", "description": "Wait up to this many milliseconds; maximum 30000"}
            },
        },
    },
}

BASH_INPUT_TOOL = {
    "type": "function",
    "function": {
        "name": "BashInput",
        "description": "Send text to the current interactive Bash command; include a newline to submit it.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
}

BASH_INTERRUPT_TOOL = {
    "type": "function",
    "function": {
        "name": "BashInterrupt",
        "description": "Send Ctrl-C to the current Bash foreground process group.",
        "parameters": {"type": "object", "properties": {}},
    },
}

BASH_TOOLS = [BASH_TOOL, BASH_READ_TOOL, BASH_INPUT_TOOL, BASH_INTERRUPT_TOOL]
SHELLSIM_BASH_TOOL = {
    **BASH_TOOL,
    "function": {
        **BASH_TOOL["function"],
        "description": (
            "Run one complete command in ShellSim's persistent Bash-like shell. "
            "ShellSim uses built-in commands and does not execute arbitrary native programs. "
            "Output is capped at 128 KiB; redirect larger output to a file."
        ),
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
        return "harbor-qemu-bash"

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
        tools = BASH_TOOLS if shell.interactive else [SHELLSIM_BASH_TOOL]
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
                        result = _shell_result(
                            await shell.execute(
                                arguments["command"],
                                wait=_wait_seconds(arguments),
                                output_limit_bytes=BASH_OUTPUT_LIMIT_BYTES,
                            )
                        )
                    elif name == "BashRead":
                        result = _shell_result(
                            await shell.read(wait=_wait_seconds(arguments), output_limit_bytes=BASH_OUTPUT_LIMIT_BYTES)
                        )
                    elif name == "BashInput":
                        await shell.write(arguments["text"].encode())
                        result = _shell_result(await shell.read(wait=0.5, output_limit_bytes=BASH_OUTPUT_LIMIT_BYTES))
                    elif name == "BashInterrupt":
                        await shell.interrupt()
                        result = _shell_result(await shell.read(wait=5, output_limit_bytes=BASH_OUTPUT_LIMIT_BYTES))
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
