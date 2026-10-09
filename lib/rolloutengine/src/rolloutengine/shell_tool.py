# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The shell tool that the default task session offers to tasks with the shell capability."""

import json
from typing import Any

from shellbox.machine import Result

SHELL_TOOL_NAME = "shell"


def shell_tool_definition() -> dict[str, Any]:
    """The chat-completions function definition of ``shell(command: string)``."""
    return {
        "type": "function",
        "function": {
            "name": SHELL_TOOL_NAME,
            "description": "Run a shell command in the task workspace. Files persist between commands.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    }


def shell_observation(result: Result) -> str:
    """The tool message content for one completed shell command."""
    return json.dumps(
        {
            "stdout": result.stdout.decode(errors="replace"),
            "stderr": result.stderr.decode(errors="replace"),
            "exit_code": result.exit_code,
            "reason": result.reason.value,
            "truncated": result.stdout_truncated or result.stderr_truncated,
        }
    )
