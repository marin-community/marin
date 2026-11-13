"""Native-tool Terminus agent, adapted from krafton-ai/KIRA.

The original Apache-2.0 TerminusKira implementation subclasses Terminus2 but
copies its model loop. This built-in variant keeps only KIRA's tool protocol and
multimodal helper; lifecycle, context recovery, terminal execution, trajectory
handling, and timeouts remain owned by Harbor's current Terminus2.
"""

import base64
import json
import math
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harbor.agents.terminus_2.terminus_2 import (
    Command,
    EpisodeLoggingPaths,
    LLMInteractionResult,
    LLMRequestContext,
    Terminus2,
)
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.llms.base import LLMResponse, OutputLengthExceededError
from harbor.llms.chat import Chat
from harbor.models.trajectories import ToolCall


@dataclass(frozen=True)
class ImageReadRequest:
    file_path: str
    instruction: str


@dataclass(frozen=True)
class _NoTextParseResult:
    commands: tuple = ()
    is_task_complete: bool = False
    error: str = ""
    warning: str = "The truncated native-tool response contained no usable tool call."
    analysis: str = ""
    plan: str = ""


class _NativeToolOverflowParser:
    def parse_response(self, response: str) -> _NoTextParseResult:
        return _NoTextParseResult()

    def salvage_truncated_response(self, truncated_response: str) -> None:
        return None


_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "execute_commands",
            "description": "Execute a batch of commands in the terminal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis": {"type": "string"},
                    "plan": {"type": "string"},
                    "commands": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "keystrokes": {
                                    "type": "string",
                                    "description": "Exact tmux keystrokes; end shell commands with a newline.",
                                },
                                "duration": {
                                    "type": "number",
                                    "description": "Seconds to wait, at most 60.",
                                },
                            },
                            "required": ["keystrokes"],
                        },
                    },
                },
                "required": ["analysis", "plan", "commands"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": "Declare the task complete after verifying the result.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "image_read",
            "description": "Read and visually analyze an image file in the task environment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "image_read_instruction": {"type": "string"},
                },
                "required": ["file_path", "image_read_instruction"],
            },
        },
    },
]


def _tool_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call.get("function", {}).get("arguments", {})
    if isinstance(raw, str):
        value = json.loads(raw)
    else:
        value = raw
    if not isinstance(value, dict):
        raise ValueError("tool arguments must be a JSON object")
    return value


class TerminusKira(Terminus2):
    """Terminus2 with KIRA's native function-calling interface."""

    def __init__(self, *args, llm_call_kwargs: dict[str, Any] | None = None, **kwargs):
        call_kwargs = dict(llm_call_kwargs or {})
        call_kwargs.update(
            tools=_TOOLS,
            tool_choice="auto",
            parallel_tool_calls=False,
        )
        super().__init__(*args, llm_call_kwargs=call_kwargs, **kwargs)

    @staticmethod
    def name() -> str:
        return "terminus-kira"

    def version(self) -> str:
        return "2.0.0"

    def _get_parser(self) -> _NativeToolOverflowParser:
        return _NativeToolOverflowParser()

    def _get_prompt_template_path(self) -> Path:
        return Path(__file__).parent / "templates" / "terminus-kira.txt"

    def _get_error_response_type(self) -> str:
        return "response with a valid native tool call"

    def _get_completion_confirmation_message(self, terminal_output: str) -> str:
        return (
            f"Current terminal state:\n{terminal_output}\n\n"
            "Are you sure the task is complete? Verify the requested result and that "
            "only necessary state changed. If so, call task_complete again."
        )

    async def _recover_output_overflow(
        self,
        *,
        chat: Chat,
        prompt: str,
        error: OutputLengthExceededError,
        request_context: LLMRequestContext,
    ) -> LLMResponse:
        # A partial native tool call cannot be resolved safely. Preserve its
        # token evidence, but retain it as plain assistant text for recovery.
        error.partial_response.tool_calls = None
        return await super()._recover_output_overflow(
            chat=chat,
            prompt=prompt,
            error=error,
            request_context=request_context,
        )

    async def _handle_llm_interaction(
        self,
        chat: Chat,
        prompt: str,
        logging_paths: EpisodeLoggingPaths,
        original_instruction: str = "",
        session: TmuxSession | None = None,
    ) -> LLMInteractionResult:
        response = await self._query_llm(
            chat,
            prompt,
            LLMRequestContext(logging_paths, original_instruction, session),
        )
        commands: list[Command] = []
        image_read = None
        is_task_complete = False
        feedback: list[str] = []
        analysis = ""
        plan = ""
        tool_calls = response.tool_calls or []

        if not tool_calls:
            feedback.append("WARNINGS: Use a supplied tool to act on the task.")

        for tool_call in tool_calls:
            name = tool_call.get("function", {}).get("name", "")
            try:
                arguments = _tool_arguments(tool_call)
            except (json.JSONDecodeError, ValueError) as error:
                feedback.append(
                    f"ERROR: Invalid arguments for {name or 'tool'}: {error}"
                )
                continue

            if name == "execute_commands":
                analysis = str(arguments.get("analysis", ""))
                plan = str(arguments.get("plan", ""))
                raw_commands = arguments.get("commands", [])
                if not isinstance(raw_commands, list):
                    feedback.append("ERROR: execute_commands.commands must be an array")
                    continue
                for command in raw_commands:
                    if not isinstance(command, dict):
                        feedback.append(
                            "ERROR: execute_commands entries must be objects"
                        )
                        continue
                    try:
                        duration = float(command.get("duration", 1.0))
                    except (TypeError, ValueError):
                        feedback.append("ERROR: command duration must be a number")
                        continue
                    if not math.isfinite(duration):
                        feedback.append("ERROR: command duration must be finite")
                        continue
                    commands.append(
                        Command(
                            keystrokes=str(command.get("keystrokes", "")),
                            duration_sec=min(max(duration, 0), 60),
                        )
                    )
            elif name == "task_complete":
                is_task_complete = True
            elif name == "image_read":
                file_path = arguments.get("file_path")
                instruction = arguments.get("image_read_instruction")
                if isinstance(file_path, str) and isinstance(instruction, str):
                    image_read = ImageReadRequest(file_path, instruction)
                else:
                    feedback.append(
                        "ERROR: image_read requires string path and instruction"
                    )
            else:
                feedback.append(f"ERROR: Unknown tool {name!r}")

        return LLMInteractionResult(
            commands=commands,
            is_task_complete=is_task_complete,
            feedback="\n".join(feedback),
            analysis=analysis,
            plan=plan,
            response=response,
            native_tool_calls=tool_calls or None,
            custom_action=image_read,
        )

    async def _execute_interaction(
        self,
        interaction: LLMInteractionResult,
        session: TmuxSession,
    ) -> str:
        image_read = interaction.custom_action
        if not isinstance(image_read, ImageReadRequest):
            return await super()._execute_interaction(interaction, session)

        suffix = Path(image_read.file_path).suffix.lower()
        mime_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(suffix)
        if mime_type is None:
            return f"ERROR: Unsupported image format {suffix!r}."

        result = await session.environment.exec(
            f"base64 -w 0 -- {shlex.quote(image_read.file_path)}",
            timeout_sec=60,
        )
        if result.return_code != 0:
            return (
                f"ERROR: Could not read {image_read.file_path!r}: {result.stderr or ''}"
            )
        try:
            base64.b64decode(result.stdout or "", validate=True)
        except ValueError:
            return f"ERROR: Invalid base64 data returned for {image_read.file_path!r}."

        response = await self._llm.call(
            prompt=[
                {"type": "text", "text": image_read.instruction},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{result.stdout or ''}"
                    },
                },
            ],
            max_tokens=self._llm.get_model_output_limit(),
        )
        if self._chat is not None:
            self._chat.add_auxiliary_usage(response)
        return f"Image analysis for {image_read.file_path!r}:\n{response.content}"

    def _append_native_tool_results(
        self,
        chat: Chat,
        interaction: LLMInteractionResult,
        observation: str,
    ) -> None:
        if interaction.native_tool_calls:
            chat.append_tool_results(
                interaction.native_tool_calls,
                "Tool call completed; its result follows in the next user message.",
            )

    def _trajectory_tool_calls(
        self,
        interaction: LLMInteractionResult,
        episode: int,
    ) -> list[ToolCall] | None:
        if not interaction.native_tool_calls:
            return None
        result = []
        for tool_call in interaction.native_tool_calls:
            function = tool_call.get("function", {})
            try:
                arguments = _tool_arguments(tool_call)
            except (json.JSONDecodeError, ValueError):
                arguments = {"raw": function.get("arguments")}
            result.append(
                ToolCall(
                    tool_call_id=str(
                        tool_call.get("id") or f"call_{episode}_{len(result) + 1}"
                    ),
                    function_name=str(function.get("name") or "unknown"),
                    arguments=arguments,
                )
            )
        return result
