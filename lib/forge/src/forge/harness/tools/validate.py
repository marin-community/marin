# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import override

from forge.effects.context import Context
from forge.harness.protocol import ReturnRequest, ToolResult
from forge.harness.tools.base import ValidatedTool
from forge.harness.tools.submit import Gate, read_submission
from forge.model.llm import ToolCall
from forge.model.workpiece import Conversation, Submission


@dataclass(frozen=True, slots=True)
class ValidateRequest:
    submission: Submission


@dataclass(frozen=True, slots=True)
class ValidateArgs:
    pass


@dataclass(frozen=True, slots=True)
class ValidateTool(ValidatedTool[ValidateArgs]):
    gate: Gate
    task_path: str
    controls_path: str
    name = "validate"
    description = "Validate the current task and controls, then continue with the evidence."
    args_type = ValidateArgs

    @override
    async def run(self, ctx: Context, args: ValidateArgs) -> ToolResult | ReturnRequest[ValidateRequest]:
        result = await read_submission(ctx, self.task_path, self.controls_path, "", self.gate)
        return result if isinstance(result, ToolResult) else ReturnRequest("validate", ValidateRequest(result))


def pending_validate(session: Conversation) -> ToolCall | None:
    """The author's unanswered validate call, when validation is running as a tool."""
    answered = {message.tool_call_id for message in session.messages if message.role == "tool"}
    calls = (call for message in reversed(session.messages) for call in message.tool_calls)
    return next((call for call in calls if call.name == "validate" and call.id not in answered), None)
