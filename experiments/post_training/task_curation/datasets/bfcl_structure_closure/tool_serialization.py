# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render structural tool cases with the tokenizer's native tool format."""

import json
import re
from typing import cast

from taskcompendium.models import FunctionDefinition
from transformers import PreTrainedTokenizerFast
from verifyit.tool_call_markup import TOOL_CLOSE, TOOL_OPEN, ToolCallStatus, tool_call_regions

RESULT_TOOL_NAME = "write_result"


def result_tool() -> FunctionDefinition:
    return FunctionDefinition(
        name=RESULT_TOOL_NAME,
        description="Record the result for a task.",
        parameters={
            "type": "object",
            "properties": {"task": {"type": "string"}, "result": {"type": "array"}},
            "required": ["task", "result"],
        },
    )


def native_tool_text(task_id: str, answer: list[object], tokenizer: PreTrainedTokenizerFast) -> str:
    """Render a tool call and ignore delimiters quoted inside its arguments."""
    body = {"name": RESULT_TOOL_NAME, "arguments": {"task": task_id, "result": answer}}
    rendered = cast(
        str,
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": body}]},
            ],
            tools=[{"type": "function", "function": result_tool().model_dump(exclude_none=True)}],
            tokenize=False,
            enable_thinking=False,
        ),
    )
    for match in reversed(list(re.finditer(re.escape(TOOL_OPEN), rendered))):
        candidate = rendered[match.start() :]
        regions = tool_call_regions(candidate, initial_thinking=False)
        if not regions or regions[0].status != ToolCallStatus.VALID:
            continue
        call = candidate[: regions[0].end]
        if json.loads(call[len(TOOL_OPEN) : -len(TOOL_CLOSE)]) == body:
            return call
    raise ValueError("Native serialization did not preserve the tool call")
