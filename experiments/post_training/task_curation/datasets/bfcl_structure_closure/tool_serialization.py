# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render structural tool cases with the tokenizer's native tool format."""

import json
import re
from dataclasses import replace
from typing import cast

from taskcompendium.models import FunctionDefinition
from transformers import PreTrainedTokenizerFast
from verifyit.tool_call_markup import TOOL_CLOSE, TOOL_OPEN, ToolCallStatus, tool_call_regions

from experiments.post_training.task_curation.datasets.bfcl_structure_closure.cases import (
    CLOSE_REQUIRED,
    CONTINUE_REQUIRED,
    OPENER_RETENTION,
    REFERENCE_SEPARATOR,
    RESULT_TOOL_NAME,
    VALUE_CONTINUE_REQUIRED,
    ClosureCase,
)

RESULT_TOOL = FunctionDefinition(
    name=RESULT_TOOL_NAME,
    description="Record the result for a task.",
    parameters={
        "type": "object",
        "properties": {"task": {"type": "string"}, "result": {"type": "array"}},
        "required": ["task", "result"],
    },
)


def _native_tool_text(body: dict, tokenizer: PreTrainedTokenizerFast, instruction: str) -> str:
    """Read the rendered call without interpreting quoted markers as boundaries."""
    rendered = cast(
        str,
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": body}]},
            ],
            tools=[{"type": "function", "function": RESULT_TOOL.model_dump(exclude_none=True)}],
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


def native_tool_case(case: ClosureCase, tokenizer: PreTrainedTokenizerFast) -> ClosureCase:
    """Keep known tool content and control boundaries in the tokenizer's tool format."""
    body = json.loads(case.expected.removeprefix(TOOL_OPEN).removesuffix(TOOL_CLOSE))
    expected = _native_tool_text(body, tokenizer, case.instruction)
    if case.control == CLOSE_REQUIRED:
        cut = len(expected) - len(TOOL_CLOSE)
    elif case.control == OPENER_RETENTION:
        cut = expected.index("{")
    elif case.control == CONTINUE_REQUIRED:
        cut = expected.index(case.task_id) + max(1, len(case.task_id) // 2)
    elif case.control == VALUE_CONTINUE_REQUIRED:
        cut = expected.index('"task":') + len('"task":')
    else:
        raise ValueError(f"Unsupported native tool control: {case.control}")
    instruction = case.instruction
    if case.control != CLOSE_REQUIRED:
        lead, reference = instruction.rsplit(REFERENCE_SEPARATOR, 1)
        if reference != case.expected:
            raise ValueError("Instruction reference differs from the case reference")
        instruction = lead + REFERENCE_SEPARATOR + expected
    return replace(case, prefix=expected[:cut], expected=expected, instruction=instruction)
