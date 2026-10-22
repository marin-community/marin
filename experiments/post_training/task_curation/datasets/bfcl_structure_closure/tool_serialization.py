# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tool-call serialization shared by curated tasks and historical token exports."""

import hashlib
import json
import re
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import cast

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
    Split,
)
from experiments.post_training.task_curation.datasets.bfcl_structure_closure.prefixes import near_boundary_case


class ToolPrompt(StrEnum):
    COMPACT = "compact"
    NATIVE_DECLARED = "native-declared"


@dataclass(frozen=True)
class ToolPromptVariant:
    name: str
    case_id: str
    prefix: str
    expected: str
    declared: bool


def paired_tool_variants(
    case_id: str, compact_prefix: str, instruction: str, tokenizer: PreTrainedTokenizerFast
) -> list[ToolPromptVariant]:
    """Cross declaration and serialization while holding tool-call content fixed."""
    body = json.loads(compact_prefix.removeprefix(TOOL_OPEN))
    native_prefix = native_tool_text(body, tokenizer, instruction=instruction, tools=None).removesuffix(TOOL_CLOSE)
    return [
        ToolPromptVariant(
            name=f"{declaration}-{serialization}",
            case_id=hashlib.sha256(f"{case_id}/{declaration}-{serialization}".encode()).hexdigest(),
            prefix=prefix,
            expected=prefix + TOOL_CLOSE,
            declared=declaration == "declared",
        )
        for declaration in ("undeclared", "declared")
        for serialization, prefix in (("compact", compact_prefix), ("native", native_prefix))
    ]


def write_result_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": RESULT_TOOL_NAME,
            "description": "Record the result for a task.",
            "parameters": {
                "type": "object",
                "properties": {"task": {"type": "string"}, "result": {"type": "array"}},
                "required": ["task", "result"],
            },
        },
    }


def native_tool_text(
    body: dict, tokenizer: PreTrainedTokenizerFast, *, instruction: str, tools: list[dict] | None
) -> str:
    """Read the rendered call without interpreting quoted markers as boundaries."""
    rendered = cast(
        str,
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": body}]},
            ],
            tools=tools,
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
    expected = native_tool_text(body, tokenizer, instruction=case.instruction, tools=[write_result_tool()])
    advanced_training_prefix = False
    if case.control == CLOSE_REQUIRED:
        cut = len(expected) - len(TOOL_CLOSE)
    elif case.control == OPENER_RETENTION:
        cut = expected.index("{")
    elif case.control == CONTINUE_REQUIRED:
        original_cut = case.expected.index(case.task_id) + max(1, len(case.task_id) // 2)
        cut = expected.index(case.task_id) + max(1, len(case.task_id) // 2)
        advanced_training_prefix = case.split == Split.TRAIN and len(case.prefix) > original_cut
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
    converted = replace(case, prefix=expected[:cut], expected=expected, instruction=instruction)
    return near_boundary_case(converted) if advanced_training_prefix else converted
