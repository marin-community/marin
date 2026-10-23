# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read prepared Nemotron pivot candidates as pass-rate rows.

A candidate is one expert turn: the conversation up to it in ``prompt``, the source request options in
``extra_info.nemotron_ultra.request_json``, and the expert action in ``record_json``. SWE and Terminal
candidates share this layout and differ only in how a reply is graded. Requests mirror MarinSkyRL's
chat path (``model_clients.DirectModelClient._chat_options``) so pass rates are measured on the
distribution training samples from. OpenHands candidates reuse the row identity, order, and message
cleanup here.
"""

import json
from dataclasses import dataclass
from typing import Any

from verifyit.adapters.nemotron_pivot import grade_terminus, grade_tool_call
from verifyit.grade import Reward


def chat_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Responses API function tools to chat-completions tools."""
    if any(tool["type"] != "function" for tool in tools):
        raise ValueError("only function tools are supported")
    return [
        {"type": "function", "function": {key: value for key, value in tool.items() if key != "type"}} for tool in tools
    ]


def without_nulls(message: dict[str, Any]) -> dict[str, Any]:
    """Drop the null fields Parquet adds to messages that never had them."""
    return {key: value for key, value in message.items() if value is not None}


def row_id(row: dict[str, Any]) -> str:
    return row["extra_info"]["source_id"]


def sort_key(row: dict[str, Any]) -> tuple:
    # Turns of one trajectory extend each other's prompt, so neighbors reuse the prefix cache.
    return row["extra_info"]["trajectory_id"], row["extra_info"]["index"]


def chat_request(row: dict[str, Any]) -> dict[str, Any]:
    options = json.loads(row["extra_info"]["nemotron_ultra"]["request_json"])
    # Drop Responses-only options; the pass-rate sampling config sets the output limit.
    body = {key: value for key, value in options.items() if key not in ("input", "max_output_tokens")}
    if tools := body.pop("tools", None):
        body["tools"] = chat_tools(tools)
    body["messages"] = [without_nulls(message) for message in row["prompt"]]
    return body


def source_record(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(row["extra_info"]["nemotron_ultra"]["record_json"])


@dataclass(frozen=True)
class PivotToolCallTask:
    """Tool-call pivots graded by argument comparison (the SWE release)."""

    components = ("tool_name", "nemo", "exact")
    row_id = staticmethod(row_id)
    sort_key = staticmethod(sort_key)
    request = staticmethod(chat_request)

    def grade(self, row: dict[str, Any], message: dict[str, Any]) -> Reward:
        return grade_tool_call(source_record(row)["expected_action"], message)


@dataclass(frozen=True)
class PivotTerminalTask:
    """Terminus-2 JSON pivots graded by keystroke similarity (the Terminal release)."""

    components = ("schema_completion", "exact_commands", "string_90")
    row_id = staticmethod(row_id)
    sort_key = staticmethod(sort_key)
    request = staticmethod(chat_request)

    def grade(self, row: dict[str, Any], message: dict[str, Any]) -> Reward:
        record = source_record(row)
        return grade_terminus(record["expected_answer"], message.get("content") or "", record.get("threshold"))


PIVOT_TOOL_CALL = PivotToolCallTask()
PIVOT_TERMINAL = PivotTerminalTask()
