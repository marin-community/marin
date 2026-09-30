"""The diagnostic replays one immutable transport request without tool execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_context_budget_probe import (
    REQUEST_SHA,
    canonical,
    load_input,
    trace_metadata,
)


def _input() -> Path:
    return Path(__file__).resolve().parents[1] / "data/context-budget-probe-001/request.json"


def test_frozen_input_exact_request_identity() -> None:
    request, input_sha = load_input(_input())
    assert hashlib.sha256(canonical(request)).hexdigest() == REQUEST_SHA
    assert input_sha == hashlib.sha256(_input().read_bytes()).hexdigest()
    assert len(request["messages"]) == 178
    assert len(request["tools"]) == 1


def test_input_tamper_fails_before_transport(tmp_path: Path) -> None:
    value = json.loads(_input().read_text())
    value["request"]["messages"][0]["content"] += " changed"
    path = tmp_path / "request.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="request hash differs"):
        load_input(path)


def test_trace_rejects_prompt_mutation_and_extracts_metadata(tmp_path: Path) -> None:
    request, _ = load_input(_input())
    trace = tmp_path / "glm-requests.jsonl"
    first = {"request": request, "http_status": 400, "error_category": "context_length"}
    second_request = {**request, "max_tokens": None}
    second = {
        "request": second_request,
        "finish_reason": "tool_calls",
        "usage": {"prompt_tokens": 229000, "completion_tokens": 100},
        "message": {"role": "assistant", "tool_calls": [{"id": "ignored"}]},
    }
    trace.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    metadata = trace_metadata(trace, request)
    assert [record["max_tokens"] for record in metadata] == [32768, None]
    assert metadata[1]["finish_reason"] == "tool_calls"
    second_request["messages"] = [{"role": "user", "content": "different"}]
    trace.write_text(json.dumps(second) + "\n")
    with pytest.raises(ValueError, match="altered frozen prompt"):
        trace_metadata(trace, request)
