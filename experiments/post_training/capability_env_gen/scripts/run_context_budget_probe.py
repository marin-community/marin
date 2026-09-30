#!/usr/bin/env python3
"""Replay one frozen GLM request to test remaining-context transport fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

INPUT_SCHEMA = "capability-context-budget-probe-input-v1"
REPORT_SCHEMA = "capability-context-budget-probe-report-v1"
SOURCE_LOG_SHA = "6e05c4e686039d59b5bd66d6208863e8ff56bde94eda0f7fb744c2271b4e216f"
SOURCE_LINE_SHA = "8ad51679a75d2d8203cc431f48fe400dc9a6fce28731c86e86d9c99a969c808c"
REQUEST_SHA = "2f4f1ee5d2121aa81cecd5de33a3bf31a6f7179618474d517777b27c3176f0ec"


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def load_input(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("schema_version") != INPUT_SCHEMA:
        raise ValueError("invalid context-budget probe input")
    if (
        payload.get("source_log_sha256") != SOURCE_LOG_SHA
        or payload.get("source_line_number") != 89
        or payload.get("source_line_sha256") != SOURCE_LINE_SHA
        or payload.get("request_sha256") != REQUEST_SHA
        or payload.get("prior_http_status") != 400
        or payload.get("prior_error_category") != "context_length"
    ):
        raise ValueError("context-budget probe source identity differs")
    request = payload.get("request")
    if not isinstance(request, dict) or hashlib.sha256(canonical(request)).hexdigest() != REQUEST_SHA:
        raise ValueError("context-budget probe request hash differs")
    if (
        set(request) != {"model", "messages", "max_tokens", "temperature", "chat_template_kwargs", "tools"}
        or request["model"] != "glm-5.3"
        or request["max_tokens"] != 32768
        or request["temperature"] != 0.0
        or not isinstance(request["messages"], list)
        or len(request["messages"]) != 178
        or not isinstance(request["tools"], list)
        or len(request["tools"]) != 1
    ):
        raise ValueError("context-budget probe request shape differs")
    return request, hashlib.sha256(raw).hexdigest()


def trace_metadata(path: Path, original: dict) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_bytes().splitlines():
        value = json.loads(line)
        body = value.get("request")
        if not isinstance(body, dict):
            raise TypeError("GLM trace lacks request body")
        if {key: val for key, val in body.items() if key != "max_tokens"} != {
            key: val for key, val in original.items() if key != "max_tokens"
        }:
            raise ValueError("GLM trace altered frozen prompt or tools")
        records.append(
            {
                "record_sha256": hashlib.sha256(line).hexdigest(),
                "request_sha256": hashlib.sha256(canonical(body)).hexdigest(),
                "max_tokens": body["max_tokens"],
                "http_status": value.get("http_status"),
                "error_category": value.get("error_category"),
                "error_body_sha256": value.get("error_body_sha256"),
                "finish_reason": value.get("finish_reason"),
                "usage": value.get("usage"),
                "response_message_sha256": (
                    hashlib.sha256(canonical(value["message"])).hexdigest()
                    if "message" in value else None
                ),
            }
        )
    return records


def run_probe(input_path: Path, out: Path) -> dict:
    from capability_pipeline.runtime_agents import GLMChatAgent

    request, input_sha = load_input(input_path)
    out.mkdir(parents=True, exist_ok=False)
    raw = out / "raw"
    agent = GLMChatAgent(
        api_key_env="GLM_API_TOKEN",
        model_name=request["model"],
        max_tokens=request["max_tokens"],
        temperature=request["temperature"],
        chat_template_kwargs=request["chat_template_kwargs"],
        request_timeout=300,
        logs_dir=raw,
    )
    limits = list(agent.token_limits)
    if limits != [32768, 65536, 131072, None]:
        raise ValueError("default solver token limits changed")
    exception_type = None
    response = None
    try:
        response = agent._completion(request["messages"], request["tools"])
    except Exception as error:  # noqa: BLE001 - preserve the single remote outcome.
        exception_type = type(error).__name__
    trace = raw / "glm-requests.jsonl"
    records = trace_metadata(trace, request)
    first_exact = bool(records) and records[0]["request_sha256"] == REQUEST_SHA
    passed = (
        exception_type is None
        and len(records) == 2
        and first_exact
        and records[0]["http_status"] == 400
        and records[0]["error_category"] == "context_length"
        and records[0]["max_tokens"] == 32768
        and records[1]["http_status"] is None
        and records[1]["max_tokens"] is None
        and records[1]["finish_reason"] in {"stop", "tool_calls"}
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "state": "passed" if passed else "failed",
        "scope": "one exact frozen request; transport fallback only; no returned tool execution",
        "source_log_sha256": SOURCE_LOG_SHA,
        "source_line_sha256": SOURCE_LINE_SHA,
        "input_sha256": input_sha,
        "frozen_request_sha256": REQUEST_SHA,
        "default_token_limits": limits,
        "transport_exception_type": exception_type,
        "response_tool_call_count": len(response.get("tool_calls", [])) if isinstance(response, dict) else None,
        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest() if trace.is_file() else None,
        "requests": records,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        load_input(args.input)
        return 0
    if args.out is None:
        parser.error("--out is required unless --validate-only is set")
    return 0 if run_probe(args.input, args.out)["state"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
