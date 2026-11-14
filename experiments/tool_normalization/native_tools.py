# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay proposed calls through OpenCode's native tools inside an actor sandbox.

The loopback server is a deterministic transport, not a model or an audit. The
caller supplies an already generated sequence; OpenCode supplies tool results.
Run this program inside the isolated Linux executor for candidate traces.
"""

import argparse
import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def observations(calls: list[dict], events: list[dict]) -> list[dict]:
    """Bind native observations to exactly the proposed calls, in order."""
    if len(calls) != len(events):
        raise ValueError("Incomplete native execution")
    messages = []
    for index, (call, event) in enumerate(zip(calls, events, strict=True)):
        part = event["part"]
        state = part["state"]
        if (part["tool"], part["callID"], state["input"]) != (call["name"], f"replay_{index}", call["arguments"]):
            raise ValueError("Native execution differs from proposed call")
        if state["status"] not in ("completed", "error"):
            raise ValueError("Nonterminal native execution")
        metadata = state.get("metadata", {})
        if metadata.get("truncated", False):
            display = metadata.get("display", {})
            # A complete requested page is normal Read behavior, not a lost execution response.
            paginated = (
                call["name"] == "read"
                and "limit" in call["arguments"]
                and display.get("type") == "file"
                and display["lineEnd"] - display["lineStart"] + 1 == call["arguments"]["limit"]
                and f"Use offset={display['lineEnd'] + 1} to continue." in state["output"]
                and "Output capped at" not in state["output"]
                and "line truncated to" not in state["output"]
            )
            if not paginated:
                raise ValueError("Truncated native observation")
        messages.append(
            {
                "role": "tool",
                "name": part["tool"],
                "tool_call_id": part["callID"],
                "content": state["output"] if state["status"] == "completed" else state["error"],
            }
        )
    return messages


def replay(binary: Path, calls: list[dict], workspace: Path, evidence: Path, timeout: int) -> dict:
    evidence.mkdir(parents=True, exist_ok=False)
    requests = []
    failures = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            # Full requests are persisted below; omit the HTTP server's stderr log.
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            available = {tool["function"]["name"] for tool in request.get("tools", [])}
            observed = [message for message in request["messages"] if message["role"] == "tool"]
            index = len(observed)
            delta = {"role": "assistant", "content": "Replay complete."}
            finish = "stop"
            if available and index < len(calls):
                call = calls[index]
                if call["name"] not in available:
                    failures.append({"unavailable_tool": call["name"], "available": sorted(available)})
                else:
                    delta = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": f"replay_{index}",
                                "type": "function",
                                "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])},
                            }
                        ],
                    }
                    finish = "tool_calls"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for content, reason in [(delta, None), ({}, finish)]:
                chunk = {
                    "id": "native-tool-replay",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "replay",
                    "choices": [{"index": 0, "delta": content, "finish_reason": reason}],
                }
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = {
        "provider": {
            "replay": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Deterministic replay",
                "options": {"baseURL": f"http://127.0.0.1:{server.server_port}/v1", "apiKey": "inert"},
                "models": {"replay": {"name": "replay", "limit": {"context": 262144, "output": 8192}}},
            }
        },
        "model": "replay/replay",
        "small_model": "replay/replay",
        "permission": "allow",
        "autoupdate": False,
        "agent": {"build": {"steps": len(calls) + 3}},
    }
    config_path = evidence / "opencode.json"
    config_path.write_text(json.dumps(config))
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(evidence / "home"),
        "XDG_CONFIG_HOME": str(evidence / "config"),
        "XDG_DATA_HOME": str(evidence / "data"),
        "XDG_CACHE_HOME": str(evidence / "cache"),
        "OPENCODE_CONFIG": str(config_path),
        "OPENCODE_DISABLE_AUTOUPDATE": "true",
        "OPENCODE_DISABLE_MODELS_FETCH": "true",
    }
    try:
        result = subprocess.run(
            [
                str(binary),
                "run",
                "--pure",
                "--format",
                "json",
                "--model",
                "replay/replay",
                "Execute the supplied calls in order.",
            ],
            cwd=workspace,
            env=env,
            capture_output=True,
            timeout=timeout,
        )
        (evidence / "events.jsonl").write_bytes(result.stdout)
        (evidence / "stderr.txt").write_bytes(result.stderr)
        events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith(b"{")]
        observed = [event for event in events if event.get("type") == "tool_use"]
        report = {
            "exit_code": result.returncode,
            "tool_events": observed,
            "dispatch_errors": failures,
            "planned_calls": len(calls),
            "observed_calls": len(observed),
        }
        (evidence / "report.json").write_text(json.dumps(report, indent=2))
        # Return raw native evidence even when validation fails; the host validates after saving it.
        return report
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        (evidence / "requests.json").write_text(json.dumps(requests, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--calls", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--timeout", type=int, required=True)
    args = parser.parse_args()
    report = replay(args.binary, json.loads(args.calls.read_text()), args.workspace, args.evidence, args.timeout)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
