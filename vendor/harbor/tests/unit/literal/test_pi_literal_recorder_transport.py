"""The trial-local Pi recorder must relay streamed tool calls unchanged."""

import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from harbor.agents.installed.pi_in_memory_capture import EXTENSION_SCRIPT
from harbor.agents.installed.pi_literal_recorder import RECORDER_SCRIPT


@pytest.mark.parametrize("actor", [None, "teacher"])
@pytest.mark.parametrize("api_prefix", ["", "/proxy/t/redacted/native-agent"])
@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize("logprobs_available", [False, True])
def test_recording_relay_captures_fragmented_stream_and_served_request(
    tmp_path: Path,
    actor: str | None,
    api_prefix: str,
    verbose: bool,
    logprobs_available: bool,
) -> None:
    requests = []
    chunks = [
        {
            "id": "sample-1",
            "model": "tiny",
            "prompt_token_ids": [1, 2, 3],
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "provider_specific_fields": {"token_ids": [9]},
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": '{"command":',
                                },
                            }
                        ],
                    },
                    **(
                        {"logprobs": {"token_logprobs": [-0.25]}}
                        if logprobs_available
                        else {}
                    ),
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "sample-1",
            "model": "tiny",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '"echo 1"}'}}
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
    ]
    stream = (
        "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        + "data: [DONE]\n\n"
    ).encode()

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            assert self.path == f"{api_prefix}/v1/chat/completions"
            requests.append(
                (
                    json.loads(self.rfile.read(int(self.headers["content-length"]))),
                    self.headers,
                )
            )
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            # Split inside both JSON and SSE framing, as a real HTTP stream can.
            for chunk in (stream[:13], stream[13:48], stream[48:-8], stream[-8:]):
                self.wfile.write(chunk)
                self.wfile.flush()

        def log_message(self, *args: object) -> None:
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=upstream.serve_forever)
    thread.start()
    script_path = tmp_path / "recorder.mjs"
    script_path.write_text(RECORDER_SCRIPT)
    harness = tmp_path / "harness.mjs"
    harness.write_text(
        "import { startRecorder } from './recorder.mjs';\n"
        "const recorder = await startRecorder();\n"
        "console.log(recorder.url);\n"
        "process.stdin.once('data', () => {\n"
        "  recorder.recordObservation({toolCallId: 'call_1'}, [{type: 'text', text: 'ok'}]);\n"
        "  try { console.log(JSON.stringify({records: recorder.records, trajectory: recorder.buildTrajectory()})); }\n"
        "  catch (error) { console.log(JSON.stringify({records: recorder.records, error: error.constructor.name})); }\n"
        "  recorder.server.close();\n"
        "});\n"
    )
    process = subprocess.Popen(
        [shutil.which("node") or "node", str(harness)],
        env={
            **os.environ,
            "HARBOR_PI_UPSTREAM_BASE": f"http://127.0.0.1:{upstream.server_port}{api_prefix}/v1",
            "HARBOR_PI_TRIAL_ID": "trial-1",
            "HARBOR_PI_POLICY_STEP": "3",
            "HARBOR_PI_SANDBOX_DIAGNOSTICS": "1" if verbose else "0",
            "HARBOR_PI_SANDBOX_DIAGNOSTICS_PATH": str(
                tmp_path / "pi-sandbox-diagnostics.jsonl"
            ),
            **({"HARBOR_PI_CAPTURE_ACTOR": actor} if actor else {}),
        },
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        recorder_url = process.stdout.readline().strip()
        assert recorder_url.startswith("http://127.0.0.1:"), recorder_url
        body = json.dumps(
            {
                "model": "tiny",
                "messages": [{"role": "user", "content": "go"}],
                "stream": True,
            }
        ).encode()
        request = Request(
            f"{recorder_url}/chat/completions",
            data=body,
            headers={"content-type": "application/json", "x-ot-trial-id": "trial-1"},
        )
        with urlopen(request, timeout=10) as response:
            assert response.read() == stream
        if actor == "teacher":
            assert "logprobs" not in requests[0][0]
            assert "top_logprobs" not in requests[0][0]
        else:
            assert requests[0][0]["logprobs"] is True
            assert requests[0][0]["top_logprobs"] == 1
        assert requests[0][0]["return_token_ids"] is True
        assert requests[0][1]["x-ot-trial-id"] == "trial-1"
        assert process.stdin is not None
        process.stdin.write("go\n")
        process.stdin.flush()
        output = json.loads(process.stdout.readline())
        [record] = output["records"]
        if actor is None and not logprobs_available:
            assert output["error"] == "Error"
            assert record["literal"]["logprobs"] is None
            return
        assert output["trajectory"]["trial_id"] == "trial-1"
        assert output["trajectory"]["policy_step"] == 3
        [turn] = output["trajectory"]["turns"]
        assert turn["observations"] == [
            {
                "tool_call_id": "call_1",
                "content": "ok",
                "authored_by": actor or "student",
            }
        ]
        assert turn["executed"]["assistant"] == record["literal"]["assistant_message"]
        assert not (tmp_path / "literal.jsonl").exists()
        assert record["literal"]["prompt_token_ids"] == [1, 2, 3]
        assert record["literal"]["completion_token_ids"] == [9]
        assert record["literal"]["logprobs"] == (
            [-0.25] if logprobs_available else None
        )
        assert record["literal"]["assistant_message"]["tool_calls"][0]["function"] == {
            "name": "bash",
            "arguments": '{"command":"echo 1"}',
        }
        assert record["request"]["messages"] == [{"role": "user", "content": "go"}]
        assert record.get("actor") == actor
        diagnostic_file = tmp_path / "pi-sandbox-diagnostics.jsonl"
        assert diagnostic_file.exists() is verbose
        if verbose:
            diagnostics = [
                json.loads(line) for line in diagnostic_file.read_text().splitlines()
            ]
            assert [item["phase"] for item in diagnostics] == [
                "upstream_status",
                "literal_evidence",
            ]
            assert diagnostics[-1]["completion_token_count"] == 1
            assert diagnostics[-1]["logprob_count"] == (
                1 if logprobs_available else None
            )
            assert "redacted" not in diagnostic_file.read_text()
            assert "echo 1" not in diagnostic_file.read_text()
    finally:
        process.terminate()
        process.communicate(timeout=10)
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=10)


def test_capture_extension_closes_server_when_evidence_is_invalid(
    tmp_path: Path,
) -> None:
    """A failed in-memory collation must not pin the Pi process until trial timeout."""
    package_root = tmp_path / "package"
    ai_package = package_root / "node_modules/@earendil-works/pi-ai"
    api_dir = ai_package / "dist/api"
    api_dir.mkdir(parents=True)
    (ai_package / "package.json").write_text('{"type":"module"}')
    (api_dir / "openai-completions.lazy.js").write_text(
        "export function openAICompletionsApi() { return { streamSimple() {} }; }\n"
    )
    (tmp_path / "recorder.mjs").write_text(RECORDER_SCRIPT)
    (tmp_path / "capture.mjs").write_text(
        EXTENSION_SCRIPT.replace(
            "'/tmp/harbor-pi-literal-recorder.mjs'", "'./recorder.mjs'"
        )
    )
    (tmp_path / "check.mjs").write_text(
        "import capture from './capture.mjs';\n"
        "const events = new Map();\n"
        "await capture({ registerProvider() {}, on(name, callback) { events.set(name, callback); } });\n"
        "try {\n"
        "  await events.get('agent_end')();\n"
        "  throw new Error('invalid trajectory unexpectedly accepted');\n"
        "} catch (error) {\n"
        "  if (!String(error).includes('Pi trajectory must include a sampled turn')) throw error;\n"
        "  console.log('incomplete evidence rejected');\n"
        "}\n"
    )
    result = subprocess.run(
        [shutil.which("node") or "node", str(tmp_path / "check.mjs")],
        env={
            **os.environ,
            "HARBOR_PI_PACKAGE_ROOT": str(package_root),
            "HARBOR_PI_PROVIDER": "hosted_vllm",
            "HARBOR_PI_UPSTREAM_BASE": "http://127.0.0.1:1/v1",
            "HARBOR_PI_SANDBOX_DIAGNOSTICS": "1",
            "HARBOR_PI_SANDBOX_DIAGNOSTICS_PATH": str(
                tmp_path / "pi-sandbox-diagnostics.jsonl"
            ),
        },
        text=True,
        capture_output=True,
        timeout=5,
        check=True,
    )
    assert result.stdout.strip() == "incomplete evidence rejected"
    diagnostics = [
        json.loads(line)
        for line in (tmp_path / "pi-sandbox-diagnostics.jsonl").read_text().splitlines()
    ]
    assert diagnostics == [
        {
            "source": "recorder",
            "phase": "capture_failed",
            "turn_count": 0,
            "missing_prompt_ids": False,
            "missing_completion_ids": False,
            "missing_logprobs": False,
            "missing_assistant": False,
        }
    ]
