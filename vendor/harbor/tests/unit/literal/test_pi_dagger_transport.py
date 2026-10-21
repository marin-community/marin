"""Pi's pre-action DAgger routing keeps private labels out of executed history."""

import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from harbor_config.models.trial.pi import PiDaggerPlan
from harbor.agents.installed.pi_literal_recorder import RECORDER_SCRIPT


@pytest.mark.parametrize(
    "protocol, executors",
    [
        ("turn_mixture", ["student", "teacher"]),
        ("prefix_takeover", ["student", "teacher"]),
    ],
)
def test_private_teacher_label_and_executed_actor(
    tmp_path: Path, protocol: str, executors: list[str]
) -> None:
    seen: dict[str, list[dict]] = {"student": [], "teacher": []}

    def server(actor: str) -> ThreadingHTTPServer:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                request = json.loads(
                    self.rfile.read(int(self.headers["content-length"]))
                )
                seen[actor].append(request)
                assert request["return_token_ids"] is True
                if actor == "teacher":
                    assert request["model"] == "teacher"
                    assert "logprobs" not in request and "top_logprobs" not in request
                else:
                    assert request["logprobs"] is True
                    assert request["top_logprobs"] == 1
                turn = len(request["messages"]) > 1
                delta = {"role": "assistant", "content": "done" if turn else actor}
                chunk = {
                    "id": f"{actor}-{len(seen[actor])}",
                    "model": actor,
                    "prompt_token_ids": [1, 2] if turn else [1],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                **delta,
                                "provider_specific_fields": {"token_ids": [8]},
                            },
                            **(
                                {"logprobs": {"token_logprobs": [-0.2]}}
                                if actor == "student"
                                else {}
                            ),
                            "finish_reason": "stop",
                        }
                    ],
                }
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                self.wfile.write(
                    f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
                )

            def log_message(self, *args: object) -> None:
                pass

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return httpd

    student, teacher = server("student"), server("teacher")
    plan = PiDaggerPlan.parse(
        {
            "protocol": protocol,
            "teacher_model": "teacher",
            "teacher_api_base": f"http://127.0.0.1:{teacher.server_port}/v1",
            "trial_id": "trial-1",
            "policy_step": 1,
            "executors": executors,
        }
    )
    (tmp_path / "recorder.mjs").write_text(RECORDER_SCRIPT)
    (tmp_path / "harness.mjs").write_text(
        "import {startRecorder} from './recorder.mjs';\n"
        "const value = await startRecorder(); console.log(value.url);\n"
        "process.stdin.once('data', () => {\n"
        " console.log(JSON.stringify({entries: value.records, turns: value.daggerTurns, trajectory: value.buildTrajectory()}));\n"
        " value.server.close();\n"
        "});\n"
    )
    process = subprocess.Popen(
        [shutil.which("node") or "node", str(tmp_path / "harness.mjs")],
        env={
            **os.environ,
            "HARBOR_PI_UPSTREAM_BASE": f"http://127.0.0.1:{student.server_port}/v1",
            "HARBOR_PI_DAGGER_PLAN": json.dumps(
                {**plan.__dict__, "executors": list(plan.executors)}
            ),
        },
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None and process.stdin is not None
        url = process.stdout.readline().strip()
        assert url.startswith("http://127.0.0.1:"), (
            process.stderr.read() if process.poll() is not None else ""
        )
        for messages, expected in [
            ([{"role": "user", "content": "act"}], "student"),
            (
                [
                    {"role": "user", "content": "act"},
                    {"role": "assistant", "content": "student"},
                ],
                "teacher",
            ),
        ]:
            request = Request(
                f"{url}/chat/completions",
                data=json.dumps(
                    {
                        "model": "student",
                        "messages": messages,
                        "stream": True,
                    }
                ).encode(),
                headers={"content-type": "application/json"},
            )
            with urlopen(request, timeout=5) as response:
                assert (
                    f'"content": "{expected if len(messages) == 1 else "done"}"'
                    in response.read().decode()
                )
        process.stdin.write("go\n")
        process.stdin.flush()
        result = json.loads(process.stdout.readline())
        assert [turn["actor"] for turn in result["turns"]] == executors
        hydrated = result["trajectory"]
        assert hydrated["schema_version"] == 1
        assert hydrated["planned_executors"] == executors
        assert [turn["actor"] for turn in hydrated["turns"]] == executors
        assert hydrated["turns"][0]["teacher_label"]["response_token_ids"] == [8]
        assert hydrated["turns"][0]["teacher_label"]["logprobs"] is None
        assert hydrated["turns"][0]["executed"]["logprobs"] == [-0.2]
        assert hydrated["turns"][1]["teacher_label"] == hydrated["turns"][1]["executed"]
        assert hydrated["turns"][1]["executed"]["logprobs"] is None
        assert all(turn["observations"] == [] for turn in hydrated["turns"])
        assert [entry["actor"] for entry in result["entries"]] == executors
        assert len(seen["teacher"]) == 2
        assert len(seen["student"]) == 1
        assert (
            result["turns"][0]["teacher_label"]["literal"]["assistant_message"][
                "content"
            ]
            == "teacher"
        )
        assert (
            result["entries"][0]["literal"]["assistant_message"]["content"] == "student"
        )
        assert (
            result["turns"][1]["teacher_label"]["request_id"]
            == result["entries"][1]["request_id"]
        )
        assert seen["student"][0]["messages"] == [{"role": "user", "content": "act"}]
        assert not (tmp_path / "literal.jsonl").exists()
    finally:
        process.terminate()
        process.communicate(timeout=10)
        student.shutdown()
        teacher.shutdown()
        student.server_close()
        teacher.server_close()
