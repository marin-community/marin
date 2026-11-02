"""Never send an unlabelled Pi revision to the student model endpoint."""

import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest

from harbor.agents.installed.pi_literal_recorder import RECORDER_SCRIPT


def test_private_critique_uses_request_correlation_not_physical_trial_identity(
    tmp_path: Path,
) -> None:
    teacher_correlations: list[str | None] = []

    class Student(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.dumps(
                {
                    "id": "student-1",
                    "model": "student",
                    "prompt_token_ids": [1],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "29"},
                            "token_ids": [2],
                            "logprobs": {"token_logprobs": [-0.2]},
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    class Teacher(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            teacher_correlations.append(self.headers.get("x-ot-trial-id"))
            body = json.dumps(
                {
                    "id": "teacher-1",
                    "model": "teacher",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Revise to 30"},
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    student = ThreadingHTTPServer(("127.0.0.1", 0), Student)
    teacher = ThreadingHTTPServer(("127.0.0.1", 0), Teacher)
    threads = [
        threading.Thread(target=server.serve_forever) for server in (student, teacher)
    ]
    for thread in threads:
        thread.start()
    (tmp_path / "recorder.mjs").write_text(RECORDER_SCRIPT)
    (tmp_path / "harness.mjs").write_text(
        "import { startRecorder } from './recorder.mjs';\n"
        "const recorder = await startRecorder();\n"
        "console.log(recorder.url);\n"
        "process.stdin.once('data', async () => {\n"
        "  try { await recorder.requestCritique([]); console.log('accepted'); }\n"
        "  catch { console.log('rejected'); }\n"
        "  recorder.server.close();\n"
        "});\n"
    )
    process = subprocess.Popen(
        [shutil.which("node") or "node", str(tmp_path / "harness.mjs")],
        env={
            **os.environ,
            "HARBOR_PI_UPSTREAM_BASE": f"http://127.0.0.1:{student.server_port}/v1",
            "HARBOR_PI_CRITIQUE_PLAN": json.dumps(
                {
                    "teacher_model": "teacher",
                    "teacher_api_base": f"http://127.0.0.1:{teacher.server_port}/v1",
                    "principle": "Correct the number",
                    "trial_id": "physical-trial",
                }
            ),
            "HARBOR_PI_CORRELATION_ID": "attempt-correlation",
        },
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        url = f"{process.stdout.readline().strip()}/chat/completions"
        request = Request(
            url,
            data=json.dumps(
                {
                    "model": "student",
                    "messages": [{"role": "user", "content": "solve"}],
                    "stream": False,
                }
            ).encode(),
            headers={"x-ot-trial-id": "attempt-correlation"},
        )
        with urlopen(request, timeout=5) as response:
            assert response.status == 200
        stdout, _ = process.communicate("go\n", timeout=10)
        assert process.returncode == 0 and stdout.strip() == "accepted"
        assert teacher_correlations == ["attempt-correlation"]
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=10)
        for server in (student, teacher):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=10)


def test_second_model_request_is_blocked_until_feedback_arrives(tmp_path: Path) -> None:
    model_requests: list[dict] = []

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            model_requests.append(
                json.loads(self.rfile.read(int(self.headers["content-length"])))
            )
            body = json.dumps(
                {
                    "id": "sample-1",
                    "model": "student",
                    "prompt_token_ids": [1],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "draft"},
                            "token_ids": [2],
                            "logprobs": {"token_logprobs": [-0.2]},
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=upstream.serve_forever)
    thread.start()
    (tmp_path / "recorder.mjs").write_text(RECORDER_SCRIPT)
    (tmp_path / "harness.mjs").write_text(
        "import { startRecorder } from './recorder.mjs';\n"
        "const recorder = await startRecorder();\n"
        "console.log(recorder.url);\n"
        "process.stdin.once('data', () => recorder.server.close());\n"
    )
    process = subprocess.Popen(
        [shutil.which("node") or "node", str(tmp_path / "harness.mjs")],
        env={
            **os.environ,
            "HARBOR_PI_UPSTREAM_BASE": f"http://127.0.0.1:{upstream.server_port}/v1",
            "HARBOR_PI_CRITIQUE_PLAN": json.dumps(
                {
                    "teacher_model": "teacher",
                    "teacher_api_base": "http://teacher:8080/v1",
                    "principle": "Be precise",
                    "trial_id": "trial-1",
                }
            ),
        },
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        recorder_url = process.stdout.readline().strip()
        first = {
            "model": "student",
            "messages": [{"role": "user", "content": "solve"}],
            "stream": False,
        }
        url = f"{recorder_url}/chat/completions"
        with urlopen(
            Request(url, data=json.dumps(first).encode()), timeout=5
        ) as response:
            assert response.status == 200
        with pytest.raises((OSError, URLError)):
            urlopen(
                Request(
                    url,
                    data=json.dumps(
                        {
                            **first,
                            "messages": [
                                *first["messages"],
                                {"role": "assistant", "content": "draft"},
                            ],
                        }
                    ).encode(),
                ),
                timeout=5,
            )
        assert len(model_requests) == 1
    finally:
        process.terminate()
        process.communicate(timeout=10)
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=10)
