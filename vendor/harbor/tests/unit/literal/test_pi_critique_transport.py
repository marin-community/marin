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


def test_second_model_request_is_blocked_until_feedback_arrives(tmp_path: Path) -> None:
    model_requests: list[dict] = []

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            model_requests.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
            body = json.dumps({
                "id": "sample-1", "model": "student", "prompt_token_ids": [1],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "draft"},
                             "token_ids": [2], "logprobs": {"token_logprobs": [-0.2]}}],
            }).encode()
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
            "HARBOR_PI_CRITIQUE_PLAN": json.dumps({
                "teacher_model": "teacher", "teacher_api_base": "http://teacher:8080/v1",
                "principle": "Be precise", "trial_id": "trial-1",
            }),
        },
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert process.stdout is not None
        recorder_url = process.stdout.readline().strip()
        first = {"model": "student", "messages": [{"role": "user", "content": "solve"}], "stream": False}
        url = f"{recorder_url}/chat/completions"
        with urlopen(Request(url, data=json.dumps(first).encode()), timeout=5) as response:
            assert response.status == 200
        with pytest.raises((OSError, URLError)):
            urlopen(Request(url, data=json.dumps({
                **first, "messages": [*first["messages"], {"role": "assistant", "content": "draft"}],
            }).encode()), timeout=5)
        assert len(model_requests) == 1
    finally:
        process.terminate()
        process.communicate(timeout=10)
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=10)
