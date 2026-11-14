# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve the native client's advertised context inside the production worker launch."""

import json
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation import difficulty_campaign


def test_native_worker_keeps_full_completion_at_protocol_input_limit(tmp_path, monkeypatch):
    class Models(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            assert self.path == "/v1/models"
            body = json.dumps({"data": [{"id": "fixture", "max_model_len": 65536}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Models)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    real_run = subprocess.run
    results = []

    def native_worker(command, *, check, env):
        # Replace the full campaign boundary with its real native client in a fresh interpreter.
        program = """
import json, sys
from harbor.llms.openai_llm import OpenAILLM
from harbor.llms.vllm_tokenization import completion_budget_from_count
model = OpenAILLM('openai/fixture', api_key='fixture', api_base=sys.argv[1], model_info={
    'max_input_tokens': 65536, 'max_output_tokens': 16384})
context = model.get_model_context_limit()
print(json.dumps([context, completion_budget_from_count(
    prompt_tokens=49152, context_window=context, max_output_tokens=model.get_model_output_limit())]))
"""
        result = real_run(
            [*command[: command.index("python") + 1], "-c", program, endpoint],
            check=check,
            capture_output=True,
            text=True,
            env=env,
        )
        results.append(json.loads(result.stdout))
        return result

    monkeypatch.setattr(difficulty_campaign, "forwarded_secrets", lambda: {})
    monkeypatch.setattr(difficulty_campaign, "require_secret", lambda name: "fixture")
    monkeypatch.setattr(difficulty_campaign.subprocess, "run", native_worker)
    try:
        difficulty_campaign.run_trial_worker(
            tmp_path / "prepared.json",
            tmp_path / "candidate.json",
            tmp_path,
            StoragePath(str(tmp_path / "output")),
            "fixture",
            recovery_arguments=[],
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    assert results == [[65536, 16384]]
