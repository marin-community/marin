# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Final traffic receipts include upstream replies that outlive their task."""

import io
import json
import tarfile
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue
from threading import Event, Thread

import httpx
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation import difficulty_relay
from experiments.post_training.task_curation.difficulty_trials import preserved_model_traffic


def test_final_archive_waits_for_inflight_upstream_reply(tmp_path, monkeypatch):
    started, release, finish, closing, finished = (Event() for _ in range(5))
    urls = Queue()
    requests = Queue()

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            requests.put(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            started.set()
            assert release.wait(10)
            body = json.dumps({"usage": {"prompt_tokens": 1, "completion_tokens": 2}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [1]

    class ClosingServer(difficulty_relay.ReceiptHTTPServer):
        def server_close(self):
            closing.set()
            super().server_close()

    monkeypatch.setattr(difficulty_relay.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(difficulty_relay, "ReceiptHTTPServer", ClosingServer)
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    upstream_thread = Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    local = tmp_path / "local"
    local.mkdir()
    output = StoragePath(str(tmp_path / "output"))

    def campaign():
        try:
            with preserved_model_traffic(local, output), difficulty_relay.budget_relay(
                endpoint=f"http://127.0.0.1:{upstream.server_port}/v1", model="fixture", api_key="fixture",
                tokenizer_model="fixture", tokenizer_revision="a" * 40, parameters={"temperature": 0.7}, receipts=local / "model-requests",
                budget=difficulty_relay.CANDIDATE_BUDGET,
            ) as relay:
                urls.put(relay)
                assert finish.wait(10)
        finally:
            finished.set()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            worker = pool.submit(campaign)
            relay = urls.get(timeout=10)
            response = pool.submit(httpx.post, relay + "/task/v1/chat/completions",
                                   json={"messages": [{"role": "user", "content": "answer"}], "temperature": 0.0}, timeout=10)
            try:
                assert started.wait(10)
                finish.set()
                assert closing.wait(10)
                # The upstream is known to be pending; final archival must await that reply.
                assert not finished.wait(1)
            finally:
                release.set()
                finish.set()
            assert response.result(timeout=10).status_code == 200
            assert requests.get(timeout=10)["temperature"] == 0.7
            worker.result(timeout=10)
    finally:
        release.set()
        finish.set()
        upstream.shutdown()
        upstream_thread.join()
        upstream.server_close()

    receipt = json.loads((output / "model-requests-final.json").read_text())
    assert (receipt["input_tokens"], receipt["output_tokens"], receipt["received_responses"]) == (1, 2, 1)
    with tarfile.open(fileobj=io.BytesIO(StoragePath(receipt["archive"]).read_bytes()), mode="r:gz") as archive:
        (member,) = [member for member in archive if member.name.endswith("/response.json")]
        assert json.load(archive.extractfile(member))["body"]["usage"]["completion_tokens"] == 2
        (request,) = [member for member in archive if member.name.endswith("/request.json")]
        recorded = json.load(archive.extractfile(request))
        assert (recorded["requested_body"]["temperature"], recorded["body"]["temperature"]) == (0.0, 0.7)
