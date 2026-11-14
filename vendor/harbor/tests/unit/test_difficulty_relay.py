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
from rigging.timing import Duration, ExponentialBackoff

import httpx
from rigging.filesystem.storage_path import StoragePath
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from experiments.post_training.task_curation import difficulty_provider, difficulty_relay
from experiments.post_training.task_curation.difficulty_provider import ProviderLimits
from experiments.post_training.task_curation.difficulty_trials import preserved_model_traffic


def test_provider_capacity_bounds_concurrent_http_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(difficulty_provider, "ADMISSION_POLL_SECONDS", .01)
    first, second, release = Event(), Event(), Event()
    calls = []

    def upstream(request):
        calls.append(request)
        (first if len(calls) == 1 else second).set()
        assert release.wait(10)
        return httpx.Response(200, json={"usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [1]

    real_client = httpx.Client
    monkeypatch.setattr(difficulty_relay.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(difficulty_relay.httpx, "Client", lambda **kwargs: real_client(
        **kwargs, transport=httpx.MockTransport(upstream)))
    try:
        with difficulty_relay.budget_relay(
            endpoint="https://provider.fixture/v1", model="fixture", api_key="fixture",
            tokenizer_model="fixture", tokenizer_revision="a"*40, parameters={},
            budget=difficulty_relay.RequestBudget(65536,49152,16384),
            provider_limits=ProviderLimits(max_in_flight=1, attempts=1), receipts=tmp_path,
        ) as relay, ThreadPoolExecutor(max_workers=2) as pool, real_client() as client:
            one = pool.submit(client.post, relay+"/one/v1/chat/completions", json={"messages": []})
            assert first.wait(10)
            two = pool.submit(client.post, relay+"/two/v1/chat/completions", json={"messages": []})
            try:
                assert ExponentialBackoff(initial=.01,maximum=.05).wait_until(
                    lambda: len(list(tmp_path.glob("*/000000/provider-capacity-wait.json"))) == 1,
                    Duration.from_seconds(5))
                assert len(calls) == 1
            finally:
                release.set()
            assert one.result(timeout=10).status_code == two.result(timeout=10).status_code == 200
    finally:
        release.set()
    admitted = [json.loads(path.read_text()) for path in tmp_path.glob("*/000000/provider-admitted.json")]
    assert len(admitted) == 2
    assert max(row["peak"] for row in admitted) == 1
    assert max(row["capacity_polls"] for row in admitted) >= 1


def test_provider_retries_rejections_with_preserved_waits(tmp_path, monkeypatch):
    calls = []
    def upstream(request):
        calls.append(request)
        status = [429,503,200][len(calls)-1]
        return httpx.Response(status, json={"usage": {"prompt_tokens": 1, "completion_tokens": 2}})

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [1]

    real_client = httpx.Client
    monkeypatch.setattr(difficulty_relay.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(difficulty_relay.httpx, "Client", lambda **kwargs: real_client(
        **kwargs, transport=httpx.MockTransport(upstream)))
    with difficulty_relay.budget_relay(
        endpoint="https://provider.fixture/v1", model="fixture", api_key="fixture",
        tokenizer_model="fixture", tokenizer_revision="a"*40, parameters={},
        budget=difficulty_relay.RequestBudget(65536,49152,16384),
        provider_limits=ProviderLimits(max_in_flight=1, attempts=3), receipts=tmp_path,
    ) as relay, real_client() as client:
        assert client.post(relay+"/task/v1/chat/completions", json={"messages": []}, timeout=10).status_code == 200
    statuses = [json.loads(path.read_text())["status"] for path in sorted(tmp_path.glob("*/000000/upstream-attempts/*/status.json"))]
    waits = [json.loads(path.read_text()) for path in sorted(tmp_path.glob("*/000000/provider-waits/*.json"))]
    assert statuses == [429,503,200]
    assert len(waits) == 2
    assert all(row["kind"] == "provider_wait" and row["task_attempt"] is False for row in waits)
    assert .75 <= waits[0]["backoff_seconds"] <= 1.25
    assert 1.5 <= waits[1]["backoff_seconds"] <= 2.5


def test_relay_renders_tool_arguments_without_changing_upstream_wire_history(tmp_path, monkeypatch):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")), unk_token="[UNK]",
        chat_template="""{% for message in messages %}{{ message.content or '' }}
{% for call in message.get('tool_calls') or [] %}
{% for key, value in call.function.arguments.items() %}{{key}}={{value}}{% endfor %}
{% endfor %}{% endfor %}""",
    )
    arguments = '{"command":"echo 42"}'
    messages = [{"role":"assistant", "content":"", "tool_calls":[{
        "id":"shell-1", "type":"function", "function":{"name":"shell", "arguments":arguments},
    }]}, {"role":"tool", "tool_call_id":"shell-1", "content":"42"}]
    upstream_requests = []

    def model_response(request):
        upstream_requests.append(json.loads(request.content))
        return httpx.Response(200, json={"usage":{"prompt_tokens":1, "completion_tokens":2}})

    real_client = httpx.Client
    monkeypatch.setattr(difficulty_relay.AutoTokenizer, "from_pretrained", lambda *args, **kwargs:tokenizer)
    monkeypatch.setattr(difficulty_relay.httpx, "Client", lambda **kwargs:real_client(
        **kwargs, transport=httpx.MockTransport(model_response)))
    with difficulty_relay.budget_relay(
        endpoint="https://model.fixture/v1", model="fixture", api_key="fixture",
        tokenizer_model="fixture", tokenizer_revision="a"*40, parameters={},
        budget=difficulty_relay.RequestBudget(65536, 49152, 16384), receipts=tmp_path,
        provider_limits=ProviderLimits(max_in_flight=1, attempts=1),
    ) as relay:
        with real_client() as client:
            response = client.post(relay+"/task/v1/chat/completions", json={"messages":messages})
        assert response.status_code == 200
    assert upstream_requests[0]["messages"] == messages
    assert upstream_requests[0]["messages"][0]["tool_calls"][0]["function"]["arguments"] == arguments


def test_final_archive_waits_for_inflight_upstream_reply(tmp_path, monkeypatch):
    started, release, finish, closing, finished = (Event() for _ in range(5))
    urls = Queue()

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
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
                tokenizer_model="fixture", tokenizer_revision="a" * 40, parameters={}, receipts=local / "model-requests",
                budget=difficulty_relay.RequestBudget(context_tokens=65536, input_tokens=49152, output_tokens=16384),
                provider_limits=ProviderLimits(max_in_flight=1, attempts=1),
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
                                   json={"messages": [{"role": "user", "content": "answer"}]}, timeout=10)
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
