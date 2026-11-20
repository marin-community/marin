import json
import socket
from contextlib import asynccontextmanager

from aiohttp import ClientSession, web

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from marinskyrl.near_window_serving import GpuMemory, NearWindowProbeSettings, run_near_window_probe


class CompletionServer:
    def __init__(self, *, drop_prompt_token=False):
        self.requests = []
        self.drop_prompt_token = drop_prompt_token
        self.served_model_name = "/regional/model-cache/qualified-weight-version"

    async def get_served_model_name(self):
        return self.served_model_name

    async def completion(self, request_payload):
        request = request_payload["json"]
        self.requests.append(request)
        assert request["model"] == self.served_model_name
        prompt = request["prompt"]
        count = len(prompt) - int(self.drop_prompt_token) if isinstance(prompt, list) else 7
        return {"model": self.served_model_name, "choices": [{"text": "ok"}], "usage": {
            "prompt_tokens": count, "completion_tokens": request["max_tokens"],
        }}


@pytest.mark.asyncio
@pytest.mark.parametrize("drop_prompt_token", [False, True])
async def test_probe_checks_server_usage_against_rendered_input_and_persists_failure(tmp_path, monkeypatch, drop_prompt_token):
    monkeypatch.setenv("IRIS_OUTPUT_DIR", str(tmp_path))
    tokenizer = probe_tokenizer()
    engines = [CompletionServer(drop_prompt_token=drop_prompt_token), CompletionServer()]

    async def memory_reader():
        return [GpuMemory(gpu_uuid=f"gpu-{rank}", used_mib=120000, total_mib=180000) for rank in range(2)]

    destination = tmp_path / "published-receipt.json"
    settings = dict(engines=2, minimum_prompt_tokens=20, max_new_tokens=4, context_tokens=64,
                    temperature=0.7, top_p=1.0,
                    request_timeout=10, memory_sample_interval=2, receipt_uri=str(destination))
    if drop_prompt_token:
        with pytest.raises(ExceptionGroup) as failure:
            await run_near_window_probe(engines, tokenizer, NearWindowProbeSettings(**settings), 3, receipt_directory=tmp_path, memory_reader=memory_reader)
        assert any(isinstance(error, ValueError) for error in failure.value.exceptions)
    else:
        await run_near_window_probe(engines, tokenizer, NearWindowProbeSettings(**settings), 3, receipt_directory=tmp_path, memory_reader=memory_reader)
    receipt = json.loads(destination.read_bytes())
    assert receipt == json.loads((tmp_path / "near-window-serving-receipt.json").read_bytes())
    assert receipt["result"] == ("failed" if drop_prompt_token else "passed")
    if drop_prompt_token:
        assert "near_window_request" not in receipt
        return
    assert receipt["served_model_names"] == [engine.served_model_name for engine in engines]
    near = engines[0].requests[-1]
    assert near["prompt"][0:2] == [1, 2] and near["prompt"][-2:] == [4, 5]
    assert receipt["near_window_request"]["usage"] == {"prompt_tokens": 24, "completion_tokens": 4}
    assert receipt["peak_used_mib_by_gpu"] == {"gpu-0": 120000, "gpu-1": 120000}
    assert len(receipt["rank_readiness_requests"]) == 2


CONTEXT_TOKENS = 262144


def probe_tokenizer():
    backend = Tokenizer(WordLevel({"[UNK]": 0, "a": 1, "b": 2, "x": 3, "c": 4, "d": 5}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, model_max_length=CONTEXT_TOKENS)
    tokenizer.chat_template = "a b {{ messages[-1]['content'].split('\\n')[0] }} c d"
    return tokenizer


class HttpCompletionEngine:
    """Loopback stand-in for the actor's registered-model and completion RPCs."""

    def __init__(self, session, url):
        self.session = session
        self.url = url

    async def get_served_model_name(self):
        async with self.session.get(self.url + "/v1/models") as response:
            return (await response.json())["data"][0]["id"]

    async def completion(self, request_payload):
        async with self.session.post(
            self.url + "/v1/completions", json=request_payload["json"], headers=request_payload["headers"]
        ) as response:
            return await response.json()


@asynccontextmanager
async def local_completion_server(served_name):
    requests = []

    async def models(request):
        return web.json_response({"data": [{"id": served_name}]})

    async def complete(request):
        body = await request.json()
        requests.append(body)
        if body["model"] != served_name:
            return web.json_response({"error": {"message": "model does not exist", "code": 404}}, status=404)
        prompt_tokens = len(body["prompt"]) if isinstance(body["prompt"], list) else 7
        if prompt_tokens + body["max_tokens"] > CONTEXT_TOKENS:
            return web.json_response({"error": {"message": "context limit", "code": 400}}, status=400)
        return web.json_response({"model": served_name, "choices": [{"text": "stub response"}], "usage": {
            "prompt_tokens": prompt_tokens, "completion_tokens": body["max_tokens"],
        }})

    app = web.Application(client_max_size=4 * 1024**2)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/completions", complete)
    runner = web.AppRunner(app)
    with socket.socket() as listener:
        await runner.setup()
        try:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            await web.SockSite(runner, listener).start()
            yield f"http://127.0.0.1:{port}", requests
        finally:
            await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("served_name", ["s3://regional-bucket/exact-streamed-model", "explicit-configured-alias"])
async def test_probe_model_matches_registered_server_name_over_http(tmp_path, monkeypatch, served_name):
    monkeypatch.setenv("IRIS_OUTPUT_DIR", str(tmp_path))
    tokenizer = probe_tokenizer()
    settings = dict(engines=16, minimum_prompt_tokens=250000, max_new_tokens=1024, context_tokens=CONTEXT_TOKENS,
                    temperature=0.7, top_p=1.0, request_timeout=10, memory_sample_interval=2,
                    receipt_uri=str(tmp_path / "published.json"))

    async def memory_reader():
        return [GpuMemory(gpu_uuid=f"stub-{rank}", used_mib=120000, total_mib=180000) for rank in range(16)]

    async with local_completion_server(served_name) as (url, requests), ClientSession() as session:
        engines = [HttpCompletionEngine(session, url) for _ in range(16)]
        await run_near_window_probe(engines, tokenizer, NearWindowProbeSettings(**settings), 1, receipt_directory=tmp_path, memory_reader=memory_reader)
    assert len(requests) == 17 and all(request["model"] == served_name for request in requests)
    near = max(requests, key=lambda request: len(request["prompt"]))
    assert len(near["prompt"]) >= 250000 and near["max_tokens"] == 1024
    receipt = json.loads((tmp_path / "published.json").read_bytes())
    assert receipt["result"] == "passed" and receipt["near_window_request"]["model"] == served_name
