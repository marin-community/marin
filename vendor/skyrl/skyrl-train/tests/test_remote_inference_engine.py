"""Exact rollout token transport against a local OpenAI-compatible server."""

import pytest
from httpx import AsyncClient, ASGITransport
from aiohttp import web
from aiohttp.test_utils import TestServer

from skyrl_train.inference_engines.remote_inference_engine import RemoteInferenceEngine, create_remote_inference_engines
from skyrl_train.inference_engines.inference_engine_client_http_endpoint import create_app, set_global_state


@pytest.mark.asyncio
async def test_remote_completion_keeps_sampled_ids_and_aligned_behavior_logprobs():
    requests = []

    async def generate(request):
        body = await request.json()
        requests.append(body)
        return web.json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "text": " X",
                        "token_ids": [3],
                        "finish_reason": "abort",
                        "logprobs": {"token_logprobs": [-0.25]},
                    },
                    {
                        "index": 1,
                        "text": " X",
                        "token_ids": [2],
                        "finish_reason": "stop",
                        "logprobs": {"token_logprobs": [-0.5]},
                    },
                ]
            }
        )

    app = web.Application()
    app.router.add_post("/v1/completions", generate)
    async with TestServer(app) as server:
        (engine,) = create_remote_inference_engines(
            urls=[str(server.make_url("")).removeprefix("http://").rstrip("/")],
            model_name="test",
        )
        result = await engine.generate({"prompt_token_ids": [[0, 1], [0, 2]], "sampling_params": {"temperature": 1.0}})
        # An abort retry extends the original prompt with served IDs, never encoded response text.
        retry_ids = [[0, 1] + result["response_ids"][0], [0, 2] + result["response_ids"][1]]
        await engine.generate({"prompt_token_ids": retry_ids, "sampling_params": {"temperature": 1.0}})
    assert result["responses"] == [" X", " X"]
    assert result["response_ids"] == [[3], [2]]
    assert result["response_logprobs"] == [[-0.25], [-0.5]]
    assert result["stop_reasons"] == ["abort", "stop"]
    assert requests[0]["return_token_ids"] is True
    assert requests[0]["logprobs"] == 0
    assert requests[1]["prompt"] == [[0, 1, 3], [0, 2, 2]]


@pytest.mark.asyncio
async def test_remote_weight_publication_and_lifecycle_use_vllm_wire_protocol():
    requests = []

    async def receive(request):
        body = await request.json() if request.can_read_body else None
        requests.append((request.path, body))
        return web.json_response({"accepted": request.path})

    app = web.Application()
    for path in (
        "init_weight_update_communicator",
        "update_weights",
        "wake_up",
        "sleep",
        "reset_prefix_cache",
        "destroy_weights_update_group",
    ):
        app.router.add_post(f"/{path}", receive)
    async with TestServer(app) as server:
        engine = RemoteInferenceEngine(str(server.make_url("")).removeprefix("http://").rstrip("/"), "test")
        await engine.init_weight_update_communicator("trainer", 29500, 1, 3, "policy", "nccl", True)
        result = await engine.update_named_weights(
            {"names": ["model.norm.weight"], "dtypes": ["bfloat16"], "shapes": [[128]], "extras": None}
        )
        await engine.sleep(level=2)
        await engine.wake_up(tags=["weights", "kv_cache"])
        await engine.reset_prefix_cache()
        await engine.teardown()
    assert result == {"accepted": "/update_weights"}
    assert requests == [
        (
            "/init_weight_update_communicator",
            {
                "master_address": "trainer",
                "master_port": 29500,
                "rank_offset": 1,
                "world_size": 3,
                "group_name": "policy",
                "backend": "nccl",
                "override_existing": True,
            },
        ),
        ("/update_weights", {"name": "model.norm.weight", "dtype": "bfloat16", "shape": [128]}),
        ("/sleep", {"level": 2}),
        ("/wake_up", {"tags": ["weights", "kv_cache"]}),
        ("/reset_prefix_cache", None),
        ("/destroy_weights_update_group", None),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["/v1/completions", "/v1/chat/completions"])
@pytest.mark.parametrize(
    "error",
    [
        {"error": {"message": "invalid request", "code": 400}},
        {"object": "error", "message": "invalid request", "code": 400},
    ],
)
async def test_openai_bridge_preserves_requests_and_maps_both_error_shapes_to_http_status(endpoint, error):
    received = []

    async def complete(request):
        received.append(await request.json())
        return web.json_response(error, status=400)

    app = web.Application()
    app.router.add_post(endpoint, complete)
    body = {"model": "external-model", "temperature": 0.3, "provider_option": {"keep": True}}
    if endpoint == "/v1/completions":
        body["prompt"] = [1, 2, 3]
    else:
        body["messages"] = [{"role": "user", "content": "hello"}]
    async with TestServer(app) as server:
        engine = RemoteInferenceEngine(str(server.make_url("")).removeprefix("http://").rstrip("/"), "external-model")
        set_global_state(engine, None)
        try:
            async with AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://bridge") as client:
                response = await client.post(endpoint, json=body)
        finally:
            set_global_state(None, None)
    assert response.status_code == 400
    assert response.json() == error
    assert received == [body]
