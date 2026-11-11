"""Exact rollout token transport against a local OpenAI-compatible server."""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from skyrl_train.inference_engines.remote_inference_engine import RemoteInferenceEngine


class AliasingTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [2]  # Both 2 and 3 decode to the same text; retokenization loses identity.

    def decode(self, tokens, skip_special_tokens=True):
        return " X"


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
        engine = RemoteInferenceEngine(
            str(server.make_url("")).removeprefix("http://").rstrip("/"), "test", "vllm", AliasingTokenizer()
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
