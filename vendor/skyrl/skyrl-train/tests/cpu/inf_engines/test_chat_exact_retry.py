import base64
import io
from copy import deepcopy

import numpy as np
import pytest
from omegaconf import OmegaConf

from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.trajectory_runners.routed_experts import normalize_routed_experts


def _make_min_cfg():
    return OmegaConf.create(
        {
            "trainer": {
                "policy": {"model": {"path": "dummy-model"}},
            },
            "generator": {
                "backend": "vllm",
                "enable_http_endpoint": False,
                "http_endpoint_host": "127.0.0.1",
                "http_endpoint_port": 0,
                "weight_sync_pause_timeout_seconds": 30.0,
            },
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("capture", ["none", "routes", "missing_routes", "wrong_prompt"])
async def test_chat_completion_retry_keeps_exact_ids_and_optional_routes(capture):
    def encoded_routes(values):
        buffer = io.BytesIO()
        np.save(buffer, np.asarray(values, dtype=np.uint8).reshape(-1, 1, 1))
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    class Engine:
        def __init__(self):
            self.requests = []
            self.responses = [
                {
                    "prompt_token_ids": [1, 2],
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "AB"},
                            "finish_reason": "abort",
                            "token_ids": [11, 12],
                            "routed_experts": encoded_routes([1, 2, 3]),
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
                },
                {
                    "prompt_token_ids": [1, 2, 11, 12],
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "C"},
                            "finish_reason": "stop",
                            "token_ids": [13],
                            "routed_experts": encoded_routes([1, 2, 3, 4]),
                        }
                    ],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
                },
            ]

        async def chat_completion(self, request):
            self.requests.append(deepcopy(request))
            response = deepcopy(self.responses.pop(0))
            if capture in ("none", "wrong_prompt") or (capture == "missing_routes" and len(self.requests) == 2):
                response["choices"][0].pop("routed_experts")
            if capture == "wrong_prompt" and len(self.requests) == 2:
                response["prompt_token_ids"][-1] = 99
            return response

    engine = Engine()
    client = InferenceEngineClient([engine], object(), _make_min_cfg())
    request = {"json": {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 3}}
    if capture in ("missing_routes", "wrong_prompt"):
        with pytest.raises(ValueError):
            await client.chat_completion(request)
        return
    response = await client.chat_completion(request)
    choice = response["choices"][0]

    assert choice["token_ids"] == [11, 12, 13]
    assert engine.requests[1]["json"]["_skyrl_exact_prompt_token_ids"] == [1, 2, 11, 12]
    if capture == "routes":
        # Predictive-position routes start at the final prompt token.
        np.testing.assert_array_equal(
            normalize_routed_experts(choice["routed_experts"], response["prompt_token_ids"], choice["token_ids"]),
            [[[2]], [[3]], [[4]]],
        )
