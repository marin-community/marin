# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Echoed likelihood transport and malformed response evidence."""

import json

import httpx
import openai
import pytest

from rolloutengine.likelihood import LikelihoodTransportError, VllmLikelihood


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) for character in text]


@pytest.mark.asyncio
async def test_malformed_nonfinite_likelihood_retains_json_safe_wire_evidence_without_retry():
    requests = []

    def serve(request):
        requests.append(json.loads(request.content))
        response = {
            "id": "invalid-number",
            "object": "text_completion",
            "created": 0,
            "model": "fixture",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length",
                    "text": "cA!",
                    "logprobs": {
                        "tokens": ["c", "A", "!"],
                        "text_offset": [0, 1, 2],
                        "token_logprobs": [None, float("nan"), -0.2],
                        "top_logprobs": [None, {"A": -0.1}, {"!": -0.2}],
                    },
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }
        return httpx.Response(
            200, content=json.dumps(response, allow_nan=True), headers={"content-type": "application/json"}
        )

    async with openai.AsyncOpenAI(
        api_key="test",
        base_url="http://fixture.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)),
    ) as client:
        with pytest.raises(LikelihoodTransportError) as failure:
            await VllmLikelihood(client, "fixture", CharacterTokenizer(), 64, 0)("c", "A", seed=None)
    evidence = failure.value.evidence
    assert requests == [evidence["request"]]
    assert evidence["response"]["id"] == "invalid-number"
    assert evidence["response"]["choices"][0]["logprobs"]["token_logprobs"][1] is None
    assert "NaN" in evidence["response_json"]
    json.dumps(evidence, allow_nan=False)
