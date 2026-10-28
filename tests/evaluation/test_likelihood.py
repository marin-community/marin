# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay independent harness token-boundary and local-completions reference cases."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from marin.evaluation.likelihood import VllmLikelihood, encode_pair
from openai import AsyncOpenAI
from taskcompendium.evaluation import LikelihoodRequest

ORACLE = json.loads((Path(__file__).parent / "fixtures/evalchemy_likelihood_oracle.json").read_text())


class FixtureTokenizer:
    eos_token_id = 99

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return ORACLE["vocabulary"][text]


@pytest.mark.parametrize("reference", ORACLE["pair_cases"])
def test_encoding_preserves_legacy_joint_boundary_and_left_truncation(reference):
    request = LikelihoodRequest(context=reference["context"], continuation=reference["continuation"])
    tokenizer = FixtureTokenizer()
    context, continuation = encode_pair(tokenizer, request, prefix_token_id=ORACLE["prefix_token_id"])
    assert (list(context), list(continuation)) == (reference["prefix"], reference["target"])
    client = AsyncOpenAI(api_key="unused", base_url="http://fixture/v1")
    scorer = VllmLikelihood(client, "fixture", tokenizer, ORACLE["context_size"], ORACLE["prefix_token_id"])
    prepared = scorer.prepare(request)
    assert list(prepared.prompt_token_ids) == reference["prompt_ids"]
    assert prepared.context_tokens == reference["context_tokens"]
    assert prepared.truncated_context_tokens == len(reference["prefix"]) - reference["context_tokens"]
    asyncio.run(client.close())


@pytest.mark.parametrize("index", [0, 1])
def test_http_echo_scores_only_target_tokens_and_preserves_greedy_ties(index):
    reference = ORACLE["score_cases"][index]
    pair = ORACLE["pair_cases"][1 if index == 0 else 0]
    requests = []

    def serve(request):
        requests.append(json.loads(request.content))
        response = {
            "id": "fixture",
            "created": 0,
            "model": "fixture",
            "object": "text_completion",
            "choices": [{**reference["response"]["choices"][0], "text": "", "finish_reason": "length"}],
        }
        return httpx.Response(200, json=response)

    async def score():
        async with AsyncOpenAI(
            api_key="unused",
            base_url="http://fixture/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)),
        ) as client:
            scorer = VllmLikelihood(client, "fixture", FixtureTokenizer(), ORACLE["context_size"], 99)
            return await scorer(LikelihoodRequest(context=pair["context"], continuation=pair["continuation"]))

    result = asyncio.run(score())
    assert (result.loglikelihood, result.is_greedy) == tuple(reference["result"])
    assert result.token_count == len(pair["target"])
    assert list(result.evidence.prompt_token_ids) == pair["prompt_ids"]
    assert result.evidence.context_tokens == pair["context_tokens"]
    assert result.token_logprobs == tuple(reference["response"]["choices"][0]["logprobs"]["token_logprobs"][1:-1])
    assert requests[0]["prompt"] == pair["prompt_ids"]
    assert requests[0]["echo"] is True
    assert requests[0]["max_tokens"] == 1


def test_endpoint_without_prompt_echo_is_an_error():
    def serve(request):
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "created": 0,
                "model": "fixture",
                "object": "text_completion",
                "choices": [
                    {
                        "index": 0,
                        "text": "generated",
                        "finish_reason": "length",
                        "logprobs": {"token_logprobs": [-0.1], "top_logprobs": [{"generated": -0.1}]},
                    }
                ],
            },
        )

    async def score():
        async with AsyncOpenAI(
            api_key="unused",
            base_url="http://fixture/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)),
        ) as client:
            scorer = VllmLikelihood(client, "fixture", FixtureTokenizer(), 4, 99)
            return await scorer(LikelihoodRequest(context="Context ", continuation="answer"))

    with pytest.raises(ValueError, match="different token boundary"):
        asyncio.run(score())
