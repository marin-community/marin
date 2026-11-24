# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Echoed likelihood transport and malformed response evidence."""

import json

import httpx
import openai
import pytest

from rolloutengine.likelihood import LikelihoodTransportError, VllmLikelihood, encode_pair


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
            await VllmLikelihood(client, "fixture", CharacterTokenizer(), 64, 0, None)("c", "A", seed=None)
    evidence = failure.value.evidence
    assert requests == [evidence["request"]]
    assert evidence["response"]["id"] == "invalid-number"
    assert evidence["response"]["choices"][0]["logprobs"]["token_logprobs"][1] is None
    assert "NaN" in evidence["response_json"]
    json.dumps(evidence, allow_nan=False)


BOS = 1


@pytest.mark.parametrize(
    ("context", "bos_token_id", "expected_context"),
    [
        # A nonempty context starts with the model's BOS, and the continuation keeps the trailing space.
        ("Q: x? ", BOS, (BOS, *map(ord, "Q: x?"))),
        ("Q: x? ", None, tuple(map(ord, "Q: x?"))),
        # An empty context is only the prefix token, whatever the BOS policy.
        ("", BOS, (0,)),
    ],
)
def test_context_starts_with_bos_only_when_the_model_expects_one(context, bos_token_id, expected_context):
    context_tokens, continuation_tokens = encode_pair(
        CharacterTokenizer(), context, "yes", prefix_token_id=0, bos_token_id=bos_token_id
    )
    assert context_tokens == expected_context
    assert continuation_tokens == tuple(map(ord, " yes" if context else "yes"))


def test_bos_counts_as_context_so_only_continuation_tokens_are_scored():
    prepared = VllmLikelihood(None, "fixture", CharacterTokenizer(), 64, 0, BOS).prepare("ab", "c")
    assert prepared.prompt_token_ids == (BOS, ord("a"), ord("b"), ord("c"))
    assert (prepared.context_tokens, prepared.continuation_tokens) == (3, 1)


def test_truncation_keeps_bos_and_drops_the_oldest_context_text():
    prepared = VllmLikelihood(None, "fixture", CharacterTokenizer(), 5, 0, BOS).prepare("abcd", "e")
    assert prepared.prompt_token_ids == (BOS, ord("c"), ord("d"), ord("e"))
    assert (prepared.context_tokens, prepared.truncated_context_tokens) == (3, 2)
