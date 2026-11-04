# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Echoed likelihood transport and malformed response evidence."""

import json

import httpx
import openai
import pytest

from rolloutengine.likelihood import LikelihoodTransportError, VllmLikelihood, special_prefix


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) for character in text]


class BosCharacterTokenizer(CharacterTokenizer):
    """Put BOS, id 1, before the text when special tokens are on, as Llama-style tokenizers do."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [1, *super().encode(text)] if add_special_tokens else super().encode(text)


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
            await VllmLikelihood(client, "fixture", CharacterTokenizer(), 64, prefix_token_id=0, context_prefix=())(
                "c", "A", seed=None
            )
    evidence = failure.value.evidence
    assert requests == [evidence["request"]]
    assert evidence["response"]["id"] == "invalid-number"
    assert evidence["response"]["choices"][0]["logprobs"]["token_logprobs"][1] is None
    assert "NaN" in evidence["response_json"]
    json.dumps(evidence, allow_nan=False)


@pytest.mark.asyncio
async def test_likelihood_context_starts_with_bos_and_scores_only_the_continuation():
    prompts = []

    def serve(request):
        prompt = json.loads(request.content)["prompt"]
        prompts.append(prompt)
        tokens = [*map(str, prompt), "!"]
        # Position i of the echoed prompt has log-probability -i / 100.
        values = [None, *(-index / 100 for index in range(1, len(prompt))), -0.5]
        response = {
            "id": "echo",
            "object": "text_completion",
            "created": 0,
            "model": "fixture",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length",
                    "text": "!",
                    "logprobs": {
                        "tokens": tokens,
                        "text_offset": list(range(len(tokens))),
                        "token_logprobs": values,
                        "top_logprobs": [
                            None,
                            *({token: value} for token, value in zip(tokens[1:], values[1:], strict=True)),
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": 1, "total_tokens": len(prompt) + 1},
        }
        return httpx.Response(200, json=response)

    tokenizer = BosCharacterTokenizer()
    async with openai.AsyncOpenAI(
        api_key="test",
        base_url="http://fixture.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)),
    ) as client:
        likelihood = VllmLikelihood(
            client, "fixture", tokenizer, 64, prefix_token_id=1, context_prefix=special_prefix(tokenizer)
        )
        score = await likelihood("Answer: ", "yes", seed=None)
    assert prompts == [[1, *map(ord, "Answer: yes")]]
    # BOS and "Answer:" are positions 0 to 7; the continuation " yes" is positions 8 to 11.
    assert score.token_count == 4
    assert score.loglikelihood == pytest.approx(-(8 + 9 + 10 + 11) / 100)
