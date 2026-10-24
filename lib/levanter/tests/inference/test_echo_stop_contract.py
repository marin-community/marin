# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure protocol-unit checks: scripted engine results, no model execution."""
import asyncio
from types import SimpleNamespace

import pytest

from levanter.inference.engine import TokenSequenceLogprobs
from levanter.inference.openai import InferenceBatch, InferenceContext, InferenceRequest


@pytest.mark.asyncio
@pytest.mark.parametrize("raw,eos,stop", [([5], (5,), None), ([5], (), [[5]]), ([5, 7], (), [[5, 7]])])
async def test_echo_keeps_raw_stop_token_while_text_strips_it(raw, eos, stop):
    class Tokenizer:
        def decode(self, tokens, **kwargs):
            return "".join({4: "A", 5: "B", 6: "C", 7: "D"}[token] for token in tokens)

    class Engine:
        def generate(self, requests):
            return SimpleNamespace(tokens=[raw], logprobs=[[-0.5] * len(raw)], total_generated=len(raw))

        def score_token_logprobs(self, ids, top_k):
            self.scored_ids = list(ids)
            return TokenSequenceLogprobs([0.0] + [-0.25] * (len(ids) - 1), [{token: -0.25} for token in ids])

    context = InferenceContext.__new__(InferenceContext)
    context.engine = Engine()
    context.tokenizer = Tokenizer()
    context.config = SimpleNamespace(eos_token_ids=eos)
    future = asyncio.get_running_loop().create_future()
    request = InferenceRequest(
        request_id="echo-eos",
        prompt_tokens=[4, 6],
        max_tokens=len(raw),
        temperature=0,
        top_p=1,
        stop_tokens=stop,
        seed=19311,
        future=future,
        echo_logprobs_top_k=1,
    )
    context._execute_batch(InferenceBatch([request]))
    result = (await future)[0]
    assert result.text == "" and result.tokens == [] and result.finish_reason == "stop"
    assert result.prompt_tokens == 2 and result.completion_tokens == len(raw)
    assert result.echo_token_ids == [4, 6, *raw]
    assert context.engine.scored_ids == [4, 6, *raw]
    # Pinned lm-eval continuation scoring discards the generated token only.
    assert result.echo_logprobs.token_logprobs[1 : -len(raw)] == [-0.25]
