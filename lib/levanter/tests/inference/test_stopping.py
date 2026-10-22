# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio

import equinox as eqx
import haliax as hax
import jax.numpy as jnp
import pytest

from levanter.inference.engine import InferenceEngine, InferenceEngineConfig
from levanter.inference.openai import (
    InferenceBatch,
    InferenceContext,
    InferenceRequest,
    InferenceServerConfig,
    _encode_stop_tokens,
)
from levanter.layers.kv_cache import KvPageCache
from levanter.testing.helpers import use_test_mesh


class ScriptedModel(eqx.Module):
    schedule: jnp.ndarray
    Vocab: hax.Axis = eqx.field(static=True, default=hax.Axis("vocab", 16))

    def initial_cache(self, spec, *, dtype):
        return KvPageCache.init(spec, hax.Axis("kv_head", 1), hax.Axis("head_size", 1), dtype)

    def decode(self, input_ids, cache, batch_info, pos_ids):
        targets = self.schedule[jnp.clip(pos_ids.array, 0, self.schedule.size - 1)]
        logits = jnp.where(jnp.arange(self.Vocab.size)[None, :] == targets[:, None], 100.0, -100.0)
        return hax.named(logits, (*input_ids.axes, self.Vocab)), cache


class ScriptedTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        return [{"A": 4, "B": 5, "C": 6, "D": 7}[character] for character in text]

    def decode(self, tokens, *, skip_special_tokens=True):
        return "".join({4: "A", 5: "B", 6: "C", 7: "D"}.get(int(token), "?") for token in tokens)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "schedule,eos,stop,max_tokens,visible,used,reason",
    [
        ([4, 5, 6, 7], (5,), None, 4, [4], 2, "stop"),
        ([4, 5, 6, 7], (), ["B", "D"], 4, [4], 2, "stop"),
        ([4, 7, 6, 5], (), ["B", "D"], 4, [4], 2, "stop"),
        ([4, 5, 6, 7], (), ["BC"], 4, [4], 3, "stop"),
        ([4, 5, 6, 7], (), None, 3, [4, 5, 6], 3, "length"),
    ],
)
async def test_server_stops_and_strips_independent_sequences(schedule, eos, stop, max_tokens, visible, used, reason):
    with use_test_mesh():
        model = ScriptedModel(jnp.array(schedule, jnp.int32))
        tokenizer = ScriptedTokenizer()
        config = InferenceServerConfig(
            eos_token_ids=eos,
            service=InferenceEngineConfig(
                max_seq_len=16,
                max_pages=4,
                page_size=4,
                max_seqs=1,
                max_seqs_in_prefill=1,
                max_prefill_size=4,
                max_queued_tokens=8,
                max_tokens_per_round=1,
                max_rounds=1,
            ),
        )
        engine = InferenceEngine.from_model_with_config(model, tokenizer, config.service, axis_resources={})
        context = InferenceContext(model, tokenizer, engine, config)
        future = asyncio.get_running_loop().create_future()
        request = InferenceRequest(
            request_id="stop-test",
            prompt_tokens=[0],
            max_tokens=max_tokens,
            temperature=0,
            top_p=1,
            stop_tokens=_encode_stop_tokens(stop, tokenizer),
            seed=17,
            future=future,
        )
        context._execute_batch(InferenceBatch([request]))
        result = (await future)[0]
        assert result.tokens == visible
        assert result.completion_tokens == used
        assert result.finish_reason == reason
        assert result.text == tokenizer.decode(visible)
        assert len(result.logprobs) == len(visible)
