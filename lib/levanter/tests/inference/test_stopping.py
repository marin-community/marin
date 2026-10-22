# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("openai")

from levanter.inference.engine import InferenceEngine, InferenceEngineConfig
from levanter.inference.openai import (
    InferenceBatch,
    InferenceContext,
    InferenceRequest,
    InferenceServerConfig,
    _encode_stop_tokens,
    _host_prng_key,
)
from levanter.inference.multihost import _encode_request
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

        class HostOnlyEngine(InferenceEngine):
            def generate(self, requests, step_callback=None):
                for request in requests:
                    _encode_request(request)  # Reject any device-backed HTTP metadata before generation.
                return super().generate(requests, step_callback=step_callback)

        engine = HostOnlyEngine.from_model_with_config(model, tokenizer, config.service, axis_resources={})
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


@pytest.mark.parametrize("x64", [False, True])
def test_host_prng_key_matches_legacy_threefry_seed_mapping(x64):
    with jax.enable_x64(x64):
        for seed in (0, 17, 19311, 2**32 - 1, 2**32 + 7, -1):
            np.testing.assert_array_equal(_host_prng_key(seed), jax.random.PRNGKey(seed))


@pytest.mark.asyncio
async def test_single_sequence_server_serializes_concurrent_requests():
    config = InferenceServerConfig(
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
        )
    )
    with use_test_mesh(mesh=config.trainer.device_mesh):
        model = ScriptedModel(jnp.array([4, 5, 6, 7], jnp.int32))
        tokenizer = ScriptedTokenizer()
        batches = []

        class SingleSequenceEngine(InferenceEngine):
            def generate(self, requests, step_callback=None):
                assert len(requests) == 1
                batches.append(requests[0].prompt_tokens)
                return super().generate(requests, step_callback=step_callback)

        engine = SingleSequenceEngine.from_model_with_config(model, tokenizer, config.service, axis_resources={})
        context = InferenceContext(model, tokenizer, engine, config)
        futures = [asyncio.get_running_loop().create_future() for _ in range(8)]
        context.start()
        try:
            for index, future in enumerate(futures):
                context.submit_request([index], 2, 0, 1, None, 19311, future)
            results = await asyncio.wait_for(asyncio.gather(*futures), timeout=60)
            assert len(batches) == 8
            assert all(result[0].completion_tokens == 2 for result in results)
        finally:
            context.shutdown()
