# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np

from levanter.grug.sharding import compact_grug_mesh
from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel


def test_window_pool_matches_full_cache_across_eviction_and_reuse():
    config = SnowballConfig(
        vocab_size=23,
        hidden_dim=16,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_layers=2,
        num_experts=4,
        num_experts_per_token=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=64,
        sliding_window=4,
        attention_implementation="reference",
        inference_attention_implementation="reference",
        inference_decode_implementation="unrolled_split_cache",
        inference_max_unrolled_prefill_tokens=64,
        moe_implementation="ring",
    )
    serving = InferenceEngineConfig(
        max_seq_len=64,
        max_prefill_size=64,
        max_pages=128,
        page_size=2,
        max_seqs=4,
        max_seqs_in_prefill=4,
        max_queued_tokens=4,
        max_rounds=8,
        compute_dtype=jnp.float32,
        prefix_cache="recent_prompts",
    )

    def request(tokens, rid):
        params = dataclasses.replace(
            SeqDecodingParams.default(),
            max_num_tokens=jnp.array(len(tokens) + 3),
            temperature=jnp.array(0.0),
        )
        return Request(tokens, rid, params, 1)

    rng = np.random.default_rng(663)
    initial, second, third = [rng.integers(0, 23, 40).tolist() for _ in range(3)]
    related = initial[:35] + [7, 8, 9, 10, 11]
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)), jax.default_matmul_precision("highest"):
        model = SnowballLMHeadModel.init(hax.Axis("vocab", 23), config, key=jax.random.key(41))
        full = InferenceEngine.from_model_with_config(model, None, serving)
        hybrid = InferenceEngine.from_model_with_config(
            model, None, dataclasses.replace(serving, window_cache_max_pages=57)
        )
        # The histories exceed the compact pool and revisit an evicted prefix.
        for rid, batch in enumerate(((initial,), (related,), (second, third), (initial, related))):
            requests = [request(tokens, rid * 2 + i) for i, tokens in enumerate(batch)]
            baseline = full.generate(requests)
            actual = hybrid.generate(requests)
            assert actual.tokens == baseline.tokens
            assert actual.finish_reasons == baseline.finish_reasons
            np.testing.assert_allclose(actual.logprobs, baseline.logprobs, rtol=1e-5, atol=1e-5)
        hybrid.reset()
        assert hybrid.generate([request(initial, 20)]).tokens == full.generate([request(initial, 20)]).tokens
