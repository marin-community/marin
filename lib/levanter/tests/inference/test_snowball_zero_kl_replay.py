# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug.sharding import compact_grug_mesh
from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.kernels.logsumexp import batch_invariant_selected_logprobs
from levanter.layers.attention import AttentionMask
from levanter.layers.sampler import LogprobMode
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel


@pytest.mark.tpu
@pytest.mark.timeout(300)
def test_snowball_strict_zero_kl_replays_all_served_policy_scores():
    if jax.default_backend() != "tpu" or jax.device_count() != 4 or jax.devices()[0].device_kind != "TPU v4":
        pytest.skip("The zero-KL fixture requires one four-chip TPUv4 host")
    cfg = SnowballConfig(
        vocab_size=23,
        hidden_dim=128,
        intermediate_dim=32,
        shared_expert_intermediate_dim=32,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        num_experts=2,
        num_experts_per_token=1,
        max_seq_len=256,
        sliding_window=128,
    ).with_zero_kl_replay()
    rng = np.random.default_rng(91)
    n_generated = 8
    prompts = [rng.integers(1, cfg.vocab_size, size=length).tolist() for length in (124, 128, 129, 131)]
    request_params = SeqDecodingParams.default()
    requests = [
        Request(
            prompt_tokens=prompt,
            request_id=index,
            decode_params=dataclasses.replace(
                request_params,
                key=jax.random.PRNGKey(index),
                temperature=jnp.float32(0.7),
                top_p=jnp.float32(0.8),
                max_num_tokens=jnp.int32(len(prompt) + n_generated),
            ),
            n_generations=1,
        )
        for index, prompt in enumerate(prompts)
    ]
    with jax.set_mesh(compact_grug_mesh()), jax.default_matmul_precision("highest"):
        model = SnowballLMHeadModel.init(hax.Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(19))
        engine = InferenceEngine.from_model_with_config(
            model,
            tokenizer=None,
            config=InferenceEngineConfig(
                max_seq_len=256,
                max_seqs=4,
                max_pages=8,
                max_seqs_in_prefill=4,
                max_prefill_size=1024,
                max_queued_tokens=4,
                max_tokens_per_round=4,
                max_rounds=4,
                page_size=128,
                compute_dtype=jnp.bfloat16,
                logprob_mode=LogprobMode.POLICY,
            ),
        )
        result = engine.generate(requests, capture_routes=True)
        assert result.logprobs is not None and result.expert_routes is not None
        tokens = np.zeros((4, 256), dtype=np.int32)
        valid = np.zeros_like(tokens, dtype=np.bool_)
        experts = np.zeros((cfg.num_layers, 4, 256, cfg.num_experts_per_token), dtype=np.int32)
        for row, (prompt, generated, capture) in enumerate(
            zip(prompts, result.tokens, result.expert_routes, strict=True)
        ):
            assert len(generated) == n_generated
            full = prompt + generated
            tokens[row, : len(full)] = full
            valid[row, : len(full)] = capture.valid
            experts[:, row, : len(full)] = capture.expert_ids
        tokens = hax.named(jnp.asarray(tokens), ("batch", "position"))
        routes = model.prepare_route_replay(experts, hax.named(jnp.asarray(valid), tokens.axes))

        @jax.jit
        def replay_scores(current, tokens, routes):
            hidden, _, _ = current._activations(tokens, False, AttentionMask.causal(), route_replay=routes)
            logits = current.project_logits(hidden).array.astype(jnp.float32)
            targets = jnp.roll(tokens.array, -1, axis=-1)
            return batch_invariant_selected_logprobs(logits, targets)

        scores = np.asarray(replay_scores(model, tokens, routes))

    for row, (prompt, generated, served) in enumerate(zip(prompts, result.tokens, result.logprobs, strict=True)):
        positions = np.arange(len(prompt) - 1, len(prompt) + len(generated) - 1)
        np.testing.assert_array_equal(
            scores[row, positions].view(np.uint32), np.asarray(served, np.float32).view(np.uint32)
        )
