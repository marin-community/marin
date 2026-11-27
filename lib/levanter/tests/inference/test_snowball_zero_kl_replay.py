# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug.sharding import compact_grug_mesh
from levanter.inference import inference_pb2
from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.inference.policy_replay import RolloutChoice, prepare_rollout_policy_batch
from levanter.inference.router_replay import write_router_replay
from levanter.layers.sampler import LogprobMode
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.policy import ClippedPolicyConfig
from levanter.policy_model import PolicyBatch, policy_model_loss, response_logprobs


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
        snapshot = "frozen-snapshot"
        tokens = np.zeros((4, 256), dtype=np.int32)
        attended = np.zeros_like(tokens)
        response = np.zeros((4, 255), dtype=bool)
        rollouts = []
        for row, (prompt, generated, capture, served) in enumerate(
            zip(prompts, result.tokens, result.expert_routes, result.logprobs, strict=True)
        ):
            assert len(generated) == n_generated
            full = prompt + generated
            tokens[row, : len(full)] = full
            attended[row, : len(full)] = 1
            response[row, len(prompt) - 1 : len(full) - 1] = True
            generation = inference_pb2.GenerateResponse(
                prompt_token_ids=prompt, logprob_mode=inference_pb2.LOGPROB_MODE_POLICY
            )
            choice = generation.choices.add(policy_id=snapshot, token_ids=generated, policy_logprobs=served)
            write_router_replay(choice.router_replay, capture.expert_ids, capture.valid)
            rollouts.append(RolloutChoice(generation, 0))

        def named(values, axis):
            return hax.named(jnp.asarray(values), ("batch", axis))

        mask = named(response, "response")
        weights = mask.astype(jnp.float32)
        batch = PolicyBatch(
            tokens=named(tokens, "position"),
            attention_mask=named(attended, "position"),
            position_ids=named(np.broadcast_to(np.arange(256, dtype=np.int32), tokens.shape), "position"),
            response_mask=mask,
            advantages=weights,
            policy_weights=weights,
            kl_weights=weights,
            old_logprobs=hax.zeros(mask.axes),
            reference_logprobs=None,
        )
        batch = prepare_rollout_policy_batch(model, batch, rollouts, policy_id=snapshot)
        batch = dataclasses.replace(batch, reference_logprobs=batch.old_logprobs)
        scores = jax.jit(response_logprobs)(model, batch).array

        @jax.jit
        def objective(model, batch):
            return policy_model_loss(
                model,
                batch,
                key=None,
                config=ClippedPolicyConfig(0.2, 0.2, 0.1),
                accumulation_steps=1,
                block_size=1024,
            )

        _, metrics = objective(model, batch)
        assert metrics["policy_kl"].value() == 0.0
        assert metrics["log_ratio_abs_max"].value() == 0.0

    expected = np.asarray(result.logprobs, dtype=np.float32).reshape(-1)
    np.testing.assert_array_equal(np.asarray(scores)[response].view(np.uint32), expected.view(np.uint32))
