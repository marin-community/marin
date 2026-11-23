# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses

import haliax as hax
import httpx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from levanter.grug.sharding import compact_grug_mesh
from levanter.inference.binary import (
    GENERATE_PATH,
    LOGPROB_MODE,
    MEDIA_TYPE,
    POLICY_ID,
    POLICY_LOGPROBS,
    PROMPT_TOKEN_IDS,
    TOKEN_IDS,
    choice_tensor_key,
    decode_frame,
    encode_frame,
)
from levanter.inference.engine import InferenceEngineConfig
from levanter.inference.openai import InferenceServer, InferenceServerConfig
from levanter.inference.policy_replay import BinaryRolloutChoice, prepare_binary_policy_batch
from levanter.layers.sampler import LogprobMode
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.policy import ClippedPolicyConfig
from levanter.policy_model import PolicyBatch, policy_model_loss, response_logprobs
from levanter.tokenizers import load_tokenizer


@pytest.mark.tpu
@pytest.mark.timeout(300)
def test_snowball_strict_zero_kl_replays_all_served_policy_scores(tmp_path):
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
    snapshot = "frozen-snapshot"
    vocabulary = {f"token{index}": index for index in range(cfg.vocab_size)}
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel(vocabulary)))
    hf_tokenizer.save_pretrained(tmp_path)
    tokenizer = load_tokenizer(str(tmp_path))
    mesh = compact_grug_mesh()
    with jax.set_mesh(mesh):
        model = SnowballLMHeadModel.init(hax.Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(19))
        server = InferenceServer.create(
            InferenceServerConfig(
                initial_policy_id=snapshot,
                service=InferenceEngineConfig(
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
            ),
            model,
            tokenizer,
            mesh=mesh,
        )
        try:

            async def generate():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=server.app), base_url="http://inference"
                ) as client:
                    return await asyncio.gather(
                        *(
                            client.post(
                                GENERATE_PATH,
                                headers={"content-type": MEDIA_TYPE},
                                content=encode_frame(
                                    {
                                        "max_tokens": n_generated,
                                        "temperature": 0.7,
                                        "top_p": 0.8,
                                        "seed": row,
                                        "return_route_captures": True,
                                        "expected_policy_id": snapshot,
                                    },
                                    {PROMPT_TOKEN_IDS: np.asarray(prompt, np.int32)},
                                ),
                            )
                            for row, prompt in enumerate(prompts)
                        )
                    )

            generations = asyncio.run(generate())
        finally:
            server.shutdown()

        tokens = np.zeros((4, 256), dtype=np.int32)
        attended = np.zeros_like(tokens)
        response = np.zeros((4, 255), dtype=bool)
        rollouts = []
        served_scores = []
        for row, (prompt, generation) in enumerate(zip(prompts, generations, strict=True)):
            assert generation.status_code == 200, generation.text
            metadata, fields = decode_frame(generation.content)
            assert metadata[LOGPROB_MODE] == LogprobMode.POLICY
            assert metadata["choices"][0][POLICY_ID] == snapshot
            generated = fields[choice_tensor_key(0, TOKEN_IDS)]
            assert len(generated) == n_generated
            full = prompt + generated.tolist()
            tokens[row, : len(full)] = full
            attended[row, : len(full)] = 1
            response[row, len(prompt) - 1 : len(full) - 1] = True
            rollouts.append(BinaryRolloutChoice(generation.content, 0))
            served_scores.append(fields[choice_tensor_key(0, POLICY_LOGPROBS)])

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
        batch = prepare_binary_policy_batch(model, batch, rollouts, policy_id=snapshot)
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

    expected = np.asarray(served_scores, dtype=np.float32).reshape(-1)
    np.testing.assert_array_equal(np.asarray(scores)[response].view(np.uint32), expected.view(np.uint32))
