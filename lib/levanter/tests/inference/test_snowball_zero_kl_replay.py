# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
import json
import time

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from levanter.grug.sharding import compact_grug_mesh
from levanter.inference.engine import InferenceEngineConfig
from levanter.layers.sampler import LogprobMode
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.policy import ClippedPolicyConfig
from levanter.policy_model import PolicyBatch, pack_policy_batch, policy_model_loss, response_logprobs
from levanter.tokenizers import load_tokenizer

try:
    import grpc
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip("The strict gRPC fixture requires the serve extra", allow_module_level=True)

from levanter.inference import inference_pb2, inference_pb2_grpc  # noqa: E402
from levanter.inference.openai import InferenceServer, InferenceServerConfig  # noqa: E402
from levanter.inference.policy_replay import RolloutChoice, prepare_rollout_policy_batch  # noqa: E402

_POLICY_ID = "frozen-snapshot"
_NEW_TOKENS = 8
_SLOTS = 4
_PAGE_SIZE = 128
_PROMPT_LENGTHS = (19, 29, 63, 131)


def _generate(model, tokenizer, mesh, prompts, measured_rounds):
    length = model.config.max_seq_len
    server = InferenceServer.create(
        InferenceServerConfig(
            initial_policy_id=_POLICY_ID,
            host="127.0.0.1",
            grpc_port=0,
            service=InferenceEngineConfig(
                max_seq_len=length,
                max_seqs=_SLOTS,
                max_pages=_SLOTS * length // _PAGE_SIZE,
                max_seqs_in_prefill=_SLOTS,
                max_prefill_size=_SLOTS * length,
                max_queued_tokens=_SLOTS,
                max_tokens_per_round=_SLOTS,
                max_rounds=4,
                page_size=_PAGE_SIZE,
                compute_dtype=jnp.bfloat16,
                logprob_mode=LogprobMode.POLICY,
            ),
        ),
        model,
        tokenizer,
        mesh=mesh,
    )

    async def run_rounds():
        async with grpc.aio.insecure_channel(f"{server.config.host}:{server.app.state.grpc_port}") as channel:
            stub = inference_pb2_grpc.InferenceStub(channel)
            rounds = []
            for _ in range(measured_rounds + 1):
                start = time.perf_counter()
                responses = await asyncio.gather(
                    *(
                        stub.Generate(
                            inference_pb2.GenerateRequest(
                                prompt_token_ids=prompt,
                                max_new_tokens=_NEW_TOKENS,
                                temperature=0.7,
                                top_p=0.8,
                                seed=row,
                                capture_router_replay=True,
                                expected_policy_id=_POLICY_ID,
                                expected_logprob_mode=inference_pb2.LOGPROB_MODE_POLICY,
                            ),
                            timeout=240,
                        )
                        for row, prompt in enumerate(prompts)
                    )
                )
                rounds.append(time.perf_counter() - start)
            return responses, rounds

    try:
        with TestClient(server.app):
            return asyncio.run(run_rounds())
    finally:
        server.shutdown()


def _rollout_batch(model, prompts, generations):
    length = model.config.max_seq_len
    tokens = np.zeros((_SLOTS, length), dtype=np.int32)
    attended = np.zeros_like(tokens)
    response = np.zeros((_SLOTS, length - 1), dtype=bool)
    rollouts = []
    for row, (prompt, generation) in enumerate(zip(prompts, generations, strict=True)):
        choice = generation.choices[0]
        assert generation.logprob_mode == inference_pb2.LOGPROB_MODE_POLICY
        assert choice.policy_id == _POLICY_ID
        assert len(choice.token_ids) == _NEW_TOKENS
        full = prompt + list(choice.token_ids)
        tokens[row, : len(full)] = full
        attended[row, : len(full)] = 1
        response[row, len(prompt) - 1 : len(full) - 1] = True
        rollouts.append(RolloutChoice(generation, 0))

    def named(values, axis):
        return hax.named(jnp.asarray(values), ("batch", axis))

    mask = named(response, "response")
    weights = mask.astype(jnp.float32)
    batch = PolicyBatch(
        tokens=named(tokens, "position"),
        attention_mask=named(attended, "position"),
        position_ids=named(np.broadcast_to(np.arange(length, dtype=np.int32), tokens.shape), "position"),
        response_mask=mask,
        advantages=weights,
        policy_weights=weights,
        kl_weights=weights,
        old_logprobs=hax.zeros(mask.axes),
        reference_logprobs=None,
    )
    batch = prepare_rollout_policy_batch(model, batch, rollouts, policy_id=_POLICY_ID)
    return dataclasses.replace(batch, reference_logprobs=batch.old_logprobs)


def _score_error(actual, expected):
    delta = np.abs(actual - expected)
    return {
        "tokens": int(delta.size),
        "mismatches": int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32))),
        "mean_abs": float(delta.mean()),
        "p95_abs": float(np.percentile(delta, 95)),
        "max_abs": float(delta.max()),
    }


@pytest.mark.tpu
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "layers,expert_axis_size",
    [
        pytest.param(1, 1, id="fixture-data4"),
        pytest.param(2, 4, id="production-width-expert4", marks=pytest.mark.manual),
        pytest.param(26, 4, id="production-depth-expert4", marks=pytest.mark.manual),
    ],
)
def test_snowball_strict_zero_kl_replays_all_served_policy_scores(tmp_path, layers, expert_axis_size):
    if jax.default_backend() != "tpu" or jax.device_count() != _SLOTS:
        pytest.skip("The strict replay gate requires one four-device TPU host")
    device_kind = jax.devices()[0].device_kind.lower()
    if device_kind not in ("tpu v4", "tpu v5", "tpu v5p", "tpu v6 lite", "tpu v6e"):
        pytest.skip("The fixture supports TPUv4, v5p, and v6e projections")
    if layers == 26 and device_kind not in ("tpu v5", "tpu v5p"):
        pytest.skip("The full-depth FP32 fixture requires the v5p HBM budget")
    production_width = layers > 1
    if production_width:
        base = SnowballConfig(num_layers=layers, max_seq_len=256, moe_capacity_factor=float(expert_axis_size))
    else:
        base = SnowballConfig(
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
        )
    cfg = base.with_zero_kl_replay()
    rng = np.random.default_rng(91)
    prompts = [rng.integers(1, cfg.vocab_size, size=length).tolist() for length in _PROMPT_LENGTHS]
    vocabulary = {f"token{index}": index for index in range(cfg.vocab_size)}
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel(vocabulary)))
    hf_tokenizer.save_pretrained(tmp_path)
    tokenizer = load_tokenizer(str(tmp_path))
    mesh = compact_grug_mesh(expert_axis_size=expert_axis_size)
    vocab = hax.Axis("vocab", cfg.vocab_size)
    key = jax.random.key(19)
    with jax.set_mesh(mesh):
        model = SnowballLMHeadModel.init(vocab, cfg, key=key)
        generations, strict_rounds = _generate(
            model, tokenizer, mesh, prompts, measured_rounds=3 if production_width else 0
        )
        if production_width:
            print(
                "SNOWBALL_STRICT_THROUGHPUT "
                + json.dumps(
                    {
                        "layers": layers,
                        "first_batch_seconds": strict_rounds[0],
                        "warmed_batch_seconds": strict_rounds[1:],
                        "output_tokens_per_second": _SLOTS * _NEW_TOKENS / np.median(strict_rounds[1:]),
                    }
                ),
                flush=True,
            )
        batch = _rollout_batch(model, prompts, generations)
        packed = pack_policy_batch(batch, sequence_length=cfg.max_seq_len, pad_token_id=0)
        assert packed.tokens.resolve_axis("batch").size < _SLOTS
        score = jax.jit(response_logprobs)

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

        parity = {}
        for name, learner_batch in (("unpacked", batch), ("packed", packed)):
            mask = np.asarray(learner_batch.response_mask.array)
            actual = np.asarray(score(model, learner_batch).array)[mask]
            expected = np.asarray(learner_batch.old_logprobs.array)[mask]
            assert actual.size == _SLOTS * _NEW_TOKENS
            _, metrics = objective(model, learner_batch)
            parity[name] = {
                **_score_error(actual, expected),
                "kl": float(metrics["policy_kl"].value()),
                "max_abs_log_ratio": float(metrics["log_ratio_abs_max"].value()),
            }
            if production_width:
                print("SNOWBALL_STRICT_PARITY " + json.dumps({name: parity[name]}), flush=True)

        if production_width and layers < 26:
            template = jax.eval_shape(lambda: SnowballLMHeadModel.init(vocab, base, key=key))
            fast = jax.tree.unflatten(jax.tree.structure(template), jax.tree.leaves(model))
            fast = fast.with_expert_sharded_kv()
            generations, fast_rounds = _generate(fast, tokenizer, mesh, prompts, measured_rounds=3)
            fast_batch = _rollout_batch(model, prompts, generations)
            mask = np.asarray(fast_batch.response_mask.array)
            expected = np.asarray(score(model, fast_batch).array)[mask]
            actual = np.asarray(fast_batch.old_logprobs.array)[mask]
            assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
            print(
                "SNOWBALL_REPLAY_GATE "
                + json.dumps(
                    {
                        "device_kind": jax.devices()[0].device_kind,
                        "mesh": dict(mesh.shape),
                        "prompt_lengths": _PROMPT_LENGTHS,
                        "context": cfg.max_seq_len,
                        "layers": cfg.num_layers,
                        "hidden": cfg.hidden_dim,
                        "experts": cfg.num_experts,
                        "heads": cfg.num_heads,
                        "vocabulary": cfg.vocab_size,
                        "strict_first_batch_seconds": strict_rounds[0],
                        "strict_output_tokens_per_second": _SLOTS * _NEW_TOKENS / np.median(strict_rounds[1:]),
                        "fast_kv_layout": "expert_sharded",
                        "fast_first_batch_seconds": fast_rounds[0],
                        "fast_output_tokens_per_second": _SLOTS * _NEW_TOKENS / np.median(fast_rounds[1:]),
                        "fast_score_error": _score_error(actual, expected),
                    },
                    sort_keys=True,
                )
            )

        for name, evidence in parity.items():
            assert evidence["mismatches"] == evidence["kl"] == evidence["max_abs_log_ratio"] == 0, (name, evidence)
