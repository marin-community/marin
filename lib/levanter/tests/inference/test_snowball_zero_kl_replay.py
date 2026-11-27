# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
import json
import time

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from levanter.grug.sharding import compact_grug_mesh
from levanter.inference.engine import InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.layers.sampler import LogprobMode
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.policy import ClippedPolicyConfig, clipped_policy_loss
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
_PAGE_SIZE = 128
_FIXTURE_CONTEXT = 256
_PRODUCTION_PROMPT_LENGTHS = (19, 29, 63, 131)
_PRODUCTION_WINDOW_PROMPT_LENGTHS = (19, 29, 63, 2051)
_FIXTURE_PROMPT_LENGTHS = (124, 128, 129, 131)
_WARMUP_ROUNDS = 3
_MEASURED_ROUNDS = 7
_TEMPERATURE = 0.7
_TOP_P = 0.8


def _measure_fixed_batch(server, prompts, measured_rounds):
    requests = [
        Request(
            prompt,
            row,
            dataclasses.replace(
                SeqDecodingParams.default(),
                max_num_tokens=jnp.int32(len(prompt) + _NEW_TOKENS),
                temperature=jnp.float32(_TEMPERATURE),
                top_p=jnp.float32(_TOP_P),
                key=jax.random.PRNGKey(row),
            ),
            n_generations=1,
        )
        for row, prompt in enumerate(prompts)
    ]
    context = server.inference_context
    rounds = []
    with context.model_lock, hax.axis_mapping(context.config.trainer.compute_axis_mapping):
        for _ in range(_WARMUP_ROUNDS + measured_rounds):
            start = time.perf_counter()
            result = context.engine.generate(requests, capture_routes=True)
            rounds.append(time.perf_counter() - start)
            assert result.total_generated == len(prompts) * _NEW_TOKENS
    return rounds


def _timing(rounds, slots):
    samples = rounds[_WARMUP_ROUNDS:]
    return {
        "warmup_batch_seconds": rounds[:_WARMUP_ROUNDS],
        "warmed_batch_seconds": samples,
        "output_tokens_per_second": slots * _NEW_TOKENS / np.median(samples),
    }


def _generate(model, tokenizer, mesh, prompts, measured_rounds):
    length = model.config.max_seq_len
    slots = len(prompts)
    rpc_timeout = 480 if length > _FIXTURE_CONTEXT or model.config.num_layers == 26 else 240
    server = InferenceServer.create(
        InferenceServerConfig(
            initial_policy_id=_POLICY_ID,
            host="127.0.0.1",
            grpc_port=0,
            batch_timeout=0.001,
            service=InferenceEngineConfig(
                max_seq_len=length,
                max_seqs=slots,
                max_pages=slots * length // _PAGE_SIZE,
                max_seqs_in_prefill=slots,
                max_prefill_size=slots * length,
                max_queued_tokens=slots,
                max_tokens_per_round=slots,
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

    async def generate_responses():
        async with grpc.aio.insecure_channel(f"{server.config.host}:{server.app.state.grpc_port}") as channel:
            stub = inference_pb2_grpc.InferenceStub(channel)
            return await asyncio.gather(
                *(
                    stub.Generate(
                        inference_pb2.GenerateRequest(
                            prompt_token_ids=prompt,
                            max_new_tokens=_NEW_TOKENS,
                            temperature=_TEMPERATURE,
                            top_p=_TOP_P,
                            seed=row,
                            capture_router_replay=True,
                            expected_policy_id=_POLICY_ID,
                            expected_logprob_mode=inference_pb2.LOGPROB_MODE_POLICY,
                        ),
                        timeout=rpc_timeout,
                    )
                    for row, prompt in enumerate(prompts)
                )
            )

    try:
        with TestClient(server.app):
            timings = _measure_fixed_batch(server, prompts, measured_rounds) if measured_rounds else []
            return asyncio.run(generate_responses()), timings
    finally:
        server.shutdown()


def _rollout_batch(model, prompts, generations):
    length = model.config.max_seq_len
    slots = len(prompts)
    tokens = np.zeros((slots, length), dtype=np.int32)
    attended = np.zeros_like(tokens)
    response = np.zeros((slots, length - 1), dtype=bool)
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


def _v6e_expert_tiles(token_rows):
    # Match the AgentX serving profile on the two fixture contexts.
    return (128 if token_rows <= 8192 else 256, 1280)


def _score_error(actual, expected):
    delta = np.abs(actual - expected)
    mismatch = np.flatnonzero(actual.view(np.uint32) != expected.view(np.uint32))
    return {
        "tokens": int(delta.size),
        "mismatches": int(mismatch.size),
        "mismatch_indices": mismatch.tolist(),
        "mean_abs": float(delta.mean()),
        "p95_abs": float(np.percentile(delta, 95)),
        "max_abs": float(delta.max()),
    }


@pytest.mark.tpu
@pytest.mark.parametrize(
    "layers,expert_axis_size,context_size,prompt_lengths",
    [
        pytest.param(
            1, 1, _FIXTURE_CONTEXT, _FIXTURE_PROMPT_LENGTHS, id="fixture-data4", marks=pytest.mark.timeout(300)
        ),
        pytest.param(
            2,
            4,
            _FIXTURE_CONTEXT,
            _PRODUCTION_PROMPT_LENGTHS,
            id="production-width-expert4",
            marks=(pytest.mark.manual, pytest.mark.timeout(900)),
        ),
        pytest.param(
            2,
            8,
            _FIXTURE_CONTEXT,
            _PRODUCTION_PROMPT_LENGTHS * 2,
            id="production-width-expert8",
            marks=(pytest.mark.manual, pytest.mark.timeout(1200)),
        ),
        pytest.param(
            2,
            4,
            2176,
            _PRODUCTION_WINDOW_PROMPT_LENGTHS,
            id="production-window-expert4",
            marks=(pytest.mark.manual, pytest.mark.timeout(900)),
        ),
        pytest.param(
            26,
            4,
            _FIXTURE_CONTEXT,
            _PRODUCTION_PROMPT_LENGTHS,
            id="production-depth-expert4",
            marks=(pytest.mark.manual, pytest.mark.timeout(1500)),
        ),
    ],
)
def test_snowball_strict_zero_kl_replays_all_served_policy_scores(
    tmp_path, layers, expert_axis_size, context_size, prompt_lengths
):
    slots = len(prompt_lengths)
    if jax.default_backend() != "tpu" or jax.local_device_count() != slots or jax.device_count() != slots:
        pytest.skip(f"The strict replay gate requires one {slots}-device TPU host")
    device_kind = jax.devices()[0].device_kind.lower()
    if device_kind not in ("tpu v4", "tpu v5", "tpu v5p", "tpu v6 lite", "tpu v6e"):
        pytest.skip("The fixture supports TPUv4, v5p, and v6e projections")
    if slots == 8 and device_kind not in ("tpu v6 lite", "tpu v6e"):
        pytest.skip("The eight-chip fixture targets v6e")
    if layers == 26 and device_kind not in ("tpu v5", "tpu v5p"):
        pytest.skip("The full-depth FP32 fixture requires the v5p HBM budget")
    production_width = layers > 1
    compare_native = layers == 2 and device_kind != "tpu v4"
    if production_width:
        base = SnowballConfig(num_layers=layers, max_seq_len=context_size, moe_capacity_factor=float(expert_axis_size))
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
            max_seq_len=context_size,
            sliding_window=128,
        )
    cfg = base.with_zero_kl_replay()
    rng = np.random.default_rng(91)
    prompts = [rng.integers(1, cfg.vocab_size, size=length).tolist() for length in prompt_lengths]
    vocabulary = {f"token{index}": index for index in range(cfg.vocab_size)}
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel(vocabulary)))
    hf_tokenizer.save_pretrained(tmp_path)
    tokenizer = load_tokenizer(str(tmp_path))
    mesh = compact_grug_mesh(expert_axis_size=expert_axis_size)
    vocab = hax.Axis("vocab", cfg.vocab_size)
    key = jax.random.key(19)
    with jax.set_mesh(mesh):
        model = SnowballLMHeadModel.init(vocab, cfg, key=key)
        if compare_native:
            # Both precision profiles start from the same representable weights.
            bf16_weights = jax.tree.map(lambda weight: weight.astype(jnp.bfloat16), model)
            model = jax.tree.map(lambda weight: weight.astype(jnp.float32), bf16_weights)
        if layers == 26:
            # Retain one packed copy of the expert gate/up weights before compiling decode.
            for index in range(layers):
                transformer = model.transformer
                block = jax.block_until_ready(transformer.blocks[index].with_packed_experts())
                blocks = (*transformer.blocks[:index], block, *transformer.blocks[index + 1 :])
                model = dataclasses.replace(model, transformer=dataclasses.replace(transformer, blocks=blocks))
                del transformer
        generations, strict_rounds = _generate(
            model, tokenizer, mesh, prompts, measured_rounds=_MEASURED_ROUNDS if production_width else 0
        )
        if production_width:
            print(
                "SNOWBALL_STRICT_THROUGHPUT "
                + json.dumps(
                    {
                        "layers": layers,
                        "admission": "fixed_batch",
                        **_timing(strict_rounds, slots),
                    }
                ),
                flush=True,
            )
        batch = _rollout_batch(model, prompts, generations)
        packing_length = cfg.max_seq_len if production_width else 2 * cfg.max_seq_len
        packed = pack_policy_batch(batch, sequence_length=packing_length, pad_token_id=0)
        assert packed.tokens.resolve_axis("batch").size < slots
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
            current_scores = score(model, learner_batch)
            values = np.asarray(current_scores.array)
            if name == "packed":
                np.testing.assert_array_equal(values[~mask], 0)
            actual = values[mask]
            expected = np.asarray(learner_batch.old_logprobs.array)[mask]
            assert actual.size == slots * _NEW_TOKENS
            if production_width:
                _, metrics = clipped_policy_loss(
                    current_scores,
                    learner_batch.old_logprobs,
                    learner_batch.reference_logprobs,
                    learner_batch.advantages,
                    learner_batch.policy_weights,
                    learner_batch.kl_weights,
                    config=ClippedPolicyConfig(0.2, 0.2, 0.1),
                    accumulation_steps=1,
                )
            else:
                _, metrics = objective(model, learner_batch)
            parity[name] = {
                **_score_error(actual, expected),
                "kl": float(metrics["policy_kl"].value()),
                "max_abs_log_ratio": float(metrics["log_ratio_abs_max"].value()),
            }
            if production_width:
                print("SNOWBALL_STRICT_PARITY " + json.dumps({name: parity[name]}), flush=True)

        if not production_width:

            def head_loss(weight, learner_batch):
                candidate = eqx.tree_at(lambda m: m.transformer.output_proj, model, weight)
                return objective(candidate, learner_batch)[0]

            head_gradient = jax.jit(jax.grad(head_loss))
            weight = model.transformer.output_proj
            reference = head_gradient(weight, batch)
            actual = head_gradient(weight, packed)
            assert np.count_nonzero(reference) > 0
            np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=1e-5)

        if compare_native:
            template = jax.eval_shape(lambda: SnowballLMHeadModel.init(vocab, base, key=key))
            weights = jax.tree.leaves(bf16_weights)
            native = jax.tree.unflatten(jax.tree.structure(template), weights).with_expert_sharded_kv()
            tuned = native.with_unrolled_decode().with_paged_attention("tpu_v3")
            if device_kind in ("tpu v6 lite", "tpu v6e"):
                tuned = tuned.with_separate_serving_experts(_v6e_expert_tiles)
            else:
                tuned = tuned.with_packed_experts()
            profiles = {}
            for name, serving in (("native_bf16", native), ("agentx_bf16", tuned)):
                generations, rounds = _generate(serving, tokenizer, mesh, prompts, measured_rounds=_MEASURED_ROUNDS)
                fast_batch = _rollout_batch(model, prompts, generations)
                mask = np.asarray(fast_batch.response_mask.array)
                expected = np.asarray(score(model, fast_batch).array)[mask]
                actual = np.asarray(fast_batch.old_logprobs.array)[mask]
                assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
                profiles[name] = {**_timing(rounds, slots), "score_error": _score_error(actual, expected)}
                print("SNOWBALL_NATIVE_PROFILE " + json.dumps({name: profiles[name]}), flush=True)

            print(
                "SNOWBALL_REPLAY_GATE "
                + json.dumps(
                    {
                        "device_kind": jax.devices()[0].device_kind,
                        "mesh": dict(mesh.shape),
                        "prompt_lengths": prompt_lengths,
                        "context": cfg.max_seq_len,
                        "layers": cfg.num_layers,
                        "hidden": cfg.hidden_dim,
                        "experts": cfg.num_experts,
                        "heads": cfg.num_heads,
                        "vocabulary": cfg.vocab_size,
                        "admission": "fixed_batch",
                        "strict_fp32": _timing(strict_rounds, slots),
                        **profiles,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        for name, evidence in parity.items():
            assert evidence["mismatches"] == evidence["kl"] == evidence["max_abs_log_ratio"] == 0, (name, evidence)
