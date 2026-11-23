# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug.sharding import compact_grug_mesh
from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import FinishReason, SeqDecodingParams
from levanter.inference.page_table import PageBatchInfo, PageTableSpec
from levanter.inference.snowball_engine_replay import (
    align_snowball_policy_routes,
    generate_snowball_replay,
    replay_snowball_logits,
    replay_snowball_loss,
)
from levanter.layers.kv_cache import StackedKvPageCache
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.policy_model import PolicyBatch, PolicyRouteCaptures, pack_policy_batch, response_logprobs


@pytest.fixture
def serving_model():
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
        max_seq_len=8,
        sliding_window=4,
        attention_implementation="reference",
        inference_attention_implementation="reference",
        moe_implementation="ring",
    )
    layout = compact_grug_mesh(expert_axis_size=1)
    devices = np.asarray(jax.local_devices()[:1], dtype=object).reshape((1,) * len(layout.axis_names))
    mesh = jax.sharding.Mesh(devices, layout.axis_names, axis_types=layout.axis_types)
    with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
        yield SnowballLMHeadModel.init(hax.Axis("vocab", 23), config, key=jax.random.key(41))


def _engine(model, max_rounds, *, prefill_bucket_sizes=()):
    return InferenceEngine.from_model_with_config(
        model,
        None,
        InferenceEngineConfig(
            max_seq_len=8,
            max_pages=4,
            max_seqs=2,
            page_size=4,
            max_queued_tokens=2,
            max_seqs_in_prefill=2,
            max_prefill_size=8,
            prefill_bucket_sizes=prefill_bucket_sizes,
            max_rounds=max_rounds,
            compute_dtype=jnp.float32,
        ),
    )


def _requests():
    return [
        Request(
            prompt,
            rid,
            dataclasses.replace(
                SeqDecodingParams.default(),
                max_num_tokens=jnp.asarray(length),
                temperature=jnp.asarray(1.0),
                key=jax.random.PRNGKey(rid),
            ),
            1,
        )
        for prompt, rid, length in [([1, 2], 101, 6), ([3, 4, 5], 202, 7), ([6], 303, 5), ([8, 9], 404, 8)]
    ]


def test_prefill_buckets_preserve_snowball_generation(serving_model):
    requests = _requests()
    full = _engine(serving_model, 3).generate(requests)
    bucketed = _engine(serving_model, 3, prefill_bucket_sizes=(4, 6)).generate(requests)
    assert bucketed == full


def test_continuous_generation_admits_another_request_while_decoding(serving_model):
    short, long, followup, _ = _requests()
    short = dataclasses.replace(
        short, decode_params=dataclasses.replace(short.decode_params, max_num_tokens=jnp.array(3))
    )
    followup = dataclasses.replace(
        followup, decode_params=dataclasses.replace(followup.decode_params, max_num_tokens=jnp.array(2))
    )
    expected = _engine(serving_model, 3).generate([short, long, followup])
    completed = {}
    remaining = [followup]

    def poll(capacity):
        if short.request_id in completed and capacity:
            return [remaining.pop()] if remaining else []
        return []

    def complete(rid, result):
        completed[rid] = result.tokens[0]
        if rid == followup.request_id:
            remaining.clear()

    _engine(serving_model, 3).generate_continuously(
        [short, long], poll, complete, should_abort=lambda: False, on_tokens=lambda *_: None
    )
    assert list(completed) == [short.request_id, followup.request_id, long.request_id]
    assert [completed[request.request_id] for request in (short, long, followup)] == expected.tokens


@pytest.mark.parametrize(
    "decode_layers,unrolled_prefill",
    [("unrolled", 0), ("unrolled_split_cache", 0), ("unrolled_split_cache", 8)],
)
def test_unrolled_decode_preserves_snowball_generation(serving_model, decode_layers, unrolled_prefill):
    requests = _requests()
    scanned = _engine(serving_model, 3).generate(requests)
    model = dataclasses.replace(
        serving_model,
        _config=dataclasses.replace(
            serving_model.config,
            inference_decode_implementation=decode_layers,
            inference_max_unrolled_prefill_tokens=unrolled_prefill,
        ),
    )
    assert _engine(model, 3).generate(requests) == scanned


@pytest.mark.parametrize("prefix_cache,max_pages", [("last_prompt", 4), ("recent_prompts", 12)])
def test_reusing_prompt_pages_preserves_generation_and_resets(serving_model, caplog, prefix_cache, max_pages):
    config = InferenceEngineConfig(
        max_seq_len=8,
        max_pages=max_pages,
        max_seqs=1,
        page_size=2,
        max_queued_tokens=2,
        max_seqs_in_prefill=1,
        max_prefill_size=8,
        prefill_bucket_sizes=(4,),
        compute_dtype=jnp.float32,
    )
    fresh = InferenceEngine.from_model_with_config(serving_model, None, config)
    cached = InferenceEngine.from_model_with_config(
        serving_model, None, dataclasses.replace(config, prefix_cache=prefix_cache)
    )
    params = dataclasses.replace(SeqDecodingParams.default(), max_num_tokens=jnp.asarray(7))
    initial = Request([1, 2, 3, 4, 5], 0, params, 1)
    next_turn = Request([1, 2, 3, 4, 6], 1, params, 1)
    cached.generate([initial])
    if prefix_cache == "recent_prompts":
        cached.generate([Request([7, 8, 9, 10, 11, 12], 2, params, 1)])
    expected = fresh.generate([next_turn])
    with caplog.at_level(logging.INFO):
        assert cached.generate([next_turn]) == expected
    assert "Reused 4 prompt tokens" in caplog.text

    caplog.clear()
    cached.reset()
    with caplog.at_level(logging.INFO):
        assert cached.generate([next_turn]) == expected
    assert "Reused" not in caplog.text


def test_retaining_packed_prompt_pages_preserves_generation_after_eviction(serving_model):
    config = InferenceEngineConfig(
        max_seq_len=8,
        max_pages=12,
        max_seqs=3,
        page_size=2,
        max_queued_tokens=3,
        max_seqs_in_prefill=3,
        prefill_bucket_sizes=(4,),
        compute_dtype=jnp.float32,
    )
    fresh = InferenceEngine.from_model_with_config(serving_model, None, config)
    cached = InferenceEngine.from_model_with_config(
        serving_model, None, dataclasses.replace(config, prefix_cache="recent_prompts")
    )
    params = dataclasses.replace(
        SeqDecodingParams.default(), temperature=jnp.asarray(0.0), max_num_tokens=jnp.asarray(7)
    )
    first = Request([1, 2, 3, 4, 5], 10, params, 1)
    second = Request([6, 7, 8, 9, 10], 20, params, 1)
    cached.generate([first, second])

    followups = [
        dataclasses.replace(first, prompt_tokens=[1, 2, 3, 4, 11]),
        dataclasses.replace(first, request_id=30, prompt_tokens=[1, 2, 3, 4, 12]),
        dataclasses.replace(second, prompt_tokens=[6, 7, 8, 9, 13]),
    ]
    # Followups can keep using selected pages after evicting their cache entries.
    assert cached.generate(followups).tokens == fresh.generate(followups).tokens
    cached.generate([Request([2, 4, 6, 8, 10, 12], 40, params, 1)])
    assert cached.generate(followups).tokens == fresh.generate(followups).tokens


def test_shared_prompt_pages_admit_three_requests_with_eight_physical_pages(serving_model):
    config = InferenceEngineConfig(
        max_seq_len=8,
        max_pages=8,
        max_seqs=3,
        page_size=2,
        max_queued_tokens=3,
        max_seqs_in_prefill=3,
        prefix_cache="recent_prompts",
        compute_dtype=jnp.float32,
    )
    cached = InferenceEngine.from_model_with_config(serving_model, None, config)
    fresh = InferenceEngine.from_model_with_config(
        serving_model, None, dataclasses.replace(config, prefix_cache="disabled")
    )
    params = dataclasses.replace(
        SeqDecodingParams.default(), temperature=jnp.asarray(0.0), max_num_tokens=jnp.asarray(7)
    )
    first = Request([1, 2, 3, 4, 5], 10, params, 1)
    second = Request([1, 2, 3, 4, 6], 20, params, 1)
    third = Request([1, 2, 3, 4, 7], 30, params, 1)
    requests = [first, second, third]
    cached.generate([first])
    events = []
    result = cached.generate(
        requests,
        on_tokens=lambda rid, _choice, _tokens: events.append(("token", rid)),
        on_complete=lambda rid, _result: events.append(("finished", rid)),
    )

    assert set(events[:3]) == {("token", request.request_id) for request in requests}
    assert result.tokens == fresh.generate(requests).tokens


def test_prompt_cache_retains_more_entries_than_live_slots(serving_model, caplog):
    config = InferenceEngineConfig(
        max_seq_len=8,
        max_pages=16,
        max_seqs=2,
        max_queued_tokens=2,
        max_seqs_in_prefill=2,
        page_size=2,
        compute_dtype=jnp.float32,
    )
    fresh = InferenceEngine.from_model_with_config(serving_model, None, config)
    cached = InferenceEngine.from_model_with_config(
        serving_model, None, dataclasses.replace(config, prefix_cache="recent_prompts", max_retained_prompts=4)
    )
    params = dataclasses.replace(
        SeqDecodingParams.default(), temperature=jnp.asarray(0.0), max_num_tokens=jnp.asarray(7)
    )
    for rid, tokens in enumerate(([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15])):
        cached.generate([Request(tokens, rid, params, 1)])

    followup = Request([1, 2, 3, 4, 16], 3, params, 1)
    expected = fresh.generate([followup])
    with caplog.at_level(logging.INFO, logger="levanter.inference.engine"):
        assert cached.generate([followup]) == expected
    assert "Reused 4 prompt tokens" in caplog.text


def test_cache_pressure_preserves_reusable_pages_of_a_live_prompt(serving_model, caplog):
    config = InferenceEngineConfig(
        max_seq_len=8,
        max_pages=8,
        max_seqs=3,
        page_size=2,
        max_queued_tokens=3,
        max_seqs_in_prefill=3,
        max_prefill_size=12,
        max_rounds=1,
        prefix_cache="recent_prompts",
        compute_dtype=jnp.float32,
    )
    cached = InferenceEngine.from_model_with_config(serving_model, None, config)
    params = dataclasses.replace(SeqDecodingParams.default(), max_num_tokens=jnp.asarray(6))
    live = Request([1, 2, 3, 4], 10, dataclasses.replace(params, max_num_tokens=jnp.asarray(8)), 1)
    short = Request([6, 7, 8, 9, 10], 20, params, 1)
    fresh = Request([11, 12, 13, 14, 15], 30, params, 1)
    reused = Request([1, 2, 3, 4, 16], 40, params, 1)
    followups = {short.request_id: fresh, fresh.request_id: reused}
    waiting = []
    completed = []

    def poll(capacity):
        return [waiting.pop(0)] if capacity and waiting else []

    def complete(rid, _result):
        completed.append(rid)
        if rid in followups:
            waiting.append(followups[rid])

    with caplog.at_level(logging.INFO, logger="levanter.inference.engine"):
        cached.generate_continuously(
            [live, short], poll, complete, should_abort=lambda: False, on_tokens=lambda *_: None
        )

    assert completed == [short.request_id, fresh.request_id, reused.request_id, live.request_id]
    assert "Reused 4 prompt tokens" in caplog.text


@pytest.mark.skipif(jax.device_count() != 4, reason="Requires a four-device expert mesh")
@pytest.mark.parametrize(
    "expert_implementation,decode_layers",
    [("grouped", "unrolled"), ("separate", "unrolled_split_cache"), ("split", "unrolled_split_cache")],
)
def test_snowball_expert_sharded_cache_preserves_five_head_generation(
    serving_model, expert_implementation, decode_layers
):
    config = dataclasses.replace(
        serving_model.config, hidden_dim=160, num_heads=20, num_kv_heads=5, inference_decode_implementation="unrolled"
    )
    serving = InferenceEngineConfig(
        max_seq_len=8,
        max_pages=4,
        max_seqs=1,
        page_size=2,
        max_queued_tokens=4,
        max_seqs_in_prefill=1,
        max_tokens_per_round=4,
        prefix_cache="last_prompt",
        compute_dtype=jnp.float32,
    )
    with (
        jax.set_mesh(compact_grug_mesh(expert_axis_size=4)),
        hax.axis_mapping({"batch": ("replica_dcn", "data", "expert"), "vocab": "model"}),
    ):
        first, second = _requests()[:2]
        second = dataclasses.replace(second, prompt_tokens=[1, 2, 5])
        vocab, key = hax.Axis("vocab", 23), jax.random.key(41)
        model = SnowballLMHeadModel.init(vocab, config, key=key)
        model_cache = InferenceEngine.from_model_with_config(model, None, serving)
        expert_config = dataclasses.replace(
            config,
            inference_kv_sharding="expert",
            inference_expert_implementation=expert_implementation,
            inference_decode_implementation=decode_layers,
        )
        sharded = SnowballLMHeadModel.init(vocab, expert_config, key=key)
        expert_cache = InferenceEngine.from_model_with_config(sharded, None, serving)
        for request in (first, second):
            assert expert_cache.generate([request]) == model_cache.generate([request])
        cache = expert_cache.gen_state.cache
        assert isinstance(cache, StackedKvPageCache)
        assert cache.kv_pages.axis_size("kv_head") == 16
        sharding = cache.kv_pages.array.sharding
        assert isinstance(sharding, jax.sharding.NamedSharding)
        assert sharding.spec[3] == "expert"


@pytest.mark.skipif(jax.device_count() != 4, reason="Requires a four-device expert mesh")
def test_packed_sampling_matches_serial_requests_on_expert_mesh(serving_model):
    config = dataclasses.replace(
        serving_model.config,
        hidden_dim=32,
        num_heads=4,
        num_kv_heads=2,
        inference_kv_sharding="expert",
        inference_decode_implementation="unrolled",
    )
    packed = InferenceEngineConfig(
        max_seq_len=8,
        max_pages=16,
        max_seqs=8,
        page_size=4,
        max_queued_tokens=8,
        max_seqs_in_prefill=4,
        max_tokens_per_round=4,
        compute_dtype=jnp.float32,
        return_policy_logprobs=True,
    )
    with (
        jax.set_mesh(compact_grug_mesh(expert_axis_size=4)),
        hax.axis_mapping({"batch": ("replica_dcn", "data", "expert"), "vocab": "model"}),
    ):
        model = SnowballLMHeadModel.init(serving_model.Vocab, config, key=jax.random.key(41))
        parallel = InferenceEngine.from_model_with_config(model, None, packed)
        serial = InferenceEngine.from_model_with_config(
            model, None, dataclasses.replace(packed, max_seqs=1, max_seqs_in_prefill=1)
        )
        requests = _requests()
        requests.append(dataclasses.replace(requests[0], request_id=505))
        expected = [serial.generate([request]).tokens[0] for request in requests]
        result = parallel.generate(requests)
        assert result.tokens == expected
        assert result.policy_logprobs == result.logprobs


@eqx.filter_jit
def _decode(model, cache, tokens, info, positions):
    return model.decode(tokens, cache, info, positions)


@eqx.filter_jit
def _capture(model, cache, tokens, info, positions):
    return model.decode_with_route_captures(tokens, cache, info, positions)


def _public_serving_oracle(model, payload):
    """Teacher-force each actual input through independent public cached decode."""
    cache = model.initial_cache(PageTableSpec(2, 4, 1), dtype=jnp.float32)
    outputs = []
    routes = []
    for position, token in enumerate(payload.tokens):
        info = PageBatchInfo(
            slot_ids=hax.named(jnp.asarray([0], jnp.int32), "seq"),
            page_indices=hax.named(jnp.asarray([[0, 1]], jnp.int32), ("seq", "page")),
            seq_lens=hax.named(jnp.asarray([position + 1], jnp.int32), "seq"),
            cu_q_lens=hax.named(jnp.asarray([0, 1], jnp.int32), "seq"),
            num_seqs=jnp.asarray(1, jnp.int32),
            new_token_dests=hax.named(jnp.asarray([position], jnp.int32), "position"),
            page_size=4,
        )
        tokens = hax.named(jnp.asarray([token], jnp.int32), "position")
        positions = hax.named(jnp.asarray([position], jnp.int32), "position")
        # Capture and noncapture use the same pre-call cache for independent
        # logit comparison before advancing the noncapturing state.
        logits, next_cache = _decode(model, cache, tokens, info, positions)
        captured_logits, _, captured = _capture(model, cache, tokens, info, positions)
        np.testing.assert_array_equal(np.asarray(logits.array), np.asarray(captured_logits.array))
        assert np.asarray(captured.token_ids).tolist() == [token]
        assert np.asarray(captured.position_ids).tolist() == [position]
        outputs.append(np.asarray(logits.array)[0])
        routes.append(jax.device_get(captured.route_captures))
        cache = next_cache
    return np.stack(outputs), routes


@pytest.mark.timeout(300)
def test_engine_replay_matches_real_scheduling_and_preserves_every_target(serving_model):
    requests = _requests()
    results = []
    for max_rounds in (1, 3):
        engine = _engine(serving_model, max_rounds)
        ordinary = engine.generate(requests)
        replay = generate_snowball_replay(engine, requests, policy_id="tiny-41")
        assert replay.generation == ordinary
        assert replay.generation.finish_reasons == [FinishReason.LENGTH] * 4
        assert [p.request_id for p in replay.payloads] == [r.request_id for r in requests]
        # More requests than slots, followed by a second call, must preserve
        # request identity even when physical pages and slot IDs are reused.
        second = generate_snowball_replay(engine, requests[-1:], policy_id="tiny-41")
        assert second.generation.tokens == ordinary.tokens[-1:]
        results.append(replay)
    assert results[0].generation == results[1].generation
    for payload, other in zip(results[0].payloads, results[1].payloads, strict=True):
        np.testing.assert_array_equal(payload.tokens, other.tokens)
        example, _ = payload.training_inputs(serving_model, policy_id="tiny-41")
        predictor_rows = np.flatnonzero(np.asarray(example.loss_weight.array))
        np.testing.assert_array_equal(predictor_rows, np.arange(payload.prompt_length - 1, len(payload.tokens) - 1))
        assert len(predictor_rows) == len(payload.sampled_logprobs)
        assert payload.positions[-1] == len(payload.tokens) - 1
        oracle, serving_routes = _public_serving_oracle(serving_model, payload)
        learner = np.asarray(replay_snowball_logits(serving_model, payload, policy_id="tiny-41").array)
        np.testing.assert_allclose(learner, oracle, rtol=1e-4, atol=1e-4)
        for position, layers in enumerate(serving_routes):
            for index, route in enumerate(layers):
                for field in ("selected_experts", "token_valid", "qb_alpha"):
                    np.testing.assert_array_equal(
                        getattr(payload.layers[index], field)[position], np.asarray(getattr(route, field))[0]
                    )
        next_logprobs = np.asarray(jax.nn.log_softmax(jnp.asarray(oracle[predictor_rows]), axis=-1))
        scored = next_logprobs[np.arange(len(predictor_rows)), payload.tokens[payload.prompt_length :]]
        np.testing.assert_allclose(payload.raw_policy_logprobs, scored, rtol=1e-4, atol=1e-4)
        # Distinct first/last weights detect either omitted endpoint or a
        # shifted causal mask without copying the learner loss implementation.
        weights = np.arange(1, len(scored) + 1, dtype=np.float32)
        with hax.axis_mapping({"batch": ("replica_dcn", "data", "expert"), "position": None, "vocab": "model"}):
            weighted = replay_snowball_loss(
                serving_model, payload, policy_id="tiny-41", target_weights=hax.named(jnp.asarray(weights), "target")
            )
        np.testing.assert_allclose(float(weighted), -np.sum(scored * weights) / weights.sum(), rtol=1e-4, atol=1e-4)


def test_served_routes_survive_padding_packing_and_policy_gradients(serving_model):
    requests = _requests()[:2]
    policy_id = "fixture:41"
    replay = generate_snowball_replay(_engine(serving_model, 3), requests, policy_id=policy_id)
    Batch, Position, Response = hax.Axis("batch", 2), hax.Axis("position", 8), hax.Axis("response", 4)
    tokens = np.zeros((Batch.size, Position.size), dtype=np.int32)
    positions = np.zeros_like(tokens)
    valid = np.zeros(tokens.shape, dtype=bool)
    for row, capture in enumerate(replay.payloads):
        start = Position.size - len(capture.tokens)
        tokens[row, start:] = capture.tokens
        positions[row, start:] = capture.positions
        valid[row, start:] = True

    selected, alpha = align_snowball_policy_routes(
        serving_model,
        replay.payloads,
        policy_id=policy_id,
        tokens=tokens,
        token_valid=valid,
        position_ids=positions,
    )
    with pytest.raises(ValueError, match="positions differ"):
        align_snowball_policy_routes(
            serving_model,
            replay.payloads,
            policy_id=policy_id,
            tokens=tokens,
            token_valid=valid,
            position_ids=positions + 1,
        )
    response = hax.ones((Batch, Response))
    batch = PolicyBatch(
        tokens=hax.named(jnp.asarray(tokens), (Batch, Position)),
        attention_mask=hax.named(jnp.asarray(valid), (Batch, Position)),
        position_ids=hax.named(jnp.asarray(positions), (Batch, Position)),
        response_mask=response,
        advantages=response,
        policy_weights=response,
        kl_weights=response * 0,
        old_logprobs=response * 0,
        reference_logprobs=None,
        route_captures=PolicyRouteCaptures(
            hax.named(jnp.asarray(selected), (Batch, Position, "layer", "top_k")),
            hax.named(jnp.asarray(alpha), (Batch, Position, "layer")),
        ),
    )
    packed = pack_policy_batch(batch, sequence_length=16, pad_token_id=0)

    def score(model, batch):
        scores = response_logprobs(model, batch)
        return (scores * batch.response_mask).sum().scalar(), scores

    with hax.axis_mapping({"batch": ("replica_dcn", "data", "expert"), "vocab": "model"}):
        _, unpacked_scores = score(serving_model, batch)
        (value, packed_scores), grads = eqx.filter_jit(eqx.filter_value_and_grad(score, has_aux=True))(
            serving_model, packed
        )
    np.testing.assert_allclose(
        np.asarray(packed_scores.array)[np.asarray(packed.response_mask.array, dtype=bool)],
        np.asarray(unpacked_scores.array).reshape(-1),
        atol=2e-5,
        rtol=2e-5,
    )
    assert jnp.isfinite(value)
    assert jnp.any(grads.transformer.blocks[0].mlp.router != 0)


@pytest.mark.timeout(300)
def test_engine_replay_stop_sequences_first_token_and_slot_reuse(serving_model):
    engine = _engine(serving_model, 3)
    requests = _requests()
    ordinary = engine.generate(requests)
    stop_requests = []
    for index, request in enumerate(requests):
        stop_length = (1, 2, 3, 1)[index]
        stop = ordinary.tokens[index][:stop_length]
        params = dataclasses.replace(
            request.decode_params, stop_tokens=hax.named(jnp.asarray([stop]), ("stop_seq", "position"))
        )
        stop_requests.append(dataclasses.replace(request, decode_params=params))
    baseline = engine.generate(stop_requests)
    replay = generate_snowball_replay(engine, stop_requests, policy_id="tiny-41")
    assert replay.generation == baseline
    assert replay.generation.finish_reasons == [FinishReason.STOP] * 4
    assert [len(tokens) for tokens in replay.generation.tokens] == [1, 2, 3, 1]
    for payload, request, generated in zip(replay.payloads, requests, baseline.tokens, strict=True):
        np.testing.assert_array_equal(payload.tokens, [*request.prompt_tokens, *generated])
        np.testing.assert_array_equal(payload.positions, np.arange(len(payload.tokens)))
        example, _ = payload.training_inputs(serving_model, policy_id="tiny-41")
        assert np.asarray(example.loss_weight.array).sum() == len(generated)
        assert all(layer.token_valid[-1] for layer in payload.layers)


@pytest.mark.timeout(180)
def test_engine_replay_rejects_unsupported_batch_before_mutation_and_handles_empty(serving_model):
    engine = _engine(serving_model, 3)
    requests = _requests()[:1]
    expected = engine.generate(requests)
    empty = generate_snowball_replay(engine, [], policy_id="tiny-41")
    assert empty.payloads == () and empty.generation.tokens == []
    for invalid in [
        dataclasses.replace(requests[0], n_generations=2),
        dataclasses.replace(requests[0], prompt_tokens=[]),
    ]:
        with pytest.raises(ValueError):
            generate_snowball_replay(engine, [invalid], policy_id="tiny-41")
        assert engine.generate(requests) == expected
    # Aborting/callbacks have no capture API: reject before entering generation.
    with pytest.raises(TypeError):
        generate_snowball_replay(engine, requests, policy_id="tiny-41", should_abort=lambda: True)
    assert engine.generate(requests) == expected
