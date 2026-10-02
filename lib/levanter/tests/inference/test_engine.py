# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import threading

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax import Axis
from jax.sharding import AxisType, Mesh, PartitionSpec as P

import levanter.inference.engine as engine_module
from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import FinishReason, SeqDecodingParams
from levanter.inference.page_table import PageTableSpec
from levanter.layers.kv_cache import KvPageCache
from levanter.utils.mesh import create_mesh_from_axis_specs


class DummyModel(eqx.Module):
    """Minimal model stub to drive GenerationService for tests.

    - `initial_cache` returns an empty KvPageCache sized to the page-table spec.
    - `decode` returns constant logits that strongly prefer token `EOS`.
    """

    Vocab: Axis = eqx.field(static=True)
    eos: int = eqx.field(static=True)

    def __init__(self, vocab_size: int, eos_id: int = 3):
        self.Vocab = Axis("vocab", vocab_size)
        self.eos = eos_id

    def initial_cache(self, spec: PageTableSpec, *, dtype):
        # The cache contents are unused by this dummy model, so only its shardability matters.
        # `KvPageCache.init` packs K and V into a single `2 * kv_heads` axis, so one kv head per
        # device keeps that axis divisible by a `model` mesh axis sized to the device count.
        kv_heads = Axis("kv_head", jax.device_count())
        head_size = Axis("embed", 1)
        return KvPageCache.init(spec, kv_heads, head_size, dtype=dtype)

    def decode(self, input_ids, kv_cache, batch_info, pos_ids):
        # Produce logits that prefer `eos` for every sampled position
        Pos = input_ids.resolve_axis("position")
        Vocab = self.Vocab
        # One-hot on vocab axis for eos token, broadcast over positions
        logits = hax.nn.one_hot(self.eos, Vocab, dtype=jnp.float32)
        logits = logits.broadcast_axis(Pos)
        return logits, kv_cache


def _build_service(vocab_size=10):
    model = DummyModel(vocab_size=vocab_size, eos_id=3)
    service = InferenceEngine.from_model_with_config(
        model=model,  # type: ignore
        tokenizer=None,
        config=InferenceEngineConfig(
            max_seq_len=32,
            max_pages=64,
            max_seqs=8,
            page_size=8,
            compute_dtype=jnp.float32,
            max_queued_tokens=64,
            max_seqs_in_prefill=4,
        ),
    )
    return service


def test_auto_page_sizing_uses_explicit_axis_resources(monkeypatch):
    # Device memory statistics are an accelerator I/O boundary. Supply a deterministic ample
    # budget so this test reaches the useful page-capacity bound on any host.
    monkeypatch.setattr(engine_module, "estimated_free_device_memory", lambda _device: 100.0)
    mesh = create_mesh_from_axis_specs(ici_axes={"model": jax.device_count()}, dcn_axes={})

    with hax.partitioning.set_mesh(mesh):
        service = InferenceEngine.from_model_with_config(
            model=DummyModel(vocab_size=10),  # type: ignore
            tokenizer=None,
            config=InferenceEngineConfig(
                max_seq_len=32,
                max_seqs=8,
                page_size=8,
                compute_dtype=jnp.float32,
                max_queued_tokens=64,
                max_seqs_in_prefill=4,
            ),
            axis_resources={"kv_head": "model"},
        )

    assert service.config.max_pages is not None
    assert service.config.max_pages == service.config.max_seqs * service.config.max_pages_per_seq


@pytest.mark.parametrize("max_rounds", [2, 32])
@pytest.mark.parametrize("generation_counts", [(1, 1, 1), (2, 2, 2), (1, 2, 1)])
def test_generate_completes_requests_across_prefill_and_slot_limits(max_rounds, generation_counts):
    model = DummyModel(vocab_size=10, eos_id=3)
    config = InferenceEngineConfig(
        max_seq_len=12,
        max_rounds=max_rounds,
        max_pages=8,
        max_seqs=2,
        page_size=4,
        max_queued_tokens=4,
        max_seqs_in_prefill=1,
        max_prefill_size=4,
        compute_dtype=jnp.float32,
    )
    service = InferenceEngine.from_model_with_config(model, None, config)
    requests = [
        Request(
            prompt_tokens=[1, 2],
            request_id=request_id,
            decode_params=dataclasses.replace(SeqDecodingParams.default(), max_num_tokens=jnp.array(5 + request_id)),
            n_generations=generation_counts[request_id],
        )
        for request_id in range(3)
    ]
    result = service.generate(requests)
    expected = [[3] * (3 + request_id) for request_id in range(3) for _ in range(generation_counts[request_id])]
    assert result.tokens == expected
    assert result.total_generated == sum(map(len, expected))
    assert [len(values) for values in result.logprobs] == list(map(len, expected))
    # A second batch must reuse the released slots and pages without retaining earlier output.
    assert service.generate(requests[:1]).tokens == expected[: generation_counts[0]]


class ShardedLogitModel(DummyModel):
    def decode(self, input_ids, kv_cache, batch_info, pos_ids):
        logits, cache = super().decode(input_ids, kv_cache, batch_info, pos_ids)
        return hax.NamedArray(jax.sharding.reshard(logits.array, P("data", "model")), logits.axes), cache


def test_generate_samples_explicit_token_and_vocabulary_shards():
    mesh = Mesh(
        np.asarray(jax.devices()).reshape(1, -1),
        ("data", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    with jax.set_mesh(mesh):
        service = InferenceEngine.from_model_with_config(
            ShardedLogitModel(vocab_size=8 * jax.device_count()),
            None,
            InferenceEngineConfig(
                max_seq_len=8,
                max_rounds=2,
                max_pages=8,
                max_seqs=2,
                page_size=4,
                max_queued_tokens=2,
                max_seqs_in_prefill=1,
                max_prefill_size=4,
                compute_dtype=jnp.float32,
            ),
        )
        request = Request(
            prompt_tokens=[1, 2, 3, 4],
            request_id=10,
            decode_params=dataclasses.replace(SeqDecodingParams.default(), max_num_tokens=jnp.array(7)),
            n_generations=2,
        )
        result = service.generate([request])
    assert result.tokens == [[3, 3, 3], [3, 3, 3]]
    assert all(np.isfinite(values).all() for values in result.logprobs)


def test_generation_preserves_stop_and_length_reasons_across_clones_and_reuse():
    service = _build_service()
    stops = hax.named(jnp.array([[-1, 4], [3, 3]]), ("stop_seq", "position"))
    requests = [
        Request(
            prompt_tokens=[1, 2],
            request_id=0,
            n_generations=2,
            decode_params=dataclasses.replace(
                SeqDecodingParams.default(), max_num_tokens=jnp.array(4), stop_tokens=stops
            ),
        ),
        Request(
            prompt_tokens=[1, 2],
            request_id=1,
            n_generations=1,
            decode_params=dataclasses.replace(SeqDecodingParams.default(), max_num_tokens=jnp.array(5)),
        ),
    ]
    result = service.generate(requests)
    assert result.tokens == [[3, 3], [3, 3], [3, 3, 3]]
    # A matched stop wins when it coincides with the length limit.
    assert result.finish_reasons == [FinishReason.STOP, FinishReason.STOP, FinishReason.LENGTH]
    assert [len(row) for row in result.logprobs] == [2, 2, 3]
    assert service.generate(requests[1:]).finish_reasons == [FinishReason.LENGTH]


@pytest.mark.parametrize("abort_before_prefill", [False, True])
def test_abort_returns_exact_partial_results_and_releases_slots(abort_before_prefill):
    service = _build_service()
    request = Request(
        prompt_tokens=[1, 2],
        request_id=10,
        n_generations=2,
        decode_params=dataclasses.replace(
            SeqDecodingParams.default(), max_num_tokens=jnp.array(7), temperature=jnp.array(0.0)
        ),
    )
    uninterrupted = service.generate([request])
    abort = threading.Event()
    if abort_before_prefill:
        abort.set()
    partial = service.generate([request], step_callback=lambda _: abort.set(), should_abort=abort.is_set)
    assert partial.finish_reasons == [FinishReason.ABORT, FinishReason.ABORT]
    expected_count = 0 if abort_before_prefill else 1
    assert [len(tokens) for tokens in partial.tokens] == [expected_count, expected_count]
    assert len(service.free_slots) == service.config.max_seqs
    assert not service.sequences
    for tokens, logprobs, complete_tokens, complete_logprobs in zip(
        partial.tokens, partial.logprobs, uninterrupted.tokens, uninterrupted.logprobs, strict=True
    ):
        continuation = dataclasses.replace(request, prompt_tokens=request.prompt_tokens + tokens, n_generations=1)
        resumed = service.generate([continuation])
        assert tokens + resumed.tokens[0] == complete_tokens
        assert logprobs + resumed.logprobs[0] == pytest.approx(complete_logprobs)
        assert resumed.finish_reasons == [FinishReason.LENGTH]


def test_abort_keeps_already_finished_stop_reasons():
    service = _build_service()
    params = dataclasses.replace(SeqDecodingParams.default(), max_num_tokens=jnp.array(7))
    requests = [
        Request(
            prompt_tokens=[1, 2],
            request_id=0,
            n_generations=1,
            decode_params=dataclasses.replace(
                params, stop_tokens=hax.named(jnp.array([[3]]), ("stop_seq", "position"))
            ),
        ),
        Request(prompt_tokens=[1, 2], request_id=1, decode_params=params, n_generations=1),
    ]
    abort = threading.Event()
    result = service.generate(requests, step_callback=lambda _: abort.set(), should_abort=abort.is_set)
    assert result.tokens == [[3], [3]]
    assert result.finish_reasons == [FinishReason.STOP, FinishReason.ABORT]
