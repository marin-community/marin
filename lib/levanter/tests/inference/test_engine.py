# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import pytest
from haliax import Axis

from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request, _streaming_greedy_lm_head
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.inference.page_table import PageTableSpec
from levanter.layers.kv_cache import KvPageCache


class DummyModel(eqx.Module):
    """Minimal model stub to drive GenerationService for tests.

    - `initial_cache` returns an empty KvPageCache sized to the page-table spec.
    - `decode` returns constant logits that strongly prefer token `EOS`.
    """

    Vocab: Axis = eqx.field(static=True)
    Embed: Axis = eqx.field(static=True)
    eos: int = eqx.field(static=True)

    def __init__(self, vocab_size: int, eos_id: int = 3):
        self.Vocab = Axis("vocab", vocab_size)
        self.Embed = Axis("embed", 1)
        self.eos = eos_id

    def initial_cache(self, spec: PageTableSpec, *, dtype):
        # Use trivial cache dimensions; the cache is unused by this dummy model
        kv_heads = Axis("kv_head", 1)
        head_size = Axis("embed", 1)
        return KvPageCache.init(spec, kv_heads, head_size, dtype=dtype)

    def decode(self, input_ids, kv_cache, batch_info, pos_ids, *, tpu_paged_attention=None):
        del tpu_paged_attention
        # Produce logits that prefer `eos` for every sampled position
        Pos = input_ids.resolve_axis("position")
        Vocab = self.Vocab
        # One-hot on vocab axis for eos token, broadcast over positions
        logits = hax.nn.one_hot(self.eos, Vocab, dtype=jnp.float32)
        logits = logits.broadcast_axis(Pos)
        return logits, kv_cache

    def decode_hidden(self, input_ids, kv_cache, batch_info, pos_ids, *, tpu_paged_attention=None):
        del batch_info, pos_ids, tpu_paged_attention
        Pos = input_ids.resolve_axis("position")
        return hax.ones((Pos, self.Embed), dtype=jnp.float32), kv_cache

    def lm_head_logits(self, hidden):
        Pos = hidden.resolve_axis("position")
        logits = hax.nn.one_hot(self.eos, self.Vocab, dtype=jnp.float32)
        return logits.broadcast_axis(Pos)

    def get_lm_head(self):
        return hax.nn.one_hot(self.eos, self.Vocab, dtype=jnp.float32).broadcast_axis(self.Embed)


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


def test_prefill_work_prompt_tokens_are_bounded_by_prefill_size():
    service = InferenceEngine.from_model_with_config(
        model=DummyModel(vocab_size=10, eos_id=3),  # type: ignore
        tokenizer=None,
        config=InferenceEngineConfig(
            max_seq_len=32,
            max_pages=64,
            max_seqs=8,
            page_size=8,
            compute_dtype=jnp.float32,
            max_queued_tokens=64,
            max_seqs_in_prefill=4,
            max_prefill_size=4,
        ),
    )
    request = Request(
        prompt_tokens=[1, 2],
        request_id=0,
        decode_params=SeqDecodingParams(
            max_num_tokens=jnp.asarray(4, dtype=jnp.int32),
            stop_tokens=None,
            temperature=jnp.asarray(0.0, dtype=jnp.float32),
            top_p=jnp.asarray(1.0, dtype=jnp.float32),
            top_k=jnp.asarray(0, dtype=jnp.int32),
            key=jax.random.PRNGKey(0),
        ),
        n_generations=1,
    )

    work = service._prefill_prompts([request])

    assert work is not None
    assert work.prompt_tokens.axis_size("position") == 4
    assert work.prompt_tokens.array.shape == (8, 4)
    assert work.prompt_tokens.array[0, :2].tolist() == [1, 2]


def test_generate_without_lm_head_emits_dummy_tokens_after_prefill():
    service = _build_service()
    request = Request(
        prompt_tokens=[1],
        request_id=0,
        decode_params=SeqDecodingParams(
            max_num_tokens=jnp.asarray(4, dtype=jnp.int32),
            stop_tokens=None,
            temperature=jnp.asarray(0.0, dtype=jnp.float32),
            top_p=jnp.asarray(1.0, dtype=jnp.float32),
            top_k=jnp.asarray(0, dtype=jnp.int32),
            key=jax.random.PRNGKey(0),
        ),
        n_generations=1,
    )

    result = service.generate_without_lm_head([request])

    assert result.total_generated == 3
    assert result.logprobs is None
    assert result.tokens == [[3, 0, 0]]


def test_sampled_top_k_requires_configured_limit():
    service = _build_service()
    request = Request(
        prompt_tokens=[1],
        request_id=0,
        decode_params=SeqDecodingParams(
            max_num_tokens=jnp.asarray(4, dtype=jnp.int32),
            stop_tokens=None,
            temperature=jnp.asarray(1.0, dtype=jnp.float32),
            top_p=jnp.asarray(1.0, dtype=jnp.float32),
            top_k=jnp.asarray(2, dtype=jnp.int32),
            key=jax.random.PRNGKey(0),
        ),
        n_generations=1,
    )

    with pytest.raises(ValueError, match="max_top_k"):
        service.generate([request])


def test_sampled_request_without_top_k_fails_when_limit_configured():
    model = DummyModel(vocab_size=10, eos_id=3)
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
            max_top_k=4,
        ),
    )
    request = Request(
        prompt_tokens=[1],
        request_id=0,
        decode_params=SeqDecodingParams(
            max_num_tokens=jnp.asarray(4, dtype=jnp.int32),
            stop_tokens=None,
            temperature=jnp.asarray(1.0, dtype=jnp.float32),
            top_p=jnp.asarray(1.0, dtype=jnp.float32),
            top_k=jnp.asarray(0, dtype=jnp.int32),
            key=jax.random.PRNGKey(0),
        ),
        n_generations=1,
    )

    with pytest.raises(ValueError, match="Sampled requests must specify top_k"):
        service.generate([request])


def test_greedy_request_without_top_k_allowed_when_limit_configured():
    model = DummyModel(vocab_size=10, eos_id=3)
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
            max_top_k=4,
        ),
    )
    request = Request(
        prompt_tokens=[1],
        request_id=0,
        decode_params=SeqDecodingParams(
            max_num_tokens=jnp.asarray(4, dtype=jnp.int32),
            stop_tokens=None,
            temperature=jnp.asarray(0.0, dtype=jnp.float32),
            top_p=jnp.asarray(1.0, dtype=jnp.float32),
            top_k=jnp.asarray(0, dtype=jnp.int32),
            key=jax.random.PRNGKey(0),
        ),
        n_generations=1,
    )

    result = service.generate([request])

    assert result.tokens == [[3, 3, 3]]


def test_requests_need_top_p_only_when_request_filters_nucleus():
    service = _build_service()
    full_top_p = Request(
        prompt_tokens=[1],
        request_id=0,
        decode_params=SeqDecodingParams(
            max_num_tokens=jnp.asarray(4, dtype=jnp.int32),
            stop_tokens=None,
            temperature=jnp.asarray(1.0, dtype=jnp.float32),
            top_p=jnp.asarray(1.0, dtype=jnp.float32),
            top_k=jnp.asarray(0, dtype=jnp.int32),
            key=jax.random.PRNGKey(0),
        ),
        n_generations=1,
    )
    filtered_top_p = dataclasses.replace(
        full_top_p,
        request_id=1,
        decode_params=dataclasses.replace(full_top_p.decode_params, top_p=jnp.asarray(0.95, dtype=jnp.float32)),
    )

    assert not service._requests_need_top_p([full_top_p])
    assert service._requests_need_top_p([full_top_p, filtered_top_p])


def test_generate_with_lm_head_no_sampling_emits_dummy_tokens_after_prefill():
    service = _build_service()
    request = Request(
        prompt_tokens=[1],
        request_id=0,
        decode_params=SeqDecodingParams(
            max_num_tokens=jnp.asarray(4, dtype=jnp.int32),
            stop_tokens=None,
            temperature=jnp.asarray(0.0, dtype=jnp.float32),
            top_p=jnp.asarray(1.0, dtype=jnp.float32),
            top_k=jnp.asarray(0, dtype=jnp.int32),
            key=jax.random.PRNGKey(0),
        ),
        n_generations=1,
    )

    result = service.generate_with_lm_head_no_sampling([request])

    assert result.total_generated == 3
    assert result.logprobs is None
    assert result.tokens == [[3, 0, 0]]


def test_generate_streaming_greedy_lm_head_emits_argmax_tokens_after_prefill():
    model = DummyModel(vocab_size=10, eos_id=3)
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
            use_streaming_greedy_lm_head=True,
        ),
    )
    request = Request(
        prompt_tokens=[1],
        request_id=0,
        decode_params=SeqDecodingParams(
            max_num_tokens=jnp.asarray(4, dtype=jnp.int32),
            stop_tokens=None,
            temperature=jnp.asarray(0.0, dtype=jnp.float32),
            top_p=jnp.asarray(1.0, dtype=jnp.float32),
            top_k=jnp.asarray(0, dtype=jnp.int32),
            key=jax.random.PRNGKey(0),
        ),
        n_generations=1,
    )

    result = service.generate([request])

    assert result.total_generated == 3
    assert result.logprobs is None
    assert result.tokens == [[3, 3, 3]]


def test_generate_admits_multiple_prefill_chunks_for_one_logical_batch():
    model = DummyModel(vocab_size=10, eos_id=3)
    service = InferenceEngine.from_model_with_config(
        model=model,  # type: ignore
        tokenizer=None,
        config=InferenceEngineConfig(
            max_seq_len=32,
            max_pages=64,
            max_seqs=4,
            page_size=8,
            compute_dtype=jnp.float32,
            max_queued_tokens=16,
            max_rounds=1,
            max_prefill_size=4,
            max_seqs_in_prefill=4,
            use_streaming_greedy_lm_head=True,
        ),
    )
    requests = [
        Request(
            prompt_tokens=[1, 2],
            request_id=request_id,
            decode_params=SeqDecodingParams(
                max_num_tokens=jnp.asarray(4, dtype=jnp.int32),
                stop_tokens=None,
                temperature=jnp.asarray(0.0, dtype=jnp.float32),
                top_p=jnp.asarray(1.0, dtype=jnp.float32),
                top_k=jnp.asarray(0, dtype=jnp.int32),
                key=jax.random.PRNGKey(request_id),
            ),
            n_generations=1,
        )
        for request_id in range(4)
    ]

    result = service.generate(requests)

    assert result.tokens == [[3, 3], [3, 3], [3, 3], [3, 3]]
    assert result.total_generated == 8
    assert result.prefill_admissions == 2
    assert result.prefill_prompt_tokens_per_admission == [4, 4]
    assert len(result.prefill_seconds_per_admission) == 2
    assert all(seconds >= 0.0 for seconds in result.prefill_seconds_per_admission)
    assert result.decode_seconds_per_iteration
    assert len(result.decode_seconds_per_iteration) == len(result.decode_device_seconds_per_iteration)
    assert len(result.decode_seconds_per_iteration) == len(result.decode_host_seconds_per_iteration)
    assert len(result.decode_seconds_per_iteration) == len(result.decode_submit_seconds_per_iteration)
    assert len(result.decode_seconds_per_iteration) == len(result.decode_extract_seconds_per_iteration)
    assert 0 < sum(result.decode_tokens_per_iteration) <= result.total_generated
    assert all(seconds >= 0.0 for seconds in result.decode_device_seconds_per_iteration)


@pytest.mark.parametrize("method_name", ["generate_without_lm_head", "generate_with_lm_head_no_sampling"])
def test_diagnostic_generation_supports_multi_prefill(method_name):
    model = DummyModel(vocab_size=10, eos_id=3)
    service = InferenceEngine.from_model_with_config(
        model=model,  # type: ignore
        tokenizer=None,
        config=InferenceEngineConfig(
            max_seq_len=32,
            max_pages=64,
            max_seqs=4,
            page_size=8,
            compute_dtype=jnp.float32,
            max_queued_tokens=16,
            max_prefill_size=4,
            max_seqs_in_prefill=4,
        ),
    )
    requests = [
        Request(
            prompt_tokens=[1, 2],
            request_id=request_id,
            decode_params=SeqDecodingParams(
                max_num_tokens=jnp.asarray(4, dtype=jnp.int32),
                stop_tokens=None,
                temperature=jnp.asarray(0.0, dtype=jnp.float32),
                top_p=jnp.asarray(1.0, dtype=jnp.float32),
                top_k=jnp.asarray(0, dtype=jnp.int32),
                key=jax.random.PRNGKey(request_id),
            ),
            n_generations=1,
        )
        for request_id in range(3)
    ]

    result = getattr(service, method_name)(requests)

    assert result.total_generated == 6
    assert result.prefill_admissions == 2
    assert result.prefill_prompt_tokens_per_admission == [4, 2]
    assert len(result.prefill_seconds_per_admission) == 2
    assert result.decode_seconds_per_iteration
    assert len(result.decode_seconds_per_iteration) == len(result.decode_device_seconds_per_iteration)
    assert len(result.decode_seconds_per_iteration) == len(result.decode_host_seconds_per_iteration)
    assert len(result.decode_seconds_per_iteration) == len(result.decode_submit_seconds_per_iteration)
    assert len(result.decode_seconds_per_iteration) == len(result.decode_extract_seconds_per_iteration)
    assert 0 < sum(result.decode_tokens_per_iteration) <= result.total_generated


def test_streaming_greedy_lm_head_returns_logprobs_without_materialized_logits():
    Pos = Axis("position", 2)
    Embed = Axis("embed", 2)
    Vocab = Axis("vocab", 3)

    class Model(eqx.Module):
        Vocab: Axis = eqx.field(static=True)
        Embed: Axis = eqx.field(static=True)

        def __init__(self):
            self.Vocab = Vocab
            self.Embed = Embed

        def get_lm_head(self):
            return hax.named(
                jnp.array(
                    [
                        [0.0, 0.0],
                        [2.0, 0.0],
                        [0.0, 3.0],
                    ],
                    dtype=jnp.float32,
                ),
                (Vocab, Embed),
            )

    hidden = hax.named(jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32), (Pos, Embed))

    tokens, logprobs = _streaming_greedy_lm_head(Model(), hidden, return_logprobs=True)  # type: ignore[arg-type]

    logits = hidden.dot(Model().get_lm_head(), axis=Embed).array
    expected_tokens = jnp.argmax(logits, axis=-1)
    expected_logprobs = jnp.take_along_axis(
        jax.nn.log_softmax(logits, axis=-1),
        expected_tokens[:, None],
        axis=-1,
    ).squeeze(-1)
    assert tokens.array.tolist() == expected_tokens.tolist()
    assert logprobs.array == pytest.approx(expected_logprobs, abs=1e-6)
