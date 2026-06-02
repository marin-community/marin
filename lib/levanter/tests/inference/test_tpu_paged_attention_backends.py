# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import math
import dataclasses

import jax.numpy as jnp
import jax.random as jr
import pytest
from chex import assert_trees_all_close

import haliax as hax
from haliax import Axis

import levanter.inference.tpu_kernels as tpu_kernels
from levanter.inference.page_table import PageBatchInfo, PageTableSpec
from levanter.inference.tpu_kernels import (
    TpuPagedAttentionBackend,
    TpuPagedAttentionConfig,
    TpuPagedAttentionFallbackWarning,
    TpuPagedAttentionShape,
    UnsupportedTpuPagedAttentionBackend,
    available_tpu_paged_attention_backends,
    paged_attention_with_kv_update,
    tpu_paged_attention_supports_shape,
)
from levanter.inference.utils import INVALID
from levanter.layers.attention import default_ragged_paged_attention
from levanter.layers.kv_cache import KvPageCache


PAGE_SIZE = 4
HEAD_SIZE = 128
SM_SCALE = 1.0 / math.sqrt(HEAD_SIZE)


def _single_sequence_case():
    Position = Axis("position", 2)
    KVHead = Axis("kv_head", 1)
    QHeadsPerGroup = Axis("q_heads_per_group", 1)
    Head = Axis("head_size", HEAD_SIZE)
    Seq = Axis("seq", 1)
    SeqPlusOne = Axis("seq", 2)
    Page = Axis("page", 1)

    rng = jr.PRNGKey(0)
    q = hax.random.normal(rng, (Position, KVHead, QHeadsPerGroup, Head), dtype=jnp.bfloat16)
    new_k = hax.random.normal(jr.fold_in(rng, 1), (Position, KVHead, Head), dtype=jnp.bfloat16)
    new_v = hax.random.normal(jr.fold_in(rng, 2), (Position, KVHead, Head), dtype=jnp.bfloat16)
    kv_cache = KvPageCache.init(PageTableSpec(num_pages=1, page_size=PAGE_SIZE), KVHead, Head, dtype=jnp.bfloat16)

    batch_info = PageBatchInfo(
        slot_ids=hax.named(jnp.asarray([0], dtype=jnp.int32), Seq),
        page_indices=hax.named(jnp.asarray([[0]], dtype=jnp.int32), (Seq, Page)),
        seq_lens=hax.named(jnp.asarray([2], dtype=jnp.int32), Seq),
        cu_q_lens=hax.named(jnp.asarray([0, 2], dtype=jnp.int32), SeqPlusOne),
        num_seqs=jnp.asarray(1, dtype=jnp.int32),
        new_token_dests=hax.named(jnp.asarray([0, 1], dtype=jnp.int32), Position),
        page_size=PAGE_SIZE,
    )
    return q, new_k, new_v, kv_cache, batch_info


def _multi_sequence_page_boundary_case():
    Position = Axis("position", 3)
    KVHead = Axis("kv_head", 2)
    QHeadsPerGroup = Axis("q_heads_per_group", 2)
    Head = Axis("head_size", HEAD_SIZE)
    Seq = Axis("seq", 2)
    SeqPlusOne = Axis("seq", 3)
    Page = Axis("page", 3)

    rng = jr.PRNGKey(1)
    q = hax.random.normal(rng, (Position, KVHead, QHeadsPerGroup, Head), dtype=jnp.bfloat16)
    new_k = hax.random.normal(jr.fold_in(rng, 1), (Position, KVHead, Head), dtype=jnp.bfloat16)
    new_v = hax.random.normal(jr.fold_in(rng, 2), (Position, KVHead, Head), dtype=jnp.bfloat16)
    kv_cache = KvPageCache.init(PageTableSpec(num_pages=3, page_size=PAGE_SIZE), KVHead, Head, dtype=jnp.bfloat16)

    batch_info = PageBatchInfo(
        slot_ids=hax.named(jnp.asarray([0, 1], dtype=jnp.int32), Seq),
        page_indices=hax.named(
            jnp.asarray(
                [
                    [0, 1, INVALID],
                    [2, INVALID, INVALID],
                ],
                dtype=jnp.int32,
            ),
            (Seq, Page),
        ),
        seq_lens=hax.named(jnp.asarray([5, 1], dtype=jnp.int32), Seq),
        cu_q_lens=hax.named(jnp.asarray([0, 2, 3], dtype=jnp.int32), SeqPlusOne),
        num_seqs=jnp.asarray(2, dtype=jnp.int32),
        new_token_dests=hax.named(jnp.asarray([3, 4, 8], dtype=jnp.int32), Position),
        page_size=PAGE_SIZE,
    )
    return q, new_k, new_v, kv_cache, batch_info


def _expected_attention_after_update(q, new_k, new_v, kv_cache, batch_info, *, soft_cap):
    expected_cache = kv_cache.update(batch_info, new_k, new_v)
    expected = default_ragged_paged_attention(
        q,
        expected_cache.kv_pages,
        batch_info.seq_lens,
        batch_info.page_indices,
        batch_info.cu_q_lens.array,
        batch_info.num_seqs,
        sm_scale=SM_SCALE,
        soft_cap=soft_cap,
    )
    return expected, expected_cache


def _assert_backend_matches_reference(case_factory, *, backend: TpuPagedAttentionBackend, soft_cap: float | None):
    expected_q, expected_new_k, expected_new_v, expected_cache_input, expected_batch_info = case_factory()
    expected, expected_cache = _expected_attention_after_update(
        expected_q,
        expected_new_k,
        expected_new_v,
        expected_cache_input,
        expected_batch_info,
        soft_cap=soft_cap,
    )
    q, new_k, new_v, kv_cache, batch_info = case_factory()

    attn, updated_cache = paged_attention_with_kv_update(
        q,
        new_k,
        new_v,
        kv_cache,
        batch_info,
        sm_scale=SM_SCALE,
        soft_cap=soft_cap,
        config=TpuPagedAttentionConfig(backend=backend),
    )

    assert attn.axes == q.axes
    assert_trees_all_close(attn.array, expected.array, atol=1e-3, rtol=1e-3)
    assert_trees_all_close(updated_cache.kv_pages.array, expected_cache.kv_pages.array)


def test_auto_backend_updates_cache_and_matches_reference_on_cpu():
    _assert_backend_matches_reference(
        _single_sequence_case,
        backend=TpuPagedAttentionBackend.AUTO,
        soft_cap=None,
    )


@pytest.mark.parametrize("backend", [TpuPagedAttentionBackend.AUTO, TpuPagedAttentionBackend.REFERENCE])
@pytest.mark.parametrize("soft_cap", [None, 30.0])
def test_backend_matches_reference_on_multi_sequence_page_boundary_grouped_heads(backend, soft_cap):
    _assert_backend_matches_reference(
        _multi_sequence_page_boundary_case,
        backend=backend,
        soft_cap=soft_cap,
    )


def test_explicit_tpu_inference_backend_fails_fast_on_cpu():
    q, new_k, new_v, kv_cache, batch_info = _single_sequence_case()

    try:
        paged_attention_with_kv_update(
            q,
            new_k,
            new_v,
            kv_cache,
            batch_info,
            sm_scale=SM_SCALE,
            soft_cap=None,
            config=TpuPagedAttentionConfig(backend=TpuPagedAttentionBackend.TPU_INFERENCE),
        )
    except UnsupportedTpuPagedAttentionBackend as exc:
        assert "only runs on TPU" in str(exc)
    else:
        raise AssertionError("explicit tpu-inference backend should fail on CPU")


def test_backend_sequence_warns_then_runs_reference_on_cpu():
    q, new_k, new_v, kv_cache, batch_info = _single_sequence_case()
    config = TpuPagedAttentionConfig(
        backend=(TpuPagedAttentionBackend.TPU_INFERENCE, TpuPagedAttentionBackend.REFERENCE),
        fail_on_reference_fallback=False,
    )

    with pytest.warns(TpuPagedAttentionFallbackWarning):
        attn, updated_cache = paged_attention_with_kv_update(
            q,
            new_k,
            new_v,
            kv_cache,
            batch_info,
            sm_scale=SM_SCALE,
            soft_cap=None,
            config=config,
        )

    assert attn.axes == q.axes
    assert updated_cache.kv_pages.array.shape == kv_cache.kv_pages.array.shape


def test_auto_backend_warns_then_uses_jax_rpa_on_tpu_when_tpu_inference_is_unavailable(monkeypatch):
    expected_q, expected_new_k, expected_new_v, expected_cache_input, expected_batch_info = _single_sequence_case()
    expected, expected_cache = _expected_attention_after_update(
        expected_q,
        expected_new_k,
        expected_new_v,
        expected_cache_input,
        expected_batch_info,
        soft_cap=None,
    )
    q, new_k, new_v, kv_cache, batch_info = _single_sequence_case()

    def unavailable_tpu_backend(*args, **kwargs):
        del args, kwargs
        raise UnsupportedTpuPagedAttentionBackend("tpu-inference import failed")

    def fake_jax_rpa_backend(q, new_k, new_v, kv_cache, batch_info, *, sm_scale, soft_cap, config):
        del sm_scale, config
        return _expected_attention_after_update(q, new_k, new_v, kv_cache, batch_info, soft_cap=soft_cap)

    monkeypatch.setattr(tpu_kernels, "_current_platform", lambda: "tpu")
    monkeypatch.setattr(tpu_kernels, "_run_tpu_inference_backend", unavailable_tpu_backend)
    monkeypatch.setattr(tpu_kernels, "_run_jax_rpa_backend", fake_jax_rpa_backend)

    with pytest.warns(TpuPagedAttentionFallbackWarning):
        attn, updated_cache = paged_attention_with_kv_update(
            q,
            new_k,
            new_v,
            kv_cache,
            batch_info,
            sm_scale=SM_SCALE,
            soft_cap=None,
            config=TpuPagedAttentionConfig(),
        )

    assert_trees_all_close(attn.array, expected.array, atol=1e-3, rtol=1e-3)
    assert_trees_all_close(updated_cache.kv_pages.array, expected_cache.kv_pages.array)


def test_reference_backend_is_not_a_silent_tpu_fallback(monkeypatch):
    q, new_k, new_v, kv_cache, batch_info = _single_sequence_case()
    monkeypatch.setattr("levanter.inference.tpu_kernels._current_platform", lambda: "tpu")

    try:
        paged_attention_with_kv_update(
            q,
            new_k,
            new_v,
            kv_cache,
            batch_info,
            sm_scale=SM_SCALE,
            soft_cap=None,
            config=TpuPagedAttentionConfig(backend=TpuPagedAttentionBackend.REFERENCE),
        )
    except ValueError as exc:
        assert "fail_on_reference_fallback=True" in str(exc)
    else:
        raise AssertionError("reference backend should fail on TPU when fail_on_reference_fallback=True")


def test_duplicate_token_destinations_raise_before_backend_dispatch():
    q, new_k, new_v, kv_cache, batch_info = _single_sequence_case()
    duplicate_batch_info = dataclasses.replace(
        batch_info,
        new_token_dests=hax.named(jnp.asarray([0, 0], dtype=jnp.int32), q.resolve_axis("position")),
    )

    try:
        paged_attention_with_kv_update(
            q,
            new_k,
            new_v,
            kv_cache,
            duplicate_batch_info,
            sm_scale=SM_SCALE,
            soft_cap=None,
            config=TpuPagedAttentionConfig(),
        )
    except ValueError as exc:
        assert "duplicate valid destinations" in str(exc)
    else:
        raise AssertionError("duplicate destinations should fail before cache update")


def test_available_backends_on_cpu_only_report_reference():
    assert available_tpu_paged_attention_backends() == (TpuPagedAttentionBackend.REFERENCE,)


def test_initial_qwen3_shape_support_matches_target_tensor_parallelism():
    target_shape = TpuPagedAttentionShape(
        platform="tpu",
        device_kind="TPU v5",
        dtype=jnp.bfloat16,
        page_size=128,
        head_size=128,
        num_q_heads=32,
        num_kv_heads=8,
        q_heads_per_group=4,
        max_model_len=4096,
        tensor_parallel_size=4,
    )

    supported, reason = tpu_paged_attention_supports_shape(TpuPagedAttentionBackend.TPU_INFERENCE, target_shape)

    assert supported
    assert reason is None


def test_tpu_inference_adapter_casts_query_to_requested_output_dtype():
    import levanter.inference.tpu_inference_adapter as adapter

    q_array = jnp.ones((1, 1, HEAD_SIZE), dtype=jnp.bfloat16)

    assert adapter._query_array_for_out_dtype(q_array, None).dtype == jnp.bfloat16
    assert adapter._query_array_for_out_dtype(q_array, jnp.bfloat16).dtype == jnp.bfloat16
    assert adapter._query_array_for_out_dtype(q_array, jnp.float32).dtype == jnp.float32
