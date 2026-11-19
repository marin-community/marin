# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import ml_dtypes
import numpy as np
import optax
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.data.dataset import ListAsyncDataset
from levanter.data.mixture import MixtureDataset, StopStrategy
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask
from levanter.schedule import BatchSchedule
from levanter.testing.devices import use_cpu_devices

from experiments.post_training.long_loop.sft_runtime.cached_topk_cache import (
    CachedTopKMixtureReader,
    CachedTopKSourceReader,
    left_shift_prompt_logprobs,
    row_tokens_sha256,
    write_cached_topk_source,
)
from experiments.post_training.long_loop.sft_runtime.state_rebind import HostRebindConfig
from experiments.post_training.long_loop.sft_runtime.temperature_matching import (
    CachedTeacherTopK,
    TeacherMatchingMode,
    TemperatureMatchingConfig,
    cached_topk_has_valid_entries,
    chunked_linear_cached_topk_kl_weighted_sum,
    global_l2_norm,
    tm_row_mask_for_mixture_indices,
)
from experiments.post_training.long_loop.sft_runtime.test_temperature_matching import (
    _assert_fp32_state,
    _dense_tempered_kl_weighted_sum,
    _tiny_model_config,
)
from experiments.post_training.long_loop.sft_runtime.train import (
    _BATCH_AXES,
    _accumulated_loss_and_grads,
    _accumulated_loss_and_grads_with_tm,
    _balance_attention_rows_with_mask,
    _compile_first_step_and_rebind,
    _make_tm_grad_ratio_step,
    _make_train_step,
    build_train_loader,
    initial_state,
)


def _dense_cached_topk_kl_reference(
    student_hidden: jax.Array,
    student_lm_head: jax.Array,
    ids: np.ndarray,
    logprobs: np.ndarray,
    weight: jax.Array,
) -> jax.Array:
    """Naive Python-loop FP32 reference for exact {top-k} U {actual} + tail KL at T=1."""
    s_logits = jnp.einsum("...h,hv->...v", student_hidden.astype(jnp.float32), student_lm_head.astype(jnp.float32))
    flat_logits = s_logits.reshape((-1, s_logits.shape[-1]))
    flat_ids = ids.reshape((-1, ids.shape[-1]))
    flat_lp = logprobs.reshape((-1, logprobs.shape[-1]))
    flat_w = weight.reshape((-1,)).astype(jnp.float32)
    sharding = getattr(jax.typeof(flat_logits), "sharding", None)
    if sharding is not None and getattr(sharding, "mesh", None) is not None and sharding.mesh.axis_names:
        flat_logits = jax.sharding.reshard(flat_logits, P())
        flat_w = jax.sharding.reshard(flat_w, P())
    vocab = flat_logits.shape[-1]
    total = jnp.zeros((), dtype=jnp.float32)
    for row in range(flat_logits.shape[0]):
        seen: dict[int, float] = {}
        corrupt = False
        for tok, lp in zip(flat_ids[row], flat_lp[row], strict=True):
            ti = int(tok)
            if ti < 0:
                continue
            if ti >= vocab or not np.isfinite(lp) or float(lp) > 1e-3:
                corrupt = True
                break
            if ti not in seen:
                seen[ti] = min(float(lp), 0.0)
        if corrupt or not seen:
            continue
        exp_ids = list(seen.keys())
        raw_p = np.exp(np.array([seen[i] for i in exp_ids], dtype=np.float64))
        if raw_p.sum() > 1.0 + 1e-2:
            continue
        if raw_p.sum() > 1.0:
            raw_p = raw_p / raw_p.sum()
        p_tail = max(1.0 - float(raw_p.sum()), 0.0)
        z = flat_logits[row]
        lse_all = jax.scipy.special.logsumexp(z)
        log_q = z - lse_all
        kl_row = jnp.zeros((), dtype=jnp.float32)
        for ti, pi in zip(exp_ids, raw_p, strict=True):
            kl_row = kl_row + jnp.asarray(pi, jnp.float32) * (jnp.asarray(np.log(pi), jnp.float32) - log_q[ti])
        rest_ids = [v for v in range(vocab) if v not in seen]
        if rest_ids and p_tail > 1e-12:
            lse_tail = jax.scipy.special.logsumexp(z[jnp.array(rest_ids, dtype=jnp.int32)])
            log_q_tail = lse_tail - lse_all
            kl_row = kl_row + jnp.asarray(p_tail, jnp.float32) * (jnp.asarray(np.log(p_tail), jnp.float32) - log_q_tail)
        total = total + flat_w[row] * kl_row
    return total


def test_cached_topk_config_validation():
    with pytest.raises(ValueError, match="temperature == 1.0"):
        TemperatureMatchingConfig(
            enabled=True,
            mode=TeacherMatchingMode.CACHED_TOPK_KL,
            ce_weight=0.5,
            weight=0.5,
            temperature=2.5,
            microbatch_count=2,
            cached_topk_paths={"mochi": "/unused"},
        )
    with pytest.raises(ValueError, match="cached_topk_paths"):
        TemperatureMatchingConfig(
            enabled=True,
            mode=TeacherMatchingMode.CACHED_TOPK_KL,
            ce_weight=0.5,
            weight=0.5,
            temperature=1.0,
            microbatch_count=2,
        )
    with pytest.raises(ValueError, match="ce_weight"):
        TemperatureMatchingConfig(
            enabled=True,
            mode=TeacherMatchingMode.CACHED_TOPK_KL,
            ce_weight=-0.1,
            weight=0.5,
            temperature=1.0,
            microbatch_count=2,
            cached_topk_paths={"mochi": "/unused"},
        )
    cfg = TemperatureMatchingConfig(
        enabled=True,
        mode=TeacherMatchingMode.CACHED_TOPK_KL,
        ce_weight=0.5,
        weight=0.5,
        temperature=1.0,
        microbatch_count=2,
        cached_topk_paths={"mochi": "/unused"},
    )
    assert cfg.mode == TeacherMatchingMode.CACHED_TOPK_KL


@pytest.mark.parametrize("token_block_size", [1, 3, 5, 1024])
def test_chunked_cached_topk_kl_matches_dense_reference_and_deduplicates(token_block_size: int):
    """Verify exact value + grad parity, duplicate actual-token dedup, token-0 safety, and DPI lower bound."""
    rng = np.random.default_rng(77)
    vocab = 24
    s_h = jnp.asarray(rng.normal(size=(2, 6, 16)).astype(np.float32))
    s_w = jnp.asarray(rng.normal(size=(16, vocab)).astype(np.float32))
    t_h = jnp.asarray(rng.normal(size=(2, 6, 16)).astype(np.float32))
    t_w = jnp.asarray(rng.normal(size=(16, vocab)).astype(np.float32))

    t_logits = np.asarray(jnp.einsum("bsh,hv->bsv", t_h, t_w))
    t_logp = np.asarray(jax.nn.log_softmax(jnp.asarray(t_logits), axis=-1))
    k_top = 4
    ids_np = np.full((2, 6, k_top + 1), -1, dtype=np.int32)
    lp_np = np.zeros((2, 6, k_top + 1), dtype=np.float32)
    for b in range(2):
        for s in range(6):
            if b == 1 and s == 5:
                continue
            top_idx = np.argsort(-t_logp[b, s])[:k_top]
            if b == 0 and s == 0 and 0 not in top_idx:
                top_idx[-1] = 0
            ids_np[b, s, :k_top] = top_idx
            lp_np[b, s, :k_top] = t_logp[b, s, top_idx]
            if s % 2 == 0:
                dup_tok = int(top_idx[1])
                ids_np[b, s, k_top] = dup_tok
                lp_np[b, s, k_top] = t_logp[b, s, dup_tok]
            elif s == 1:
                outside = next(v for v in range(vocab) if v not in set(top_idx.tolist()))
                ids_np[b, s, k_top] = outside
                lp_np[b, s, k_top] = t_logp[b, s, outside]

    weight = jnp.asarray(np.array([[1, 1, 0, 1, 1, 1], [1, 1, 1, 0, 1, 1]], dtype=np.float32))
    topk = CachedTeacherTopK(ids=jnp.asarray(ids_np), logprobs=jnp.asarray(lp_np))

    ref_val, (ref_g_sh, ref_g_sw) = jax.value_and_grad(
        lambda sh, sw: _dense_cached_topk_kl_reference(sh, sw, ids_np, lp_np, weight), argnums=(0, 1)
    )(s_h, s_w)
    chunk_val, (chunk_g_sh, chunk_g_sw) = jax.value_and_grad(
        lambda sh, sw: chunked_linear_cached_topk_kl_weighted_sum(
            sh, sw, topk, weight, token_block_size=token_block_size
        ),
        argnums=(0, 1),
    )(s_h, s_w)

    np.testing.assert_allclose(chunk_val, ref_val, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(chunk_g_sh, ref_g_sh, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(chunk_g_sw, ref_g_sw, rtol=1e-5, atol=1e-5)

    full_kl = _dense_tempered_kl_weighted_sum(s_h, s_w, t_h, t_w, weight.at[1, 5].set(0.0), temperature=1.0)
    assert 0.0 <= float(chunk_val) <= float(full_kl) + 1e-5

    zero_val, (zero_g_sh, zero_g_sw) = jax.value_and_grad(
        lambda sh, sw: chunked_linear_cached_topk_kl_weighted_sum(
            sh, sw, topk, weight, token_block_size=token_block_size
        ),
        argnums=(0, 1),
    )(t_h, t_w)
    np.testing.assert_allclose(zero_val, 0.0, atol=1e-5)
    np.testing.assert_allclose(zero_g_sh, 0.0, atol=1e-5)
    np.testing.assert_allclose(zero_g_sw, 0.0, atol=1e-5)


def test_cached_topk_kl_stable_when_student_tail_mass_underflows_fp32():
    """Verify complement logsumexp avoids -inf/NaN even when student puts ~1.0 mass on top-k (q_tail < 1e-20)."""
    vocab = 16
    s_h = jnp.ones((1, 2, 1), dtype=jnp.float32)
    s_w = jnp.zeros((1, vocab), dtype=jnp.float32).at[0, 0].set(60.0)
    ids_np = np.array([[[0, 1, -1], [0, -1, -1]]], dtype=np.int32)
    lp_np = np.log(np.array([[[0.8, 0.15, 1.0], [0.9, 1.0, 1.0]]], dtype=np.float32))
    weight = jnp.ones((1, 2), dtype=jnp.float32)
    topk = CachedTeacherTopK(ids=jnp.asarray(ids_np), logprobs=jnp.asarray(lp_np))

    val, (g_sh, g_sw) = jax.value_and_grad(
        lambda sh, sw: chunked_linear_cached_topk_kl_weighted_sum(sh, sw, topk, weight, token_block_size=2),
        argnums=(0, 1),
    )(s_h, s_w)
    ref_val, (ref_g_sh, ref_g_sw) = jax.value_and_grad(
        lambda sh, sw: _dense_cached_topk_kl_reference(sh, sw, ids_np, lp_np, weight), argnums=(0, 1)
    )(s_h, s_w)

    assert np.isfinite(float(val)) and float(val) > 0.0
    assert np.all(np.isfinite(np.asarray(g_sh)))
    assert np.all(np.isfinite(np.asarray(g_sw)))
    np.testing.assert_allclose(val, ref_val, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(g_sh, ref_g_sh, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(g_sw, ref_g_sw, rtol=1e-5, atol=1e-5)


def test_chunked_cached_topk_kl_sharded_matches_single_device():
    """Verify 4-device (data=2, context=2) sharded cached top-k KL matches single-device reference."""
    rng = np.random.default_rng(88)
    vocab = 24
    s_h_np = rng.normal(size=(4, 8, 16)).astype(np.float32)
    s_w_np = rng.normal(size=(16, vocab)).astype(np.float32)
    t_logits = rng.normal(size=(4, 8, vocab)).astype(np.float32)
    t_logp = np.asarray(jax.nn.log_softmax(jnp.asarray(t_logits), axis=-1))
    k_slots = 5
    ids_np = np.full((4, 8, k_slots), -1, dtype=np.int32)
    lp_np = np.zeros((4, 8, k_slots), dtype=np.float32)
    for b in range(4):
        for s in range(8):
            top = np.argsort(-t_logp[b, s])[:k_slots]
            ids_np[b, s] = top
            lp_np[b, s] = t_logp[b, s, top]
    w_np = (rng.uniform(size=(4, 8)) > 0.25).astype(np.float32)

    ref_val, (ref_g_sh, ref_g_sw) = jax.value_and_grad(
        lambda sh, sw: _dense_cached_topk_kl_reference(sh, sw, ids_np, lp_np, jnp.asarray(w_np)), argnums=(0, 1)
    )(jnp.asarray(s_h_np), jnp.asarray(s_w_np))

    with use_cpu_devices(4):
        mesh = Mesh(
            np.asarray(jax.devices()).reshape(1, 2, 2, 1, 1),
            ("replica_dcn", "data", "context", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 5,
        )
        with jax.set_mesh(mesh):
            h_shard = NamedSharding(mesh, P("data", "context", None))
            slot_shard = NamedSharding(mesh, P("data", "context", None))
            w_shard = NamedSharding(mesh, P("data", "context"))
            lm_shard = NamedSharding(mesh, P(None, None))
            s_h = jax.device_put(s_h_np, h_shard)
            s_w = jax.device_put(s_w_np, lm_shard)
            topk = CachedTeacherTopK(
                ids=jax.device_put(ids_np, slot_shard),
                logprobs=jax.device_put(lp_np, slot_shard),
            )
            w = jax.device_put(w_np, w_shard)

            val, (g_sh, g_sw) = jax.value_and_grad(
                lambda sh, sw: chunked_linear_cached_topk_kl_weighted_sum(sh, sw, topk, w, token_block_size=3),
                argnums=(0, 1),
            )(s_h, s_w)

            np.testing.assert_allclose(np.asarray(val), np.asarray(ref_val), rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(np.asarray(g_sh), np.asarray(ref_g_sh), rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(np.asarray(g_sw), np.asarray(ref_g_sw), rtol=1e-5, atol=1e-5)


def test_cached_topk_kl_rejects_corrupt_rows_and_preserves_mass_closure():
    """Verify corrupt cached rows are rejected consistently, BF16 roundoff is tolerated, and full-vocab E grad is exact."""
    vocab = 4
    rng = np.random.default_rng(99)
    s_h = jnp.asarray(rng.normal(size=(1, 6, 8)).astype(np.float32))
    s_w = jnp.asarray(rng.normal(size=(8, vocab)).astype(np.float32))

    ids_np = np.array(
        [[
            [0, 1, -1, -1],
            [0, 1, -1, -1],
            [0, 1, -1, -1],
            [0, 1, -1, -1],
            [0, 4, -1, -1],
            [0, 1, 2, 3],
        ]],
        dtype=np.int32,
    )
    lp_np = np.array(
        [[
            [np.log(0.5), np.log(0.3), np.nan, np.nan],
            [np.log(0.601), np.log(0.401), 0.0, 0.0],
            [0.5, np.log(0.1), 0.0, 0.0],
            [np.log(0.7), np.log(0.6), 0.0, 0.0],
            [np.log(0.5), np.log(0.3), 0.0, 0.0],
            [np.log(0.4), np.log(0.2), np.log(0.2), np.log(0.1)],
        ]],
        dtype=np.float32,
    )
    topk = CachedTeacherTopK(ids=jnp.asarray(ids_np), logprobs=jnp.asarray(lp_np))
    valid_rows = np.asarray(cached_topk_has_valid_entries(topk, vocab_size=vocab))
    np.testing.assert_array_equal(valid_rows, np.array([[1.0, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32))

    w_all = jnp.ones((1, 6), dtype=jnp.float32)
    w_valid_only = jnp.asarray([[1.0, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=jnp.float32)

    val, (g_sh, g_sw) = jax.value_and_grad(
        lambda sh, sw: chunked_linear_cached_topk_kl_weighted_sum(sh, sw, topk, w_all, token_block_size=2),
        argnums=(0, 1),
    )(s_h, s_w)
    ref_val, (ref_g_sh, ref_g_sw) = jax.value_and_grad(
        lambda sh, sw: _dense_cached_topk_kl_reference(sh, sw, ids_np, lp_np, w_valid_only), argnums=(0, 1)
    )(s_h, s_w)

    assert np.isfinite(float(val))
    np.testing.assert_allclose(val, ref_val, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(g_sh, ref_g_sh, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(g_sw, ref_g_sw, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(g_sh)[0, [2, 3, 4]], 0.0)


@pytest.mark.parametrize("lp_dtype", ["bfloat16", "float16", "float32"])
def test_cached_topk_store_roundtrip_dtypes_shards_and_left_shift(tmp_path, lp_dtype: str):
    """Verify cached_teacher_topk_v1 writer/reader round-trip across dtypes, multi-shard LRU, and left-shift."""
    rng = np.random.default_rng(123)
    num_rows, seq_len, k_slots, vocab = 5, 8, 4, 24
    tokens = rng.integers(1, vocab, size=(num_rows, seq_len), dtype=np.int32)
    raw_ids = rng.integers(0, vocab, size=(num_rows, seq_len, k_slots), dtype=np.int32)
    raw_lp = -np.abs(rng.normal(size=(num_rows, seq_len, k_slots)).astype(np.float32)) - 0.05

    shifted_ids, shifted_lp = left_shift_prompt_logprobs(raw_ids, raw_lp)
    np.testing.assert_array_equal(shifted_ids[:, :-1, :], raw_ids[:, 1:, :])
    np.testing.assert_array_equal(shifted_ids[:, -1, :], -1)
    np.testing.assert_array_equal(shifted_lp[:, :-1, :], raw_lp[:, 1:, :])
    np.testing.assert_array_equal(shifted_lp[:, -1, :], 0.0)

    out_dir = str(tmp_path / f"cache_{lp_dtype}")
    manifest = write_cached_topk_source(
        out_dir,
        ids=shifted_ids,
        logprobs=shifted_lp,
        vocab_size=vocab,
        tokens=tokens,
        logprobs_dtype=lp_dtype,  # type: ignore[arg-type]
        rows_per_shard=2,
    )
    assert manifest["num_shards"] == 3

    reader = CachedTopKSourceReader(out_dir, max_cached_shards=1)
    expected_np_dtype = {"bfloat16": ml_dtypes.bfloat16, "float16": np.float16, "float32": np.float32}[lp_dtype]
    for row_idx in (4, 0, 2, 1, 3):
        got_ids, got_lp, got_hash = reader.read_row(row_idx)
        assert got_ids.dtype == np.int32
        assert got_lp.dtype == expected_np_dtype
        np.testing.assert_array_equal(got_ids, shifted_ids[row_idx])
        np.testing.assert_array_equal(got_lp, shifted_lp[row_idx].astype(expected_np_dtype))
        assert got_hash == row_tokens_sha256(tokens[row_idx])


def test_cached_topk_mixture_reader_wraparound_partial_coverage_and_token_hash_gate(tmp_path):
    """Verify MixtureDataset source/row resolution, finite wraparound, partial coverage, and token-ID hash gating."""
    seq_len, vocab, k_slots = 6, 24, 3
    general_tokens = np.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]], dtype=np.int32)
    mochi_tokens = np.array([[10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21]], dtype=np.int32)

    general_ds = ListAsyncDataset(
        [GrugLmExample.causal(general_tokens[i], loss_weight=np.ones(seq_len, dtype=np.float32)) for i in range(2)]
    )
    mochi_ds = ListAsyncDataset(
        [GrugLmExample.causal(mochi_tokens[i], loss_weight=np.ones(seq_len, dtype=np.float32)) for i in range(2)]
    )
    mixture = MixtureDataset(
        datasets={"general": general_ds, "mochi": mochi_ds},
        weights={"general": 0.5, "mochi": 0.5},
        block_size=4,
        key=jax.random.key(0),
        stop_strategy=StopStrategy.RESTART_STRATEGY,
        randomize_blocks=False,
    )

    # Write cache covering ONLY row 0 of mochi (partial coverage: row 1 is uncached).
    mochi_ids = np.full((1, seq_len, k_slots), 7, dtype=np.int32)
    mochi_lp = np.full((1, seq_len, k_slots), np.log(0.25), dtype=np.float32)
    cache_dir = str(tmp_path / "mochi_partial")
    write_cached_topk_source(
        cache_dir, ids=mochi_ids, logprobs=mochi_lp, vocab_size=vocab, tokens=mochi_tokens[:1], logprobs_dtype="float32"
    )

    # Unknown source fails fast.
    with pytest.raises(ValueError, match="unknown mixture sources"):
        CachedTopKMixtureReader(mixture, {"missing": cache_dir}, seq_len=seq_len, vocab_size=vocab)

    with use_cpu_devices(2):
        mesh = Mesh(
            np.asarray(jax.devices()).reshape(1, 2, 1, 1, 1),
            ("replica_dcn", "data", "context", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 5,
        )
        with jax.set_mesh(mesh):
            # Pull 4 rows spanning 2 mochi rows (0, 1) and 2 general rows (0, 1), plus second block wrapping to mochi row 0.
            reader = CachedTopKMixtureReader(
                mixture,
                {"mochi": cache_dir},
                seq_len=seq_len,
                vocab_size=vocab,
                verify_token_hashes=True,
                require_full_source_coverage=False,
            )
            resolved_block0 = reader.resolve_mixture_indices(range(0, 4))
            resolved_block1 = reader.resolve_mixture_indices(range(4, 8))
            assert resolved_block0 == resolved_block1  # finite length 2 wraps identically in block 1!

            # Build real sharded tokens matching indices 0..3.
            tok_shard = NamedSharding(mesh, P(_BATCH_AXES, None))
            rows_tok = np.stack(
                [general_tokens[r] if src == "general" else mochi_tokens[r] for src, r in resolved_block0], axis=0
            )
            batch_tokens = jax.device_put(rows_tok, tok_shard)

            topk = reader.load_batch_topk(range(0, 4), mesh=mesh, batch_axis_spec=_BATCH_AXES, batch_tokens=batch_tokens)
            assert reader.last_uncovered_rows == 1
            ids_out = np.asarray(topk.ids)
            for b, (src, r) in enumerate(resolved_block0):
                if src == "mochi" and r == 0:
                    np.testing.assert_array_equal(ids_out[b], mochi_ids[0])
                else:
                    np.testing.assert_array_equal(ids_out[b], -1)

            # Corrupt one token in mochi row 0 -> hash verification must fail fast!
            bad_rows_tok = rows_tok.copy()
            mochi0_pos = next(b for b, (src, r) in enumerate(resolved_block0) if src == "mochi" and r == 0)
            bad_rows_tok[mochi0_pos, 0] ^= 1
            with pytest.raises(ValueError, match="Token-ID parity mismatch"):
                reader.load_batch_topk(
                    range(0, 4),
                    mesh=mesh,
                    batch_axis_spec=_BATCH_AXES,
                    batch_tokens=jax.device_put(bad_rows_tok, tok_shard),
                )

            # require_full_source_coverage=True fails fast on uncovered mochi row 1.
            strict_reader = CachedTopKMixtureReader(
                mixture,
                {"mochi": cache_dir},
                seq_len=seq_len,
                vocab_size=vocab,
                require_full_source_coverage=True,
            )
            with pytest.raises(IndexError, match="covers 1 rows"):
                strict_reader.load_batch_topk(
                    range(0, 4), mesh=mesh, batch_axis_spec=_BATCH_AXES, batch_tokens=batch_tokens
                )


def test_cached_topk_token_hash_verification_fails_closed(tmp_path):
    """Verify verify_token_hashes=True never fails open on missing manifest digests or unhashable token shards."""
    seq_len, vocab, k_slots = 8, 24, 3
    tokens = np.arange(1, 1 + 2 * seq_len, dtype=np.int32).reshape(2, seq_len)
    ids = np.full((2, seq_len, k_slots), 5, dtype=np.int32)
    lp = np.full((2, seq_len, k_slots), np.log(0.2), dtype=np.float32)

    # Writer requires tokens (no silent hashless cache creation).
    with pytest.raises(TypeError):
        write_cached_topk_source(str(tmp_path / "no_tokens"), ids=ids, logprobs=lp, vocab_size=vocab)  # type: ignore[call-arg]

    cache_dir = tmp_path / "mochi_cache"
    write_cached_topk_source(
        str(cache_dir), ids=ids, logprobs=lp, vocab_size=vocab, tokens=tokens, logprobs_dtype="float32"
    )
    ds = ListAsyncDataset([GrugLmExample.causal(tokens[i], loss_weight=np.ones(seq_len, dtype=np.float32)) for i in range(2)])
    mixture = MixtureDataset(
        datasets={"mochi": ds},
        weights={"mochi": 1.0},
        block_size=2,
        key=jax.random.key(0),
        stop_strategy=StopStrategy.RESTART_STRATEGY,
        randomize_blocks=False,
    )

    # Refusal 1: manifest stripped of token_sha256_per_row must fail closed when verify_token_hashes=True.
    stripped_dir = tmp_path / "mochi_no_hashes"
    write_cached_topk_source(
        str(stripped_dir), ids=ids, logprobs=lp, vocab_size=vocab, tokens=tokens, logprobs_dtype="float32"
    )
    manifest_path = stripped_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["token_sha256_per_row"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="missing token_sha256_per_row"):
        CachedTopKMixtureReader(
            mixture, {"mochi": str(stripped_dir)}, seq_len=seq_len, vocab_size=vocab, verify_token_hashes=True
        )
    # Explicitly opting out of hash verification still allows reading a hashless manifest.
    unverified_reader = CachedTopKMixtureReader(
        mixture, {"mochi": str(stripped_dir)}, seq_len=seq_len, vocab_size=vocab, verify_token_hashes=False
    )
    assert unverified_reader.k_slots == k_slots

    # Refusal 2: batch_tokens=None and context-sharded (partial-sequence) batch_tokens must both fail closed.
    reader = CachedTopKMixtureReader(
        mixture, {"mochi": str(cache_dir)}, seq_len=seq_len, vocab_size=vocab, verify_token_hashes=True
    )
    with use_cpu_devices(4):
        mesh = Mesh(
            np.asarray(jax.devices()).reshape(1, 2, 2, 1, 1),
            ("replica_dcn", "data", "context", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 5,
        )
        with jax.set_mesh(mesh):
            with pytest.raises(ValueError, match="batch_tokens must be provided"):
                reader.load_batch_topk(range(0, 2), mesh=mesh, batch_axis_spec=_BATCH_AXES, batch_tokens=None)

            ctx_sharded_tokens = jax.device_put(tokens, NamedSharding(mesh, P("data", "context")))
            with pytest.raises(ValueError, match="full-sequence batch_tokens shards"):
                reader.load_batch_topk(
                    range(0, 2), mesh=mesh, batch_axis_spec=_BATCH_AXES, batch_tokens=ctx_sharded_tokens
                )


@pytest.mark.timeout(300)
def test_cached_topk_end_to_end_reader_rebind_train_step_and_watch_ratio(tmp_path):
    """End-to-end test: MixtureDataset -> DataLoader + CachedTopKMixtureReader -> host rebind -> train_step + watch."""

    class _RecordingTiming:
        def __init__(self):
            self.spans: list[str] = []

        def span(self, name: str, **_kwargs):
            self.spans.append(name)
            return nullcontext()

    (tmp_path / "memory.current").write_text("54321")
    (tmp_path / "memory.max").write_text("99999")

    with use_cpu_devices(2):
        config = _tiny_model_config()
        mesh = Mesh(
            np.asarray(jax.devices()).reshape(1, 2, 1, 1, 1),
            ("replica_dcn", "data", "context", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 5,
        )
        with jax.set_mesh(mesh):
            optimizer = optax.adam(1e-3)
            mp = jmp.get_policy("float32")
            student_state = initial_state(config, optimizer=optimizer, mp=mp, key=jax.random.key(31), ema_beta=None)
            student_ref = initial_state(config, optimizer=optimizer, mp=mp, key=jax.random.key(31), ema_beta=None)
            teacher_state = initial_state(config, optimizer=optimizer, mp=mp, key=jax.random.key(32), ema_beta=None)

            seq_len, vocab, k_slots = 8, config.vocab_size, 5
            rng = np.random.default_rng(303)
            general_tokens = rng.integers(1, vocab - 2, size=(2, seq_len), dtype=np.int32)
            mochi_tokens = rng.integers(1, vocab - 2, size=(2, seq_len), dtype=np.int32)
            general_weights = np.array([[1, 1, 0, 1, 1, 0, 1, 0], [1, 0, 1, 1, 0, 1, 1, 0]], dtype=np.float32)
            mochi_weights = np.array([[1, 1, 1, 0, 1, 1, 1, 0], [0, 1, 1, 1, 1, 0, 1, 0]], dtype=np.float32)
            general_segs = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
            mochi_segs = np.array([[0, 0, 1, 1, 2, 2, 3, 3], [0, 0, 0, 0, 1, 1, 1, 1]], dtype=np.int32)

            # Score mochi_tokens with teacher_state and persist cached_teacher_topk_v1 in bfloat16 with token hashes.
            b_shard = NamedSharding(mesh, P("data", None))
            mochi_t_logits = np.asarray(
                teacher_state.params.logits(
                    jax.device_put(mochi_tokens, b_shard),
                    mask=AttentionMask.causal().with_segment_ids(jax.device_put(mochi_segs, b_shard)),
                )
            )
            mochi_t_logp = np.asarray(jax.nn.log_softmax(jnp.asarray(mochi_t_logits), axis=-1))
            mochi_ids = np.full((2, seq_len, k_slots), -1, dtype=np.int32)
            mochi_lp = np.zeros((2, seq_len, k_slots), dtype=np.float32)
            for b in range(2):
                for s in range(seq_len):
                    top = np.argsort(-mochi_t_logp[b, s])[:k_slots]
                    mochi_ids[b, s] = top
                    mochi_lp[b, s] = mochi_t_logp[b, s, top]

            mochi_cache_dir = str(tmp_path / "mochi_topk_cache")
            write_cached_topk_source(
                mochi_cache_dir,
                ids=mochi_ids,
                logprobs=mochi_lp,
                vocab_size=vocab,
                tokens=mochi_tokens,
                logprobs_dtype="bfloat16",
                rows_per_shard=1,
            )

            general_ds = ListAsyncDataset(
                [
                    GrugLmExample(
                        tokens=general_tokens[i],
                        loss_weight=general_weights[i],
                        attn_mask=AttentionMask.causal().with_segment_ids(general_segs[i]),
                    )
                    for i in range(2)
                ]
            )
            mochi_ds = ListAsyncDataset(
                [
                    GrugLmExample(
                        tokens=mochi_tokens[i],
                        loss_weight=mochi_weights[i],
                        attn_mask=AttentionMask.causal().with_segment_ids(mochi_segs[i]),
                    )
                    for i in range(2)
                ]
            )
            mixture = MixtureDataset(
                datasets={"general": general_ds, "mochi": mochi_ds},
                weights={"general": 0.5, "mochi": 0.5},
                block_size=4,
                key=jax.random.key(0),
                stop_strategy=StopStrategy.RESTART_STRATEGY,
                randomize_blocks=False,
            )
            batch_schedule = BatchSchedule(4)
            loader = build_train_loader(mixture, batch_schedule=batch_schedule, mesh=mesh)
            batch = next(iter(loader.iter_from_step(0)))

            tm_cfg = TemperatureMatchingConfig(
                enabled=True,
                mode=TeacherMatchingMode.CACHED_TOPK_KL,
                ce_weight=0.5,
                weight=0.5,
                temperature=1.0,
                microbatch_count=2,
                token_block_size=4,
                offload_teacher=False,
                cached_topk_paths={"mochi": mochi_cache_dir},
                enabled_sources=("mochi",),
                log_grad_norm_ratio=True,
            )
            reader = CachedTopKMixtureReader(
                mixture,
                tm_cfg.cached_topk_paths,
                seq_len=seq_len,
                vocab_size=vocab,
                verify_token_hashes=True,
            )
            teacher_topk = reader.load_batch_topk(
                range(0, 4), mesh=mesh, batch_axis_spec=_BATCH_AXES, batch_tokens=batch.tokens
            )
            row_mask_np = tm_row_mask_for_mixture_indices(mixture, range(0, 4), tm_cfg)
            row_mask = jax.make_array_from_callback(
                row_mask_np.shape, NamedSharding(mesh, P(_BATCH_AXES)), lambda idx, a=row_mask_np: a[idx]
            )

            # Verify lockstep balance_attention_rows permutes teacher_topk with batch and row_mask.
            bal_b, bal_m, bal_topk = _balance_attention_rows_with_mask(batch, row_mask, teacher_topk)
            assert bal_topk is not None
            batch_toks_np = np.asarray(batch.tokens)
            bal_toks_np = np.asarray(bal_b.tokens)
            topk_ids_np = np.asarray(teacher_topk.ids)
            bal_ids_np = np.asarray(bal_topk.ids)
            for r in range(4):
                orig = next(j for j in range(4) if np.array_equal(bal_toks_np[r], batch_toks_np[j]))
                np.testing.assert_array_equal(bal_ids_np[r], topk_ids_np[orig])

            # Gradient & watch-ratio parity against dense reference using the actual bfloat16 cache values.
            (total_loss, metrics), grads_comb = _accumulated_loss_and_grads_with_tm(
                student_state.params, batch, None, row_mask, mp, None, 2, tm_cfg, teacher_topk=teacher_topk
            )
            (ce_loss, _), grads_ce = _accumulated_loss_and_grads(student_state.params, batch, mp, None, 2)

            topk_lp_np = np.asarray(teacher_topk.logprobs).astype(np.float32)
            active_w_np = np.asarray(batch.loss_weight) * row_mask_np[:, None]
            active_w = jax.device_put(active_w_np, b_shard)

            def ref_kl(model):
                model = eqx.tree_at(
                    lambda m: m.stacked_blocks.stacked.mlp.router_bias, model, replace_fn=jax.lax.stop_gradient
                )
                s_h, _ = mp.cast_to_compute(model)(batch.tokens, mask=batch.attn_mask)
                kl_sum = _dense_cached_topk_kl_reference(
                    s_h, mp.cast_to_compute(model).output_proj, topk_ids_np, topk_lp_np, active_w
                )
                return 0.5 * kl_sum / float(active_w_np.sum())

            ref_kl_weighted, grads_kl_ref = jax.value_and_grad(ref_kl)(student_state.params)
            np.testing.assert_allclose(metrics["train/cross_entropy_loss"], ce_loss, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(metrics["train/tm/weighted_loss"], ref_kl_weighted, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(total_loss, 0.5 * ce_loss + ref_kl_weighted, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(metrics["train/tm/active_tokens"], float(active_w_np.sum()), atol=1e-6)
            for gc, gce, gkl in zip(
                jax.tree.leaves(grads_comb),
                jax.tree.leaves(grads_ce),
                jax.tree.leaves(grads_kl_ref),
                strict=True,
            ):
                np.testing.assert_allclose(np.asarray(gc), np.asarray(0.5 * gce + gkl), rtol=2e-4, atol=1e-5)

            ratio_step = _make_tm_grad_ratio_step(
                mp, z_loss_weight=0.0, accumulation_steps=2, temperature_matching=tm_cfg
            )
            watch_stats = ratio_step(student_state.params, batch, None, row_mask, teacher_topk)
            exp_ce = 0.5 * float(global_l2_norm(grads_ce))
            exp_kl = float(global_l2_norm(grads_kl_ref))
            np.testing.assert_allclose(watch_stats["watch/tm/ce_grad_norm"], exp_ce, rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(watch_stats["watch/tm/grad_norm"], exp_kl, rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(watch_stats["watch/tm/grad_norm_ratio"], exp_kl / exp_ce, rtol=1e-4, atol=1e-6)

            # Exercise first_step_host_rebind with (None, row_mask, teacher_topk) and run train_step cleanly.
            train_step = _make_train_step(
                optimizer, mp, z_loss_weight=0.0, ema_beta=None, accumulation_steps=2, temperature_matching=tm_cfg
            )
            timing = _RecordingTiming()
            original_state_leaves = [leaf for leaf in jax.tree.leaves(student_state) if isinstance(leaf, jax.Array)]
            rebound_state = _compile_first_step_and_rebind(
                train_step,
                student_state,
                batch,
                extra_args=(None, row_mask, teacher_topk),
                compile_first_step=False,
                rebind_config=HostRebindConfig(cgroup_path=str(tmp_path)),
                timing=timing,
                step=0,
            )
            assert timing.spans == ["first_step_compile", "first_step_host_rebind"]
            assert all(leaf.is_deleted() for leaf in original_state_leaves)
            assert not teacher_topk.ids.is_deleted() and not teacher_topk.logprobs.is_deleted()

            next_rebound, step_metrics, _ = train_step(rebound_state, batch, None, row_mask, teacher_topk)
            next_ref, ref_metrics, _ = train_step(student_ref, batch, None, row_mask, teacher_topk)
            assert int(next_rebound.step) == 1
            assert np.isfinite(float(step_metrics["train/loss"]))
            _assert_fp32_state(next_rebound)
            np.testing.assert_allclose(step_metrics["train/loss"], ref_metrics["train/loss"], rtol=1e-6, atol=1e-6)
            for got, wanted in zip(
                jax.tree.leaves(next_rebound.params), jax.tree.leaves(next_ref.params), strict=True
            ):
                if isinstance(got, jax.Array):
                    np.testing.assert_allclose(np.asarray(got), np.asarray(wanted), rtol=1e-6, atol=1e-6)
