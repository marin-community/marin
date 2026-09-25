# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Numerical parity for the Kimi Delta Attention (KDA) kernels.

The chunked-parallel kernel is the MFU-friendly training path; its correctness is
anchored to the sequential recurrence (an independent, obviously-correct oracle),
and in the scalar-gate limit to levanter's HF-validated gated-delta-rule kernel.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.moe.kda import chunk_kda, kda_fused, recurrent_kda

jax.config.update("jax_default_matmul_precision", "float32")


def _inputs(batch, heads, length, dk, dv, *, seed=0):
    rng = np.random.RandomState(seed)
    q = jnp.asarray(rng.randn(batch, heads, length, dk), jnp.float32)
    k = jnp.asarray(rng.randn(batch, heads, length, dk), jnp.float32)
    v = jnp.asarray(rng.randn(batch, heads, length, dv), jnp.float32)
    # Per-channel log-decay g <= 0 (alpha = exp(g) in (0, 1]).
    g = -0.1 * jnp.abs(jnp.asarray(rng.randn(batch, heads, length, dk), jnp.float32))
    beta = jnp.asarray(rng.rand(batch, heads, length), jnp.float32)
    return q, k, v, g, beta


@pytest.mark.parametrize(
    ("length", "chunk_size"),
    [(64, 64), (128, 64), (57, 16), (61, 32), (29, 7), (48, 16)],
)
def test_chunk_kda_matches_recurrent(length, chunk_size):
    """Chunkwise-parallel KDA (exact fp32 GEMMs) equals the sequential recurrence."""
    q, k, v, g, beta = _inputs(2, 3, length, 16, 16)
    out_chunk, state_chunk = chunk_kda(q, k, v, g, beta, chunk_size=chunk_size, matmul_dtype=jnp.float32)
    out_recur, state_recur = recurrent_kda(q, k, v, g, beta)
    np.testing.assert_allclose(np.asarray(out_chunk), np.asarray(out_recur), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(state_chunk), np.asarray(state_recur), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(("length", "chunk_size"), [(64, 64), (256, 64), (512, 128), (200, 64)])
def test_parallel_scan_matches_sequential(length, chunk_size):
    """The log-depth ``scan_impl='parallel'`` recurrence equals the serial one (fp32).

    Both are exact reassociations of the same linear inter-chunk recurrence, so in
    fp32 they agree to ~1e-6; this guards the associative-scan derivation."""
    q, k, v, g, beta = _inputs(2, 3, length, 32, 32, seed=length)
    kw = dict(chunk_size=chunk_size, matmul_dtype=jnp.float32)
    out_par, st_par = chunk_kda(q, k, v, g, beta, scan_impl="parallel", **kw)
    out_seq, st_seq = chunk_kda(q, k, v, g, beta, scan_impl="sequential", **kw)
    np.testing.assert_allclose(np.asarray(out_par), np.asarray(out_seq), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(st_par), np.asarray(st_seq), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(("length", "chunk_size"), [(128, 64), (256, 64), (192, 32), (256, 128)])
def test_chunk_kda_bf16_matmuls_match_recurrent(length, chunk_size):
    """The default bf16 intra-chunk GEMMs stay close to the fp32 recurrence.

    bf16 has ~3 decimal digits of mantissa, so the chunk kernel tracks the fp32
    oracle to ~1e-2 relative -- the accuracy of the activations it feeds anyway."""
    q, k, v, g, beta = _inputs(2, 3, length, 32, 32)
    out_chunk, _ = chunk_kda(q, k, v, g, beta, chunk_size=chunk_size, matmul_dtype=jnp.bfloat16)
    out_recur, _ = recurrent_kda(q, k, v, g, beta)
    np.testing.assert_allclose(np.asarray(out_chunk), np.asarray(out_recur), rtol=2e-2, atol=2e-2)


def test_chunk_kda_initial_state_continuation():
    """A non-zero initial state carries through both kernels identically."""
    q, k, v, g, beta = _inputs(1, 2, 48, 16, 16, seed=3)
    s0 = jnp.asarray(np.random.RandomState(9).randn(1, 2, 16, 16) * 0.1, jnp.float32)
    out_chunk, _ = chunk_kda(q, k, v, g, beta, chunk_size=16, initial_state=s0, matmul_dtype=jnp.float32)
    out_recur, _ = recurrent_kda(q, k, v, g, beta, initial_state=s0)
    np.testing.assert_allclose(np.asarray(out_chunk), np.asarray(out_recur), rtol=1e-4, atol=1e-4)


def test_scalar_gate_limit_matches_levanter_gdn():
    """With g broadcast over channels (scalar per-head decay), KDA reduces to the
    Gated DeltaNet rule already validated against HF in levanter -- an independent
    oracle for the whole per-channel derivation."""
    haliax = pytest.importorskip("haliax")
    from levanter.layers.gated_deltanet import recurrent_gated_delta_rule  # noqa: PLC0415

    batch, heads, length, dk, dv = 2, 3, 40, 8, 8
    q, k, v, _, beta = _inputs(batch, heads, length, dk, dv, seed=5)
    g_scalar = -0.1 * jnp.abs(jnp.asarray(np.random.RandomState(7).randn(batch, heads, length), jnp.float32))
    g_perchannel = jnp.broadcast_to(g_scalar[..., None], (batch, heads, length, dk))

    out_kda, _ = recurrent_kda(q, k, v, g_perchannel, beta)

    def to_named(arr, dim_name):
        return haliax.named(jnp.moveaxis(arr, 1, 2), ("batch", "position", "heads", dim_name))

    out_lev, _ = recurrent_gated_delta_rule(
        to_named(q, "k_head_dim"),
        to_named(k, "k_head_dim"),
        to_named(v, "v_head_dim"),
        haliax.named(jnp.moveaxis(g_scalar, 1, 2), ("batch", "position", "heads")),
        haliax.named(jnp.moveaxis(beta, 1, 2), ("batch", "position", "heads")),
        use_qk_l2norm_in_kernel=True,
    )
    out_lev = jnp.moveaxis(out_lev.array, 2, 1)  # -> (batch, heads, length, dv)
    np.testing.assert_allclose(np.asarray(out_kda), np.asarray(out_lev), rtol=1e-4, atol=1e-4)


def test_chunk_kda_finite_under_strong_decay():
    """Strong per-channel decay (alpha ~ 0) and extreme beta stay finite (fp32)."""
    q, k, v, _, _ = _inputs(1, 2, 37, 16, 8, seed=1)
    g = -jnp.asarray(np.random.RandomState(2).uniform(2.0, 6.0, size=(1, 2, 37, 16)), jnp.float32)
    for beta_val in (1e-4, 1.0 - 1e-6):
        beta = jnp.full((1, 2, 37), beta_val, jnp.float32)
        out_chunk, _ = chunk_kda(q, k, v, g, beta, chunk_size=32)
        assert np.isfinite(np.asarray(out_chunk)).all()


def test_chunk_kda_gradients_finite():
    """The chunked kernel is differentiable w.r.t. its inputs without NaNs."""
    q, k, v, g, beta = _inputs(1, 1, 16, 8, 8, seed=4)

    def loss(q_arr, g_arr, b_arr):
        out, _ = chunk_kda(q_arr, k, v, g_arr, b_arr, chunk_size=8)
        return jnp.sum(out)

    grads = jax.grad(loss, argnums=(0, 1, 2))(q, g, beta)
    assert all(jnp.all(jnp.isfinite(grad)) for grad in grads)


# --- Fused prep (kda_prep_pallas) and state-pass (kda_state_pallas) kernels, CPU interpreter ---
_FUSED_SCAN_IMPLS = ["parallel", "pallas_interpret"]


@pytest.mark.parametrize("scan_impl", _FUSED_SCAN_IMPLS)
@pytest.mark.parametrize(("length", "chunk_size", "dk", "dv"), [(128, 32, 32, 16), (100, 32, 16, 32), (256, 64, 16, 16)])
def test_fused_prep_matches_recurrent(length, chunk_size, dk, dv, scan_impl):
    """chunk_kda with the fused Pallas prep (fp32), on either inter-chunk recurrence,
    equals the sequential recurrence, including a padded tail and a non-zero initial state."""
    q, k, v, g, beta = _inputs(2, 2, length, dk, dv, seed=length)
    s0 = jnp.asarray(np.random.RandomState(4).randn(2, 2, dk, dv) * 0.1, jnp.float32)
    out, state = chunk_kda(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=chunk_size,
        initial_state=s0,
        matmul_dtype=jnp.float32,
        prep_impl="pallas_interpret",
        scan_impl=scan_impl,
    )
    out_ref, state_ref = recurrent_kda(q, k, v, g, beta, initial_state=s0)
    np.testing.assert_allclose(np.asarray(out), np.asarray(out_ref), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(state), np.asarray(state_ref), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("scan_impl", _FUSED_SCAN_IMPLS)
@pytest.mark.parametrize(("decay_scale", "dk"), [(0.1, 32), (4.0, 32), (0.1, 64)])
def test_fused_kernels_grad_match_xla(decay_scale, dk, scan_impl):
    """The hand-written VJPs of the fused prep (and of the Pallas state pass) equal
    autodiff through the all-XLA path (fp32) for every input and the initial state,
    for mild decay, for per-token decay up to the model's -5 floor, and for a d_k the
    backward tail walks in several column blocks."""
    q, k, v, g, beta = _inputs(1, 2, 96, dk, 16, seed=7)
    g = jnp.maximum(g * (decay_scale / 0.1), -5.0)
    chunk_size = 16 if decay_scale > 1.0 else 32
    s0 = jnp.asarray(np.random.RandomState(5).randn(1, 2, dk, 16) * 0.1, jnp.float32)
    w_out = jnp.asarray(np.random.RandomState(6).randn(1, 2, 96, 16), jnp.float32)
    w_state = jnp.asarray(np.random.RandomState(8).randn(1, 2, dk, 16), jnp.float32)

    def loss(prep_impl, impl_scan, *args):
        out, state = chunk_kda(
            *args[:5],
            chunk_size=chunk_size,
            initial_state=args[5],
            matmul_dtype=jnp.float32,
            prep_impl=prep_impl,
            scan_impl=impl_scan,
        )
        return jnp.sum(w_out * out) + jnp.sum(w_state * state)

    argnums = (2, 3, 4, 5, 6, 7)
    grads_fused = jax.grad(loss, argnums=argnums)("pallas_interpret", scan_impl, q, k, v, g, beta, s0)
    grads_xla = jax.grad(loss, argnums=argnums)("xla", "parallel", q, k, v, g, beta, s0)
    for fused, ref in zip(grads_fused, grads_xla, strict=True):
        assert np.isfinite(np.asarray(fused)).all()
        np.testing.assert_allclose(np.asarray(fused), np.asarray(ref), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("scan_impl", _FUSED_SCAN_IMPLS)
def test_fused_prep_bf16_matches_recurrent(scan_impl):
    """The default bf16 GEMM operands through the fused kernels stay within bf16 accuracy."""
    q, k, v, g, beta = _inputs(2, 2, 256, 32, 32, seed=2)
    out, _ = chunk_kda(q, k, v, g, beta, chunk_size=64, prep_impl="pallas_interpret", scan_impl=scan_impl)
    out_ref, _ = recurrent_kda(q, k, v, g, beta)
    np.testing.assert_allclose(np.asarray(out), np.asarray(out_ref), rtol=2e-2, atol=2e-2)


def test_fused_kernels_take_bf16_inputs():
    """bf16 activations go straight into the fused kernels: the output tracks the fp32-input
    run to bf16 accuracy and the input cotangents come back in bf16."""
    q, k, v, g, beta = _inputs(1, 2, 128, 32, 32, seed=9)
    bf = tuple(x.astype(jnp.bfloat16) for x in (q, k, v, g, beta))
    kw = dict(chunk_size=32, prep_impl="pallas_interpret", scan_impl="pallas_interpret")
    out_bf, _ = chunk_kda(*bf, **kw)
    out_ref, _ = chunk_kda(*(x.astype(jnp.float32) for x in bf), **kw)
    np.testing.assert_allclose(np.asarray(out_bf), np.asarray(out_ref), rtol=1e-5, atol=1e-5)

    grads = jax.grad(lambda *a: jnp.sum(chunk_kda(*a, **kw)[0]), argnums=(0, 1, 2, 3, 4))(*bf)
    assert all(grad.dtype == jnp.bfloat16 for grad in grads)


def test_fused_kernels_without_qk_l2norm_match_xla():
    """With use_qk_l2norm=False the fused kernels skip the norm (and its backward) but
    still scale q, matching the XLA path in value and gradient (small k keeps the
    un-normalized delta rule stable)."""
    q, k, v, g, beta = _inputs(1, 2, 64, 32, 16, seed=11)
    k = 0.15 * k

    def run(prep_impl, scan_impl, *args):
        out, _ = chunk_kda(
            *args,
            chunk_size=32,
            use_qk_l2norm=False,
            matmul_dtype=jnp.float32,
            prep_impl=prep_impl,
            scan_impl=scan_impl,
        )
        return out

    out_fused = run("pallas_interpret", "pallas_interpret", q, k, v, g, beta)
    out_xla = run("xla", "parallel", q, k, v, g, beta)
    np.testing.assert_allclose(np.asarray(out_fused), np.asarray(out_xla), rtol=1e-4, atol=1e-4)

    def loss(prep_impl, scan_impl, *args):
        return jnp.sum(jnp.sin(run(prep_impl, scan_impl, *args)))

    grads_fused = jax.grad(loss, argnums=(2, 3, 4, 5, 6))("pallas_interpret", "pallas_interpret", q, k, v, g, beta)
    grads_xla = jax.grad(loss, argnums=(2, 3, 4, 5, 6))("xla", "parallel", q, k, v, g, beta)
    for fused, ref in zip(grads_fused, grads_xla, strict=True):
        np.testing.assert_allclose(np.asarray(fused), np.asarray(ref), rtol=1e-4, atol=1e-4)


def test_kda_fused_model_layout_matches_recurrent():
    """kda_fused takes (B, L, H, d) activations directly (the kernels index heads in place)
    and matches the recurrence in value and in gradient, including a padded tail."""

    q, k, v, g, beta = _inputs(2, 3, 80, 32, 16, seed=14)  # (B, H, L, d); 80 = 2.5 chunks of 32

    def model_layout(x):
        return jnp.swapaxes(x, 1, 2)

    def fused(*args):
        out = kda_fused(*(model_layout(x) for x in args), chunk_size=32, matmul_dtype=jnp.float32, interpret=True)
        return model_layout(out)

    def recurrent(*args):
        return recurrent_kda(*args)[0]

    np.testing.assert_allclose(
        np.asarray(fused(q, k, v, g, beta)), np.asarray(recurrent(q, k, v, g, beta)), rtol=1e-4, atol=1e-4
    )
    w = jnp.asarray(np.random.RandomState(15).randn(2, 3, 80, 16), jnp.float32)
    grads = jax.grad(lambda *a: jnp.sum(w * fused(*a)), argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
    want = jax.grad(lambda *a: jnp.sum(w * recurrent(*a)), argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
    for got, ref in zip(grads, want, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=2e-4, atol=2e-4)


def test_kda_fused_saved_chunk_states_give_identical_gradients():
    """Keeping the state pass's per-chunk states for backward (instead of re-running the forward
    pass there) reproduces the recomputing backward exactly."""
    q, k, v, g, beta = (jnp.swapaxes(x, 1, 2) for x in _inputs(2, 3, 80, 32, 16, seed=21))  # (B, L, H, ...)
    w = jnp.asarray(np.random.RandomState(22).randn(*v.shape), jnp.float32)

    def loss(save, *args):
        out = kda_fused(*args, chunk_size=32, matmul_dtype=jnp.float32, interpret=True, save_chunk_states=save)
        return jnp.sum(w * out)

    argnums = (1, 2, 3, 4, 5)
    saved = jax.grad(loss, argnums=argnums)(True, q, k, v, g, beta)
    recomputed = jax.grad(loss, argnums=argnums)(False, q, k, v, g, beta)
    for got, want in zip(saved, recomputed, strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.parametrize("packed", [False, True])
def test_kda_fused_gate_and_grouped_kv_match_unfused(packed):
    """The on-chip gate (g = rate * softplus(a + bias)) and in-place grouped-query k/v give
    the same values and gradients -- including for rate, bias and the kv heads -- as
    computing g and expanding k/v in JAX first, with and without packed documents."""

    rng = np.random.RandomState(16)
    b, length, heads, kv_heads, d = 2, 64, 4, 2, 16
    q = jnp.asarray(rng.randn(b, length, heads, d), jnp.float32)
    k = jnp.asarray(rng.randn(b, length, kv_heads, d), jnp.float32)
    v = jnp.asarray(rng.randn(b, length, kv_heads, d), jnp.float32)
    a = jnp.asarray(rng.randn(b, length, heads, d), jnp.float32)
    beta = jnp.asarray(rng.rand(b, length, heads), jnp.float32)
    rate = jnp.asarray(-rng.uniform(0.05, 0.5, (heads, d)), jnp.float32)
    bias = jnp.asarray(rng.randn(heads, d) * 0.5, jnp.float32)
    w = jnp.asarray(rng.randn(b, length, heads, d), jnp.float32)
    seg = jnp.asarray(np.repeat(np.arange(4), [7, 30, 1, 26])[None].repeat(b, 0), jnp.int32) if packed else None
    kw = dict(chunk_size=32, matmul_dtype=jnp.float32, interpret=True, segment_ids=seg)

    def fused(q, k, v, a, beta, rate, bias):
        return jnp.sum(w * kda_fused(q, k, v, a, beta, gate=(rate, bias), **kw))

    def unfused(q, k, v, a, beta, rate, bias):
        g = rate * jax.nn.softplus(a + bias)
        k, v = (jnp.repeat(x, heads // kv_heads, axis=2) for x in (k, v))
        return jnp.sum(w * kda_fused(q, k, v, g, beta, **kw))

    args = (q, k, v, a, beta, rate, bias)
    np.testing.assert_allclose(float(fused(*args)), float(unfused(*args)), rtol=1e-5)
    got = jax.grad(fused, argnums=tuple(range(7)))(*args)
    want = jax.grad(unfused, argnums=tuple(range(7)))(*args)
    for x, y in zip(got, want, strict=True):
        assert x.shape == y.shape
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-4, atol=1e-4)


# --- Packed documents: hard state resets at document boundaries ---
# chunk_size 16; lengths cover chunk-aligned boundaries, mid-chunk boundaries, several
# boundaries inside one chunk (1 and 3 share a chunk), and a length-1 document.
_DOC_LAYOUTS = {"aligned": [16, 32, 16], "mid_chunk": [5, 20, 1, 3, 35]}


def _packed(doc_lengths, *, dk=16, dv=16, heads=2, seed=21):
    length = sum(doc_lengths)
    q, k, v, g, beta = _inputs(1, heads, length, dk, dv, seed=seed)
    seg = jnp.asarray(np.repeat(np.arange(len(doc_lengths)), doc_lengths)[None], jnp.int32)  # (1, L)
    return (q, k, v, g, beta), seg


def _per_document(fn, args, doc_lengths):
    """Run ``fn`` on each document separately (zero initial state) and concatenate."""
    bounds = np.cumsum([0, *doc_lengths])
    outs = [
        fn(*(x[..., lo:hi, :] if x.ndim == 4 else x[..., lo:hi] for x in args)) for lo, hi in itertools.pairwise(bounds)
    ]
    return jnp.concatenate(outs, axis=-2)


def _fused_model_layout(q, k, v, g, beta, seg):

    out = kda_fused(
        *(jnp.swapaxes(x, 1, 2) for x in (q, k, v, g, beta)),
        segment_ids=seg,
        chunk_size=16,
        matmul_dtype=jnp.float32,
        interpret=True,
    )
    return jnp.swapaxes(out, 1, 2)


_PACKED_IMPLS = {
    "recurrent": lambda *a, seg: recurrent_kda(*a, segment_ids=seg)[0],
    "xla": lambda *a, seg: chunk_kda(*a, chunk_size=16, matmul_dtype=jnp.float32, segment_ids=seg)[0],
    "pallas": lambda *a, seg: chunk_kda(
        *a,
        chunk_size=16,
        matmul_dtype=jnp.float32,
        segment_ids=seg,
        prep_impl="pallas_interpret",
        scan_impl="pallas_interpret",
    )[0],
    "pallas_prep_assoc_scan": lambda *a, seg: chunk_kda(
        *a, chunk_size=16, matmul_dtype=jnp.float32, segment_ids=seg, prep_impl="pallas_interpret"
    )[0],
    "kda_fused": lambda *a, seg: _fused_model_layout(*a, seg),
}


@pytest.mark.parametrize("impl", list(_PACKED_IMPLS))
@pytest.mark.parametrize("layout", list(_DOC_LAYOUTS))
def test_packed_documents_match_separate_runs(layout, impl):
    """A packed row equals running each document on its own from a zero state."""
    doc_lengths = _DOC_LAYOUTS[layout]
    args, seg = _packed(doc_lengths)
    want = _per_document(lambda *a: recurrent_kda(*a)[0], args, doc_lengths)
    got = _PACKED_IMPLS[impl](*args, seg=seg)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("impl", ["xla", "pallas", "kda_fused"])
def test_packed_document_gradients_match_separate_runs(impl):
    """Gradients through the reset (all five inputs) equal those of separate documents."""
    doc_lengths = _DOC_LAYOUTS["mid_chunk"]
    args, seg = _packed(doc_lengths, seed=22)
    w = jnp.asarray(np.random.RandomState(23).randn(1, 2, sum(doc_lengths), 16), jnp.float32)

    def packed_loss(*a):
        return jnp.sum(w * _PACKED_IMPLS[impl](*a, seg=seg))

    def separate_loss(*a):
        return jnp.sum(w * _per_document(lambda *d: recurrent_kda(*d)[0], a, doc_lengths))

    got = jax.grad(packed_loss, argnums=(0, 1, 2, 3, 4))(*args)
    want = jax.grad(separate_loss, argnums=(0, 1, 2, 3, 4))(*args)
    for x, y in zip(got, want, strict=True):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("impl", ["xla", "pallas", "kda_fused"])
def test_packed_documents_with_strong_decay_stay_finite(impl):
    """Strong decay (|g| ~ 24 per token; the gate is unbounded below) must not overflow
    the masked-off rows of earlier documents in a chunk into NaN, in values or gradients."""
    doc_lengths = _DOC_LAYOUTS["mid_chunk"]
    (q, k, v, g, beta), seg = _packed(doc_lengths, seed=24)
    g = g * 300.0

    def loss(*a):
        return jnp.sum(_PACKED_IMPLS[impl](*a, seg=seg))

    value, grads = jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
    assert np.isfinite(float(value))
    assert all(np.isfinite(np.asarray(x)).all() for x in grads)


@pytest.mark.parametrize("beta_value", [0.3, 0.95])
def test_nearly_parallel_keys_match_recurrent(beta_value):
    """A run of (almost) identical keys -- repeated tokens -- makes the intra-chunk inverse's
    power series blow up; the chunked kernels must still match the recurrence (the Neumann
    product form overflowed to inf/NaN here, which NaN'd real-text training)."""
    rng = np.random.RandomState(0)
    b, h, length, d = 1, 1, 256, 32
    k = np.tile(rng.randn(d), (length, 1))[None, None] + 0.01 * rng.randn(b, h, length, d)
    q, v = rng.randn(b, h, length, d), rng.randn(b, h, length, d)
    g = np.full((b, h, length, d), -0.001)
    beta = np.full((b, h, length), beta_value)
    q, k, v, g, beta = (jnp.asarray(x, jnp.float32) for x in (q, k, v, g, beta))
    ref, _ = recurrent_kda(q, k, v, g, beta)
    for chunk_size in (64, 128):
        out, _ = chunk_kda(q, k, v, g, beta, chunk_size=chunk_size)
        np.testing.assert_allclose(np.asarray(out), np.asarray(ref), atol=1e-2)
    fused = kda_fused(*(jnp.swapaxes(x, 1, 2) for x in (q, k, v, g, beta)), chunk_size=64, interpret=True)
    np.testing.assert_allclose(np.asarray(jnp.swapaxes(fused, 1, 2)), np.asarray(ref), atol=1e-2)


def test_floor_decay_is_exact_with_16_token_chunks():
    """With the per-token log-decay floored at -5 (K3's -5*sigmoid gate), 16-token chunks keep the
    cumulative decay within the deflation cap, so strong-decay chunks stay exact (larger chunks
    under-count exp(G_r - G_i) past the cap)."""
    rng = np.random.RandomState(1)
    b, h, length, d = 2, 2, 96, 16
    q, k, v = (jnp.asarray(rng.randn(b, h, length, d), jnp.float32) for _ in range(3))
    g = jnp.asarray(-5.0 * rng.uniform(0.9, 1.0, (b, h, length, d)), jnp.float32)
    g = g.at[..., : d // 2].set(-0.01)  # half the channels barely decay
    beta = jnp.asarray(rng.uniform(0.1, 0.9, (b, h, length)), jnp.float32)
    ref, _ = recurrent_kda(q, k, v, g, beta)
    out, _ = chunk_kda(q, k, v, g, beta, chunk_size=16, matmul_dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-4, atol=1e-4)
    fused = kda_fused(
        *(jnp.swapaxes(x, 1, 2) for x in (q, k, v, g, beta)), chunk_size=16, matmul_dtype=jnp.float32, interpret=True
    )
    np.testing.assert_allclose(np.asarray(jnp.swapaxes(fused, 1, 2)), np.asarray(ref), rtol=1e-4, atol=1e-4)
