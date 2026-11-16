# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX-only tests; PyTorch is needed only to regenerate the reference fixture."""

from dataclasses import replace
from pathlib import Path
import pickle
import sys
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from levanter.optim import randrand_core as rr


def shared_sketch(size, width):
    return np.random.default_rng(size * 1000 + width).standard_normal((size, width)).astype(np.float32)


@pytest.mark.parametrize(
    "enabled,scale,tag",
    [
        (True, 0.5, "normal"),
        (False, 0.5, "disabled"),
        (True, 0.0, "zero"),
        (True, 5.0, "equal"),
    ],
)
def test_shared_sketch_pytorch_291_trajectory(monkeypatch, enabled, scale, tag):
    data = np.load(Path(__file__).parent / "data" / "randrand" / "torch_reference.npz")
    real_basis = rr.lowrank_basis_from_sketch

    def with_shared_sketch(gram, sketch, rank):
        return real_basis(gram, jnp.asarray(shared_sketch(*sketch.shape)), rank)

    monkeypatch.setattr(rr, "lowrank_basis_from_sketch", with_shared_sketch)
    labels = ("matrix", "matrix", "matrix", "embed", "vector", "head")
    config = replace(
        rr.RandRANDConfig(),
        rank=4,
        rank_sketch=4,
        pair_k=3,
        min_dim_for_deflation=1,
        low_rank_correction=enabled,
        correction_scale=scale,
    )
    tx = rr.randrand(labels, config)
    params = tuple(jnp.asarray(data[f"initial_{i}"]) for i in range(6))
    state = tx.init(params)
    step = jax.jit(lambda p, s, g: (lambda u, ns: (optax.apply_updates(p, u), ns))(*tx.update(g, s, p)))
    max_parameter_error = 0.0
    for t in range(125):
        grads = tuple(jnp.asarray(data[f"grads_{i}"][t]) for i in range(6))
        params, state = step(params, state, grads)
        if t not in (0, 1, 98, 99, 100, 101, 109, 110, 120, 124):
            continue
        for i, p in enumerate(params):
            expected = data[f"{tag}_{t}_p_{i}"]
            max_parameter_error = max(max_parameter_error, float(np.max(np.abs(np.asarray(p) - expected))))
            np.testing.assert_allclose(p, expected, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(state.first[i], data[f"{tag}_{t}_exp_avg_{i}"], atol=1e-6, rtol=1e-5)
            np.testing.assert_allclose(state.second[i], data[f"{tag}_{t}_exp_avg_sq_{i}"], atol=1e-7, rtol=1e-5)
            ms = state.matrices[i]
            if ms is not None:
                np.testing.assert_allclose(ms.left_gram, data[f"{tag}_{t}_gram_left_{i}"], atol=1e-7, rtol=1e-5)
                np.testing.assert_allclose(
                    ms.right_gram,
                    data[f"{tag}_{t}_gram_right_{i}"],
                    atol=1e-7,
                    rtol=1e-5,
                )
                if t >= 100:
                    np.testing.assert_array_equal(ms.pair_mask, data[f"{tag}_{t}_pair_mask_{i}"])
                    # SVD signs can differ. Compare the actual selected operator.
                    x = jnp.asarray(data[f"grads_{i}"][0])
                    actual = ms.left_basis @ ((ms.left_basis.T @ x @ ms.right_basis) * ms.pair_mask) @ ms.right_basis.T
                    v, u, m = (data[f"{tag}_{t}_{k}_{i}"] for k in ("V", "U", "pair_mask"))
                    expected = v @ ((v.T @ np.asarray(x) @ u) * m) @ u.T
                    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
    print(f"{tag}: max parameter absolute error {max_parameter_error:.3e}")


def small_config(**kwargs):
    return replace(
        rr.RandRANDConfig(),
        rank=2,
        rank_sketch=3,
        pair_k=1,
        min_dim_for_deflation=1,
        randrand_warmup=2,
        basis_every=2,
        warmup_steps=0,
        **kwargs,
    )


def test_jit_resume_and_explicit_rng():
    p = {"matrix": jnp.ones((6, 4), jnp.float32)}
    tx = rr.randrand({"matrix": "matrix"}, small_config(), total_steps=20)
    state = tx.init(p)
    run = jax.jit(lambda p, s, g: (lambda u, ns: (optax.apply_updates(p, u), ns))(*tx.update(g, s, p)))
    grads = [{"matrix": jax.random.normal(jax.random.PRNGKey(i), (6, 4))} for i in range(9)]
    for g in grads[:3]:
        p, state = run(p, state, g)
    restored_p, restored_s = jax.tree.map(jnp.asarray, pickle.loads(pickle.dumps(jax.device_get((p, state)))))
    for g in grads[3:]:
        p, state = run(p, state, g)
        restored_p, restored_s = run(restored_p, restored_s, g)
    for a, b in zip(jax.tree.leaves((p, state)), jax.tree.leaves((restored_p, restored_s)), strict=True):
        np.testing.assert_array_equal(a, b)


def test_refresh_timing_and_zero_gradients():
    p = (jnp.ones((6, 4), jnp.float32),)
    tx = rr.randrand(("matrix",), small_config(), total_steps=10)
    state = tx.init(p)
    key = np.asarray(state.key).copy()
    update = jax.jit(tx.update)
    for t in range(6):
        delta, state = update((jnp.zeros_like(p[0]),), state, p)
        p = optax.apply_updates(p, delta)
        assert all(np.isfinite(x).all() for x in jax.tree.leaves(state))
        if t < 2:
            np.testing.assert_array_equal(state.key, key)
        if t == 2:
            assert not np.array_equal(state.key, key)
            assert float(state.matrices[0].pair_mask.sum()) == 1.0
            key = np.asarray(state.key).copy()
        if t == 3:
            np.testing.assert_array_equal(state.key, key)
        if t == 4:
            assert not np.array_equal(state.key, key)


def test_clipping_buckets_and_schedule():
    gs = tuple(jnp.array([v], jnp.float32) for v in (3.0, 4.0, 3.0, 4.0))
    clipped = rr._clip(gs, ("matrix", "embed", "vector", "head"), 1.0)
    expected = (3 / (3 + 1e-6), 4 / (5 + 1e-6), 3 / (5 + 1e-6), 4 / (4 + 1e-6))
    for a, b in zip(clipped, expected, strict=True):
        np.testing.assert_allclose(a, [b], atol=1e-7)
    schedule = jax.jit(rr.lr_multiplier)
    np.testing.assert_allclose(
        [schedule(0), schedule(500), schedule(1000), schedule(4959)],
        [0, 0.5, 1, 0],
        atol=1e-7,
    )


def test_adamw_routes_and_constant_lr():
    ps = {"e": jnp.ones((7, 4)), "v": jnp.ones(4), "h": jnp.ones((7, 4))}
    config = replace(
        rr.RandRANDConfig(),
        embed_lr=0.003,
        vector_lr=0.003,
        head_lr=0.003,
        embed_weight_decay=0.1,
        vector_weight_decay=0.1,
        head_weight_decay=0.1,
        max_grad_norm=None,
    )
    tx = rr.randrand({"e": "embed", "v": "vector", "h": "head"}, config, schedule=lambda _: 1.0)
    ref = optax.adamw(0.003, b1=0.9, b2=0.98, eps=1e-15, weight_decay=0.1)
    state, refstate = tx.init(ps), ref.init(ps)
    reference = ps
    for i in range(20):
        gs = jax.tree.map(lambda p, i=i: jax.random.normal(jax.random.PRNGKey(i), p.shape), ps)
        u, state = tx.update(gs, state, ps)
        v, refstate = ref.update(gs, refstate, reference)
        ps, reference = optax.apply_updates(ps, u), optax.apply_updates(reference, v)
    for a, b in zip(jax.tree.leaves(ps), jax.tree.leaves(reference), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-5)
    assert all(ms is None for ms in state.matrices)


def test_runtime_does_not_require_torch():
    program = f"""
import sys
sys.modules['torch'] = None
import importlib
import jax.numpy as jnp
sys.path.insert(0, {str(Path(rr.__file__).resolve().parent)!r})
rr = importlib.import_module('randrand_core')
p = (jnp.ones(4, dtype=jnp.float32),)
tx = rr.randrand(('vector',))
u, s = tx.update(p, tx.init(p), p)
u[0].block_until_ready()
assert int(s.count) == 1
"""
    subprocess.run([sys.executable, "-c", program], check=True, timeout=60)
