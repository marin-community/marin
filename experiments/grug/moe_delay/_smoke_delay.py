# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU smoke test for the delayed-gradient optimizer wrapper. Run directly:

JAX_PLATFORMS=cpu .venv/bin/python -m experiments.grug.moe_delay._smoke_delay
"""

import jax
import jax.numpy as jnp
import optax
from jax.tree_util import GetAttrKey, SequenceKey

from experiments.grug.moe_delay.delay_optim import DelayedGrugMuonConfig, grug_stage_tau, wrap_delayed


def _params():
    return {"a": jnp.arange(6.0).reshape(2, 3), "b": jnp.ones(4)}


def _grads(key):
    k1, k2 = jax.random.split(key)
    return {"a": jax.random.normal(k1, (2, 3)), "b": jax.random.normal(k2, (4,))}


def test_tau0_identical():
    inner = optax.sgd(0.1)
    wrapped = wrap_delayed(inner, tau=0)
    p = _params()
    si, sw = inner.init(p), wrapped.init(p)
    key = jax.random.PRNGKey(0)
    for _ in range(5):
        key, sub = jax.random.split(key)
        g = _grads(sub)
        ui, si = inner.update(g, si, params=p)
        uw, sw = wrapped.update(g, sw, params=p)
        for k in g:
            assert jnp.allclose(ui[k], uw[k]), f"tau=0 mismatch on {k}"
    print("OK  tau=0 is bit-identical to inner")


def test_tau1_delays():
    lr = 0.1
    inner = optax.sgd(lr)
    wrapped = wrap_delayed(inner, tau=1)
    p = _params()
    sw = wrapped.init(p)
    key = jax.random.PRNGKey(1)
    g0 = _grads(jax.random.split(key)[0])
    g1 = _grads(jax.random.split(key)[1])
    # step 0: FIFO is full of zeros -> update applies the zero gradient.
    u0, sw = wrapped.update(g0, sw, params=p)
    assert jnp.allclose(u0["a"], 0.0), "step0 should apply zero (FIFO fill)"
    # step 1: should apply g0 (the gradient from the previous step), not g1.
    u1, sw = wrapped.update(g1, sw, params=p)
    assert jnp.allclose(u1["a"], -lr * g0["a"]), "step1 should apply the delayed g0"
    assert not jnp.allclose(u1["a"], -lr * g1["a"]), "step1 must NOT apply the fresh g1"
    print("OK  tau=1 applies the 1-step-delayed gradient")


def test_dc_asgd_runs():
    inner = optax.sgd(0.1)
    for corrector in ("dc_asgd", "dc_asgd_ema"):
        wrapped = wrap_delayed(inner, tau=2, corrector=corrector, dc_lambda=0.5)
        p = _params()
        sw = wrapped.init(p)
        key = jax.random.PRNGKey(2)
        for _ in range(4):
            key, sub = jax.random.split(key)
            g = _grads(sub)
            u, sw = wrapped.update(g, sw, params=p)
            p = optax.apply_updates(p, u)
        assert jnp.all(jnp.isfinite(p["a"])), f"{corrector} produced non-finite params"
        print(f"OK  corrector={corrector} runs and stays finite")


def test_weight_pred_predictor():
    # weight_pred exposes a forward predictor; other correctors do not.
    assert DelayedGrugMuonConfig(tau=4, corrector="none").make_forward_predictor() is None
    cfg = DelayedGrugMuonConfig(tau=4, corrector="weight_pred", pred_scale=1.0)
    predict = cfg.make_forward_predictor()
    assert predict is not None

    lr = 0.1
    wrapped = wrap_delayed(optax.sgd(lr), tau=4, corrector="weight_pred")
    p = _params()
    sw = wrapped.init(p)
    # Initial last_update is zero, so the predicted offset is zero.
    assert jnp.allclose(predict(sw)["a"], 0.0), "initial predicted offset must be zero"
    key = jax.random.PRNGKey(5)
    last_u = None
    for _ in range(6):
        key, sub = jax.random.split(key)
        last_u, sw = wrapped.update(_grads(sub), sw, params=p)
    # Predicted offset must equal tau * pred_scale * the last applied update.
    assert jnp.allclose(predict(sw)["a"], 4.0 * last_u["a"]), "predictor must be tau * last_update"
    print("OK  weight_pred forward predictor tracks tau * last_update")


def test_grug_stage_tau():
    # 6 layers, one stage per layer: last block fresh (τ=0), first block stalest.
    lt = grug_stage_tau(num_layers=6, num_stages=6)
    assert lt((GetAttrKey("blocks"), SequenceKey(5), GetAttrKey("w"))) == 0
    assert lt((GetAttrKey("blocks"), SequenceKey(4), GetAttrKey("w"))) == 1
    assert lt((GetAttrKey("blocks"), SequenceKey(0), GetAttrKey("w"))) == 5
    # input embedding group is in the first stage (stalest); output group is fresh.
    assert lt((GetAttrKey("token_embed"),)) == 5
    assert lt((GetAttrKey("output_proj"),)) == 0
    # Fewer stages than layers: contiguous chunks, still last stage fresh.
    lt3 = grug_stage_tau(num_layers=6, num_stages=3)
    assert lt3((GetAttrKey("blocks"), SequenceKey(0), GetAttrKey("w"))) == 2
    assert lt3((GetAttrKey("blocks"), SequenceKey(5), GetAttrKey("w"))) == 0
    print("OK  grug_stage_tau: last stage fresh, τ grows toward the first stage")


def test_per_leaf_delay():
    # Per-leaf profile: delay "a" by 2 steps, keep "b" fresh (τ=0).
    lr = 0.1
    leaf_tau = lambda path: 2 if path[0].key == "a" else 0  # noqa: E731
    wrapped = wrap_delayed(optax.sgd(lr), tau=0, leaf_tau=leaf_tau)
    p = _params()
    sw = wrapped.init(p)
    gs = [_grads(jax.random.PRNGKey(i)) for i in range(4)]
    for t in range(4):
        u, sw = wrapped.update(gs[t], sw, params=p)
        # "b" always applies the *current* gradient (fresh stage).
        assert jnp.allclose(u["b"], -lr * gs[t]["b"]), f"b must be fresh at step {t}"
        # "a" applies the gradient from 2 steps ago (zeros during the fill).
        expected_a = -lr * (gs[t - 2]["a"] if t >= 2 else jnp.zeros_like(gs[t]["a"]))
        assert jnp.allclose(u["a"], expected_a), f"a must be 2-step-delayed at step {t}"
    print("OK  per-leaf delay: fresh last stage + 2-step-delayed early stage in one tree")


def test_jit_and_config():
    # The wrapper must be jit-able (the grug loop jits the whole step) and the
    # config subclass must instantiate (multiple-inheritance dataclass sanity).
    cfg = DelayedGrugMuonConfig(tau=2, corrector="dc_asgd_ema", dc_lambda=0.3)
    assert cfg.tau == 2 and cfg.corrector == "dc_asgd_ema"
    inner = optax.sgd(0.1)
    wrapped = wrap_delayed(inner, tau=2, corrector="dc_asgd_ema")
    p = _params()
    sw = wrapped.init(p)

    @jax.jit
    def step(g, s, params):
        return wrapped.update(g, s, params=params)

    g = _grads(jax.random.PRNGKey(3))
    u, sw = step(g, sw, p)
    assert jnp.all(jnp.isfinite(u["a"]))
    print("OK  jit + DelayedGrugMuonConfig instantiation")


if __name__ == "__main__":
    test_tau0_identical()
    test_tau1_delays()
    test_dc_asgd_runs()
    test_weight_pred_predictor()
    test_grug_stage_tau()
    test_per_leaf_delay()
    test_jit_and_config()
    print("\nall delay-wrapper smoke checks passed")
