# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Value, gradient and checkpoint behavior of the June model's fused norms."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend import core as jex_core
from levanter.grug.sharding import compact_grug_mesh
from levanter.testing.devices import use_test_devices

from experiments.june_tpu_67b_a2b.moe.model import GatedNorm, GrugModelConfig, RMSNorm, Transformer, fused_norm_and_gate

# The Pallas gate kernels tile the hidden width in 128 columns.
_SMALL = GrugModelConfig(
    vocab_size=200,
    hidden_dim=128,
    intermediate_dim=16,
    shared_expert_intermediate_dim=16,
    num_experts=4,
    num_experts_per_token=2,
    num_layers=2,
    num_heads=2,
    num_kv_heads=1,
    head_dim=16,
    max_seq_len=16,
    sliding_window=8,
    capacity_factor=8.0,
    attention_implementation="reference",
    use_array_stacked_blocks=True,
    disable_pko=True,
    ce_implementation="xla",
)


def _perturbed(model, dtype):
    # Perturb every weight (the norm weights start at 1), then cast.
    return jax.tree.map(
        lambda a: (
            (a * (1 + 0.1 * jnp.sin(jnp.arange(a.size).reshape(a.shape)))).astype(dtype)
            if eqx.is_inexact_array(a)
            else a
        ),
        model,
    )


def _setup(cfg: GrugModelConfig, dtype):
    tokens = jax.random.randint(jax.random.key(1), (4, cfg.max_seq_len), 0, cfg.vocab_size)
    loss_weight = jnp.ones(tokens.shape, jnp.float32)
    model = _perturbed(Transformer.init(cfg, key=jax.random.key(0)), dtype)
    params, static = eqx.partition(model, eqx.is_array)

    def loss(p):
        return eqx.combine(p, static).next_token_loss(tokens, loss_weight, logsumexp_weight=1e-3)

    return jax.value_and_grad(loss), params


def _loss_and_grads(cfg: GrugModelConfig, dtype=jnp.float32):
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        step, params = _setup(cfg, dtype)
        value, grads = jax.jit(step)(params)
    return value, jax.tree.leaves(grads)


def _projections(cfg: GrugModelConfig, width: int) -> int:
    """GEMMs [..., D] x [D, width] -> [..., width] in the differentiated step's jaxpr (GatedNorm's down projection for
    width 128, the head gate for width num_heads), each loop body and conditional branch counted once."""
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        step, params = _setup(cfg, jnp.float32)
        jaxpr = jax.make_jaxpr(step)(params).jaxpr

    def count(jaxpr) -> int:
        total = 0
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "dot_general":
                rhs, out = eqn.invars[1].aval, eqn.outvars[0].aval
                total += rhs.shape == (cfg.hidden_dim, width) and out.ndim >= 3 and out.shape[-1] == width
            for param in eqn.params.values():
                for sub in param if isinstance(param, (tuple, list)) else (param,):
                    if isinstance(sub, jex_core.ClosedJaxpr):
                        total += count(sub.jaxpr)
                    elif isinstance(sub, jex_core.Jaxpr):
                        total += count(sub)
        return total

    return count(jaxpr)


def _assert_grads_close(grads, base, rtol):
    assert len(grads) == len(base)
    for g, b in zip(grads, base, strict=True):
        g, b = np.asarray(g, np.float32), np.asarray(b, np.float32)
        np.testing.assert_allclose(g, b, rtol=rtol, atol=rtol * max(np.abs(b).max(), 1e-30))


@pytest.mark.parametrize("impl", ["xla", "pallas"])
@pytest.mark.parametrize("stacked", [True, False])
def test_fused_norm_matches_separate_modules_float32(impl, stacked):
    cfg = dataclasses.replace(_SMALL, use_array_stacked_blocks=stacked)
    base_loss, base_grads = _loss_and_grads(cfg)
    loss, grads = _loss_and_grads(dataclasses.replace(cfg, fused_norm_impl=impl))
    np.testing.assert_allclose(loss, base_loss, rtol=1e-6)
    _assert_grads_close(grads, base_grads, rtol=1e-4)


@pytest.mark.parametrize("impl", ["xla", "pallas"])
@pytest.mark.timeout(300)
def test_fused_norm_matches_separate_modules_bfloat16(impl):
    base_loss, base_grads = _loss_and_grads(_SMALL, jnp.bfloat16)
    loss, grads = _loss_and_grads(dataclasses.replace(_SMALL, fused_norm_impl=impl), jnp.bfloat16)
    np.testing.assert_allclose(np.float32(loss), np.float32(base_loss), rtol=2e-2)
    _assert_grads_close(grads, base_grads, rtol=6e-2)


def test_fused_norm_returns_the_residual_sum_and_the_gated_norm():
    k = jax.random.split(jax.random.key(2), 4)
    x = jax.random.normal(k[0], (2, 8, 32)).astype(jnp.bfloat16)
    residual = jax.random.normal(k[1], (2, 8, 32)).astype(jnp.bfloat16)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        norm = RMSNorm.init(32, 1e-5)
        gated = jax.tree.map(lambda a: a.astype(jnp.bfloat16), GatedNorm.init(32, 0.5, key=k[2]))
        want = np.asarray(gated(norm(x + residual)), np.float32)
        for impl in ("xla", "pallas"):
            h, out = fused_norm_and_gate(x, residual, norm, gated, impl, return_h=True)
            np.testing.assert_array_equal(np.asarray(h, np.float32), np.asarray(x + residual, np.float32))
            np.testing.assert_allclose(np.asarray(out, np.float32), want, rtol=2**-7, atol=2**-7 * np.abs(want).max())


@pytest.mark.parametrize("fused", ["xla", "pallas"])
def test_pallas_gate_matches_xla_gate_float32(fused):
    cfg = dataclasses.replace(_SMALL, fused_norm_impl=fused)
    base_loss, base_grads = _loss_and_grads(cfg)
    loss, grads = _loss_and_grads(dataclasses.replace(cfg, gated_norm_impl="pallas"))
    np.testing.assert_allclose(loss, base_loss, rtol=1e-6)
    _assert_grads_close(grads, base_grads, rtol=1e-4)


@pytest.mark.timeout(300)
def test_pallas_gate_in_bfloat16_is_as_close_to_float32_as_xla_gate():
    # bfloat16 rounds the two paths differently; each leaf of the kernels' run must be about as close to a float32
    # run as the JAX ops' run is.
    cfg = dataclasses.replace(_SMALL, fused_norm_impl="pallas")
    anchor_loss, anchor = _loss_and_grads(cfg)
    base_loss, base_grads = _loss_and_grads(cfg, jnp.bfloat16)
    loss, grads = _loss_and_grads(dataclasses.replace(cfg, gated_norm_impl="pallas"), jnp.bfloat16)
    for value in (base_loss, loss):
        np.testing.assert_allclose(np.float32(value), np.float32(anchor_loss), rtol=2e-2)

    def rel(x, ref):
        x, ref = np.asarray(x, np.float64), np.asarray(ref, np.float64)
        return np.linalg.norm(x - ref) / max(np.linalg.norm(ref), 1e-30)

    for g, b, ref in zip(grads, base_grads, anchor, strict=True):
        assert rel(g, ref) <= 1.5 * rel(b, ref) + 1e-3


@pytest.mark.parametrize(
    ("fused", "gate", "remat_mode"),
    [
        ("off", "xla", "recompute_all"),
        ("pallas", "xla", "save_moe"),
        ("pallas", "pallas", "save_moe"),
        ("xla", "pallas", "recompute_all"),
    ],
)
def test_save_hidden_skips_the_down_projection_rerun(fused, gate, remat_mode):
    cfg = dataclasses.replace(_SMALL, fused_norm_impl=fused, gated_norm_impl=gate, remat_mode=remat_mode)
    saved = dataclasses.replace(cfg, gated_norm_remat="save_hidden")
    # Each block's two GatedNorms run their down projection in the forward and again in the backward's rerun of the
    # block; with the projection saved, only in the forward.
    assert _projections(cfg, 128) - _projections(saved, 128) == 2
    base_loss, base_grads = _loss_and_grads(cfg)
    loss, grads = _loss_and_grads(saved)
    np.testing.assert_allclose(loss, base_loss, rtol=1e-6)
    _assert_grads_close(grads, base_grads, rtol=1e-5)


def test_saved_head_gate_skips_its_gemm_rerun():
    cfg = dataclasses.replace(_SMALL, remat_mode="save_moe")
    saved = dataclasses.replace(cfg, head_gate_remat="save")
    # The backward's rerun of the block reruns the head-gate GEMM in each attention branch (long and short under the
    # conditional); saved, it does not.
    assert _projections(cfg, cfg.num_heads) - _projections(saved, cfg.num_heads) == 2
    base_loss, base_grads = _loss_and_grads(cfg)
    loss, grads = _loss_and_grads(saved)
    np.testing.assert_allclose(loss, base_loss, rtol=1e-6)
    _assert_grads_close(grads, base_grads, rtol=1e-5)


@pytest.mark.timeout(300)
@pytest.mark.multidevice
@pytest.mark.parametrize(
    "overrides",
    [
        {"fused_norm_impl": "xla"},
        {"fused_norm_impl": "pallas"},
        {"fused_norm_impl": "pallas", "gated_norm_impl": "pallas"},
        {"fused_norm_impl": "pallas", "gated_norm_impl": "pallas", "gated_norm_remat": "save_hidden"},
        {"gated_norm_remat": "save_hidden", "head_gate_remat": "save"},
    ],
)
def test_fused_norm_options_on_four_devices_match_separate_modules(overrides):
    # Batch-sharded activations must agree with the separate path, including weight gradients summed across devices.
    if overrides.get("fused_norm_impl") == "pallas" and jax.default_backend() == "tpu":
        pytest.skip("Pallas-Triton kernels run on GPU (and in the interpreter on CPU)")
    base = GrugModelConfig(
        vocab_size=32,
        hidden_dim=128,
        intermediate_dim=8,
        shared_expert_intermediate_dim=8,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=3,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=8,
        sliding_window=4,
        capacity_factor=4.0,
        attention_implementation="reference",
        moe_implementation="ring",
        use_array_stacked_blocks=True,
        disable_pko=True,
        remat_mode="save_moe",
    )
    with use_test_devices(4):
        tokens = jax.random.randint(jax.random.key(1), (8, 8), 0, base.vocab_size)
        weight = jnp.ones((8, 8), dtype=jnp.float32)

        def run(cfg):
            with jax.set_mesh(compact_grug_mesh(expert_axis_size=2, replica_axis_size=1)):
                model = _perturbed(Transformer.init(cfg, key=jax.random.key(0)), jnp.float32)
                step = jax.value_and_grad(lambda m: m.next_token_loss(tokens, weight, reduction="mean"))
                loss, grads = jax.jit(step)(model)
                jaxpr = str(jax.make_jaxpr(step)(model))
            return loss, jax.tree.leaves(grads), jaxpr

        loss, grads, _ = run(base)
        f_loss, f_grads, f_jaxpr = run(dataclasses.replace(base, **overrides))
        if overrides.get("fused_norm_impl", "off") != "off":
            assert "shard_map" in f_jaxpr
    np.testing.assert_allclose(f_loss, loss, rtol=1e-6)
    for got, want in zip(f_grads, grads, strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4 * float(np.max(np.abs(want))))
