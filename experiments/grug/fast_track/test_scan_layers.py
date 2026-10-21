# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``scan_layers``: the AttnRes layer stack as a ``jax.lax.scan`` over layer units computes the unrolled loop's
loss, gradients and AttnRes gate logs."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import experiments.grug.fast_track.test_kda_local as t


def _loss_grads_metrics(dtype=jnp.float32, **overrides):
    with jax.enable_x64(dtype == jnp.float64):
        mesh, model = t._model(**overrides)
        model = jax.tree.map(lambda x: x.astype(dtype) if eqx.is_inexact_array(x) else x, model)
        tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)

        def loss_fn(m, x):
            return m.next_token_loss(x, jnp.ones(x.shape, dtype), return_router_metrics=True)

        with jax.set_mesh(mesh):
            (loss, metrics), grads = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn, has_aux=True))(model, tokens)
        return float(loss), [np.asarray(g) for g in _per_layer_leaves(grads)], jax.tree.map(np.asarray, metrics)


def _per_layer_leaves(grads):
    """Gradient leaves layer by layer, then the non-layer leaves: independent of how layers are grouped into
    stacks (``scan_layers`` splits off ``*_blocks_rest``)."""
    layers = [leaf for layer in grads.layers() for leaf in jax.tree.leaves(eqx.filter(layer, eqx.is_inexact_array))]
    stack_fields = [f for f in vars(grads) if f.endswith(("blocks", "blocks_tail", "blocks_rest"))]
    rest = eqx.tree_at(
        lambda m: [getattr(m, f) for f in stack_fields],
        grads,
        replace=[None] * len(stack_fields),
        is_leaf=lambda x: x is None,
    )
    return layers + jax.tree.leaves(eqx.filter(rest, eqx.is_inexact_array))


@pytest.mark.parametrize(
    "overrides",
    [
        # Every layer scanned, two layers per AttnRes block (the boundary falls mid-unit).
        dict(num_layers=8, attn_res_num_blocks=4),
        # The d512 layout: one scanned unit, then the latent_out_full tail layers unrolled.
        dict(num_layers=6, latent_out_full_layers=(4, 5)),
        # Two scanned units, then one layer in the rest stack ahead of the tail.
        dict(num_layers=10, attn_res_num_blocks=3, latent_out_full_layers=(9,)),
        # Three scanned units with an extra AttnRes source and the per-layer AttnRes knobs.
        dict(
            num_layers=14,
            attn_res_num_blocks=4,
            latent_out_full_layers=(12, 13),
            second_embed=True,
            second_embed_bigram=True,
            embed2_rows=64,
            attn_res_shared_query=True,
            attn_res_partial_scale=True,
        ),
    ],
)
@pytest.mark.timeout(240)  # three compiles per case: float32 unrolled and scanned, float64 unrolled
@pytest.mark.parametrize("two_phase", [False, True])
def test_scanned_layers_match_the_unrolled_loop(overrides, two_phase):
    loss, grads, metrics = _loss_grads_metrics(**overrides)
    scan_loss, scan_grads, scan_metrics = _loss_grads_metrics(
        scan_layers=True, attn_res_two_phase=two_phase, **overrides
    )
    _, exact_grads, _ = _loss_grads_metrics(jnp.float64, **overrides)
    np.testing.assert_allclose(scan_loss, loss, rtol=1e-6)
    # The scan accumulates the source cotangents in a different order, so its float32 gradients are not the
    # unrolled loop's bit for bit; they must be as close to the float64 gradients as the unrolled loop's are.
    for unrolled, scanned, exact in zip(grads, scan_grads, exact_grads, strict=True):
        unrolled_err = np.max(np.abs(unrolled - exact))
        assert np.max(np.abs(scanned - exact)) <= 2 * unrolled_err + 1e-6 * np.max(np.abs(exact)) + 1e-12
    gate_logs = sorted(k for k in metrics if k.startswith("train/attn_res/w_g"))
    assert gate_logs and gate_logs == sorted(k for k in scan_metrics if k.startswith("train/attn_res/w_g"))
    for key in gate_logs:
        np.testing.assert_allclose(np.asarray(scan_metrics[key]), np.asarray(metrics[key]), rtol=1e-5, atol=1e-7)


@pytest.mark.timeout(240)
def test_prefetched_partials_match_the_unprefetched_scan():
    overrides = dict(num_layers=10, attn_res_num_blocks=3, latent_out_full_layers=(9,), scan_layers=True)
    loss, grads, _ = _loss_grads_metrics(scan_offload_partials=True, **overrides)
    prefetched_loss, prefetched_grads, _ = _loss_grads_metrics(
        scan_offload_partials=True, scan_prefetch_partials=True, **overrides
    )
    assert prefetched_loss == loss
    for want, got in zip(grads, prefetched_grads, strict=True):
        np.testing.assert_array_equal(got, want)


@pytest.mark.timeout(240)
def test_device_sources_match_the_offloaded_scan():
    overrides = dict(num_layers=10, attn_res_num_blocks=3, latent_out_full_layers=(9,), scan_layers=True)
    loss, grads, _ = _loss_grads_metrics(scan_offload_partials=True, **overrides)
    device_loss, device_grads, _ = _loss_grads_metrics(scan_offload_partials=True, scan_device_sources=True, **overrides)
    assert device_loss == loss
    for want, got in zip(grads, device_grads, strict=True):
        np.testing.assert_array_equal(got, want)


@pytest.mark.timeout(240)
def test_released_weight_grads_match_the_barriered_scan():
    overrides = dict(num_layers=10, attn_res_num_blocks=3, latent_out_full_layers=(9,), scan_layers=True)
    loss, grads, _ = _loss_grads_metrics(**overrides)
    released_loss, released_grads, _ = _loss_grads_metrics(scan_release_weight_grads=True, **overrides)
    assert released_loss == loss
    for want, got in zip(grads, released_grads, strict=True):
        np.testing.assert_array_equal(got, want)


@pytest.mark.timeout(400)
@pytest.mark.parametrize("fused_source_grads", [False, True])
def test_static_boundaries_match_the_traced_boundaries(fused_source_grads):
    # Blocks of 6 layers in units of 4: the traced boundary falls in different unit layers from unit to unit; the
    # static schedule scans 12-layer units instead, so the boundaries sit at unit layers 0 and 6 in every unit.
    overrides = dict(
        num_layers=24, attn_res_num_blocks=4, scan_layers=True, attn_res_fused_source_grads=fused_source_grads
    )
    loss, grads, metrics = _loss_grads_metrics(**overrides)
    static_loss, static_grads, static_metrics = _loss_grads_metrics(attn_res_static_boundaries=True, **overrides)
    _, exact_grads, _ = _loss_grads_metrics(
        jnp.float64, num_layers=24, attn_res_num_blocks=4, scan_layers=True, attn_res_static_boundaries=True
    )
    np.testing.assert_allclose(static_loss, loss, rtol=1e-6)
    # XLA fuses the two programs differently, so float32 gradients are compared by their error against float64.
    for traced, static, exact in zip(grads, static_grads, exact_grads, strict=True):
        traced_err = np.max(np.abs(traced - exact))
        assert np.max(np.abs(static - exact)) <= 2 * traced_err + 1e-6 * np.max(np.abs(exact)) + 1e-12
    gate_logs = sorted(k for k in metrics if k.startswith("train/attn_res/w_g"))
    assert gate_logs
    for key in gate_logs:
        np.testing.assert_allclose(np.asarray(static_metrics[key]), np.asarray(metrics[key]), rtol=1e-5, atol=1e-7)
