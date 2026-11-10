# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Options that cut the June model's per-layer routing statistics, on four devices (data=2 x expert=2), ring_filtered.

- ``qb_threshold_impl="radix_select"``: the QB thresholds come from an exact radix select instead of a top_k sort.
- ``router_diagnostics=False`` and ``skip_zero_router_z_loss``: no diagnostic statistics, no z-loss term at 0.
"""

import dataclasses
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.sharding import compact_grug_mesh
from levanter.testing.devices import use_test_devices

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer

# Transport and expert capacity well below the demand, so ring_filtered drops on both sides.
BASE = GrugModelConfig(
    vocab_size=32,
    hidden_dim=16,
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
    capacity_factor=0.5,
    pooled_transport_capacity_factor=0.6,
    attention_implementation="reference",
    moe_implementation="ring_filtered",
    use_array_stacked_blocks=True,
    disable_pko=True,
)


def _run(cfg):
    """Loss, summarized metrics, gradients and compiled HLO of one differentiated step."""
    mesh = compact_grug_mesh(expert_axis_size=2, replica_axis_size=1)
    tokens = jax.random.randint(jax.random.key(1), (8, 8), 0, cfg.vocab_size)
    weight = jnp.ones((8, 8), dtype=jnp.float32)
    with jax.set_mesh(mesh):
        model = Transformer.init(cfg, key=jax.random.key(0))

        def loss_fn(m):
            return m.next_token_loss(tokens, weight, reduction="mean", return_router_metrics=True)

        step = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
        (loss, metrics), grads = step(model)
        hlo = step.lower(model).compile().as_text()
    return np.asarray(loss), jax.tree.map(np.asarray, metrics), jax.tree.map(np.asarray, grads), hlo


def _assert_same(got, want):
    for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_array_equal(g, w)


@pytest.mark.timeout(300)
@pytest.mark.multidevice
def test_radix_select_qb_thresholds_match_top_k():
    with use_test_devices(4):
        loss, metrics, grads, _ = _run(BASE)
        r_loss, r_metrics, r_grads, _ = _run(dataclasses.replace(BASE, qb_threshold_impl="radix_select"))
    assert (metrics["moe/dropped_assignments"] > 0).all(), metrics["moe/dropped_assignments"]
    np.testing.assert_array_equal(r_loss, loss)
    _assert_same(r_metrics, metrics)
    _assert_same(r_grads, grads)


@pytest.mark.timeout(300)
@pytest.mark.multidevice
def test_without_router_diagnostics_training_is_unchanged():
    with use_test_devices(4):
        loss, metrics, grads, _ = _run(BASE)
        o_loss, o_metrics, o_grads, _ = _run(dataclasses.replace(BASE, router_diagnostics=False))
        # A weighted z-loss is still computed, as before.
        weighted = dataclasses.replace(BASE, router_z_loss_coef=1e-2)
        w_loss, w_metrics, w_grads, _ = _run(weighted)
        ow_loss, ow_metrics, ow_grads, _ = _run(dataclasses.replace(weighted, router_diagnostics=False))

    np.testing.assert_array_equal(o_loss, loss)
    _assert_same(o_grads, grads)
    kept = {
        "qb_beta_per_layer",
        "moe/dropped_assignments",
        "moe/skipped_padding_assignments",
        "train/cross_entropy_loss",
        "train/router/aux_loss_weighted",
    }
    assert set(o_metrics) == kept, sorted(set(o_metrics) ^ kept)
    for name in kept:
        np.testing.assert_array_equal(o_metrics[name], metrics[name])

    np.testing.assert_allclose(ow_loss, w_loss, rtol=1e-6)
    z_loss = "train/router/router_z_loss"
    np.testing.assert_allclose(ow_metrics[z_loss], w_metrics[z_loss], rtol=1e-6)
    for got, want in zip(jax.tree.leaves(ow_grads), jax.tree.leaves(w_grads), strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-7)


@pytest.mark.timeout(300)
@pytest.mark.multidevice
def test_skipping_a_zero_router_z_loss_drops_its_backward():
    with use_test_devices(4):
        loss, metrics, grads, hlo = _run(BASE)
        s_loss, s_metrics, s_grads, s_hlo = _run(dataclasses.replace(BASE, skip_zero_router_z_loss=True))
    np.testing.assert_array_equal(s_loss, loss)
    _assert_same(s_metrics, metrics)
    _assert_same(s_grads, grads)

    def exponentials(text):
        return len(re.findall(r"\bexponential\(", text))

    assert exponentials(s_hlo) < exponentials(hlo), (exponentials(s_hlo), exponentials(hlo))
