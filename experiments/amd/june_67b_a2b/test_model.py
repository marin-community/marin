# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The AMD copy of the June model against June's own, on four CPU devices (its Pallas kernels in the interpreter)."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import Pembed_vocab, compact_grug_mesh
from levanter.testing.devices import use_test_devices

from experiments.amd.june_67b_a2b import model as amd
from experiments.june_tpu_67b_a2b.moe import model as june

# Widths the kernels tile (hidden a multiple of 128, head_dim // 4 a power of two >= 16); five layers, so the
# scan runs both attention kinds (layers 3 and 4 are long); the long layers' own options and a router z-loss on.
CONFIG = june.GrugModelConfig(
    vocab_size=64,
    hidden_dim=128,
    intermediate_dim=32,
    shared_expert_intermediate_dim=32,
    num_experts=4,
    num_experts_per_token=2,
    num_layers=5,
    num_heads=2,
    num_kv_heads=1,
    head_dim=64,
    max_seq_len=16,
    sliding_window=8,
    qk_mult_long_scale=1.5,
    disable_long_rope=True,
    router_z_loss_coef=0.01,
    capacity_factor=1.0,
    pooled_transport_capacity_factor=1.0,
    attention_implementation="reference",
    moe_implementation="ring_filtered",
    use_array_stacked_blocks=True,
    disable_pko=True,
)
BATCH = 4


def _loss_and_grads(params_like, tokens, loss_weight, mask):
    params, static = eqx.partition(params_like, eqx.is_array)

    def loss(p):
        return eqx.combine(p, static).next_token_loss(
            tokens, loss_weight, mask=mask, return_router_metrics=True, logsumexp_weight=1e-3
        )

    compiled = jax.jit(jax.value_and_grad(loss, has_aux=True)).lower(params).compile()
    (value, metrics), grads = compiled(params)
    # The result types of each all-reduce, one string per collective (a tuple when XLA combined several).
    all_reduces = [
        line.split(" = ", 1)[1].split(" all-reduce(")[0]
        for line in compiled.as_text().splitlines()
        if " all-reduce(" in line
    ]
    return value, metrics, grads, all_reduces


def test_amd_model_trains_like_june_from_the_same_parameters():
    with use_test_devices(4), jax.set_mesh(compact_grug_mesh(expert_axis_size=2)):
        tokens = jax.random.randint(jax.random.key(1), (BATCH, CONFIG.max_seq_len), 0, CONFIG.vocab_size)
        loss_weight = jnp.ones(tokens.shape).at[:, -1].set(0)
        # Two documents per row, the second ending in padding (segment -1), so masking and token validity matter.
        segments = np.zeros(tokens.shape, np.int32)
        segments[:, 6:] = 1
        segments[::2, -3:] = -1
        mask = AttentionMask(is_causal=True, segment_ids=(jnp.asarray(segments), jnp.asarray(segments)))
        loss_weight = jnp.where(segments >= 0, loss_weight, 0)

        june_model = jax.jit(june.Transformer.init, static_argnums=0)(CONFIG, key=jax.random.key(0))
        amd_model = jax.jit(amd.Transformer.init, static_argnums=0)(CONFIG, key=jax.random.key(0))
        # The same parameter tree: paths (hence checkpoint layout), shapes and values.
        june_leaves = jax.tree_util.tree_flatten_with_path(june_model)[0]
        amd_leaves = jax.tree_util.tree_flatten_with_path(amd_model)[0]
        assert [jax.tree_util.keystr(p) for p, _ in amd_leaves] == [jax.tree_util.keystr(p) for p, _ in june_leaves]
        for (_, a), (_, b) in zip(amd_leaves, june_leaves, strict=True):
            np.testing.assert_array_equal(a, b)

        june_loss, june_metrics, june_grads, _ = _loss_and_grads(june_model, tokens, loss_weight, mask)
        amd_loss, amd_metrics, amd_grads, amd_all_reduces = _loss_and_grads(amd_model, tokens, loss_weight, mask)

    # The kernels compute the norms' float32 statistics in another order than XLA, so values agree to float32
    # rounding; routing and capacity drops come out the same.
    np.testing.assert_allclose(amd_loss, june_loss, rtol=1e-5)
    np.testing.assert_allclose(
        amd_metrics["train/cross_entropy_loss"], june_metrics["train/cross_entropy_loss"], rtol=1e-5
    )
    np.testing.assert_allclose(
        amd_metrics["train/router/router_z_loss"], june_metrics["train/router/router_z_loss"], rtol=1e-5
    )
    np.testing.assert_allclose(amd_metrics["qb_beta_per_layer"], june_metrics["qb_beta_per_layer"], rtol=1e-5, atol=1e-6)
    for key in (amd.MOE_DROPPED_ASSIGNMENTS_METRIC, amd.MOE_SKIPPED_PADDING_ASSIGNMENTS_METRIC):
        np.testing.assert_array_equal(amd_metrics[key], june_metrics[key])
    # The drop counts are reduced in float32 with the other routing statistics: no integer all-reduce.
    assert not any("s32" in types for types in amd_all_reduces), amd_all_reduces
    for (path, g), (_, want) in zip(
        jax.tree_util.tree_flatten_with_path(amd_grads)[0],
        jax.tree_util.tree_flatten_with_path(june_grads)[0],
        strict=True,
    ):
        g, want = np.asarray(g, np.float64), np.asarray(want, np.float64)
        assert np.linalg.norm(g - want) <= 1e-4 * max(np.linalg.norm(want), 1e-30), jax.tree_util.keystr(path)


@pytest.mark.parametrize("model_axis", [1, 2])
def test_embedding_lookup_gradient_is_the_scatter_add_and_repeats_bitwise(model_axis):
    vocab, hidden = 64, 128
    with use_test_devices(4):
        mesh = compact_grug_mesh(expert_axis_size=2, model_axis_size=model_axis)
    with use_test_devices(4), jax.set_mesh(mesh):
        embed = jax.device_put(
            jax.random.normal(jax.random.key(0), (vocab, hidden)), jax.NamedSharding(mesh, Pembed_vocab)
        )
        # Zipf-distributed ids, so most ids repeat within and across devices.
        ids = (np.random.default_rng(0).zipf(1.3, size=(BATCH, 32)) % vocab).astype(np.int32)
        tokens = jax.device_put(ids, jax.NamedSharding(mesh, P(june._BATCH_AXES, None)))
        cotangent = jax.device_put(
            jax.random.normal(jax.random.key(1), (BATCH, 32, hidden)), jax.NamedSharding(mesh, P(june._BATCH_AXES))
        )

        @jax.jit
        def grads(embed, tokens, cotangent):
            spec = P(june._BATCH_AXES)
            ours = jax.vjp(lambda e: amd.embedding_lookup(e, tokens, spec), embed)[1](cotangent)[0]
            scatter = jax.vjp(lambda e: e.at[tokens].get(out_sharding=spec), embed)[1](cotangent)[0]
            return ours, scatter

        first, scatter = grads(embed, tokens, cotangent)
        second, _ = grads(embed, tokens, cotangent)

    np.testing.assert_allclose(first, scatter, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(first).view(np.uint32), np.asarray(second).view(np.uint32))


@pytest.mark.parametrize(
    "change",
    [
        {"moe_implementation": "ring"},
        {"pooled_transport_capacity_factor": None},
        {"use_array_stacked_blocks": False},
        {"ce_implementation": "batched_xla"},
        {"remat_mode": "save_moe"},
        {"scan_unroll": 2},
        {"head_dim": 32},
        {"hidden_dim": 192, "head_dim": 64},
    ],
)
def test_amd_model_rejects_june_configs_it_does_not_implement(change):
    # The copy shares June's GrugModelConfig, so a field it silently ignored would break that contract: it must raise.
    with pytest.raises(ValueError):
        amd.Transformer.init(dataclasses.replace(CONFIG, **change), key=jax.random.key(0))
