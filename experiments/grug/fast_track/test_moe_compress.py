# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MLP compression terms: the shadow probe trains only itself, the transfer loss moves only the shared expert,
and the routed penalty reaches the routed experts."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import experiments.grug.fast_track.test_ngram_stat as t
from experiments.grug.fast_track.model import MoeCompress


def _grads(**overrides):
    mesh, model = t._model(ngram_stat_rows=0, shared_ungated_relu2=True, **overrides)
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    weight = jnp.ones(tokens.shape, jnp.float32)

    def grads(train_terms):
        loss = lambda m: m.next_token_loss(tokens, weight, train_terms=train_terms)  # noqa: E731
        return eqx.filter_jit(eqx.filter_grad(loss))(model)

    with jax.set_mesh(mesh):
        with_terms, without = grads(True), grads(False)
    paths = jax.tree_util.tree_leaves_with_path(with_terms)
    diff = {
        jax.tree_util.keystr(p): float(jnp.max(jnp.abs(a - b)))
        for (p, a), b in zip(paths, jax.tree.leaves(without), strict=True)
    }
    return diff, with_terms


def _moved(diff, needle):
    return [k for k, v in diff.items() if needle in k and v > 0]


def test_shadow_probe_trains_only_itself():
    diff, grads = _grads(moe_shadow_width=24)
    assert _moved(diff, ".shadow.")
    assert all(".shadow." in k for k, v in diff.items() if v > 0)
    assert float(jnp.abs(grads.kda_blocks.stacked.shadow.w_up).max()) > 0


def test_transfer_moves_the_shared_expert_but_not_the_routed_experts():
    diff, _ = _grads(moe_compress=MoeCompress.TRANSFER, moe_compress_weight=0.1)
    assert _moved(diff, ".shared[0].w_up") and _moved(diff, ".shared[0].w_down")
    assert not _moved(diff, ".mlp.")
    assert not _moved(diff, "attn")


def test_routed_penalty_reaches_the_routed_experts():
    diff, _ = _grads(moe_compress=MoeCompress.ROUTED_PENALTY, moe_compress_weight=0.1)
    assert _moved(diff, ".mlp.expert_mlp.")
    # The last layer's penalty reaches only its own routed path and the layers below it.
    assert not _moved(diff, ".stacked_blocks.stacked.shared[0].")


def test_compress_needs_a_weight():
    try:
        t._config(moe_compress=MoeCompress.TRANSFER)
    except ValueError:
        return
    raise AssertionError("moe_compress without a weight must fail")


def test_shadow_r2_is_logged_per_layer():
    mesh, model = t._model(ngram_stat_rows=0, shared_ungated_relu2=True, moe_shadow_width=24)
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    with jax.set_mesh(mesh):
        _, metrics = eqx.filter_jit(
            lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape), train_terms=True, return_router_metrics=True)
        )(model)
    r2 = [float(metrics[f"train/compress/shadow_r2_L{i}"]) for i in range(2)]
    assert all(np.isfinite(r2)) and all(v < 1.0 for v in r2)
    assert 0.0 < float(metrics["train/compress/r_share_L0"]) < 10.0
