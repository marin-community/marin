# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SwitchHead: per-head sigmoid top-k mixtures of value and output experts on the GQA layers."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug.attention import AttentionMask

import experiments.grug.fast_track.test_ngram_stat as t
from experiments.grug.fast_track.model import _switchhead_weights
from experiments.grug.fast_track.test_head_probe import _DENSE


def test_weights_keep_sigmoid_of_the_top_k_logits():
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 8))
    gate = jax.random.normal(jax.random.PRNGKey(1), (8, 3 * 4))
    weights = np.asarray(_switchhead_weights(x, gate, experts=4, topk=2))
    logits = np.asarray(jnp.einsum("bsd,dg->bsg", x, gate)).reshape(2, 5, 3, 4)
    top2 = np.argsort(-logits, axis=-1)[..., :2]
    expected = np.zeros_like(logits)
    np.put_along_axis(expected, top2, 1 / (1 + np.exp(-np.take_along_axis(logits, top2, -1))), -1)
    np.testing.assert_allclose(weights, expected, rtol=1e-5, atol=1e-6)


def _loss_and_grads(model, tokens, mesh):
    with jax.set_mesh(mesh):
        return eqx.filter_jit(eqx.filter_value_and_grad(lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape))))(
            model
        )


def test_vo_switchhead_trains_every_expert_bank_and_gate():
    mesh, model = t._model(**_DENSE, switchhead_experts=4, switchhead_topk=2, switchhead_sites="vo")
    attn = model.stacked_blocks.stacked.attn
    head_dim = model.config.inferred_head_dim
    assert attn.w_v.shape[-1] == 1 * 4 * head_dim
    assert attn.w_o.shape[-2] == 4 * 4 * head_dim
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, 16), 0, t._VOCAB)
    loss, grads = _loss_and_grads(model, tokens, mesh)
    assert np.isfinite(float(loss))
    g = grads.stacked_blocks.stacked.attn
    for leaf in (g.switch_v_gate, g.switch_o_gate):
        assert float(jnp.abs(leaf).max()) > 0
    # Every output expert of every head is picked by some token, so all of w_o's (n e h) row blocks learn.
    rows = np.abs(np.asarray(g.w_o)).reshape(g.w_o.shape[0], 4, 4, head_dim, -1).max(axis=(0, 3, 4))
    assert (rows > 0).all()


def _layer0_attn(model):
    return jax.tree.map(lambda a: a[0] if eqx.is_array(a) else a, model.stacked_blocks.stacked.attn)


def test_one_expert_with_zero_gates_is_quarter_scaled_plain_attention():
    """E=1, k=1 and zero gates weight V and O by sigmoid(0) = 1/2 each, so the layer is 1/4 of plain attention."""
    mesh, plain = t._model(**_DENSE)
    _, switched = t._model(**_DENSE, switchhead_experts=1, switchhead_topk=1, switchhead_sites="vo")
    plain_attn, switched_attn = _layer0_attn(plain), _layer0_attn(switched)
    switched_attn = eqx.tree_at(
        lambda a: (a.w_q, a.w_k, a.w_v, a.w_o, a.switch_v_gate, a.switch_o_gate),
        switched_attn,
        (
            plain_attn.w_q,
            plain_attn.w_k,
            plain_attn.w_v,
            plain_attn.w_o,
            jnp.zeros_like(switched_attn.switch_v_gate),
            jnp.zeros_like(switched_attn.switch_o_gate),
        ),
    )
    x = jax.random.normal(jax.random.PRNGKey(2), (2, 16, plain.config.hidden_dim))
    with jax.set_mesh(mesh):
        out_plain, _ = plain_attn(x, AttentionMask.causal())
        out_switched, stats = switched_attn(x, AttentionMask.causal())
    np.testing.assert_allclose(np.asarray(out_switched), 0.25 * np.asarray(out_plain), rtol=2e-4, atol=1e-6)
    assert any(k.endswith("switch_o_load_max_ratio") for k in stats)
