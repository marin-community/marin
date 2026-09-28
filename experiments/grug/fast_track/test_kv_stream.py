# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""KV side stream (``kv_stream_dim``) on the KDA + MLA hybrid: the side stream is the only K/V source,
the model stays causal and document-isolated through both streams, every side-stream parameter trains
and routes to its main-stream analogue's optimizer group, and the flag off leaves the model unchanged."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.attention import AttentionMask

import experiments.grug.fast_track.test_kda_local as t
from experiments.grug.fast_track.model import AttnResLayerBackward, CausalSelfAttention, KimiDeltaAttention
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

_W = 16
_MLA = dict(
    mla=True,
    mla_kv_latent_dim=16,
    qk_norm=False,
    mla_k_norm=True,
    mla_q_norm=True,
    attn_res_layer_backward=AttnResLayerBackward.SAVE,
)


def _model(kv_stream_dim: int):
    return t._model(**_MLA, kv_stream_dim=kv_stream_dim, kv_stream_heads=2)


def _loss_and_grads(mesh, model, tokens):
    weights = jnp.ones(tokens.shape, jnp.float32)
    with jax.set_mesh(mesh):
        return eqx.filter_jit(eqx.filter_value_and_grad(lambda m: m.next_token_loss(tokens, weights)))(model)


def _tokens(seed: int) -> jax.Array:
    return jax.random.randint(jax.random.PRNGKey(seed), (2, t._SEQ), 0, t._VOCAB)


def test_flag_off_is_the_base_model():
    tokens = _tokens(2)
    mesh, base = t._model(**_MLA)
    _, off = _model(0)
    assert off.kv_stream is None
    assert [x.shape for x in jax.tree.leaves(off)] == [x.shape for x in jax.tree.leaves(base)]
    np.testing.assert_array_equal(
        float(_loss_and_grads(mesh, off, tokens)[0]), float(_loss_and_grads(mesh, base, tokens)[0])
    )


def test_kv_projections_read_the_side_stream_and_every_side_param_trains():
    mesh, model = _model(_W)
    for layer in model.layers():
        attn = layer.attn
        if isinstance(attn, KimiDeltaAttention):
            assert attn.w_k.shape[0] == attn.w_v.shape[0] == _W
            assert attn.w_q.shape[0] == model.config.hidden_dim
        else:
            assert isinstance(attn, CausalSelfAttention)
            assert attn.w_dkv is not None and attn.w_dkv.shape[0] == _W
    tokens = _tokens(2)
    loss, grads = _loss_and_grads(mesh, model, tokens)
    assert np.isfinite(float(loss))
    embed_grad = np.asarray(grads.kv_stream.token_embed)
    # Exactly the rows of tokens in the batch get a gradient.
    np.testing.assert_array_equal(np.abs(embed_grad).max(axis=1) > 0, np.isin(np.arange(t._VOCAB), tokens))
    side = eqx.filter(grads.kv_stream.blocks, eqx.is_inexact_array)
    for path, grad in jax.tree_util.tree_leaves_with_path(side):
        grad = np.asarray(grad)
        assert np.isfinite(grad).all(), path
        # Every layer's slice of each stacked side-block leaf gets a gradient.
        per_layer = np.abs(grad.reshape(grad.shape[0], -1)).max(axis=1) if grad.ndim > 1 else np.abs(grad)
        assert (per_layer > 0).all(), path
    for name in ("w_k", "w_v"):
        assert np.abs(np.asarray(getattr(grads.kda_blocks.stacked.attn, name))).max() > 0, name
    assert np.abs(np.asarray(grads.stacked_blocks.stacked.attn.w_dkv)).max() > 0


def test_side_stream_optimizer_routing():
    _, model = _model(_W)
    mask = GrugMoeMuonHConfig().create_mask(eqx.filter(model, eqx.is_inexact_array))
    side = mask.kv_stream
    assert side.token_embed == mask.token_embed
    blocks = side.blocks.stacked
    for name in ("w_q", "w_k", "w_v", "w_o", "w_up", "w_down"):
        assert getattr(blocks, name) == "muonh", name
    for norm in (blocks.rms_attn, blocks.rms_mlp, blocks.kv_norm):
        assert set(jax.tree.leaves(norm)) == {"adam"}
    assert mask.kda_blocks.stacked.attn.w_k == mask.kda_blocks.stacked.attn.w_v == "muonh"
    assert mask.stacked_blocks.stacked.attn.w_dkv == "muonh"


@pytest.mark.parametrize("kv_stream_dim", [0, _W])
def test_causal_through_both_streams(kv_stream_dim):
    mesh, model = _model(kv_stream_dim)
    tokens = _tokens(1)
    perturbed = tokens.at[:, -1].set((tokens[:, -1] + 1) % t._VOCAB)
    with jax.set_mesh(mesh):
        forward = eqx.filter_jit(lambda m, x: m(x)[0])
        hidden, hidden_perturbed = forward(model, tokens), forward(model, perturbed)
    np.testing.assert_allclose(np.asarray(hidden[:, :-1]), np.asarray(hidden_perturbed[:, :-1]), rtol=1e-5, atol=1e-5)
    assert not np.allclose(np.asarray(hidden[:, -1]), np.asarray(hidden_perturbed[:, -1]))


def test_documents_isolated_through_the_side_stream():
    """Perturb a doc-1 token: doc 2 is unchanged. In the second model the perturbed token's main-stream
    embedding row equals the original's, so the perturbation reaches the outputs only via the side stream."""
    mesh, model = _model(_W)
    tokens = _tokens(3).at[:, 3].set(0)
    perturbed = tokens.at[:, 3].set(1)
    segment_ids = jnp.asarray(np.repeat([0, 1], [11, 13])[None].repeat(2, 0), jnp.int32)
    mask = AttentionMask(is_causal=True, segment_ids=(segment_ids, segment_ids))
    side_only = eqx.tree_at(lambda m: m.token_embed, model, model.token_embed.at[1].set(model.token_embed[0]))
    with jax.set_mesh(mesh):
        forward = eqx.filter_jit(lambda m, x: m(x, mask=mask)[0])
        for m in (model, side_only):
            hidden, hidden_perturbed = forward(m, tokens), forward(m, perturbed)
            np.testing.assert_allclose(
                np.asarray(hidden[:, 11:]), np.asarray(hidden_perturbed[:, 11:]), rtol=1e-5, atol=1e-5
            )
            # Positions after the perturbed one (in doc 1) see it, through the K/V of the side stream.
            assert not np.allclose(np.asarray(hidden[:, 4:11]), np.asarray(hidden_perturbed[:, 4:11]))
