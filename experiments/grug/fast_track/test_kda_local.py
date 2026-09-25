# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track with KDA local layers: only the local layers become KDA, the model stays causal, documents
stay independent, every KDA parameter trains, and per-layer router state follows layer order across the
two layer stacks."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh
from levanter.grug.attention import AttentionMask

from experiments.grug.fast_track.model import (
    CausalSelfAttention,
    GrugModelConfig,
    KimiDeltaAttention,
    LocalMixer,
    Transformer,
)
from experiments.grug.fast_track.train import _apply_qb_betas

_SEQ = 24
_VOCAB = 32


def _config(**overrides) -> GrugModelConfig:
    kwargs = dict(
        vocab_size=_VOCAB,
        hidden_dim=32,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=2,
        latent_dim=16,
        num_layers=6,
        num_heads=2,
        num_kv_heads=1,
        local_kv_heads=1,
        global_kv_heads=1,
        head_dim=16,
        max_seq_len=_SEQ,
        sliding_window=_SEQ,
        global_every=4,
        local_mixer=LocalMixer.KDA,
        attn_res=True,
        attn_res_num_blocks=3,
    )
    kwargs.update(overrides)
    return GrugModelConfig(**kwargs)


def _mesh() -> Mesh:
    return Mesh(
        np.array(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )


def _model(**overrides) -> tuple[Mesh, Transformer]:
    mesh = _mesh()
    with jax.set_mesh(mesh):
        model = Transformer.init(_config(**overrides), key=jax.random.PRNGKey(0))
    return mesh, model


def test_kda_replaces_only_the_local_layers():
    """6 layers with global_every=4: layers 3 and 5 stay softmax, 0/1/2/4 become KDA; KDA blocks have no
    branch-output SConv."""
    _, model = _model()
    layers = model.layers()
    kda, softmax = KimiDeltaAttention, CausalSelfAttention
    assert [type(layer.attn) for layer in layers] == [kda, kda, kda, softmax, kda, softmax]
    assert [layer.sconv_attn is None for layer in layers] == [True, True, True, False, True, False]


def test_hybrid_model_is_causal():
    mesh, model = _model()
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, _SEQ), 0, _VOCAB)
    perturbed = tokens.at[:, -1].set((tokens[:, -1] + 1) % _VOCAB)
    with jax.set_mesh(mesh):
        forward = eqx.filter_jit(lambda m, t: m(t)[0])
        hidden = forward(model, tokens)
        hidden_perturbed = forward(model, perturbed)
    np.testing.assert_allclose(np.asarray(hidden[:, :-1]), np.asarray(hidden_perturbed[:, :-1]), rtol=1e-5, atol=1e-5)
    assert not np.allclose(np.asarray(hidden[:, -1]), np.asarray(hidden_perturbed[:, -1]))


def test_packed_documents_do_not_see_each_other():
    mesh, model = _model()
    tokens = jax.random.randint(jax.random.PRNGKey(3), (2, _SEQ), 0, _VOCAB)
    segment_ids = jnp.asarray(np.repeat([0, 1], [11, 13])[None].repeat(2, 0), jnp.int32)
    mask = AttentionMask(is_causal=True, segment_ids=(segment_ids, segment_ids))
    perturbed = tokens.at[:, 3].set((tokens[:, 3] + 1) % _VOCAB)
    with jax.set_mesh(mesh):
        forward = eqx.filter_jit(lambda m, t: m(t, mask=mask)[0])
        hidden = forward(model, tokens)
        hidden_perturbed = forward(model, perturbed)
    np.testing.assert_allclose(np.asarray(hidden[:, 11:]), np.asarray(hidden_perturbed[:, 11:]), rtol=1e-5, atol=1e-5)
    assert not np.allclose(np.asarray(hidden[:, 3:11]), np.asarray(hidden_perturbed[:, 3:11]))


def test_every_kda_parameter_gets_gradient():
    mesh, model = _model(kda_dt_range=(0.001, 0.1))
    tokens = jax.random.randint(jax.random.PRNGKey(2), (2, _SEQ), 0, _VOCAB)
    weights = jnp.ones(tokens.shape, jnp.float32)
    with jax.set_mesh(mesh):
        loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(lambda m: m.next_token_loss(tokens, weights)))(model)
    assert np.isfinite(float(loss))
    kda_grads = grads.kda_blocks.stacked.attn
    for name in ("w_q", "w_k", "w_v", "w_o", "w_g", "w_a_down", "w_a_up", "a_log", "dt_bias", "w_beta"):
        grad = np.asarray(getattr(kda_grads, name))
        assert np.isfinite(grad).all(), name
        assert np.abs(grad).max() > 0, name


def test_router_stats_and_qb_biases_follow_layer_order():
    """Router stats come back in layer order, and QB biases computed per layer land on that layer's
    block in whichever stack holds it."""
    mesh, model = _model()
    tokens = jax.random.randint(jax.random.PRNGKey(4), (2, _SEQ), 0, _VOCAB)
    betas = jnp.arange(6 * 4, dtype=jnp.float32).reshape(6, 4) ** 1.5
    with jax.set_mesh(mesh):
        _, metrics = eqx.filter_jit(lambda m: m(tokens))(model)
        updated = _apply_qb_betas(model, betas)
    assert metrics["qb_beta_per_layer"].shape == (6, 4)
    expected = np.asarray(-betas - jnp.mean(-betas, axis=-1, keepdims=True))
    biases = np.stack([np.asarray(layer.mlp.router_bias) for layer in updated.layers()])
    np.testing.assert_allclose(biases, expected)
