# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the per-sequence-routed FFN mixture."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from experiments.datakit.cluster.quality.fast_transformer.model import FastTransformer, FastTransformerConfig

BASE_KW = dict(
    vocab_size=128,
    max_tokens=64,
    pool_window=16,
    pool_kind="meanmaxmin",
    embed_dim=16,
    hidden_dim=32,
    num_layers=2,
    num_heads=4,
    dropout=0.0,
    doc_embed_dim=24,
    doc_embed_super_token=True,
)


def _batch(n=5, seed=0):
    rng = np.random.default_rng(seed)
    ids = jnp.asarray(rng.integers(1, BASE_KW["vocab_size"], size=(n, BASE_KW["max_tokens"])).astype(np.int32))
    emb = jnp.asarray(rng.normal(size=(n, BASE_KW["doc_embed_dim"])).astype(np.float32))
    return ids, emb


def _open_gates(model):
    return eqx.tree_at(lambda m: [layer.moe_gate for layer in m.layers], model, [jnp.asarray(1.0)] * len(model.layers))


def test_moe_model_starts_as_the_dense_model():
    """The fusion-arm precedent: zero-init gates make the MoE forward exactly
    the dense arm's at init — same seed, same trunk weights, same outputs — so
    early training matches the dense arm."""
    dense = FastTransformer(FastTransformerConfig(**BASE_KW), key=jr.PRNGKey(0))
    moe = FastTransformer(FastTransformerConfig(**BASE_KW, moe_experts=4), key=jr.PRNGKey(0))
    ids, emb = _batch()
    np.testing.assert_array_equal(np.asarray(dense(ids, doc_embed=emb)), np.asarray(moe(ids, doc_embed=emb)))


def test_routing_is_top_k_sparse_per_document():
    """Off-top-k experts must not touch the output at all: deployment gathers
    only the top-k, so training's compute-all-and-mask must be identical."""
    model = _open_gates(FastTransformer(FastTransformerConfig(**BASE_KW, moe_experts=4, moe_top_k=2), key=jr.PRNGKey(0)))
    ids, emb = _batch(n=3)
    mix = model.layers[0].expert_mixture(emb)
    assert mix.shape == (3, 4)
    np.testing.assert_allclose(np.asarray(mix.sum(axis=1)), 1.0, rtol=1e-6)
    assert (np.asarray((mix > 0).sum(axis=1)) == 2).all()

    baseline = np.asarray(model(ids, doc_embed=emb))
    # Perturb one expert that no document routed to; the forward must not move.
    unused = int(np.asarray(mix).sum(axis=0).argmin())
    assert np.asarray(mix)[:, unused].max() == 0.0
    perturbed = eqx.tree_at(
        lambda m: m.layers[0].expert_w1,
        model,
        model.layers[0].expert_w1.at[unused].set(model.layers[0].expert_w1[unused] + 100.0),
    )
    np.testing.assert_array_equal(np.asarray(perturbed(ids, doc_embed=emb)), baseline)


def test_open_gate_routes_by_document_embedding():
    """One routing decision per document: with gates open, documents that share
    token ids but route differently must score differently."""
    model = _open_gates(FastTransformer(FastTransformerConfig(**BASE_KW, moe_experts=4), key=jr.PRNGKey(0)))
    ids, emb = _batch(n=2)
    same_ids = jnp.stack([ids[0], ids[0]])
    mix = model.layers[0].expert_mixture(emb)
    assert not np.allclose(np.asarray(mix[0]), np.asarray(mix[1])), "distinct embeddings should route differently"
    out = np.asarray(model(same_ids, doc_embed=emb))
    assert out[0] != out[1]


def test_moe_checkpoint_round_trips_through_the_scorer_format(tmp_path):
    """The serialized config must rebuild the exact MoE architecture — a field
    silently falling back to its dataclass default would deserialize a
    shape-mismatched or dense model."""
    from experiments.datakit.cluster.quality.fast_transformer.scorer import PooledScorer, artifact_names
    from experiments.datakit.cluster.quality.fast_transformer.train import _save_scorer

    config = FastTransformerConfig(**BASE_KW, moe_experts=4, moe_top_k=2, moe_expert_ratio=1)
    model = _open_gates(FastTransformer(config, key=jr.PRNGKey(0)))
    remap = {t: t + 2 for t in range(BASE_KW["vocab_size"] - 2)}
    _save_scorer(model, remap, "unused-tokenizer", config, str(tmp_path), "moe_test")
    eqx_name, remap_name, meta_name = artifact_names("moe_test")
    scorer = PooledScorer.load(str(tmp_path / eqx_name), str(tmp_path / remap_name), str(tmp_path / meta_name))
    assert scorer.model.config == config
    ids, emb = _batch(n=4)
    np.testing.assert_array_equal(
        np.asarray(scorer.model(ids, doc_embed=emb)), np.asarray(model(ids, doc_embed=emb))
    )
