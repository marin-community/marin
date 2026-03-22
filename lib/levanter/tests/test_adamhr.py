# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for AdamHR, MuonHR optimizers and constrained geometry helpers."""

import jax
import jax.numpy as jnp
import pytest

import haliax as hax

from levanter.optim.constrained_geometry import parallel_transport, project_tangent, retract


class TestConstrainedGeometry:
    def test_project_tangent_removes_radial_component_2d(self):
        """After projection, the result should be orthogonal to p."""
        p = jnp.array([[1.0, 0.0], [0.0, 2.0]])
        g = jnp.array([[1.0, 1.0], [1.0, 1.0]])
        g_tan = project_tangent(g, p)
        dot = jnp.sum(g_tan * p)
        assert jnp.abs(dot) < 1e-6

    def test_project_tangent_removes_radial_component_3d(self):
        """Rowwise projection for >2D: each row's projection is orthogonal."""
        p = jnp.array([[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]])
        g = jnp.ones_like(p)
        g_tan = project_tangent(g, p)
        for i in range(p.shape[0]):
            dot = jnp.sum(g_tan[i] * p[i])
            assert jnp.abs(dot) < 1e-5

    def test_retract_preserves_norm_2d(self):
        """Retract should preserve the original norm."""
        p = jnp.array([[3.0, 4.0], [1.0, 0.0]])
        orig_norm = jnp.linalg.norm(p)
        tangent = jnp.array([[0.1, -0.1], [0.05, 0.05]])
        delta = retract(p, tangent, learning_rate=0.1)
        p_new = p + delta
        new_norm = jnp.linalg.norm(p_new)
        assert jnp.abs(new_norm - orig_norm) < 1e-5

    def test_retract_preserves_norm_3d_rowwise(self):
        """Retract should preserve rowwise norms for >2D."""
        key = jax.random.PRNGKey(42)
        p = jax.random.normal(key, (3, 4, 5))
        tangent = jax.random.normal(jax.random.PRNGKey(1), (3, 4, 5)) * 0.01
        orig_norms = jnp.sqrt(jnp.sum(jnp.square(p), axis=(1, 2)))
        delta = retract(p, tangent, learning_rate=0.1)
        p_new = p + delta
        new_norms = jnp.sqrt(jnp.sum(jnp.square(p_new), axis=(1, 2)))
        assert jnp.allclose(orig_norms, new_norms, atol=1e-5)

    def test_parallel_transport_stays_tangent(self):
        """Transported vector should be tangent to the new point."""
        p_old = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        p_new = jnp.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]])
        p_new = p_new / jnp.linalg.norm(p_new) * jnp.linalg.norm(p_old)
        v = jnp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        v_transported = parallel_transport(v, p_old, p_new)
        dot = jnp.sum(v_transported * p_new)
        assert jnp.abs(dot) < 1e-5


class TestAdamHRConfig:
    def test_mask_routes_embeddings_to_adamhr(self):
        """AdamHR should route embeddings to the constrained path."""
        from levanter.optim.adamhr import AdamHRConfig

        In = hax.Axis("in", 4)
        Out = hax.Axis("out", 3)

        class _FakeModel:
            embeddings: hax.nn.Embedding
            linear: hax.nn.Linear

        model = _FakeModel()
        model.embeddings = hax.nn.Embedding.init(hax.Axis("vocab", 10), In, key=jax.random.PRNGKey(0))
        model.linear = hax.nn.Linear.init(In=In, Out=Out, key=jax.random.PRNGKey(1))

        # AdamHR mask routing is path-based; we need a proper tree
        import equinox as eqx

        class FakeModel(eqx.Module):
            Embedding: hax.nn.Embedding
            linear: hax.nn.Linear

        model = FakeModel(
            Embedding=hax.nn.Embedding.init(hax.Axis("vocab", 10), In, key=jax.random.PRNGKey(0)),
            linear=hax.nn.Linear.init(In=In, Out=Out, key=jax.random.PRNGKey(1)),
        )
        config = AdamHRConfig()
        mask = config.create_mask(model)
        # Embedding weight goes to adamhr
        assert mask.Embedding.weight == "adamhr"
        # Linear weight goes to adamhr
        assert mask.linear.weight == "adamhr"

    def test_adamhr_preserves_param_norm_after_step(self):
        """A single AdamHR step should preserve parameter norms."""
        from levanter.optim.adamhr import scale_by_adamhr

        key = jax.random.PRNGKey(0)
        params = {"w": jax.random.normal(key, (4, 3))}
        grads = {"w": jax.random.normal(jax.random.PRNGKey(1), (4, 3))}
        orig_norm = jnp.linalg.norm(params["w"])

        tx = scale_by_adamhr(b1=0.9, b2=0.999, eps=1e-8, learning_rate=0.01)
        state = tx.init(params)
        updates, _ = tx.update(grads, state, params)
        new_params = {"w": params["w"] + updates["w"]}
        new_norm = jnp.linalg.norm(new_params["w"])
        assert jnp.abs(new_norm - orig_norm) < 1e-4


class TestMuonHRConfig:
    def test_mask_routes_correctly(self):
        """MuonHR should route linear to muonhr, embeddings to adamhr."""
        from levanter.optim.muonhr import MuonHRConfig

        import equinox as eqx

        In = hax.Axis("in", 4)
        Out = hax.Axis("out", 3)

        class FakeModel(eqx.Module):
            Embedding: hax.nn.Embedding
            linear: hax.nn.Linear

        model = FakeModel(
            Embedding=hax.nn.Embedding.init(hax.Axis("vocab", 10), In, key=jax.random.PRNGKey(0)),
            linear=hax.nn.Linear.init(In=In, Out=Out, key=jax.random.PRNGKey(1)),
        )
        config = MuonHRConfig()
        mask = config.create_mask(model)
        assert mask.Embedding.weight == "adamhr"
        assert mask.linear.weight == "muonhr"


class TestLlamaEmbeddingGain:
    def test_embedding_gain_applied(self):
        """When use_embedding_gain=True, LlamaEmbedding should have a gain parameter."""
        import dataclasses

        from levanter.models.llama import LlamaConfig, LlamaEmbedding

        config = dataclasses.replace(
            LlamaConfig(hidden_dim=16, num_layers=1, num_heads=2, num_kv_heads=2),
            use_embedding_gain=True,
        )
        Vocab = hax.Axis("vocab", 32)
        emb = LlamaEmbedding.init(Vocab, config, key=jax.random.PRNGKey(0))
        assert emb.embedding_gain is not None
        # gain should be all ones at init
        assert jnp.allclose(emb.embedding_gain.array, 1.0)

    def test_embedding_no_gain_by_default(self):
        """By default, embedding_gain should be None."""
        from levanter.models.llama import LlamaConfig, LlamaEmbedding

        config = LlamaConfig(hidden_dim=16, num_layers=1, num_heads=2, num_kv_heads=2)
        Vocab = hax.Axis("vocab", 32)
        emb = LlamaEmbedding.init(Vocab, config, key=jax.random.PRNGKey(0))
        assert emb.embedding_gain is None
