# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Truncating the weakest singular directions of the MuonH update, per matrix type."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

import experiments.grug.fast_track.test_ngram_stat as t
import experiments.grug.fast_track.test_optimizer_group_knobs as knobs
from experiments.grug.fast_track.grugmuon_stacked import _truncate_bottom_directions
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig


def _svd_pair(shape, seed):
    """A matrix with distinct singular values and a same-basis "Newton-Schulz output" g(Σ)."""
    rng = np.random.default_rng(seed)
    m, n = shape
    u, _ = np.linalg.qr(rng.standard_normal((m, min(m, n))))
    v, _ = np.linalg.qr(rng.standard_normal((n, min(m, n))))
    sigma = np.geomspace(1.0, 1e-3, min(m, n))
    out_sigma = np.linspace(1.1, 0.2, min(m, n))
    return (u * sigma) @ v.T, u, out_sigma, v


@pytest.mark.parametrize("shape", [(12, 8), (8, 12)])
def test_truncation_drops_exactly_the_weakest_directions(shape):
    m, u, out_sigma, v = _svd_pair(shape, 0)
    direction = (u * out_sigma) @ v.T
    got = np.asarray(_truncate_bottom_directions(jnp.asarray(direction), jnp.asarray(m), 0.25))
    keep = min(shape) - 2
    expected = (u[:, :keep] * out_sigma[:keep]) @ v[:, :keep].T
    np.testing.assert_allclose(got, expected, atol=1e-4)


def test_truncation_runs_per_expert_on_a_sharded_stack():
    pairs = [_svd_pair((8, 6), seed) for seed in range(4)]
    m = np.stack([p[0] for p in pairs]).reshape(1, 4, 8, 6)
    direction = np.stack([(u * s) @ v.T for _, u, s, v in pairs]).reshape(1, 4, 8, 6)
    expected = np.stack([(u[:, :3] * s[:3]) @ v[:, :3].T for _, u, s, v in pairs]).reshape(1, 4, 8, 6)

    @jax.jit
    def run(d, m):
        spec = P(None, "expert", None, None)
        return _truncate_bottom_directions(reshard(d, spec), reshard(m, spec), 0.5)

    with jax.set_mesh(t._mesh()):
        got = np.asarray(run(jnp.asarray(direction, jnp.float32), jnp.asarray(m, jnp.float32)))
    np.testing.assert_allclose(got, expected, atol=1e-4)


def test_truncation_group_takes_only_its_matrix_type():
    _, model = t._model(second_embed=True, second_embed_bigram=True, embed2_rows=64, ngram_stat_rows=0)
    params = eqx.filter(model, eqx.is_inexact_array)
    routed = GrugMoeMuonHConfig(muon_truncate_family="routed", muon_truncate_frac=0.2).create_mask(params)
    assert routed.kda_blocks.stacked.mlp.expert_mlp.w_up == "muonh_trunc"
    assert routed.kda_blocks.stacked.attn.w_q == "muonh"
    kda = GrugMoeMuonHConfig(muon_truncate_family="kda", muon_truncate_frac=0.2).create_mask(params)
    assert kda.kda_blocks.stacked.attn.w_q == "muonh_trunc"
    assert kda.kda_blocks.stacked.mlp.expert_mlp.w_up == "muonh"


def test_truncation_settings_are_validated():
    with pytest.raises(ValueError):
        GrugMoeMuonHConfig(muon_truncate_family="routed")
    with pytest.raises(ValueError):
        GrugMoeMuonHConfig(muon_truncate_frac=0.2)
    with pytest.raises(ValueError):
        GrugMoeMuonHConfig(muon_truncate_family="lm_head", muon_truncate_frac=0.2)


@pytest.mark.parametrize("family", ["routed", "kda", "latent"])
def test_truncation_changes_only_its_family_in_the_full_optimizer(family):
    mesh, model = t._model(**knobs._BIGRAM)
    params = eqx.filter(model, eqx.is_inexact_array)
    with jax.set_mesh(mesh):
        base = knobs._two_steps(GrugMoeMuonHConfig(), params)
        out = knobs._two_steps(GrugMoeMuonHConfig(muon_truncate_family=family, muon_truncate_frac=0.5), params)
    leaves = {
        "routed": lambda u: u.kda_blocks.stacked.mlp.expert_mlp.w_up,
        "kda": lambda u: u.kda_blocks.stacked.attn.w_q,
        "latent": lambda u: u.kda_blocks.stacked.mlp.w_latent_up,
    }
    for name, leaf in leaves.items():
        same = np.allclose(np.asarray(leaf(out)), np.asarray(leaf(base)), rtol=1e-5, atol=1e-7)
        assert same == (name != family), name
