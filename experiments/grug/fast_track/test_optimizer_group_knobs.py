# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-group optimizer knobs: routed-expert momentum / consistency scaling / cautious masking / Adam, and Sinkhorn
momentum on the bigram table."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

import experiments.grug.fast_track.test_ngram_stat as t
from experiments.grug.fast_track.optimizer import (
    GrugMoeMuonHConfig,
    cautious_matrix_deltas,
    expert_consistency_metrics,
    retract_to_param_sphere,
    scale_by_expert_consistency,
)

_BIGRAM = dict(ngram_stat_rows=0, second_embed=True, second_embed_bigram=True, embed2_rows=64)


def _two_steps(config: GrugMoeMuonHConfig, params):
    """The second step's updates after two independent random gradients (momentum sets how much of step one stays)."""
    opt = config.build(10)

    @eqx.filter_jit
    def run(params):
        # Replicated leaves: the tiny test model's layouts are ambiguous for the stacked Newton-Schulz contraction.
        params = jax.tree.map(lambda p: reshard(p, P(*(None,) * p.ndim)), params)
        leaves, treedef = jax.tree.flatten(params)
        keys = jax.random.split(jax.random.PRNGKey(0), 2 * len(leaves))
        first = jax.tree.unflatten(
            treedef, [0.01 * jax.random.normal(k, p.shape) for k, p in zip(keys[::2], leaves, strict=True)]
        )
        second = jax.tree.unflatten(
            treedef, [0.01 * jax.random.normal(k, p.shape) for k, p in zip(keys[1::2], leaves, strict=True)]
        )
        state = opt.init(params)
        _, state = opt.update(first, state, params)
        updates, _ = opt.update(second, state, params)
        return updates

    return run(params)


def test_routed_momentum_changes_only_the_routed_experts():
    mesh, model = t._model(**_BIGRAM)
    params = eqx.filter(model, eqx.is_inexact_array)
    mask = GrugMoeMuonHConfig(muonh_routed_momentum=0.5).create_mask(params)
    mlp_mask = mask.kda_blocks.stacked.mlp
    assert mlp_mask.expert_mlp.w_up == "muonh_routed"
    assert mask.kda_blocks.stacked.attn.w_q == "muonh"
    with jax.set_mesh(mesh):
        base = _two_steps(GrugMoeMuonHConfig(), params)
        low = _two_steps(GrugMoeMuonHConfig(muonh_routed_momentum=0.0), params)
    np.testing.assert_allclose(
        np.asarray(low.kda_blocks.stacked.attn.w_q), np.asarray(base.kda_blocks.stacked.attn.w_q), rtol=1e-6
    )
    assert not np.allclose(
        np.asarray(low.kda_blocks.stacked.mlp.expert_mlp.w_up), np.asarray(base.kda_blocks.stacked.mlp.expert_mlp.w_up)
    )


def test_bigram_table_can_train_with_sinkhorn_momentum():
    mesh, model = t._model(**_BIGRAM)
    params = eqx.filter(model, eqx.is_inexact_array)
    config = GrugMoeMuonHConfig(embed2_update="sinkhorn")
    assert config.create_mask(params).token_embed2 == "embed2"
    with jax.set_mesh(mesh):
        sinkhorn = _two_steps(config, params)
        adam = _two_steps(GrugMoeMuonHConfig(), params)
    table = np.asarray(sinkhorn.token_embed2)
    assert np.isfinite(table).all() and np.abs(table).sum() > 0
    assert not np.allclose(table, np.asarray(adam.token_embed2))
    # Sinkhorn keeps one momentum buffer for the table; AdEMAMix keeps three (m, v and the slow m).
    with jax.set_mesh(mesh):
        floats = lambda c: sum(x.size for x in jax.tree.leaves(c.build(10).init(params)))  # noqa: E731
        assert floats(GrugMoeMuonHConfig(adam_ademamix_alpha=5.0)) - floats(config) >= model.token_embed2.size


def test_embed2_update_is_validated():
    with pytest.raises(ValueError):
        GrugMoeMuonHConfig(embed2_update="lion")
    with pytest.raises(ValueError):
        GrugMoeMuonHConfig(embed2_update="sinkhorn", embed2_row_sparse_adam=True)


def test_qk_group_takes_its_own_lr_and_leaves_other_matrices_alone():
    mesh, model = t._model(**_BIGRAM)
    params = eqx.filter(model, eqx.is_inexact_array)
    mask = GrugMoeMuonHConfig(muonh_qk_lr_mult=2.0).create_mask(params)
    attn = mask.kda_blocks.stacked.attn
    assert attn.w_q == "muonh_qk" and attn.w_k == "muonh_qk"
    assert attn.w_v == "muonh" and attn.w_o == "muonh"
    with jax.set_mesh(mesh):
        base = _two_steps(GrugMoeMuonHConfig(), params)
        bold = _two_steps(GrugMoeMuonHConfig(muonh_qk_lr_mult=2.0), params)
    np.testing.assert_allclose(
        np.asarray(bold.kda_blocks.stacked.attn.w_v), np.asarray(base.kda_blocks.stacked.attn.w_v), rtol=1e-6
    )
    ratio = np.linalg.norm(np.asarray(bold.kda_blocks.stacked.attn.w_q)) / np.linalg.norm(
        np.asarray(base.kda_blocks.stacked.attn.w_q)
    )
    assert 1.5 < ratio < 2.5


def test_expert_consistency_keeps_a_repeating_expert_and_mutes_a_noisy_one():
    opt = scale_by_expert_consistency(optax.identity(), momentum=0.9, beta2=0.99)
    fixed = jax.random.normal(jax.random.PRNGKey(0), (6, 5))
    state = opt.init(jnp.zeros((2, 6, 5)))
    for step in range(60):
        noise = jax.random.normal(jax.random.PRNGKey(step + 1), (6, 5))
        grad = jnp.stack([fixed, noise])
        update, state = opt.update(grad, state)
    np.testing.assert_allclose(np.asarray(update[0]), np.asarray(fixed), rtol=1e-4)
    assert np.linalg.norm(update[1]) < 0.3 * np.linalg.norm(noise)


def test_cautious_deltas_drop_uphill_coordinates_per_matrix():
    grad = jnp.array([[[1.0, -1.0], [1.0, -1.0]], [[1.0, 1.0], [1.0, 1.0]]])
    # The inner transform's delta descends on expert 0 but moves uphill on half of expert 1.
    delta = jnp.array([[[-1.0, 1.0], [-1.0, 1.0]], [[-1.0, 1.0], [-1.0, 1.0]]])
    opt = cautious_matrix_deltas(
        optax.GradientTransformation(lambda p: optax.EmptyState(), lambda u, s, p=None: (delta, s))
    )
    out, _ = opt.update(grad, opt.init(grad))
    np.testing.assert_allclose(np.asarray(out[0]), np.asarray(delta[0]))
    np.testing.assert_allclose(np.asarray(out[1]), np.array([[-2.0, 0.0], [-2.0, 0.0]]))


def test_routed_experts_can_train_with_adam():
    _, model = t._model(**_BIGRAM)
    params = eqx.filter(model, eqx.is_inexact_array)
    mask = GrugMoeMuonHConfig(routed_expert_optimizer="adam").create_mask(params)
    assert mask.kda_blocks.stacked.mlp.expert_mlp.w_up == "adam"
    assert mask.kda_blocks.stacked.attn.w_q == "muonh"
    with pytest.raises(ValueError):
        GrugMoeMuonHConfig(routed_expert_optimizer="sgd")


def test_routed_consistency_and_cautious_change_only_the_routed_experts():
    mesh, model = t._model(**_BIGRAM)
    params = eqx.filter(model, eqx.is_inexact_array)
    with jax.set_mesh(mesh):
        base = _two_steps(GrugMoeMuonHConfig(), params)
        for config in (
            GrugMoeMuonHConfig(muonh_routed_consistency_beta2=0.99),
            GrugMoeMuonHConfig(muonh_routed_cautious=True),
        ):
            assert config.create_mask(params).kda_blocks.stacked.mlp.expert_mlp.w_up == "muonh_routed"
            out = _two_steps(config, params)
            np.testing.assert_allclose(
                np.asarray(out.kda_blocks.stacked.attn.w_q), np.asarray(base.kda_blocks.stacked.attn.w_q), rtol=1e-6
            )
            routed, routed_base = (np.asarray(u.kda_blocks.stacked.mlp.expert_mlp.w_up) for u in (out, base))
            assert not np.allclose(routed, routed_base)


def _fixed_delta(delta):
    return optax.GradientTransformation(lambda p: optax.EmptyState(), lambda u, s, p=None: (delta, s))


@pytest.mark.parametrize("per_expert", [True, False])
def test_retraction_puts_scaled_updates_back_on_the_sphere(per_expert):
    params = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 6, 5))
    delta = 0.3 * jax.random.normal(jax.random.PRNGKey(1), params.shape)
    out, _ = retract_to_param_sphere(_fixed_delta(delta), per_expert).update(params, optax.EmptyState(), params)
    axes = (2, 3) if per_expert else (1, 2, 3)
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(params + out).reshape(*params.shape[: axes[0]], -1), axis=-1),
        np.linalg.norm(np.asarray(params).reshape(*params.shape[: axes[0]], -1), axis=-1),
        rtol=1e-5,
    )


def test_consistency_metrics_report_the_per_expert_multipliers():
    opt = scale_by_expert_consistency(optax.identity(), momentum=0.9, beta2=0.99)

    @jax.jit
    def run(ones, noise_keys):
        # Expert-sharded like the model's [L, E, in, out] stacks, so the metrics must flatten a sharded leaf.
        grads = reshard(jnp.stack([ones, ones]), P(None, "expert", None, None))
        state = opt.init(grads)
        for key in noise_keys:
            noisy = jnp.stack([ones, jax.random.normal(key, ones.shape)])
            _, state = opt.update(reshard(noisy, P(None, "expert", None, None)), state)
        return expert_consistency_metrics(state)

    with jax.set_mesh(t._mesh()):
        metrics = run(jnp.ones((2, 6, 5)), list(jax.random.split(jax.random.PRNGKey(0), 40)))
    assert float(metrics["train/optim/expert_consistency_p90"]) > 0.9
    assert float(metrics["train/optim/expert_consistency_p10"]) < 0.3
