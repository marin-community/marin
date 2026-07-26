# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh
from jax.sharding import PartitionSpec as P
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe.model import CausalSelfAttention, DenseMLP, GrugModelConfig, MoEMLP
from experiments.grug.moe.train import _compute_flops


def test_nonexpert_weights_are_sharded_over_data_and_expert_axes():
    mesh = Mesh(
        np.asarray([jax.devices()[0]]).reshape((1, 1, 1)),
        ("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )

    with jax.set_mesh(mesh):
        dense = DenseMLP.init(16, 32, 0.02, key=jax.random.key(0))
        config = GrugModelConfig(
            vocab_size=128,
            hidden_dim=16,
            intermediate_dim=8,
            shared_expert_intermediate_dim=16,
            num_experts=4,
            num_experts_per_token=2,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
        )
        attention = CausalSelfAttention.init(config, key=jax.random.key(1))

    column_sharding = P(("data", "expert"), "model")
    row_sharding = P("model", ("data", "expert"))
    assert dense.w_gate.sharding.spec == column_sharding
    assert dense.w_up.sharding.spec == column_sharding
    assert dense.w_down.sharding.spec == row_sharding
    assert attention.w_q.sharding.spec == column_sharding
    assert attention.w_k.sharding.spec == column_sharding
    assert attention.w_v.sharding.spec == column_sharding
    assert attention.w_o.sharding.spec == row_sharding


def _latent_config(**overrides) -> GrugModelConfig:
    base = dict(
        vocab_size=128,
        hidden_dim=16,
        intermediate_dim=8,
        shared_expert_intermediate_dim=16,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=8,
    )
    base.update(overrides)
    return GrugModelConfig(**base)


def test_latent_moe_narrows_the_dispatched_expert_width():
    """The whole mechanism: with moe_latent_dim set, the expert MLP -- and therefore the tensor
    that crosses the expert-parallel all-to-all inside it -- is latent-wide, while the MoE block
    still consumes and produces hidden-wide activations."""
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    cfg = _latent_config(moe_latent_dim=8)

    with jax.set_mesh(mesh):
        latent_mlp = MoEMLP.init(cfg, key=jax.random.key(0))
        dense_mlp = MoEMLP.init(_latent_config(), key=jax.random.key(0))
        x = jnp.zeros((1, 8, cfg.hidden_dim), dtype=jnp.float32)
        x = jax.sharding.reshard(x, P(("replica_dcn", "data", "expert"), None, None))
        routed, _, _ = latent_mlp(x)

    # Expert weights contract over the latent width, not hidden.
    assert latent_mlp.expert_mlp.w_gate.shape == (cfg.num_experts, 8, cfg.intermediate_dim)
    assert dense_mlp.expert_mlp.w_gate.shape == (cfg.num_experts, cfg.hidden_dim, cfg.intermediate_dim)
    assert latent_mlp.moe_down.shape == (cfg.hidden_dim, 8)
    assert latent_mlp.moe_up.shape == (8, cfg.hidden_dim)
    # The router stays at hidden width: only the expert input is projected.
    assert latent_mlp.router.shape == dense_mlp.router.shape
    assert routed.shape == x.shape


def test_latent_moe_norm_is_opt_in():
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    with jax.set_mesh(mesh):
        assert MoEMLP.init(_latent_config(moe_latent_dim=8), key=jax.random.key(0)).moe_latent_rms is None
        normed = MoEMLP.init(_latent_config(moe_latent_dim=8, moe_latent_norm=True), key=jax.random.key(0))
        assert normed.moe_latent_rms is not None
        assert normed.moe_latent_rms.weight.shape == (8,)


@pytest.mark.parametrize(
    "overrides",
    [
        {"moe_latent_dim": 0},
        {"moe_latent_dim": 16},  # == hidden_dim, so it saves nothing
        {"moe_latent_dim": 6},  # does not divide hidden_dim
        {"moe_latent_norm": True},  # norm without a latent dim
    ],
)
def test_latent_moe_config_rejects_unusable_widths(overrides):
    with pytest.raises(ValueError):
        _latent_config(**overrides)


def test_latent_moe_lowers_analytic_flops_per_token_by_the_expected_amount():
    """Latent MoE moves the denominator, so MFU is not comparable across the pair without it.

    At the param-preserving hero pair (4-of-128 at i3072 vs latent L3072, 4-of-256 at i3072) the
    routed-expert term halves and two [D, L] projections per layer are added back.
    """
    dense = _latent_config(
        vocab_size=128256,
        hidden_dim=6144,
        intermediate_dim=3072,
        shared_expert_intermediate_dim=6144,
        num_experts=128,
        num_experts_per_token=4,
        num_layers=48,
        num_heads=48,
        num_kv_heads=12,
        max_seq_len=4096,
    )
    latent = dataclasses.replace(dense, num_experts=256, moe_latent_dim=3072)

    _, dense_summary = _compute_flops(model_config=dense)
    _, latent_summary = _compute_flops(model_config=latent)
    dense_fpt = dense_summary["throughput/flops_per_token_analytic"]
    latent_fpt = latent_summary["throughput/flops_per_token_analytic"]

    per_layer_delta = (
        6 * 3072 * 3072 * 4  # latent routed GLU
        + 4 * 6144 * 3072  # the two [D, L] projections
        + 2 * 6144 * 128  # the router doubles with the expert count
    ) - (
        6 * 6144 * 3072 * 4
    )  # dense routed GLU
    assert latent_fpt == pytest.approx(dense_fpt + 48 * per_layer_delta)
    assert latent_fpt < dense_fpt

    # Routed parameters are preserved across the pair: halving the expert width and doubling the
    # expert count leaves the routed parameter budget unchanged.
    assert 128 * 3 * 6144 * 3072 == 256 * 3 * 3072 * 3072
