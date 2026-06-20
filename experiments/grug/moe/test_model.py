# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from jax._src import config as jax_config
from jax.sharding import AbstractMesh, AxisType, use_abstract_mesh
from levanter.grug.attention import AttentionMask, ThdSegmentMetadata
from levanter.grug.grug_moe import GroupedMoEExpertMlp, MoEExpertMlp
from levanter.utils.activation import ActivationFunctionEnum

from experiments.grug.moe.model import GroupedMoEMLP, GrugModelConfig, MoEMLP, _layer_attention_masks


class _reset_abstract_mesh:
    def __enter__(self):
        self._prev = jax_config.abstract_mesh_context_manager.swap_local(jax_config.config_ext.unset)
        return self

    def __exit__(self, exc_type, exc, tb):
        jax_config.abstract_mesh_context_manager.set_local(self._prev)
        return False


def test_layer_attention_masks_use_direct_short_window_and_full_long_mask():
    metadata = ThdSegmentMetadata(
        segment_lengths=jnp.array([[4, 4]], dtype=jnp.int32),
        num_segments=jnp.array([2], dtype=jnp.int32),
    )
    mask = AttentionMask(is_causal=True, thd_segment_metadata=metadata)

    short_mask, long_mask = _layer_attention_masks(mask, sliding_window=8)

    assert short_mask.sliding_window == 8
    assert long_mask.sliding_window is None
    assert short_mask.thd_segment_metadata is metadata
    assert long_mask.thd_segment_metadata is metadata
    assert short_mask.segment_ids is None
    assert long_mask.segment_ids is None


def test_grouped_expert_mlp_from_layers_matches_individual_layers():
    key = jax.random.PRNGKey(0)
    layer_keys = jax.random.split(key, 2)
    layers = tuple(
        MoEExpertMlp(
            w_gate_up=jax.random.normal(layer_key, (4, 6, 10), dtype=jnp.float32),
            w_down=jax.random.normal(layer_key, (4, 5, 6), dtype=jnp.float32),
            implementation="scatter",
            activation=ActivationFunctionEnum.silu,
            capacity_factor=1.0,
            remat_mode="none",
        )
        for layer_key in layer_keys
    )
    grouped = GroupedMoEExpertMlp.from_layers(layers)
    x = jax.random.normal(jax.random.PRNGKey(1), (7, 6), dtype=jnp.float32)
    selected_experts = jnp.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [0, 2],
            [1, 3],
            [2, 0],
        ],
        dtype=jnp.int32,
    )
    combine_weights = jnp.array(
        [
            [0.7, 0.3],
            [0.6, 0.4],
            [0.5, 0.5],
            [0.4, 0.6],
            [0.3, 0.7],
            [0.8, 0.2],
            [0.2, 0.8],
        ],
        dtype=jnp.float32,
    )

    expected = jnp.stack([layer(x, selected_experts, combine_weights) for layer in layers], axis=0)
    actual = grouped(
        jnp.stack([x, x], axis=0),
        jnp.stack([selected_experts, selected_experts], axis=0),
        jnp.stack([combine_weights, combine_weights], axis=0),
    )

    assert grouped.valid_group_size == len(layers)
    assert actual.shape == expected.shape
    assert jnp.max(jnp.abs(actual - expected)) < 1e-4


def test_grouped_expert_mlp_from_layers_rejects_mixed_backends():
    layer = MoEExpertMlp(
        w_gate_up=jnp.zeros((2, 3, 8), dtype=jnp.float32),
        w_down=jnp.zeros((2, 4, 3), dtype=jnp.float32),
        implementation="scatter",
        activation=ActivationFunctionEnum.silu,
        capacity_factor=1.0,
        remat_mode="none",
    )
    mismatched = MoEExpertMlp(
        w_gate_up=jnp.zeros((2, 3, 8), dtype=jnp.float32),
        w_down=jnp.zeros((2, 4, 3), dtype=jnp.float32),
        implementation="ring",
        activation=ActivationFunctionEnum.silu,
        capacity_factor=1.0,
        remat_mode="none",
    )

    with pytest.raises(ValueError, match="same implementation"):
        GroupedMoEExpertMlp.from_layers((layer, mismatched))


def test_grouped_moe_mlp_from_layers_shapes_with_grouped_expert_bank():
    cfg = GrugModelConfig(
        vocab_size=16,
        hidden_dim=6,
        intermediate_dim=5,
        shared_expert_intermediate_dim=0,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=1,
        num_kv_heads=1,
        head_dim=6,
        moe_implementation="scatter",
        remat_mode="none",
    )
    layer_keys = jax.random.split(jax.random.PRNGKey(10), 2)
    layers = tuple(
        MoEMLP(
            router=jax.random.normal(layer_key, (cfg.hidden_dim, cfg.num_experts), dtype=jnp.float32),
            router_bias=jax.random.normal(layer_key, (cfg.num_experts,), dtype=jnp.float32) * 0.01,
            expert_mlp=MoEExpertMlp(
                w_gate_up=jax.random.normal(layer_key, (cfg.num_experts, cfg.hidden_dim, 2 * cfg.intermediate_dim)),
                w_down=jax.random.normal(layer_key, (cfg.num_experts, cfg.intermediate_dim, cfg.hidden_dim)),
                implementation="scatter",
                activation=ActivationFunctionEnum.silu,
                capacity_factor=1.0,
                remat_mode="none",
            ),
            cfg=cfg,
        )
        for layer_key in layer_keys
    )
    grouped = GroupedMoEMLP.from_layers(layers)
    x_group = jax.random.normal(jax.random.PRNGKey(11), (2, 1, 8, cfg.hidden_dim), dtype=jnp.float32)
    mesh = AbstractMesh(
        axis_sizes=(1, 1, 1, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        lowered = jax.jit(lambda y: grouped(y)).trace(x_group).lower(lowering_platforms=(platform,))
        actual_shape = jax.eval_shape(grouped, x_group)

    assert grouped.valid_group_size == len(layers)
    assert lowered is not None
    assert actual_shape[0].shape == x_group.shape
    assert actual_shape[1]["routing_counts"].shape == (len(layers), cfg.num_experts)
    assert actual_shape[1]["qb_beta"].shape == (len(layers), cfg.num_experts)


def test_grouped_moe_mlp_from_layers_rejects_mixed_configs():
    base = GrugModelConfig(vocab_size=16, hidden_dim=4, num_heads=1, num_kv_heads=1, head_dim=4)
    other = GrugModelConfig(vocab_size=16, hidden_dim=4, num_heads=1, num_kv_heads=1, head_dim=4, num_experts=2)
    expert = MoEExpertMlp(
        w_gate_up=jnp.zeros((base.num_experts, base.hidden_dim, 2 * base.intermediate_dim), dtype=jnp.float32),
        w_down=jnp.zeros((base.num_experts, base.intermediate_dim, base.hidden_dim), dtype=jnp.float32),
        implementation="scatter",
        activation=ActivationFunctionEnum.silu,
        capacity_factor=1.0,
        remat_mode="none",
    )
    layer = MoEMLP(
        router=jnp.zeros((base.hidden_dim, base.num_experts), dtype=jnp.float32),
        router_bias=jnp.zeros((base.num_experts,), dtype=jnp.float32),
        expert_mlp=expert,
        cfg=base,
    )
    mismatched = MoEMLP(
        router=jnp.zeros((base.hidden_dim, base.num_experts), dtype=jnp.float32),
        router_bias=jnp.zeros((base.num_experts,), dtype=jnp.float32),
        expert_mlp=expert,
        cfg=other,
    )

    with pytest.raises(ValueError, match="same GrugModelConfig"):
        GroupedMoEMLP.from_layers((layer, mismatched))
