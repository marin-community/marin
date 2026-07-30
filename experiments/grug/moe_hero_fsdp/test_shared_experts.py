# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import jax
import pytest
from jax.sharding import PartitionSpec as P
from jax.sharding import use_abstract_mesh

from experiments.grug.moe_hero_fsdp.model import GrugModelConfig, Transformer, debug_mesh_and_token_pspec


def _small_config(*, num_shared_experts: int) -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=32,
        hidden_dim=8,
        intermediate_dim=4,
        shared_expert_intermediate_dim=12,
        num_shared_experts=num_shared_experts,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=8,
    )


def _model_shape(config: GrugModelConfig) -> Transformer:
    mesh, _ = debug_mesh_and_token_pspec(num_devices=4)
    key = jax.random.PRNGKey(0)
    with use_abstract_mesh(mesh):
        return eqx.filter_eval_shape(Transformer.init, config, key=key)


def _parameter_count(model: Transformer) -> int:
    return sum(leaf.size for leaf in jax.tree.leaves(model) if isinstance(leaf, jax.ShapeDtypeStruct))


def test_split_shared_experts_preserve_total_parameter_count():
    unsplit_count = _parameter_count(_model_shape(_small_config(num_shared_experts=1)))
    split_count = _parameter_count(_model_shape(_small_config(num_shared_experts=2)))

    assert split_count == unsplit_count


def test_shared_expert_count_must_divide_intermediate_dim():
    config = _small_config(num_shared_experts=2)

    with pytest.raises(ValueError, match="shared_expert_intermediate_dim must be divisible"):
        dataclasses.replace(config, num_shared_experts=5)


def test_shared_expert_count_must_be_positive():
    config = _small_config(num_shared_experts=2)

    with pytest.raises(ValueError, match="num_shared_experts must be positive"):
        dataclasses.replace(config, num_shared_experts=0)


def test_split_shared_experts_preserve_weight_sharding():
    unsplit = _model_shape(_small_config(num_shared_experts=1)).blocks[0].shared
    split = _model_shape(_small_config(num_shared_experts=2)).blocks[0].shared

    assert unsplit is not None
    assert split is not None
    expected_specs = (P("data", "model"), P("data", "model"), P("model", "data"))
    unsplit_specs = tuple(weight.sharding.spec for weight in (unsplit[0].w_gate, unsplit[0].w_up, unsplit[0].w_down))
    split_specs = tuple(
        tuple(weight.sharding.spec for weight in (expert.w_gate, expert.w_up, expert.w_down)) for expert in split
    )

    assert unsplit_specs == expected_specs
    assert split_specs == (expected_specs, expected_specs)
