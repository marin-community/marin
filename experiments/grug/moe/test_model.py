# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import pytest
from jax.sharding import AbstractMesh, AxisType, use_abstract_mesh
from jax.sharding import PartitionSpec as P

from experiments.grug.moe.model import CausalSelfAttention, DenseMLP, GrugModelConfig, Transformer


def _mesh(*, replica_dcn: int = 1, data: int = 1, expert: int = 1, model: int = 1) -> AbstractMesh:
    return AbstractMesh(
        axis_sizes=(replica_dcn, data, expert, model),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )


def _config(hidden_dim: int) -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=128,
        hidden_dim=hidden_dim,
        intermediate_dim=hidden_dim // 2,
        shared_expert_intermediate_dim=hidden_dim,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=4,
        num_kv_heads=1,
    )


def _dense_specs(mesh: AbstractMesh, hidden_dim: int) -> tuple[P, P]:
    with use_abstract_mesh(mesh):
        dense = eqx.filter_eval_shape(lambda: DenseMLP.init(hidden_dim, 16, 0.02, key=jax.random.key(0)))
    return dense.w_gate.sharding.spec, dense.w_down.sharding.spec


def _attention_specs(mesh: AbstractMesh, hidden_dim: int) -> tuple[P, P]:
    with use_abstract_mesh(mesh):
        attention = eqx.filter_eval_shape(lambda: CausalSelfAttention.init(_config(hidden_dim), key=jax.random.key(1)))
    return attention.w_q.sharding.spec, attention.w_o.sharding.spec


def _lm_head_spec(mesh: AbstractMesh, hidden_dim: int) -> P:
    with use_abstract_mesh(mesh):
        model = eqx.filter_eval_shape(lambda: Transformer.init(_config(hidden_dim), key=jax.random.key(2)))
    return model.output_proj.sharding.spec


def test_nonexpert_weights_shard_over_data_and_expert():
    """A divisible hidden dim uses both FSDP axes, so EP does not leave these replicated."""
    mesh = _mesh(data=2, expert=2)
    column, row = P(("data", "expert"), "model"), P("model", ("data", "expert"))

    assert _dense_specs(mesh, 32) == (column, row)
    assert _attention_specs(mesh, 32) == (column, row)


def test_indivisible_hidden_dim_falls_back_to_the_pre_ep_layout():
    """Widening the shard group must not add a divisibility precondition.

    hidden=12 divides data=4 but not data*expert=8, the shape class that would
    otherwise abort model init once "expert" joined the group.
    """
    mesh = _mesh(data=4, expert=2)
    column, row = P("data", "model"), P("model", "data")

    assert _dense_specs(mesh, 12) == (column, row)
    assert _attention_specs(mesh, 12) == (column, row)


def test_hidden_dim_indivisible_by_the_pre_ep_group_still_fails_loudly():
    """Falling back stops at the pre-EP group; it does not slide on to replication.

    A geometry the pre-EP layout could not shard raised at init before EP and has to
    keep doing so. Silently replicating every non-expert weight and its optimizer state
    would turn a fail-fast config error into a mid-run memory blowup.
    """
    mesh = _mesh(data=3, expert=2)

    with pytest.raises(ValueError, match="does not evenly divide"):
        _dense_specs(mesh, 32)


def test_a_mesh_without_an_expert_axis_uses_the_pre_ep_layout():
    """Non-EP meshes still initialize; an axis the mesh lacks drops out of the group."""
    mesh = AbstractMesh(
        axis_sizes=(2, 1),
        axis_names=("data", "model"),
        axis_types=(AxisType.Explicit,) * 2,
    )

    assert _dense_specs(mesh, 32) == (P("data", "model"), P("model", "data"))


def test_lm_head_widens_only_when_the_hidden_dim_divides():
    """The head keeps the replica axis it had before EP and gains "expert" when it fits."""
    # 8 divides replica_dcn*data*expert == 8.
    assert _lm_head_spec(_mesh(replica_dcn=2, data=2, expert=2), 8) == P(("replica_dcn", "data", "expert"), "model")
    # 8 divides replica_dcn*data == 4 but not replica_dcn*data*expert == 16.
    assert _lm_head_spec(_mesh(replica_dcn=2, data=2, expert=4), 8) == P(("replica_dcn", "data"), "model")
