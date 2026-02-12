# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

import jax
import jax.numpy as jnp
from jax.sharding import AbstractMesh, AxisType, NamedSharding, PartitionSpec as P, use_abstract_mesh

from jax._src import config as jax_config

from levanter.grug.attention import AttentionMask
from levanter.grug.model import GrugModelConfig
from levanter.grug.sharding import Pbatch
from levanter.grug.model import init_parameters, loss_fn


def _make_abstract_mesh(*, data: int, model: int) -> AbstractMesh:
    return AbstractMesh(
        axis_sizes=(data, model),
        axis_names=("data", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


class _reset_abstract_mesh:
    def __enter__(self):
        self._prev = jax_config.abstract_mesh_context_manager.swap_local(jax_config.config_ext.unset)
        return self

    def __exit__(self, exc_type, exc, tb):
        jax_config.abstract_mesh_context_manager.set_local(self._prev)
        return False


@pytest.mark.parametrize(
    ("data", "model"),
    [
        (4, 1),
        (2, 2),
    ],
)
def test_grug_loss_can_lower_on_abstract_4_device_mesh(data: int, model: int):
    cfg = GrugModelConfig(
        vocab_size=256,
        hidden_dim=128,
        intermediate_dim=256,
        num_layers=1,
        num_heads=8,
        num_kv_heads=8,
        max_seq_len=256,  # splash attention requires KV seq multiple of 128
    )

    mesh = _make_abstract_mesh(data=data, model=model)
    # Some test setups establish a size-1 abstract mesh globally; JAX forbids changing the mesh
    # size under `use_abstract_mesh`. Reset to "unset" so we can test other sizes here.
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        # Build shaped params via eval_shape so we exercise init sharding rules under AbstractMesh.
        key = jax.ShapeDtypeStruct(shape=(2,), dtype=jnp.uint32, sharding=NamedSharding(mesh, P()))
        params = jax.eval_shape(lambda k: init_parameters(cfg, key=k), key)

        batch = 8
        seq = 256

        def f(p):
            token_ids = jnp.zeros((batch, seq), dtype=jnp.int32)
            token_ids = jax.sharding.reshard(token_ids, Pbatch)
            loss_weight = jnp.ones((batch, seq), dtype=jnp.float32)
            loss_weight = jax.sharding.reshard(loss_weight, Pbatch)
            return loss_fn(p, token_ids, loss_weight, cfg, mask=AttentionMask.causal(), reduction="mean")

        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = jax.jit(f).trace(params).lower(lowering_platforms=(platform,))
        # Lowering is the point of this test; don't force full compilation.
        assert lowered is not None
