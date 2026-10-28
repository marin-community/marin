# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Transient feature sharding for the single-replica Belay embedding probe."""

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard

from experiments.grug.fast_track.model import _BATCH_AXES, _FSDP_AXES


@jax.custom_vjp
def column_sharded_embedding(table: jax.Array, ids: jax.Array) -> jax.Array:
    """Look up column shards and return token-row shards, retaining row storage."""
    mesh = get_abstract_mesh()
    if mesh.shape["replica_dcn"] != 1 or mesh.shape["model"] != 1:
        raise ValueError("The column embedding probe requires one replica and model shard")
    columns = reshard(table, P(None, _FSDP_AXES))
    ids = reshard(ids, P(None, None))
    output = shard_map(
        lambda t, i: t[i],
        mesh=mesh,
        in_specs=(P(None, _FSDP_AXES), P(None, None)),
        out_specs=P(None, None, _FSDP_AXES),
    )(columns, ids)
    return reshard(output, P(_BATCH_AXES, None, None))


def _forward(table, ids):
    return column_sharded_embedding(table, ids), (ids, jnp.zeros((0, *table.shape), table.dtype))


def _backward(residual, cotangent):
    ids, table_like = residual
    rows = table_like.shape[1]

    def local_scatter(local_ids, local_cotangent):
        accumulator = jnp.zeros((rows, local_cotangent.shape[-1]), jnp.float32)
        return accumulator.at[local_ids].add(local_cotangent.astype(jnp.float32)).astype(table_like.dtype)

    columns = shard_map(
        local_scatter,
        mesh=get_abstract_mesh(),
        in_specs=(P(None, None), P(None, None, _FSDP_AXES)),
        out_specs=P(None, _FSDP_AXES),
    )(reshard(ids, P(None, None)), reshard(cotangent, P(None, None, _FSDP_AXES)))
    return reshard(columns, P(_FSDP_AXES, None)), None


column_sharded_embedding.defvjp(_forward, _backward)
