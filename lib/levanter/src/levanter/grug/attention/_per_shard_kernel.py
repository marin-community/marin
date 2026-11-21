# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""Run a per-device self-attention kernel under Grug's explicit meshes."""

from collections.abc import Callable

import jax
from jax import shard_map
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from jaxtyping import Array, Float

from levanter.sharding import partition_spec_of, partitioned_dims


def run_self_attention_per_shard(
    kernel_name: str,
    run: Callable[..., Float[Array, "B S Hq D"]],
    q: Float[Array, "B S Hq D"],
    k: Float[Array, "B S Hkv D"],
    v: Float[Array, "B S Hkv D"],
    *batch_operands: Array,
) -> Float[Array, "B S Hq D"]:
    """Call ``run(q, k, v, *batch_operands)`` on each shard of the current mesh.

    Batch and heads may be sharded; sequence and head_dim may not, and k and v must be sharded like q. Each of
    ``batch_operands`` has a leading batch axis and is resharded to q's batch sharding.
    """
    mesh = get_abstract_mesh()
    if mesh is None or mesh.empty:
        return run(q, k, v, *batch_operands)
    q_dims = partitioned_dims(q, mesh)
    if q_dims[1] or q_dims[3]:
        raise ValueError(f"{kernel_name} needs unsharded sequence and head_dim, got {partition_spec_of(q)}")
    for name, x in (("k", k), ("v", v)):
        if partitioned_dims(x, mesh) != q_dims:
            raise ValueError(
                f"{kernel_name} needs {name} sharded like q, got q={partition_spec_of(q)} {name}={partition_spec_of(x)}"
            )
    # pallas_call rejects operands whose type names any Explicit mesh axis, even one of size 1, and Grug's meshes keep
    # length-1 axes in every spec, so only skip shard_map when q's spec names no axis at all.
    if not any(q_dims) and not any(jax.typeof(q).sharding.spec):
        return run(q, k, v, *batch_operands)
    q_spec = tuple(partition_spec_of(q))
    q_spec = P(*q_spec, *((None,) * (q.ndim - len(q_spec))))
    batch_operands = tuple(reshard(x, P(q_spec[0], *((None,) * (x.ndim - 1)))) for x in batch_operands)

    @shard_map(mesh=mesh, out_specs=q_spec, check_vma=False)
    def _local(q_local, k_local, v_local, *batch_local):
        return run(q_local, k_local, v_local, *batch_local)

    return _local(q, k, v, *batch_operands)
