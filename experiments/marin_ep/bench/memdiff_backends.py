# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile-only HBM diff of EP MoE backends at hero per-device shapes.

The EP64 hero OOMs on the marin_ep ragged path (183 GiB needed vs 138 GiB
budget) while fixed_all_to_all fits. This compiles fwd+bwd of one MoE
layer per backend on an EP4 tray with per-device shapes matched to the
hero (T=65536/device, El=3, A_global/E = 87,381 so capacity and pool rows
are identical) and prints XLA's memory analysis — no execution, inputs are
ShapeDtypeStructs.

  uv run python experiments/marin_ep/bench/memdiff_backends.py [capacity_factor]
"""

import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import shard_map
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug._moe.ep_fixed_all_to_all import _moe_mlp_ep_fixed_a2a_local
from levanter.grug._moe.ep_marin import marin_ep_moe_local
from levanter.grug._moe.ep_ragged_all_to_all import _moe_mlp_ep_ragged_a2a_local

TOKENS = 65536  # per device
TOPK = 4
HIDDEN = 3072
INTERMEDIATE = 6272


def main() -> None:
    cf = float(sys.argv[1]) if len(sys.argv) > 1 else 1.33
    devices = jax.device_count()
    num_experts = devices * 3  # keeps A_global/E at the hero's 87,381 rows/expert

    mesh = Mesh(
        np.asarray(jax.devices()).reshape(1, 1, devices, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    batch_spec = P(("replica_dcn", "data", "expert"))
    weight_spec = P("expert", None, None)

    def sds(shape, dtype, spec):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=NamedSharding(mesh, spec))

    total = devices * TOKENS
    args = (
        sds((total, HIDDEN), jnp.bfloat16, batch_spec),
        sds((total, TOPK), jnp.int32, batch_spec),
        sds((total, TOPK), jnp.float32, batch_spec),
        sds((num_experts, HIDDEN, 2 * INTERMEDIATE), jnp.bfloat16, weight_spec),
        sds((num_experts, INTERMEDIATE, HIDDEN), jnp.bfloat16, weight_spec),
        sds((total, HIDDEN), jnp.bfloat16, batch_spec),  # cotangent carrier
    )

    backends = {
        "marin_ep(ragged)": partial(marin_ep_moe_local, pool_group_size=3, transport="ragged"),
        "ep_ragged": _moe_mlp_ep_ragged_a2a_local,
        "ep_fixed": _moe_mlp_ep_fixed_a2a_local,
    }

    for name, local_fn in backends.items():
        shard_fn = shard_map(
            partial(local_fn, activation_fn=jax.nn.silu, num_experts=num_experts, capacity_factor=cf),
            mesh=mesh,
            in_specs=(batch_spec, batch_spec, batch_spec, weight_spec, weight_spec),
            out_specs=(batch_spec, P()),
            check_vma=False,
        )

        def loss(x, e, w, w13, w2, cot, _fn=shard_fn):
            y, _ = _fn(x, e, w, w13, w2)
            return jnp.sum((y * cot).astype(jnp.float32))

        with jax.set_mesh(mesh):
            compiled = jax.jit(jax.grad(loss, argnums=(0, 3, 4))).lower(*args).compile()
        ma = compiled.memory_analysis()
        gib = 1024**3
        print(
            f"{name:18s} temp {ma.temp_size_in_bytes / gib:7.2f} GiB  "
            f"args {ma.argument_size_in_bytes / gib:6.2f}  out {ma.output_size_in_bytes / gib:6.2f}  "
            f"alias {ma.alias_size_in_bytes / gib:6.2f}  cf={cf}",
            flush=True,
        )


if __name__ == "__main__":
    main()
