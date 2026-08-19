# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-process ragged-path crash repro at hero per-device shapes.

The EP64 hero crashes with CUDA_ERROR_ILLEGAL_ADDRESS inside the NCCL
window/symmetric-memory ragged all-to-all on every ragged-class flavor.
Single-process EP4 at identical shapes is fine, so the trigger is the
multi-process NCCL path. Run one process per GPU:

  MARIN_EP_COORD=<host:port> MARIN_EP_NUM_PROCS=4 MARIN_EP_PROC_ID=<i>
  CUDA_VISIBLE_DEVICES=<i> uv run python .../repro_ragged_multiproc.py [tokens]
"""

import os
import sys
from functools import partial

import jax

jax.distributed.initialize(
    coordinator_address=os.environ["MARIN_EP_COORD"],
    num_processes=int(os.environ["MARIN_EP_NUM_PROCS"]),
    process_id=int(os.environ["MARIN_EP_PROC_ID"]),
)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import shard_map  # noqa: E402
from jax.sharding import AxisType, Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402
from levanter.grug._moe.ep_marin import _moe_mlp_ep_marin_local  # noqa: E402

HIDDEN = 3072
INTERMEDIATE = 6272
TOPK = 4
CAPACITY_FACTOR = 1.1


def main() -> None:
    tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 65536
    proc = jax.process_index()
    devices = jax.device_count()
    local = jax.local_device_count()
    num_experts = devices * 3

    rng = np.random.default_rng(seed=proc)
    mesh = Mesh(
        np.asarray(jax.devices()).reshape(1, 1, devices, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    batch_spec = P(("replica_dcn", "data", "expert"))
    weight_spec = P("expert", None, None)
    shard_fn = shard_map(
        partial(
            _moe_mlp_ep_marin_local,
            activation_fn=jax.nn.silu,
            num_experts=num_experts,
            capacity_factor=CAPACITY_FACTOR,
        ),
        mesh=mesh,
        in_specs=(batch_spec, batch_spec, batch_spec, weight_spec, weight_spec),
        out_specs=(batch_spec, P()),
        check_vma=False,
    )

    def make(shape, dtype, spec, gen):
        local_shape = (shape[0] // devices * local, *shape[1:])
        return jax.make_array_from_process_local_data(NamedSharding(mesh, spec), gen(local_shape).astype(dtype), shape)

    total = devices * tokens
    x = make((total, HIDDEN), jnp.bfloat16, batch_spec, lambda s: rng.standard_normal(s, dtype=np.float32))
    experts = make((total, TOPK), jnp.int32, batch_spec, lambda s: rng.integers(0, num_experts, s))
    weights = make((total, TOPK), jnp.float32, batch_spec, lambda s: rng.random(s, dtype=np.float32))
    w13 = make(
        (num_experts, HIDDEN, 2 * INTERMEDIATE),
        jnp.bfloat16,
        weight_spec,
        lambda s: 0.02 * rng.standard_normal(s, dtype=np.float32),
    )
    w2 = make(
        (num_experts, INTERMEDIATE, HIDDEN),
        jnp.bfloat16,
        weight_spec,
        lambda s: 0.02 * rng.standard_normal(s, dtype=np.float32),
    )

    def loss(x_, e_, w_, w13_, w2_):
        y, dropped = shard_fn(x_, e_, w_, w13_, w2_)
        return jnp.sum(y.astype(jnp.float32)), dropped

    with jax.set_mesh(mesh):
        for step in range(3):
            (val, dropped), grads = jax.jit(jax.value_and_grad(loss, argnums=(0, 3, 4), has_aux=True))(
                x, experts, weights, w13, w2
            )
            jax.block_until_ready(grads)
            print(f"[proc {proc}] step {step} OK loss={float(val):.3f} dropped={int(dropped)}", flush=True)


if __name__ == "__main__":
    main()
