# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""EP4 reproduction of the fused2 EP64 combine wedge (MEP-069 hypothesis).

Pallas zero-initializes the GMEM semaphore scratch of each pallas_call with an
XLA op that runs when the call is scheduled. ``fused_gemm_combine``'s
``received_sem`` is signalled by REMOTE ranks, and a peer starts sending as
soon as its own GEMM tiles land. If a fast peer's signals arrive before the
slow rank's zeroing op has run, they are erased and the slow rank waits at
``received_sem`` forever. Heavily skewed routing at EP4 (rank 0's experts get
almost all tokens, the last rank's almost none) recreates the rank-speed
divergence that makes the race window wide at EP64.

Run once per GPU (1 process/GPU):

  MARIN_EP_COORD=<host:port> MARIN_EP_NUM_PROCS=4 MARIN_EP_PROC_ID=<i>
  CUDA_VISIBLE_DEVICES=<i> python .../repro_fused2_zeroing_race.py [iters]

Prediction under the hypothesis: some iterations wedge on the OVERLOADED rank
(late zeroing) while underloaded peers print their iteration lines.
"""

import os
import sys
import time

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

from levanter.grug._moe.ep_marin import marin_ep_moe_local  # noqa: E402


def main() -> None:
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    proc = jax.process_index()
    devices = jax.device_count()
    assert jax.local_device_count() == 1
    tokens, topk, hidden, intermediate = 8192, 4, 6144, 6272
    num_experts = devices * 3
    local_experts = 3
    capacity_factor = 1.2

    rng = np.random.default_rng(seed=11)
    # Skew: rank 0's experts take almost all assignments, the last rank's
    # almost none, so the last rank reaches its combine sends while rank 0 is
    # still deep in GEMM work.
    w = np.full(num_experts, 0.01)
    w[:local_experts] = 10.0
    probs = w / w.sum()
    experts = rng.choice(num_experts, size=(devices * tokens, topk), p=probs).astype(np.int32)
    x = rng.standard_normal((devices * tokens, hidden)).astype(np.float32)
    weights = (rng.random((devices * tokens, topk)) + 0.05).astype(np.float32)
    w13 = (0.1 * rng.standard_normal((num_experts, hidden, 2 * intermediate))).astype(np.float32)
    w2 = (0.1 * rng.standard_normal((num_experts, intermediate, hidden))).astype(np.float32)

    mesh = Mesh(
        np.asarray(jax.devices()).reshape(1, 1, devices, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    batch_spec = P(("replica_dcn", "data", "expert"))
    weight_spec = P("expert", None, None)

    shard_fn = shard_map(
        lambda x_, e_, w_, w13_, w2_: marin_ep_moe_local(
            x_,
            e_,
            w_,
            w13_,
            w2_,
            activation_fn=jax.nn.silu,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
            pool_group_size=local_experts,
            transport="mgpu_fused2",
        ),
        mesh=mesh,
        in_specs=(batch_spec, batch_spec, batch_spec, weight_spec, weight_spec),
        out_specs=(batch_spec, P()),
        check_vma=False,
    )

    def put_batch(a, dtype=None):
        mine = a.reshape(devices, tokens, *a.shape[1:])[proc]
        if dtype is not None:
            mine = mine.astype(dtype)
        return jax.make_array_from_process_local_data(
            NamedSharding(mesh, batch_spec), mine, a.shape
        )

    def put_weight(a):
        mine = a.reshape(devices, local_experts, *a.shape[1:])[proc].astype(jnp.bfloat16)
        return jax.make_array_from_process_local_data(
            NamedSharding(mesh, weight_spec), mine, a.shape
        )

    args = (
        put_batch(x, jnp.bfloat16),
        put_batch(experts),
        put_batch(weights),
        put_weight(w13),
        put_weight(w2),
    )

    # Squeeze the heap so XLA must reuse scratch buffers across layers,
    # chaining each layer's semaphore zeroing behind earlier work (the hero
    # regime that opens the race window).
    filler_gb = int(os.environ.get("MARIN_EP_FILLER_GB", "120"))
    filler = jnp.zeros((filler_gb * 2**27,), jnp.float64) if filler_gb else None  # 8B elems

    def layers(x0, e_, w_, w13_, w2_, n_layers=6):
        def body(x_c, _):
            y_, d_ = shard_fn(x_c, e_, w_, w13_, w2_)
            return (x_c + y_).astype(x_c.dtype), d_

        xf, ds = jax.lax.scan(body, x0, None, length=n_layers)
        return xf, jnp.sum(ds)

    with jax.set_mesh(mesh):
        fn = jax.jit(layers)
        t0 = time.time()
        y, dropped = jax.block_until_ready(fn(*args))
        print(f"[{proc}] warmup ok {time.time() - t0:.1f}s dropped={int(dropped)}", flush=True)
        for it in range(iters):
            t1 = time.time()
            jax.block_until_ready(fn(*args))
            print(f"[{proc}] iter {it} ok {time.time() - t1:.2f}s", flush=True)
        del filler
    print(f"[{proc}] REPRO_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
