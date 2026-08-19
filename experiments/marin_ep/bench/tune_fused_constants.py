# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Screen the fused dispatch kernel's transport constants at the 8/384 geometry.

TRANSPORT_STAGE_ROWS / TRANSPORT_LANE_CHUNK / _FUSED_MCS were tuned at the
4/192 hero shard shape (3 local experts, 2I=12544). The 8/384 hero runs 6
local experts at 2I=6144 with ~576k pool rows per device; MEP-080's
differential profile puts the parity gap inside the fused kernels' SM time,
so re-screen the constants at a representative per-device workload: EP4,
num_experts=24 (6 local), hero tokens/topk/cf.

Run one process per GPU (the zeroing-race driver pattern):

  MARIN_EP_COORD=<host:port> MARIN_EP_NUM_PROCS=4 MARIN_EP_PROC_ID=<i>
  CUDA_VISIBLE_DEVICES=<i> python .../tune_fused_constants.py
"""

import os
import sys
import time

import jax

collective_mb = int(os.environ.get("MARIN_EP_COLLECTIVE_MEMORY_MB", "0"))
if collective_mb:
    # Bound the NCCL-window collective arena; must precede client creation.
    jax.config.update("jax_pjrt_client_create_options", {"collective_memory_size": collective_mb * 1024 * 1024})

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
from levanter.grug._moe import fused_dispatch_brd  # noqa: E402
from levanter.grug._moe.ep_marin import marin_ep_moe_local  # noqa: E402

TOKENS, TOPK, HIDDEN, INTER = 65536, 8, 6144, 3072
LOCAL_EXPERTS = 6
CF = 1.1
ITERS = 6

# One (stage_rows, lane_chunk, mcs) config per process invocation, from
# MARIN_EP_TUNE_CFG="sr,lc,mcs"; a failed config must not pollute the next
# one's collective arena.
CFG = tuple(int(v) for v in os.environ["MARIN_EP_TUNE_CFG"].split(","))


def main() -> None:
    proc = jax.process_index()
    devices = jax.device_count()
    assert jax.local_device_count() == 1
    num_experts = devices * LOCAL_EXPERTS

    rng = np.random.default_rng(seed=7)
    probs = rng.dirichlet(np.full(num_experts, 0.7))
    experts = rng.choice(num_experts, size=(devices * TOKENS, TOPK), p=probs).astype(np.int32)
    x = rng.standard_normal((devices * TOKENS, HIDDEN)).astype(np.float32)
    weights = (rng.random((devices * TOKENS, TOPK)) + 0.05).astype(np.float32)
    w13 = (0.02 * rng.standard_normal((num_experts, HIDDEN, 2 * INTER))).astype(np.float32)
    w2 = (0.02 * rng.standard_normal((num_experts, INTER, HIDDEN))).astype(np.float32)
    cot = rng.standard_normal((devices * TOKENS, HIDDEN)).astype(np.float32)

    mesh = Mesh(
        np.asarray(jax.devices()).reshape(1, 1, devices, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    batch_spec = P(("replica_dcn", "data", "expert"))
    weight_spec = P("expert", None, None)

    def put_batch(a, dtype=None):
        mine = a.reshape(devices, TOKENS, *a.shape[1:])[proc]
        if dtype is not None:
            mine = mine.astype(dtype)
        return jax.make_array_from_process_local_data(NamedSharding(mesh, batch_spec), mine, a.shape)

    def put_weight(a):
        mine = a.reshape(devices, LOCAL_EXPERTS, *a.shape[1:])[proc].astype(jnp.bfloat16)
        return jax.make_array_from_process_local_data(NamedSharding(mesh, weight_spec), mine, a.shape)

    xb = put_batch(x, jnp.bfloat16)
    eb = put_batch(experts)
    wb = put_batch(weights)
    cb = put_batch(cot, jnp.bfloat16)
    w13b = put_weight(w13)
    w2b = put_weight(w2)

    results = []
    for stage_rows, lane_chunk, mcs in [CFG]:
        fused_dispatch_brd.TRANSPORT_STAGE_ROWS = stage_rows
        fused_dispatch_brd.TRANSPORT_LANE_CHUNK = lane_chunk
        fused_dispatch_brd._FUSED_MCS = mcs

        shard_fn = shard_map(
            lambda x_, e_, w_, w13_, w2_: marin_ep_moe_local(
                x_,
                e_,
                w_,
                w13_,
                w2_,
                activation_fn=jax.nn.silu,
                num_experts=num_experts,
                capacity_factor=CF,
                pool_group_size=LOCAL_EXPERTS,
                transport="mgpu_fused",
            ),
            mesh=mesh,
            in_specs=(batch_spec, batch_spec, batch_spec, weight_spec, weight_spec),
            out_specs=(batch_spec, P()),
            check_vma=False,
        )

        def loss(x_, e_, w_, c_, w13_, w2_, shard_fn=shard_fn):
            y, _ = shard_fn(x_, e_, w_, w13_, w2_)
            return jnp.sum(y * c_)

        with jax.set_mesh(mesh):
            step = jax.jit(jax.grad(loss, argnums=(0, 4, 5)))
            try:
                jax.block_until_ready(step(xb, eb, wb, cb, w13b, w2b))
                times = []
                for _ in range(ITERS):
                    t0 = time.perf_counter()
                    jax.block_until_ready(step(xb, eb, wb, cb, w13b, w2b))
                    times.append(time.perf_counter() - t0)
            except Exception as exc:
                if proc == 0:
                    print(f"sr={stage_rows} lc={lane_chunk} mcs={mcs}: FAIL {type(exc).__name__}: {exc}", flush=True)
                continue
        ms = min(times) * 1e3
        results.append((ms, stage_rows, lane_chunk, mcs))
        if proc == 0:
            print(f"sr={stage_rows} lc={lane_chunk} mcs={mcs}: {ms:.2f} ms fwd+bwd", flush=True)

    if proc == 0:
        print("CFG_COMPLETED", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
