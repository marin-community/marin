# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hierarchical-transport gate: fused intranode puts inside a multi-process mesh.

The fused `put_segments` kernel is validated on a single-process mesh (M6).
The hierarchical two-hop design (ragged internode + fused intranode) needs
one more property: the kernel must run over a process-LOCAL "gpu" sub-axis
while the mesh's other axis spans processes. Upstream removed multi-process
NVSHMEM from Mosaic, but every `remote_ref`/semaphore target here stays
inside one process, so the local symmetric-memory path should suffice.

Run one process per NODE (4 local GPUs each):

  MARIN_EP_COORD=<host:port> MARIN_EP_NUM_PROCS=<nodes> MARIN_EP_PROC_ID=<i>
  uv run python experiments/marin_ep/bench/spike_hier_intranode.py

Every process builds the same node-local plan (fixed seed), runs the fused
intranode shuffle on its 4 GPUs concurrently with the other nodes, and
asserts all four of its local pool shards against the NumPy plan reference.
"""

import os

import jax
import numpy as np

jax.distributed.initialize(
    coordinator_address=os.environ["MARIN_EP_COORD"],
    num_processes=int(os.environ["MARIN_EP_NUM_PROCS"]),
    process_id=int(os.environ["MARIN_EP_PROC_ID"]),
)

import jax.numpy as jnp  # noqa: E402
from jax import lax, shard_map  # noqa: E402
from jax.sharding import AxisType, Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402
from levanter.grug._moe.marin_ep_transport import dispatch_segments, put_segments  # noqa: E402

from experiments.marin_ep.planref import execute_plans  # noqa: E402

HIDDEN = 1024
LOCAL_EXPERTS = 3
GPUS_PER_NODE = 4
MAX_SEG_ROWS = 2000


def main() -> None:
    proc = jax.process_index()
    nodes = jax.process_count()
    assert jax.local_device_count() == GPUS_PER_NODE, jax.local_device_count()
    node_experts = GPUS_PER_NODE * LOCAL_EXPERTS

    # One intranode plan, shared by every node (same seed): accepted[g, e]
    # rows flow from intranode gpu g to the owner of node-local expert e.
    rng = np.random.default_rng(seed=7)
    accepted = rng.integers(0, MAX_SEG_ROWS, size=(GPUS_PER_NODE, node_experts)).astype(np.int32)
    accepted[0, 1] = 0
    kept = accepted.sum(axis=0)
    kept_by_owner = kept.reshape(GPUS_PER_NODE, LOCAL_EXPERTS)
    region = (np.cumsum(kept_by_owner, axis=1) - kept_by_owner).reshape(node_experts).astype(np.int32)
    pool_rows = int(kept_by_owner.sum(axis=1).max())
    send_rows = int(accepted.sum(axis=1).max())

    # Sends are tagged by GLOBAL device index so intranode cross-talk from
    # another node's shuffle cannot silently pass the assert.
    def send_buf(global_dev: int) -> np.ndarray:
        n = int(accepted[global_dev % GPUS_PER_NODE].sum())
        buf = np.zeros((send_rows, HIDDEN), np.float32)
        buf[:n] = global_dev * 1e6 + np.arange(n)[:, None] + np.arange(HIDDEN)[None, :] / 1e3
        return buf

    mesh = Mesh(
        np.asarray(jax.devices()).reshape(nodes, GPUS_PER_NODE),
        ("node", "gpu"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    accepted_j = jnp.asarray(accepted)
    region_j = jnp.asarray(region)

    def local_fn(src):
        gpu_id = lax.axis_index("gpu")
        plan = dispatch_segments(accepted_j, region_j, gpu_id, local_experts=LOCAL_EXPERTS)
        return put_segments(src, plan, out_rows=pool_rows, axis_name="gpu", num_devices=GPUS_PER_NODE)

    run = jax.jit(
        shard_map(
            local_fn,
            mesh=mesh,
            in_specs=P(("node", "gpu"), None),
            out_specs=P(("node", "gpu"), None),
            check_vma=False,
        )
    )

    local_sends = np.concatenate([send_buf(proc * GPUS_PER_NODE + g) for g in range(GPUS_PER_NODE)], axis=0)
    src_global = jax.make_array_from_process_local_data(
        NamedSharding(mesh, P(("node", "gpu"), None)),
        local_sends,
        (nodes * GPUS_PER_NODE * send_rows, HIDDEN),
    )

    plans = [
        dispatch_segments(accepted_j, region_j, jnp.int32(g), local_experts=LOCAL_EXPERTS) for g in range(GPUS_PER_NODE)
    ]

    for attempt in range(2):
        pool = run(src_global)
        node_sends = [send_buf(proc * GPUS_PER_NODE + g) for g in range(GPUS_PER_NODE)]
        want = execute_plans(plans, node_sends, pool_rows)
        for shard in pool.addressable_shards:
            gpu = shard.index[0].start // pool_rows % GPUS_PER_NODE
            covered = int(kept_by_owner[gpu].sum())
            np.testing.assert_array_equal(
                np.asarray(shard.data)[:covered],
                want[gpu][:covered],
                err_msg=f"attempt {attempt} proc {proc} gpu {gpu}",
            )
    print(f"[proc {proc}] HIER-INTRANODE CORRECT ({nodes} nodes x {GPUS_PER_NODE} GPUs)", flush=True)


if __name__ == "__main__":
    main()
