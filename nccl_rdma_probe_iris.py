"""Minimal multi-node NCCL collective-correctness probe (issue #7012, B200MFU-020).

Differential test for the "Schmidt inter-node RDMA is broken at the fabric
level" claim: run the *same* collectives as the Schmidt `nccl_probe.py` (one
256 MB reduce + one all-gather over the global mesh) on another cluster with
the same JAX/NCCL stack. Correct results over the IB/RDMA transport here mean
the software stack is fine cross-node and the Schmidt failure is
environment-specific.

Transport is selected by env vars on the submit command:
    (none)               NCCL default — on GB200 NVL72 cross-node may ride MNNVL
    NCCL_MNNVL_ENABLE=0  force cross-node traffic onto NET/IB (RDMA)
    NCCL_IB_DISABLE=1    TCP sockets control

Run with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET to record which transport
actually carried the collectives.
"""

import faulthandler
import math
import os
import sys
import time

import jax
import numpy as np
from iris.runtime.jax_init import initialize_jax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

initialize_jax()
faulthandler.dump_traceback_later(240, repeat=True)


def phase(msg: str) -> None:
    print(f"PROBE[p{jax.process_index()}] {msg}", file=sys.stderr, flush=True)


phase(
    f"init: {jax.process_count()} procs, {jax.device_count()} devices, "
    f"host={os.uname().nodename}, "
    f"NCCL_MNNVL_ENABLE={os.environ.get('NCCL_MNNVL_ENABLE', '<unset>')}, "
    f"NCCL_IB_DISABLE={os.environ.get('NCCL_IB_DISABLE', '<unset>')}"
)

devices = np.array(jax.devices()).reshape(jax.device_count())
mesh = Mesh(devices, ("data",))
sharding = NamedSharding(mesh, P("data"))

n = 64 * 1024 * 1024  # 256 MB f32 global, same as the Schmidt probe
x = jax.make_array_from_callback((n,), sharding, lambda idx: np.ones((n,), np.float32)[idx])
phase("array built")


@jax.jit
def do_psum(x):
    return jax.lax.with_sharding_constraint(x * 2.0, sharding).sum()


@jax.jit
def do_allgather(x):
    y = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P(None)))
    return y[::1024].sum()


# Schmidt failure signature was gross divergence (one rank reads 0.0), so a
# loose relative tolerance cleanly separates corruption from reduction rounding.
expected = {"psum-ish reduce": 2.0 * n, "all-gather": n / 1024}
failures = 0
for name, fn in [("psum-ish reduce", do_psum), ("all-gather", do_allgather)]:
    t0 = time.perf_counter()
    out = float(jax.block_until_ready(fn(x)))
    ok = math.isclose(out, expected[name], rel_tol=1e-3)
    failures += not ok
    phase(
        f"{name}: {time.perf_counter() - t0:.2f}s -> {out:.1f} "
        f"(expected {expected[name]:.1f}) {'OK' if ok else 'WRONG'}"
    )

faulthandler.cancel_dump_traceback_later()
phase("ALL COLLECTIVES CORRECT" if failures == 0 else f"{failures} COLLECTIVE(S) WRONG")
if failures:
    sys.exit(1)
