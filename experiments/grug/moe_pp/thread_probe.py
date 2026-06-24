# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""De-risk threaded host dispatch: can a background thread move arrays (device_get /
device_put) while the main thread keeps dispatching jit compute on the same devices?

The device-group multi-host pipeline serializes at the host boundary because the
transport (``device_get`` + cross-host broadcast) runs on the single dispatch thread
and blocks it. The proposed fix is a per-host transport thread that runs the boundary
hop off the critical path while the main thread keeps the local GPUs busy with other
microbatches -- safe to interleave because every host follows the same 1f1b schedule,
so the cross-host collectives stay ordered.

This probe tests only the *mechanics* on one host (no cross-host collective): a worker
thread pulls a still-computing array to the host and pushes it back, while the main
thread dispatches a long chain of independent jit matmuls. It asserts (1) no deadlock /
error from concurrent dispatch + transfer, and (2) the worker's transported result is
numerically correct. It does NOT prove GPU NCCL-concurrency safety -- that needs the
2-node run -- but a failure here kills the approach before spending a cycle.

    XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu \\
      uv run python -m experiments.grug.moe_pp.thread_probe
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

N = 2048
CHAIN = 200
ITEMS = 16


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    devs = jax.local_devices()
    if len(devs) < 2:
        raise RuntimeError(f"need >=2 local devices, got {len(devs)}")
    dst = devs[1]
    logger.info("thread_probe on %d %s device(s)", len(devs), devs[0].platform)

    @jax.jit
    def heavy(x):
        # A dependent chain so the array is genuinely still computing when the worker
        # asks for it -- forces device_get to wait, the exact blocking we must hide.
        for _ in range(CHAIN):
            x = jnp.tanh(x @ x.T) / N
        return x

    inbox: queue.Queue = queue.Queue()
    outbox: queue.Queue = queue.Queue()
    errors: list = []

    def worker():
        # Mirrors the transport thread: pull a (possibly still-computing) on-device array
        # to the host, then place it onto another device -- the device_get/device_put pair
        # that a real boundary hop runs (minus the cross-host broadcast).
        try:
            while True:
                item = inbox.get()
                if item is None:
                    return
                idx, arr = item
                host = np.asarray(jax.device_get(arr))
                back = jax.device_put(host, dst)
                outbox.put((idx, back, host))
        except Exception as exc:  # surface to the main thread, then re-raise
            errors.append(exc)
            outbox.put(None)

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    rng = np.random.default_rng(0)
    inputs = [jnp.asarray(rng.standard_normal((N, N), np.float32)) for _ in range(ITEMS)]

    start = time.perf_counter()
    # Main thread: dispatch every heavy() (async) and hand each result to the worker,
    # never blocking on a transfer -- then drain the worker's results.
    dispatched = [heavy(x) for x in inputs]
    for idx, arr in enumerate(dispatched):
        inbox.put((idx, arr))
    inbox.put(None)

    results: dict = {}
    for _ in range(ITEMS):
        got = outbox.get()
        if got is None:
            break
        idx, back, host = got
        results[idx] = (back, host)
    t.join(timeout=30)
    elapsed = time.perf_counter() - start

    if errors:
        raise errors[0]
    if t.is_alive():
        raise RuntimeError("worker thread did not finish -- deadlock between dispatch and transfer")
    if len(results) != ITEMS:
        raise RuntimeError(f"transported {len(results)}/{ITEMS} items")

    # The transported array (host->dst) must equal a direct device_get of the source.
    max_err = 0.0
    for idx in range(ITEMS):
        back, host = results[idx]
        ref = np.asarray(jax.device_get(dispatched[idx]))
        max_err = max(max_err, float(np.max(np.abs(np.asarray(back) - ref))))
        assert back.devices() == {dst}, f"item {idx} landed on {back.devices()}, not {dst}"
    logger.info(
        "PASS: %d items transported concurrently with %d dispatched chains in %.2fs | max_err=%.2e",
        ITEMS,
        ITEMS,
        elapsed,
        max_err,
    )
    if max_err > 1e-5:
        raise RuntimeError(f"transported values diverged: max_err={max_err:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
