# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Isolate why the device-group pipeline serializes: does JAX overlap GPUs at all?

Three micro-probes, independent of the grug model:

1. **2-GPU overlap** -- dispatch a heavy kernel on dev0 and dev1, block on both.
   If wall time ~= one kernel, the two GPUs overlap (async works); if ~= two
   kernels, the single-thread eager dispatch is serializing them.
2. **device_put non-blocking** -- time the *dispatch* of a cross-GPU transfer of a
   still-computing array. If it returns immediately the transfer is async; if it
   takes ~a kernel time it is blocking the host thread (which would serialize a
   pipeline that transports between every stage).
3. **8-GPU chain vs fanout** -- a dependent chain dev0->..->dev7 (transport each
   hop) vs 8 independent kernels; the ratio shows whether transported dependencies
   pipeline.

    iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --extra gpu \\
      -- python -m experiments.grug.moe_pp.overlap_probe
"""

from __future__ import annotations

import logging
import time

import jax
import jax.numpy as jnp

from experiments.grug.moe_pp.benchmark import init_distributed

logger = logging.getLogger(__name__)

N = 8192
DEPTH = 40


def _heavy(x):
    for _ in range(DEPTH):
        x = jnp.tanh(x @ x) * 1e-4 + 1.0
    return x


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    init_distributed()
    devs = jax.devices()
    logger.info("overlap_probe on %d %s device(s)", len(devs), devs[0].platform)
    heavy = jax.jit(_heavy)

    a = [jax.device_put(jnp.ones((N, N), jnp.float32), d) for d in devs]
    for d in range(min(8, len(devs))):
        jax.block_until_ready(heavy(a[d]))  # warmup-compile per device

    # --- probe 1: 2-GPU overlap ---
    iters = 10
    jax.block_until_ready((heavy(a[0]), heavy(a[1])))
    t = time.perf_counter()
    for _ in range(iters):
        jax.block_until_ready(heavy(a[0]))
        jax.block_until_ready(heavy(a[1]))
    serial2 = (time.perf_counter() - t) / iters
    t = time.perf_counter()
    for _ in range(iters):
        r0, r1 = heavy(a[0]), heavy(a[1])
        jax.block_until_ready((r0, r1))
    par2 = (time.perf_counter() - t) / iters
    logger.info(
        "PROBE1 2-GPU: serial=%.1fms parallel=%.1fms speedup=%.2fx (2.0=perfect overlap, 1.0=serialized)",
        serial2 * 1e3,
        par2 * 1e3,
        serial2 / par2,
    )

    # --- probe 2: is cross-GPU device_put blocking the host thread? ---
    x = heavy(a[0])  # async, still computing on dev0
    t = time.perf_counter()
    y = jax.device_put(x, devs[1])  # dispatch only
    dispatch = time.perf_counter() - t
    jax.block_until_ready(y)
    one_kernel = serial2 / 2
    logger.info(
        "PROBE2 device_put dispatch=%.1fms (one kernel=%.1fms). %s",
        dispatch * 1e3,
        one_kernel * 1e3,
        "BLOCKS host (serializes pipeline)" if dispatch > 0.4 * one_kernel else "async (does not block)",
    )

    # --- probe 3: transported 8-stage chain vs 8 independent kernels ---
    p = min(8, len(devs))
    t = time.perf_counter()
    for _ in range(iters):
        h = a[0]
        for s in range(p):
            h = jax.device_put(h, devs[s])
            h = heavy(h)
        jax.block_until_ready(h)
    chain = (time.perf_counter() - t) / iters
    t = time.perf_counter()
    for _ in range(iters):
        outs = [heavy(a[s]) for s in range(p)]
        jax.block_until_ready(outs)
    fanout = (time.perf_counter() - t) / iters
    logger.info(
        "PROBE3 %d-hop transported chain=%.1fms | %d independent kernels=%.1fms | chain/fanout=%.2fx",
        p,
        chain * 1e3,
        p,
        fanout * 1e3,
        chain / fanout,
    )

    # --- probe 4: M microbatches x P stages, mb-major dispatch (EXACTLY the pipeline pattern) ---
    # Each microbatch is a transported chain dev0->..->dev(P-1). Dispatched mb-by-mb without
    # blocking. If the runtime fills the pipeline, wall time ~ (M+P-1) kernels; if it serializes,
    # ~ M*P kernels.
    m_count = 8
    t = time.perf_counter()
    for _ in range(iters):
        finals = []
        for _m in range(m_count):
            h = a[0]
            for s in range(p):
                h = jax.device_put(h, devs[s])
                h = heavy(h)
            finals.append(h)
        jax.block_until_ready(finals)
    pipe = (time.perf_counter() - t) / iters
    ideal = (m_count + p - 1) * one_kernel
    serial = m_count * p * one_kernel
    logger.info(
        "PROBE4 %dmb x %dstage mb-major=%.0fms | pipelined-ideal=%.0fms serial=%.0fms | %s (overlap eff=%.0f%%)",
        m_count,
        p,
        pipe * 1e3,
        ideal * 1e3,
        serial * 1e3,
        "PIPELINES" if pipe < 0.6 * serial else "SERIALIZES",
        100.0 * (serial - pipe) / (serial - ideal),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
