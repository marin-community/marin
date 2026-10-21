# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Time ``prodquant_probe.producer_probe`` (producer half of a quantize-on-load wgrad GEMM, no MMA) at the hero
chunk's operand shapes and reuse factors; see ``bench_mxfp8_prodquant_bounds.py`` for the GEMM/quantizer baselines.

    python lib/levanter/scripts/bench/bench_prodquant_probe.py
"""

import os
import statistics
import sys
import time

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(__file__))
from prodquant_probe import producer_probe  # noqa: E402
from quack.cute_dsl_utils import get_device_multiprocessor_count  # noqa: E402


def _time(fn, x, iters=5, reps=3):
    jax.block_until_ready(fn(x))
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        for _ in range(iters):
            out = fn(x)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - start) / iters)
    return statistics.median(samples)


def main():
    sms = get_device_multiprocessor_count()
    rows = 262_144
    print(jax.devices()[0].device_kind, "SMs", sms, flush=True)
    quick = "--quick" in sys.argv
    for width in (3072,) if quick else (3072, 4608):
        x = jax.random.normal(jax.random.PRNGKey(width), (rows, width), jnp.bfloat16)
        for reuse in (1,) if quick else (1, 6, 9, 18):
            for warps, per_sm, pad in (
                ((8, 1, 160 * 1024),) if quick else ((4, 1, 160 * 1024), (8, 1, 160 * 1024), (4, 2, 64 * 1024))
            ):
                for quantize in (False, True):
                    ctas = sms * per_sm
                    fn = lambda a: producer_probe(
                        a, reuse=reuse, warps=warps, quantize=quantize, ctas=ctas, pad_bytes=pad
                    )  # noqa: E731
                    try:
                        sec = _time(fn, x)
                    except Exception as exc:
                        print(
                            f"FAILED width {width} reuse {reuse} warps {warps}: {type(exc).__name__}: {str(exc)[:2000]}",
                            flush=True,
                        )
                        raise
                    elements = rows * width * reuse
                    tag = "quant" if quantize else "load "
                    print(
                        f"width {width} reuse {reuse:2d} warps {warps} ctas/SM {per_sm} {tag}: {sec * 1e3:8.3f} ms "
                        f"{elements / sec / 1e12:6.2f} T elem/s  {2 * elements / sec / 1e12:6.2f} TB/s",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
