# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify native paged attention on an eight-H100 head mesh without model weights."""

import json
import os
import statistics
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug.attention import ragged_paged_attention

PAGE_SIZE = 128
PAGES = 512
HEADS = 8
GROUPS = 4
HEAD_DIM = 128
TIMED_CALLS = 7
CASES = ((8, 1, 2048), (8, 1, 16384), (8, 1, 64512), (256, 31, 643), (4096, 1024, 51840), (65536, 64512, 64512))


def main() -> None:
    devices = jax.devices()
    if len(devices) != HEADS or any("H100" not in device.device_kind for device in devices):
        raise RuntimeError(f"Expected eight H100 devices, got {devices}")
    mesh = Mesh(np.asarray(devices), ("expert",), axis_types=(AxisType.Explicit,))
    query_sharding = NamedSharding(mesh, P(None, "expert", None, None))
    cache_sharding = NamedSharding(mesh, P(None, None, "expert", None))
    metadata_sharding = NamedSharding(mesh, P())
    output = Path(os.environ["IRIS_OUTPUT_DIR"]) / "gpu-attention.json"
    measurements = []
    with jax.set_mesh(mesh):
        key = jax.random.key(42)
        cache = jax.random.normal(
            key, (PAGES, PAGE_SIZE, 2 * HEADS, HEAD_DIM), jnp.bfloat16, out_sharding=cache_sharding
        )
        page_order = jax.random.permutation(jax.random.fold_in(key, 1), PAGES)
        for q_capacity, q_length, context in CASES:
            q = jax.random.normal(
                jax.random.fold_in(key, q_capacity),
                (q_capacity, HEADS, GROUPS, HEAD_DIM),
                jnp.bfloat16,
                out_sharding=query_sharding,
            )
            live_pages = (context + PAGE_SIZE - 1) // PAGE_SIZE
            indices = jnp.where(jnp.arange(PAGES) < live_pages, page_order, -1)[None]
            metadata = (jnp.array([context]), indices, jnp.array([0, q_length]), jnp.array(1))
            inputs = (q, cache, *(jax.device_put(a, metadata_sharding) for a in metadata))
            jax.block_until_ready(inputs)
            for window in (None, 2048):
                attention = partial(ragged_paged_attention, sm_scale=HEAD_DIM**-0.5, sliding_window=window)
                candidate = jax.jit(partial(attention, implementation="gpu"))
                start = time.perf_counter()
                compiled = candidate.lower(*inputs).compile()
                compile_seconds = time.perf_counter() - start
                start = time.perf_counter()
                actual = compiled(*inputs).block_until_ready()
                first_seconds = time.perf_counter() - start
                residual = None
                if q_capacity <= 256:
                    oracle = jax.jit(partial(attention, implementation="reference"))(*inputs)
                    actual_fp32 = np.asarray(actual, dtype=np.float32)
                    oracle_fp32 = np.asarray(oracle, dtype=np.float32)
                    np.testing.assert_allclose(actual_fp32, oracle_fp32, atol=2e-4, rtol=8e-3)
                    residual = float(np.max(np.abs(actual_fp32 - oracle_fp32)))
                assert bool(jnp.all(jnp.isfinite(actual)))
                assert not bool(jnp.any(actual[q_length:]))
                samples = []
                for _ in range(TIMED_CALLS):
                    start = time.perf_counter()
                    compiled(*inputs).block_until_ready()
                    samples.append(time.perf_counter() - start)
                measurements.append(
                    {
                        "kernel": "snowball_gpu_paged_attention",
                        "implementation": "gpu",
                        "device": devices[0].device_kind,
                        "device_count": len(devices),
                        "jax": jax.__version__,
                        "dtype": str(q.dtype),
                        "shape": {"query_capacity": q_capacity, "query_length": q_length, "context": context},
                        "block_sizes": {"page": PAGE_SIZE, "window": window},
                        "xla_flags": os.environ.get("XLA_FLAGS", ""),
                        "compile_time": compile_seconds,
                        "first_run_time": first_seconds,
                        "steady_state_time": statistics.median(samples),
                        "samples": samples,
                        "max_absolute_difference": residual,
                        "error": None,
                    }
                )
                output.write_text(json.dumps(measurements, indent=2) + "\n")
                print(
                    {"q": q_length, "context": context, "window": window, "median_ms": 1000 * statistics.median(samples)}
                )


if __name__ == "__main__":
    main()
