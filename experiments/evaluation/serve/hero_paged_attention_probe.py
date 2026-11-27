# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure BF16 Hero paged attention on one four-device GB200 node."""

import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug.attention import ragged_paged_attention

_PAGE_SIZE = 128
_KV_HEADS = 12
_QUERY_GROUPS = 4
_HEAD_DIM = 128
_TIMED_CALLS = 7


@dataclass(frozen=True)
class AttentionCase:
    query_capacity: int
    query_length: int
    context: int
    compare_reference: bool


_CASES = (
    AttentionCase(4, 2, 2048, True),
    AttentionCase(4, 2, 65536, True),
    AttentionCase(256, 63, 2048, True),
    AttentionCase(128, 127, 65536, True),
    AttentionCase(65536, 65535, 65535, False),
)


def _run_case(case: AttentionCase, window: int | None, mesh: Mesh) -> dict:
    query_sharding = NamedSharding(mesh, P(None, "model", None, None))
    cache_sharding = NamedSharding(mesh, P(None, None, "model", None))
    metadata_sharding = NamedSharding(mesh, P())
    pages = (case.context + _PAGE_SIZE - 1) // _PAGE_SIZE
    key = jax.random.key(42)
    cache = jax.random.normal(
        key, (pages, _PAGE_SIZE, 2 * _KV_HEADS, _HEAD_DIM), jnp.bfloat16, out_sharding=cache_sharding
    )
    q = jax.random.normal(
        jax.random.fold_in(key, 1),
        (case.query_capacity, _KV_HEADS, _QUERY_GROUPS, _HEAD_DIM),
        jnp.bfloat16,
        out_sharding=query_sharding,
    )
    page_order = jax.random.permutation(jax.random.fold_in(key, 2), pages)
    # Leave a spare slot, as the inference engine does for one prompt with clones.
    table = jnp.stack([page_order, jnp.full_like(page_order, -1)])
    metadata = (jnp.array([case.context, -1]), table, jnp.array([0, case.query_length, -1]), jnp.array(1))
    inputs = (q, cache, *(jax.device_put(a, metadata_sharding) for a in metadata))
    jax.block_until_ready(inputs)
    attention = partial(ragged_paged_attention, sm_scale=_HEAD_DIM**-0.5, sliding_window=window)
    candidate = jax.jit(partial(attention, implementation="gpu"))

    start = time.perf_counter()
    compiled = candidate.lower(*inputs).compile()
    compile_time = time.perf_counter() - start
    start = time.perf_counter()
    actual = compiled(*inputs).block_until_ready()
    first_run = time.perf_counter() - start
    assert bool(jnp.all(jnp.isfinite(actual)))
    assert not bool(jnp.any(actual[case.query_length :]))

    error = None
    reference_seconds = None
    if case.compare_reference:
        reference = jax.jit(partial(attention, implementation="reference")).lower(*inputs).compile()
        expected = reference(*inputs).block_until_ready()
        start = time.perf_counter()
        reference(*inputs).block_until_ready()
        reference_seconds = time.perf_counter() - start
        actual_fp32 = np.asarray(actual, np.float32)
        expected_fp32 = np.asarray(expected, np.float32)
        np.testing.assert_allclose(actual_fp32, expected_fp32, atol=2e-4, rtol=8e-3)
        error = float(np.max(np.abs(actual_fp32 - expected_fp32)))

    timed = []
    for _ in range(_TIMED_CALLS):
        start = time.perf_counter()
        compiled(*inputs).block_until_ready()
        timed.append(time.perf_counter() - start)
    return {
        "kernel": "hero_bf16_paged_attention",
        "implementation": "split_k" if case.query_capacity <= 8 else "fa4",
        "backend": jax.default_backend(),
        "device_type": jax.devices()[0].device_kind,
        "device_count": mesh.size,
        "dtype": str(q.dtype),
        "shape": asdict(case),
        "block_sizes": {"page": _PAGE_SIZE, "window": window, "kv_heads_per_rank": _KV_HEADS // mesh.size},
        "cache_bytes_per_rank": cache.nbytes // mesh.size,
        "compile_time": compile_time,
        "first_run_time": first_run,
        "steady_state_time": statistics.median(timed),
        "reference_steady_state_time": reference_seconds,
        "max_absolute_difference": error,
        "error": None,
        "jax": jax.__version__,
        "git_sha": os.environ.get("IRIS_GIT_COMMIT"),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
    }


def main() -> None:
    devices = jax.devices()
    if len(devices) != 4 or any("GB200" not in device.device_kind for device in devices):
        raise RuntimeError(f"Expected four GB200 devices, got {devices}")
    mesh = Mesh(np.asarray(devices), ("model",), axis_types=(AxisType.Explicit,))
    output = Path(os.environ["IRIS_OUTPUT_DIR"]) / "hero-paged-attention.json"
    measurements = []
    with jax.set_mesh(mesh):
        for case in _CASES:
            for window in (None, 2048):
                row = _run_case(case, window, mesh)
                measurements.append(row)
                output.write_text(json.dumps(measurements, indent=2) + "\n")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
