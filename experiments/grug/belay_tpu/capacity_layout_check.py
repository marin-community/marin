# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure exact local-route counting and capacity layout on TPU."""

import functools
import json
import os
import time

import jax
import jax.numpy as jnp
import levanter.tracker as tracker
import numpy as np

from experiments.grug.belay_tpu.capacity_layout import factorized_route_counts, interval_route_layout
from experiments.grug.belay_tpu.local_experts import route_counts, route_layout
from experiments.grug.belay_tpu.operator_check import initialize_operator_check


def evaluate(mode, candidate, flat, counts, accepted):
    count = factorized_route_counts if candidate else route_counts
    layout = interval_route_layout if candidate else route_layout
    if mode == "counts":
        return (count(flat, counts.size),)
    if mode == "combined":
        counts = count(flat, counts.size)
    return (counts, *layout(flat, counts, accepted))


def main():
    output, operator_topology, reference_topology = initialize_operator_check()
    tracker.log_summary({"kernel_check/input_seed": 427})
    failures = []
    with jax.default_device(jax.local_devices()[0]):
        for case in ("balanced", "skewed"):
            rng = np.random.default_rng(427)
            flat = rng.integers(384, size=327680, dtype=np.int32)
            if case == "skewed":
                flat[rng.random(flat.size) < 0.95] = 0
            counts = np.bincount(flat, minlength=384).astype(np.int32)
            accepted = (counts * 0.93).astype(np.int32)
            accepted[::11] = 0
            keep = np.zeros(flat.shape, bool)
            for expert, count in enumerate(accepted):
                keep[np.flatnonzero(flat == expert)[:count]] = True
            order = np.argsort(np.where(keep, flat, 384), kind="stable")
            oracle = (counts, order, np.argsort(order), keep)
            inputs = tuple(jnp.asarray(value) for value in (flat, counts, accepted))
            for mode in ("counts", "layout", "combined"):
                expected = (counts,) if mode == "counts" else oracle
                for candidate in (False, True):
                    function = jax.jit(functools.partial(evaluate, mode, candidate))
                    start = time.perf_counter()
                    compiled = function.lower(*inputs).compile()
                    compile_seconds = time.perf_counter() - start
                    actual = jax.block_until_ready(compiled(*inputs))
                    passed = all(np.array_equal(got, want) for got, want in zip(actual, expected, strict=True))
                    if not passed:
                        failures.append((case, mode, candidate))
                    samples = []
                    for _ in range(9):
                        start = time.perf_counter()
                        jax.block_until_ready(compiled(*inputs))
                        samples.append(time.perf_counter() - start)
                    record = {
                        "stage": "capacity_layout_comparison",
                        "case": case,
                        "mode": mode,
                        "implementation": "intervals_factorized" if candidate else "source",
                        "shape": [32768, 10, 384],
                        "seed": 427,
                        "passed": passed,
                        "exact_integer_oracle": True,
                        "optimizer_updates": 0,
                        "compile_time": compile_seconds,
                        "steady_state_time": float(np.median(samples[2:])),
                        "samples": samples,
                        "temporary_bytes": compiled.memory_analysis().temp_size_in_bytes,
                        "memory": str(compiled.memory_analysis()),
                        "backend": jax.default_backend(),
                        "device_type": jax.local_devices()[0].device_kind,
                        "rank": jax.process_index(),
                        "operator_topology": operator_topology,
                        "reference_topology": reference_topology,
                        "git_sha": os.environ["BELAY_SOURCE_COMMIT"],
                        "xla_flags": os.environ.get("XLA_FLAGS", ""),
                        "backend_env": {"LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS", "")},
                    }
                    print(json.dumps(record), flush=True)
                    with (output / "capacity-layout-check.jsonl").open("a") as file:
                        file.write(json.dumps(record) + "\n")
    tracker.log_summary({"kernel_check/capacity_layout_pass": not failures})
    tracker.current_tracker().finish()
    if failures:
        raise AssertionError(f"Capacity layout failures: {failures}")


if __name__ == "__main__":
    main()
