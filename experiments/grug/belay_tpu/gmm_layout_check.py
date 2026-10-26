# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure whole-width grouped GEMM tiles without optimizer updates."""

import argparse
import functools
import json
import os
import time

import jax
import jax.numpy as jnp
import levanter.tracker as tracker
import numpy as np
from jax.experimental.pallas.ops.tpu.megablox.gmm import gmm

from experiments.grug.belay_tpu.kernel_check import measure_errors
from experiments.grug.belay_tpu.local_experts import _tiling
from experiments.grug.belay_tpu.operator_check import initialize_operator_check
from experiments.grug.belay_tpu.row_partitioned_gmm import row_partitioned_gmm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-partitions", action="store_true")
    args = parser.parse_args()
    output, operator_topology, reference_topology = initialize_operator_check()
    tracker.log_summary({"kernel_check/input_seed": 420})
    with jax.default_device(jax.local_devices()[0]):
        rows, experts = 327680, 384
        counts = np.full(experts, (rows - 1024) // experts, np.int32)
        counts[: (rows - 1024) % experts] += 1
        counts[-1] += 1024
        groups = jnp.asarray(counts)
        cases = (
            ("forward_up", 256, 384, False),
            ("forward_down", 384, 256, False),
            ("backward_up_input", 384, 256, True),
            ("backward_down_input", 256, 384, True),
        )
        for case, width, out_width, transpose in cases:
            rng = np.random.default_rng(420)
            x = jnp.asarray(0.1 * rng.standard_normal((rows, width), dtype=np.float32), jnp.bfloat16)
            weight_shape = (experts, out_width, width) if transpose else (experts, width, out_width)
            weights = jnp.asarray(0.1 * rng.standard_normal(weight_shape, dtype=np.float32), jnp.bfloat16)
            variants = [("source", _tiling)]
            variants.extend((f"whole_width_{tile}", (tile, width, out_width)) for tile in (128, 256, 512, 1024))
            variants = [
                (name, functools.partial(gmm, preferred_element_type=x.dtype, tiling=tiling, transpose_rhs=transpose))
                for name, tiling in variants
            ]
            if args.row_partitions:
                variants.extend(
                    (
                        f"row_partitions_{count}",
                        functools.partial(row_partitioned_gmm, partitions=count, transpose_rhs=transpose),
                    )
                    for count in (1, 2, 4)
                )
            expected = None
            for name, function in variants:
                start = time.perf_counter()
                compiled = jax.jit(function).lower(x, weights, groups).compile()
                compile_seconds = time.perf_counter() - start
                actual = jax.block_until_ready(compiled(x, weights, groups))
                if expected is None:
                    expected = actual
                deviations = measure_errors([actual], [expected], 2e-2)
                passed = all(np.isfinite(row["normalized_max"]) and row["normalized_max"] <= 1 for row in deviations)
                timings = []
                for _ in range(9):
                    start = time.perf_counter()
                    jax.block_until_ready(compiled(x, weights, groups))
                    timings.append(time.perf_counter() - start)
                record = {
                    "stage": "gmm_layout_comparison",
                    "rank": jax.process_index(),
                    "operator_topology": operator_topology,
                    "reference_topology": reference_topology,
                    "case": case,
                    "implementation": name,
                    "shape": [rows, width, out_width, experts],
                    "transpose_rhs": transpose,
                    "passed": passed,
                    "errors": deviations,
                    "compile_seconds": compile_seconds,
                    "median_seconds": float(np.median(timings[2:])),
                    "timings": timings,
                    "memory": str(compiled.memory_analysis()),
                    "git_sha": os.environ["BELAY_SOURCE_COMMIT"],
                }
                print(json.dumps(record), flush=True)
                with (output / "gmm-layout-check.jsonl").open("a") as file:
                    file.write(json.dumps(record) + "\n")
                if not passed:
                    raise AssertionError(f"Grouped GEMM layout mismatch: {case}/{name}")
    tracker.log_summary({"kernel_check/gmm_layout_pass": True})
    tracker.current_tracker().finish()


if __name__ == "__main__":
    main()
