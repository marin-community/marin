# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure output-combination fusion with and without producer conversion."""

import json
import os
import time

import jax
import jax.numpy as jnp
import levanter.tracker as tracker
import numpy as np

from experiments.grug.belay_tpu.feature_combine_tpu import combine_feature_major, combine_with_transpose
from experiments.grug.belay_tpu.kernel_check import measure_errors
from experiments.grug.belay_tpu.local_experts import combine_local
from experiments.grug.belay_tpu.operator_check import initialize_operator_check


def main():
    output, operator_topology, reference_topology = initialize_operator_check()
    tracker.log_summary({"kernel_check/input_seed": 422})
    with jax.default_device(jax.local_devices()[0]):
        for case, tokens, topk, features, dtype in (
            ("small", 128, 10, 16, jnp.float32),
            ("v432-target", 32768, 10, 256, jnp.bfloat16),
        ):
            rng = np.random.default_rng(422)
            order = jnp.asarray(np.argsort(rng.integers(384, size=tokens * topk), stable=True), jnp.int32)
            inverse = jnp.argsort(order)
            rows = jnp.asarray(rng.normal(size=(tokens * topk, features)), dtype).at[-31:].set(0)
            weights = jnp.asarray(rng.random((tokens, topk)), jnp.float32).at[:, 0].set(0)
            transposed = jax.block_until_ready(jax.jit(lambda x: x.T)(rows))
            positions = inverse.reshape(weights.shape).T

            def source(x, w, order=order, inverse=inverse):
                return combine_local(x, order, inverse, w)

            def with_conversion(x, w, order=order, inverse=inverse):
                return combine_with_transpose(x, order, inverse, w)

            def pretransposed(x, w, positions=positions):
                return combine_feature_major(x, positions, w.T).T

            expected = None
            for name, function, data in (
                ("source", source, rows),
                ("fused_with_conversion", with_conversion, rows),
                ("fused_pretransposed", pretransposed, transposed),
            ):
                start = time.perf_counter()
                compiled = jax.jit(function).lower(data, weights).compile()
                compile_seconds = time.perf_counter() - start
                actual = jax.block_until_ready(compiled(data, weights))
                if expected is None:
                    expected = actual
                tolerance = 2e-2 if dtype == jnp.bfloat16 else 1e-5
                errors = measure_errors([actual], [expected], tolerance)
                passed = all(np.isfinite(row["normalized_max"]) and row["normalized_max"] <= 1 for row in errors)
                timings = []
                for _ in range(9):
                    start = time.perf_counter()
                    jax.block_until_ready(compiled(data, weights))
                    timings.append(time.perf_counter() - start)
                record = {
                    "stage": "feature_combine_comparison",
                    "rank": jax.process_index(),
                    "case": case,
                    "implementation": name,
                    "operator_topology": operator_topology,
                    "reference_topology": reference_topology,
                    "shape": [tokens, topk, features],
                    "dtype": str(dtype),
                    "passed": passed,
                    "errors": errors,
                    "compile_seconds": compile_seconds,
                    "median_seconds": float(np.median(timings[2:])),
                    "timings": timings,
                    "memory": str(compiled.memory_analysis()),
                    "git_sha": os.environ["BELAY_SOURCE_COMMIT"],
                }
                print(json.dumps(record), flush=True)
                with (output / "feature-combine-check.jsonl").open("a") as file:
                    file.write(json.dumps(record) + "\n")
                if not passed:
                    raise AssertionError(f"Feature-major combine mismatch: {case}/{name}")
    tracker.log_summary({"kernel_check/feature_combine_pass": True})
    tracker.current_tracker().finish()


if __name__ == "__main__":
    main()
