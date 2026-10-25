# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure local expert combine values, pullbacks and memory on TPU."""

import functools
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import levanter.tracker as tracker
import numpy as np

from experiments.grug.belay_tpu.belay_ckpt1_tpu import DATA_ROOT, build_config, require_tpu_rounding, verified_tokenizer
from experiments.grug.belay_tpu.kernel_check import measure_errors
from experiments.grug.belay_tpu.local_experts import combine_local
from experiments.grug.belay_tpu.local_movement import combine_expert_order


def evaluate(combine, rows, order, inverse, weights, gradient):
    output, backward = jax.vjp(lambda x, w: combine(x, order, inverse, w), rows, weights)
    return output, *backward(gradient)


def main():
    require_tpu_rounding()
    run_id = os.environ["BELAY_RUN_ID"]
    config = build_config(
        run_id, f"{DATA_ROOT}/kaiyuew/belay-ckpt1-20261025/{run_id}", verified_tokenizer(), os.environ["BELAY_TPU"], 0
    )
    config.trainer.trainer.initialize()
    if jax.default_backend() != "tpu":
        raise RuntimeError("Combine qualification requires TPU")
    tracker.log_summary({"kernel_check/optimizer_updates": 0})
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    device = jax.local_devices()[0]
    with jax.default_device(device):
        for case, tokens, topk, width in (("mixed", 31, 3, 128), ("empty", 31, 3, 128), ("v432-target", 32768, 10, 256)):
            for dtype, tolerance in ((jnp.float32, 1e-4), (jnp.bfloat16, 2e-2)):
                if case.endswith("target") and dtype == jnp.float32:
                    continue
                rng = np.random.default_rng(391)
                count = tokens * topk
                order = jnp.asarray(np.argsort(rng.integers(384, size=count), kind="stable"), jnp.int32)
                inverse = jnp.argsort(order)
                rows = jnp.asarray(rng.standard_normal((count, width), dtype=np.float32), dtype)
                rows = rows.at[count * 9 // 10 :].set(0)
                if case == "empty":
                    rows = jnp.zeros_like(rows)
                weights = jnp.asarray(rng.random((tokens, topk), dtype=np.float32))
                weights = weights.at[:, 0].set(0)
                gradient = jnp.asarray(rng.standard_normal((tokens, width), dtype=np.float32), dtype)
                inputs = (rows, order, inverse, weights, gradient)
                expected = None
                for name, combine in (("source", combine_local), ("expert_order", combine_expert_order)):
                    function = jax.jit(functools.partial(evaluate, combine))
                    start = time.perf_counter()
                    compiled = function.lower(*inputs).compile()
                    compile_seconds = time.perf_counter() - start
                    actual = jax.block_until_ready(compiled(*inputs))
                    if expected is None:
                        expected = actual
                    deviations = measure_errors(actual, expected, tolerance)
                    passed = all(np.isfinite(row["normalized_max"]) and row["normalized_max"] <= 1 for row in deviations)
                    timings = []
                    for _ in range(9):
                        start = time.perf_counter()
                        jax.block_until_ready(compiled(*inputs))
                        timings.append(time.perf_counter() - start)
                    row = {
                        "stage": "local_movement_comparison",
                        "rank": jax.process_index(),
                        "case": case,
                        "implementation": name,
                        "dtype": str(dtype),
                        "shape": [tokens, topk, width],
                        "passed": passed,
                        "errors": deviations,
                        "compile_seconds": compile_seconds,
                        "median_seconds": float(np.median(timings[2:])),
                        "timings": timings,
                        "memory": str(compiled.memory_analysis()),
                        "git_sha": os.environ["BELAY_SOURCE_COMMIT"],
                    }
                    print(json.dumps(row), flush=True)
                    with (output / "local-movement-check.jsonl").open("a") as file:
                        file.write(json.dumps(row) + "\n")
                    if not passed:
                        raise AssertionError(f"Combine mismatch: {case}/{dtype}/{name}")
        tracker.log_summary({"kernel_check/local_movement_pass": True})
    tracker.current_tracker().finish()


if __name__ == "__main__":
    main()
