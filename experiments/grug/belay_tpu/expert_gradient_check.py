# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify expert-partitioned weight gradients on TPU without optimizer steps."""

import functools
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import levanter.tracker as tracker
import numpy as np
from jax.experimental.pallas.ops.tpu.megablox.gmm import tgmm

from experiments.grug.belay_tpu.belay_ckpt1_tpu import DATA_ROOT, build_config, require_tpu_rounding, verified_tokenizer
from experiments.grug.belay_tpu.expert_gradient_tpu import expert_weight_gradient
from experiments.grug.belay_tpu.kernel_check import measure_errors
from experiments.grug.belay_tpu.local_experts import _tiling


def source_gradient(x, dy, groups):
    return tgmm(x.T, dy, groups, preferred_element_type=x.dtype, tiling=_tiling)


def main():
    require_tpu_rounding()
    run_id = os.environ["BELAY_RUN_ID"]
    config = build_config(
        run_id, f"{DATA_ROOT}/kaiyuew/belay-ckpt1-20261025/{run_id}", verified_tokenizer(), os.environ["BELAY_TPU"], 0
    )
    config.trainer.trainer.initialize()
    if jax.default_backend() != "tpu":
        raise RuntimeError("Expert-gradient qualification requires TPU")
    tracker.log_summary({"kernel_check/optimizer_updates": 0})
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    with jax.default_device(jax.local_devices()[0]):
        cases = (
            ("mixed", 128, 128, 128, 4),
            ("empty_half", 128, 128, 128, 4),
            ("up_target", 327680, 256, 384, 384),
            ("down_target", 327680, 384, 256, 384),
        )
        for case, rows, width, out_width, experts in cases:
            for dtype, tolerance in ((jnp.float32, 1e-4), (jnp.bfloat16, 2e-2)):
                if case.endswith("target") and dtype == jnp.float32:
                    continue
                rng = np.random.default_rng(417)
                if case.endswith("target"):
                    active = rows - 1024
                    groups = np.full(experts, active // experts, np.int32)
                    groups[: active % experts] += 1
                    groups[-1] += rows - active
                else:
                    groups = np.asarray([1, 0, 31, 96] if case == "mixed" else [0, 0, 0, 128], np.int32)
                x = jnp.asarray(0.1 * rng.standard_normal((rows, width), dtype=np.float32), dtype)
                dy = jnp.asarray(0.1 * rng.standard_normal((rows, out_width), dtype=np.float32), dtype)
                inputs = (x, dy, jnp.asarray(groups))
                variants = [("source", source_gradient)]
                variants.extend(
                    (
                        f"partition_{count}",
                        functools.partial(expert_weight_gradient, partitions=count, tile_rows=512, output_dtype=dtype),
                    )
                    for count in (1, 2, 4)
                )
                expected = None
                for name, function in variants:
                    precision = "highest" if dtype == jnp.float32 else "default"
                    start = time.perf_counter()
                    with jax.default_matmul_precision(precision):
                        compiled = jax.jit(function).lower(*inputs).compile()
                    compile_seconds = time.perf_counter() - start
                    actual = jax.block_until_ready(compiled(*inputs))
                    if expected is None:
                        expected = actual
                    deviations = measure_errors([actual], [expected], tolerance)
                    passed = all(np.isfinite(r["normalized_max"]) and r["normalized_max"] <= 1 for r in deviations)
                    timings = []
                    for _ in range(9):
                        start = time.perf_counter()
                        jax.block_until_ready(compiled(*inputs))
                        timings.append(time.perf_counter() - start)
                    record = {
                        "stage": "expert_gradient_comparison",
                        "rank": jax.process_index(),
                        "case": case,
                        "implementation": name,
                        "dtype": str(dtype),
                        "shape": [rows, width, out_width, experts],
                        "passed": passed,
                        "errors": deviations,
                        "compile_seconds": compile_seconds,
                        "median_seconds": float(np.median(timings[2:])),
                        "timings": timings,
                        "memory": str(compiled.memory_analysis()),
                        "git_sha": os.environ["BELAY_SOURCE_COMMIT"],
                    }
                    print(json.dumps(record), flush=True)
                    with (output / "expert-gradient-check.jsonl").open("a") as file:
                        file.write(json.dumps(record) + "\n")
                    if not passed:
                        raise AssertionError(f"Expert-gradient mismatch: {case}/{dtype}/{name}")
    tracker.log_summary({"kernel_check/expert_gradient_pass": True})
    tracker.current_tracker().finish()


if __name__ == "__main__":
    main()
