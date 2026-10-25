# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify state DMA tiles against the already validated C=16 TPU KDA path."""

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
from experiments.grug.belay_tpu.kda_tpu import chunk_kda_tpu
from experiments.grug.belay_tpu.kernel_check import evaluate, measure_errors


def main():
    require_tpu_rounding()
    run_id = os.environ["BELAY_RUN_ID"]
    config = build_config(
        run_id,
        f"{DATA_ROOT}/kaiyuew/belay-ckpt1-20261025/{run_id}",
        verified_tokenizer(),
        os.environ["BELAY_TPU"],
        0,
    )
    config.trainer.trainer.initialize()
    if jax.default_backend() != "tpu":
        raise RuntimeError("KDA DMA tiling requires TPU qualification")
    tracker.log_summary({"kernel_check/optimizer_updates": 0})
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    device = jax.local_devices()[0]
    failures = []

    def emit(row):
        row.update(rank=jax.process_index(), device=device.device_kind, source_commit=os.environ["BELAY_SOURCE_COMMIT"])
        print(json.dumps(row), flush=True)
        with (output / "kda-tile-check.jsonl").open("a") as file:
            file.write(json.dumps(row) + "\n")

    with jax.default_device(device), jax.default_matmul_precision("highest"):
        for case, length in (("packed", 128), ("strong_decay", 128), ("ckpt1_target", 4096)):
            shape = (8, 2, length, 128) if length == 4096 else (1, 2, length, 128)
            for dtype, tolerance in ((jnp.float32, 1e-4), (jnp.bfloat16, 2e-2)):
                if length == 4096 and dtype == jnp.float32:
                    continue
                rng = np.random.RandomState(151)
                q, k, v = (jnp.asarray(rng.normal(size=shape), dtype) for _ in range(3))
                decay = jnp.asarray(-(5 if case == "strong_decay" else 0.1) * rng.rand(*shape), jnp.float32)
                beta = jnp.asarray(rng.rand(*shape[:-1]), jnp.float32)
                initial = jnp.asarray(0.01 * rng.normal(size=(*shape[:2], 128, 128)), jnp.float32)
                erase = jnp.asarray(0.5 + rng.rand(*shape), jnp.float32)
                if case == "ckpt1_target":
                    initial, erase = jnp.zeros_like(initial), jnp.ones_like(erase)
                ids = np.searchsorted([7, 16, 17, 33, 56], np.arange(length), side="right")
                segments = jnp.asarray(ids[None, None, :], jnp.int32)
                inputs = (q, k, v, decay, beta, initial, erase)
                cotangents = (
                    jnp.asarray(rng.normal(size=shape), jnp.float32),
                    jnp.asarray(rng.normal(size=initial.shape), jnp.float32),
                )
                expected = None
                for tile in (1, 2, 4, 8, 16):
                    if (length // 16) % tile:
                        continue
                    kernel = functools.partial(chunk_kda_tpu, matmul_dtype=dtype, state_tile=tile)
                    function = jax.jit(functools.partial(evaluate, kernel))
                    start = time.perf_counter()
                    compiled = function.lower(inputs, segments, cotangents).compile()
                    compile_seconds = time.perf_counter() - start
                    actual = jax.block_until_ready(compiled(inputs, segments, cotangents))
                    if expected is None:
                        expected = actual
                    deviations = measure_errors(actual, expected, tolerance)
                    passed = all(np.isfinite(row["normalized_max"]) and row["normalized_max"] <= 1 for row in deviations)
                    if not passed:
                        failures.append((case, str(dtype), tile))
                    samples = []
                    for _ in range(7):
                        start = time.perf_counter()
                        jax.block_until_ready(compiled(inputs, segments, cotangents))
                        samples.append(time.perf_counter() - start)
                    emit(
                        {
                            "stage": "kda_tile_comparison",
                            "case": case,
                            "shape": shape,
                            "dtype": str(dtype),
                            "tile": tile,
                            "source_parity_pass": passed,
                            "errors": deviations,
                            "compile_seconds": compile_seconds,
                            "median_seconds": float(np.median(samples[2:])),
                            "samples": samples,
                            "memory": str(compiled.memory_analysis()),
                        }
                    )
        tracker.log_summary({"kernel_check/kda_tile_pass": not failures})
    tracker.current_tracker().finish()
    if failures:
        raise AssertionError(f"KDA DMA tiling failures: {failures}")


if __name__ == "__main__":
    main()
