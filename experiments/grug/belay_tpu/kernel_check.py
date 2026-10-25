# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Numerical and memory checks for Ckpt 1 TPU kernels; no optimizer steps."""

import functools
import json
import os
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import levanter.tracker as tracker
import numpy as np

from experiments.grug.belay_tpu.belay_ckpt1_tpu import DATA_ROOT, build_config, verified_tokenizer
from experiments.grug.belay_tpu.kda_tpu import chunk_kda_tpu
from experiments.grug.moe.kda import chunk_kda, recurrent_kda


def evaluate(kernel, inputs, segments, cotangents):
    def forward(*values):
        return kernel(*values[:5], initial_state=values[5], erase=values[6], segment_ids=segments)

    outputs, backward = jax.vjp(forward, *inputs)
    return (*outputs, *backward(cotangents))


def measure_errors(actual, expected, tolerance):
    result = []
    for got, want in zip(actual, expected, strict=True):
        got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
        delta = np.abs(got - want)
        result.append(
            {
                "max_abs": float(delta.max()),
                "mean_abs": float(delta.mean()),
                "normalized_max": float((delta / (tolerance + tolerance * np.abs(want))).max()),
            }
        )
    return result


def errors(actual, expected, tolerance):
    result = measure_errors(actual, expected, tolerance)
    if not all(np.isfinite(row["normalized_max"]) and row["normalized_max"] <= 1 for row in result):
        raise AssertionError(f"Kernel numerical mismatch: {result}")
    return result


def main():
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
        raise RuntimeError("Kernel qualification requires TPU")
    tracker.log_summary({"kernel_check/optimizer_updates": 0})
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    device = jax.local_devices()[0]
    failures = []

    def record(row):
        row.update(
            rank=jax.process_index(),
            backend="tpu",
            device_type=device.device_kind,
            device_count=1,
            git_sha=os.environ["BELAY_SOURCE_COMMIT"],
            xla_flags=os.environ.get("XLA_FLAGS", ""),
            backend_env={"LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS", "")},
        )
        print(json.dumps(row), flush=True)
        with (output / "kernel-check.jsonl").open("a") as file:
            file.write(json.dumps(row) + "\n")

    with jax.default_device(device), jax.default_matmul_precision("highest"):
        for scenario, length in (
            ("packed", 57),
            ("strong_decay", 128),
            ("repeated", 128),
            ("ckpt1_packed", 128),
            ("ckpt1_long", 128),
            ("target", 4096),
            ("ckpt1_target", 4096),
        ):
            shape = (8, 2, length, 128) if length == 4096 else (1, 2, length, 128)
            rng = np.random.RandomState(51)
            for dtype, tolerance in ((jnp.float32, 1e-4), (jnp.bfloat16, 2e-2)):
                if length == 4096 and dtype == jnp.float32:
                    continue
                q, k, v = (jnp.asarray(rng.randn(*shape), dtype) for _ in range(3))
                if scenario == "repeated":
                    k = jnp.broadcast_to(k[..., :1, :], shape)
                gate_scale = 5 if scenario == "strong_decay" else 0.1
                g = jnp.asarray(-gate_scale * rng.rand(*shape), jnp.float32)
                beta = jnp.asarray(rng.rand(*shape[:-1]), jnp.float32)
                initial = jnp.asarray(0.01 * rng.randn(*shape[:2], 128, 128), jnp.float32)
                erase = jnp.asarray(0.5 + rng.rand(*shape), jnp.float32)
                if scenario.startswith("ckpt1"):
                    # Ckpt 1 has no erase gate and starts every sequence at zero.
                    initial, erase = jnp.zeros_like(initial), jnp.ones_like(erase)
                inputs = (q, k, v, g, beta, initial, erase)
                ids = np.searchsorted([7, 16, 17, 33, 56], np.arange(length), side="right")
                if scenario == "ckpt1_long":
                    ids = np.zeros(length, np.int32)
                segments = jnp.asarray(ids[None, None, :], jnp.int32)
                ct = (jnp.asarray(rng.randn(*shape), jnp.float32), jnp.asarray(rng.randn(*initial.shape), jnp.float32))
                source = functools.partial(chunk_kda, chunk_size=16, matmul_dtype=dtype)
                candidate = functools.partial(chunk_kda_tpu, matmul_dtype=dtype)
                expected = jax.jit(functools.partial(evaluate, source))(inputs, segments, ct)
                for name, kernel in (("source", source), ("pallas_saved_state", candidate)):
                    fn = jax.jit(functools.partial(evaluate, kernel))
                    start = time.perf_counter()
                    compiled = fn.lower(inputs, segments, ct).compile()
                    compile_time = time.perf_counter() - start
                    actual = jax.block_until_ready(compiled(inputs, segments, ct))
                    source_errors = measure_errors(actual, expected, tolerance)
                    oracle_errors = None
                    if length < 200:
                        oracle = jax.jit(functools.partial(evaluate, recurrent_kda))(inputs, segments, ct)
                        oracle_errors = measure_errors(actual, oracle, tolerance)
                    passed = all(
                        np.isfinite(row["normalized_max"]) and row["normalized_max"] <= 1
                        for row in [*source_errors, *(oracle_errors or [])]
                    )
                    if not passed:
                        failures.append((scenario, str(jnp.dtype(dtype)), name))
                    samples = []
                    for index in range(7):
                        start = time.perf_counter()
                        jax.block_until_ready(compiled(inputs, segments, ct))
                        if index >= 2:
                            samples.append(time.perf_counter() - start)
                    record(
                        {
                            "stage": "kernel_pass" if passed else "kernel_numerical_failure",
                            "kernel": "kda",
                            "implementation": name,
                            "shape": shape,
                            "dtype": str(jnp.dtype(dtype)),
                            "scenario": scenario,
                            "error_leaves": [
                                "output",
                                "final_state",
                                "dq",
                                "dk",
                                "dv",
                                "dg",
                                "dbeta",
                                "dinitial",
                                "derase",
                            ],
                            "block_sizes": {"chunk": 16, "state": [128, 128]},
                            "compile_time": compile_time,
                            "steady_state_time": statistics.median(samples),
                            "samples": samples,
                            "error": None if passed else "Frozen numerical bound exceeded; diagnostic timing only",
                            "source_errors": source_errors,
                            "oracle_errors": oracle_errors,
                            "tolerance": tolerance,
                            "temporary_bytes": compiled.memory_analysis().temp_size_in_bytes,
                        }
                    )
    tracker.log_summary({"kernel_check/kda_pass": not failures})
    tracker.current_tracker().finish()
    if failures:
        raise AssertionError(f"KDA qualification failures retained without stopping the remaining grid: {failures}")


if __name__ == "__main__":
    main()
