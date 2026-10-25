# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check saved expert pullbacks against the source recomputing backward on TPU."""

import argparse
import functools
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import levanter.tracker as tracker
import numpy as np
from levanter.grug._moe.ep_ragged_all_to_all import _RaggedDotExpertMlp, _SavedRaggedDotExpertMlp

from experiments.grug.belay_tpu.belay_ckpt1_tpu import DATA_ROOT, build_config, require_tpu_rounding, verified_tokenizer
from experiments.grug.belay_tpu.kernel_check import measure_errors
from experiments.grug.belay_tpu.saved_relu2_adapter import SavedRelu2ExpertMlp


def relu2(x):
    return jnp.square(jax.nn.relu(x))


def evaluate(expert_mlp, inputs, cotangent):
    out, residuals = expert_mlp.forward(*inputs)
    return out, *expert_mlp.backward(residuals, cotangent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused-relu2", action="store_true")
    args = parser.parse_args()
    require_tpu_rounding()
    run_id = os.environ["BELAY_RUN_ID"]
    config = build_config(
        run_id, f"{DATA_ROOT}/kaiyuew/belay-ckpt1-20261025/{run_id}", verified_tokenizer(), os.environ["BELAY_TPU"], 0
    )
    config.trainer.trainer.initialize()
    if jax.default_backend() != "tpu":
        raise RuntimeError("Saved expert pullback qualification requires TPU")
    tracker.log_summary({"kernel_check/optimizer_updates": 0})
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    device = jax.local_devices()[0]
    with jax.default_device(device):
        cases = (
            ("padding", "empty", "skewed", "target", "local-target", "local-v432-target")
            if args.fused_relu2
            else ("padding", "empty", "skewed", "target")
        )
        for case in cases:
            rows, width, hidden, experts = (344064, 256, 384, 24) if case == "target" else (128, 128, 128, 3)
            if case == "local-target":
                rows, width, hidden, experts = 163840, 256, 384, 384
            if case == "local-v432-target":
                rows, width, hidden, experts = 327680, 256, 384, 384
            for dtype, tolerance in ((jnp.float32, 1e-4), (jnp.bfloat16, 2e-2)):
                if case.endswith("target") and dtype == jnp.float32:
                    continue
                rng = np.random.RandomState(171)
                active_host = np.asarray([32, 48, 0], np.int32)
                if case == "empty":
                    active_host.fill(0)
                elif case == "skewed":
                    active_host = np.asarray([0, 128, 0], np.int32)
                elif case.endswith("target"):
                    active_rows = 327680 if case == "target" else rows - 1024
                    active_host = np.full(experts, active_rows // experts, np.int32)
                    active_host[: active_rows % experts] += 1
                active = jnp.asarray(active_host)
                physical = active.at[-1].add(rows - active.sum())
                x = jnp.asarray(rng.normal(size=(rows, width)), dtype)
                x = x.at[int(active_host.sum()) :].set(jnp.nan)
                up = jnp.asarray(0.1 * rng.normal(size=(experts, width, hidden)), dtype)
                down = jnp.asarray(0.1 * rng.normal(size=(experts, hidden, width)), dtype)
                cotangent = jnp.asarray(rng.normal(size=x.shape), dtype)
                inputs = (x, up, down, physical, active)
                expected = None
                implementations = [
                    ("source", _RaggedDotExpertMlp(relu2)),
                    ("saved", _SavedRaggedDotExpertMlp(relu2)),
                ]
                if args.fused_relu2:
                    implementations.append(("fused_saved", SavedRelu2ExpertMlp()))
                for implementation, expert_mlp in implementations:
                    function = jax.jit(functools.partial(evaluate, expert_mlp))
                    start = time.perf_counter()
                    # TPU BF16 MXU inputs use the training precision. Requesting
                    # FP32 contract precision for BF16 Mosaic matmuls is invalid.
                    precision = "highest" if dtype == jnp.float32 else "default"
                    with jax.default_matmul_precision(precision):
                        compiled = function.lower(inputs, cotangent).compile()
                    compile_seconds = time.perf_counter() - start
                    actual = jax.block_until_ready(compiled(inputs, cotangent))
                    if expected is None:
                        expected = actual
                    deviations = measure_errors(actual, expected, tolerance)
                    passed = all(np.isfinite(row["normalized_max"]) and row["normalized_max"] <= 1 for row in deviations)
                    timings = []
                    for _ in range(7):
                        start = time.perf_counter()
                        jax.block_until_ready(compiled(inputs, cotangent))
                        timings.append(time.perf_counter() - start)
                    row = {
                        "stage": "saved_moe_comparison",
                        "rank": jax.process_index(),
                        "case": case,
                        "implementation": implementation,
                        "dtype": str(dtype),
                        "matmul_precision": precision,
                        "shape": [rows, width, hidden, experts],
                        "passed": passed,
                        "errors": deviations,
                        "compile_seconds": compile_seconds,
                        "median_seconds": float(np.median(timings[2:])),
                        "timings": timings,
                        "memory": str(compiled.memory_analysis()),
                    }
                    print(json.dumps(row), flush=True)
                    with (output / "saved-moe-check.jsonl").open("a") as file:
                        file.write(json.dumps(row) + "\n")
                    if not passed:
                        raise AssertionError(f"Saved MoE pullback mismatch: {case}/{dtype}/{implementation}")
        tracker.log_summary({"kernel_check/saved_moe_pass": True})
    tracker.current_tracker().finish()


if __name__ == "__main__":
    main()
