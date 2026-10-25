# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify saved-logit cross-entropy against Ckpt 1's streaming TPU loss head."""

import functools
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import levanter.tracker as tracker
import numpy as np
from levanter.kernels.pallas.fused_cross_entropy_loss import BlockSizes
from levanter.kernels.pallas.fused_cross_entropy_loss.xla import linear_softmax_cross_entropy_loss_xla

from experiments.grug.belay_tpu.belay_ckpt1_tpu import DATA_ROOT, build_config, require_tpu_rounding, verified_tokenizer
from experiments.grug.belay_tpu.kernel_check import measure_errors
from experiments.grug.belay_tpu.saved_ce_tpu import saved_linear_cross_entropy


def evaluate(function, x, labels, w, cotangent):
    output, pullback = jax.vjp(lambda a, b: function(a, labels, b), x, w)
    return output, pullback(cotangent)


def main():
    require_tpu_rounding()
    run_id = os.environ["BELAY_RUN_ID"]
    config = build_config(
        run_id, f"{DATA_ROOT}/kaiyuew/belay-ckpt1-20261025/{run_id}", verified_tokenizer(), os.environ["BELAY_TPU"], 0
    )
    config.trainer.trainer.initialize()
    if jax.default_backend() != "tpu":
        raise RuntimeError("Saved-logit cross-entropy qualification requires TPU")
    tracker.log_summary({"kernel_check/optimizer_updates": 0})
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    with jax.default_device(jax.local_devices()[0]), jax.default_matmul_precision("highest"):
        for case, rows, hidden, vocab, tile, scale in (
            ("small", 128, 128, 384, 128, 0.1),
            ("strong", 128, 128, 384, 128, 2.0),
            ("target", 32768, 512, 16384, 8192, 0.1),
        ):
            for dtype, tolerance in ((jnp.float32, 1e-4), (jnp.bfloat16, 2e-2)):
                if case == "target" and dtype == jnp.float32:
                    continue
                rng = np.random.RandomState(182)
                x = jnp.asarray(rng.normal(size=(rows, hidden)), dtype)
                w = jnp.asarray(scale * rng.normal(size=(hidden, vocab)), dtype)
                labels = jnp.asarray(rng.randint(0, vocab, size=rows), jnp.int32)
                g_loss = jnp.asarray(rng.normal(size=rows), jnp.float32).at[::3].set(0)
                g_lse = jnp.asarray(rng.normal(size=rows), jnp.float32)
                source = functools.partial(
                    linear_softmax_cross_entropy_loss_xla,
                    block_sizes=BlockSizes(b_block_size=rows, v_block_size=tile),
                    fast_backward=True,
                    precision="highest",
                    logit_soft_cap=10.0,
                )
                candidate = functools.partial(
                    saved_linear_cross_entropy, block_size=tile, precision="highest", logit_soft_cap=10.0
                )
                inputs = (x, labels, w, (g_loss, g_lse))
                expected = None
                for name, function in (("source", source), ("saved", candidate)):
                    start = time.perf_counter()
                    compiled = jax.jit(functools.partial(evaluate, function)).lower(*inputs).compile()
                    compile_seconds = time.perf_counter() - start
                    actual = jax.block_until_ready(compiled(*inputs))
                    if expected is None:
                        expected = actual
                    errors = measure_errors(actual, expected, tolerance)
                    passed = all(np.isfinite(row["normalized_max"]) and row["normalized_max"] <= 1 for row in errors)
                    timings = []
                    for _ in range(7):
                        start = time.perf_counter()
                        jax.block_until_ready(compiled(*inputs))
                        timings.append(time.perf_counter() - start)
                    row = {
                        "stage": "saved_ce_comparison",
                        "rank": jax.process_index(),
                        "case": case,
                        "implementation": name,
                        "dtype": str(dtype),
                        "shape": [rows, hidden, vocab],
                        "passed": passed,
                        "errors": errors,
                        "compile_seconds": compile_seconds,
                        "median_seconds": float(np.median(timings[2:])),
                        "timings": timings,
                        "memory": str(compiled.memory_analysis()),
                    }
                    print(json.dumps(row), flush=True)
                    with (output / "saved-ce-check.jsonl").open("a") as file:
                        file.write(json.dumps(row) + "\n")
                    if not passed:
                        raise AssertionError(f"Saved-logit loss-head mismatch: {case}/{dtype}/{name}")
        tracker.log_summary({"kernel_check/saved_ce_pass": True})
    tracker.current_tracker().finish()


if __name__ == "__main__":
    main()
