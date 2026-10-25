# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure BF16 MoE row-gather representations on TPU without training."""

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
from experiments.grug.belay_tpu.gather_layout import fp32_gather, packed_gather


def main():
    require_tpu_rounding()
    run_id = os.environ["BELAY_RUN_ID"]
    config = build_config(
        run_id, f"{DATA_ROOT}/kaiyuew/belay-ckpt1-20261025/{run_id}", verified_tokenizer(), os.environ["BELAY_TPU"], 0
    )
    config.trainer.trainer.initialize()
    if jax.default_backend() != "tpu":
        raise RuntimeError("Gather layout qualification requires TPU")
    tracker.log_summary({"kernel_check/optimizer_updates": 0})
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    variants = (
        ("source", lambda rows, ids: rows[ids]),
        ("packed", packed_gather),
        ("packed_barrier", functools.partial(packed_gather, barrier=True)),
        ("fp32_barrier", fp32_gather),
    )
    with jax.default_device(jax.local_devices()[0]):
        rng = np.random.default_rng(419)
        tokens, topk, width = 32768, 10, 256
        order = np.argsort(rng.integers(384, size=tokens * topk), kind="stable").astype(np.int32)
        inverse = np.argsort(order).astype(np.int32)
        cases = (
            ("dispatch", tokens, order // topk),
            ("restore", tokens * topk, inverse),
            ("backward_order", tokens * topk, order),
        )
        for case, count, indices in cases:
            rows = jnp.asarray(rng.standard_normal((count, width), dtype=np.float32), jnp.bfloat16)
            ids = jnp.asarray(indices)
            expected = None
            for name, function in variants:
                start = time.perf_counter()
                compiled = jax.jit(function).lower(rows, ids).compile()
                compile_seconds = time.perf_counter() - start
                actual = jax.block_until_ready(compiled(rows, ids))
                actual_bits = jax.lax.bitcast_convert_type(actual, jnp.uint16)
                if expected is None:
                    expected = actual_bits
                passed = bool(jnp.all(actual_bits == expected))
                timings = []
                for _ in range(11):
                    start = time.perf_counter()
                    jax.block_until_ready(compiled(rows, ids))
                    timings.append(time.perf_counter() - start)
                record = {
                    "stage": "gather_layout_comparison",
                    "rank": jax.process_index(),
                    "case": case,
                    "implementation": name,
                    "shape": list(rows.shape),
                    "output_rows": len(indices),
                    "passed": passed,
                    "compile_seconds": compile_seconds,
                    "median_seconds": float(np.median(timings[3:])),
                    "timings": timings,
                    "memory": str(compiled.memory_analysis()),
                    "git_sha": os.environ["BELAY_SOURCE_COMMIT"],
                }
                print(json.dumps(record), flush=True)
                with (output / "gather-layout-check.jsonl").open("a") as file:
                    file.write(json.dumps(record) + "\n")
                if not passed:
                    raise AssertionError(f"Gather layout changed output bits: {case}/{name}")
    tracker.log_summary({"kernel_check/gather_layout_pass": True})
    tracker.current_tracker().finish()


if __name__ == "__main__":
    main()
