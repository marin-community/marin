# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify saved-probability TPU attention without optimizer steps."""

import argparse
import functools
import json
import math
import os
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import levanter.tracker as tracker
import numpy as np
from levanter.grug.attention import AttentionMask, reference_attention
from levanter.grug.attention._inkling_relpos import dense_rel_bias

from experiments.grug.belay_tpu.belay_ckpt1_tpu import DATA_ROOT, build_config, require_tpu_rounding, verified_tokenizer
from experiments.grug.belay_tpu.inkling_fused import fused_attention_local
from experiments.grug.belay_tpu.inkling_pallas import pallas_attention_local, score_tiles
from experiments.grug.belay_tpu.inkling_tpu import saved_attention_local
from experiments.grug.belay_tpu.kernel_check import measure_errors


def dense(q, k, v, band, ids):
    return reference_attention(
        q[:, :, None],
        k[:, :, None],
        v[:, :, None],
        AttentionMask(is_causal=True, segment_ids=(ids, ids)),
        logits_dtype=jnp.float32,
        rel_bias=band[:, None],
    )[:, :, 0]


def evaluate(kernel, inputs, ids, cotangent):
    output, backward = jax.vjp(lambda *xs: kernel(*xs, ids), *inputs)
    return (output, *backward(cotangent))


def saved(q, k, v, band, ids):
    return saved_attention_local(q, k, v, band, ids, ids)


def pallas(q, k, v, band, ids):
    return pallas_attention_local(q, k, v, band, ids, ids)


def fused(q, k, v, band, ids):
    return fused_attention_local(q, k, v, band, ids, ids)


def source_scores(q, k, band, ids):
    length = q.shape[1]
    scores = jnp.einsum("bqhd,bkhd->bhqk", q[:, :, None] * (1 / math.sqrt(q.shape[-1])), k[:, :, None])
    scores = scores + dense_rel_bias(band[:, None], length, length)
    valid = AttentionMask(is_causal=True, segment_ids=(ids, ids)).materialize_mask(length, length)
    return jnp.where(valid[:, None], scores, jnp.asarray(-1e9, scores.dtype))[:, 0]


def main():
    require_tpu_rounding()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-source", action="store_true", help="Remeasure the unchanged dense source baseline")
    parser.add_argument("--fused", action="store_true", help="Compare fused attention at the v4-32 local shape")
    args = parser.parse_args()
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
        raise RuntimeError("Attention kernel checks require TPU")
    tracker.log_summary({"kernel_check/optimizer_updates": 0})
    device = jax.local_devices()[0]
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    failures = []
    with jax.default_device(device), jax.default_matmul_precision("highest"):
        for groups, length, width, scenario in (
            (2, 128, 256, "packed"),
            (2, 256, 1152, "strong_logits"),
            (2, 256, 1152, "unpacked"),
            (32 if args.fused else 16, 4096, 1152, "target"),
            (32 if args.fused else 16, 4096, 1152, "target_unpacked"),
        ):
            for dtype, tolerance in ((jnp.float32, 1e-4), (jnp.bfloat16, 2e-2)):
                if length == 4096 and dtype == jnp.float32:
                    continue
                rng = np.random.RandomState(71)
                shape = (groups, length, 128)
                scale = 4 if scenario == "strong_logits" else 1
                q, k = (jnp.asarray(scale * rng.randn(*shape), dtype) for _ in range(2))
                v = jnp.asarray(rng.randn(*shape), dtype)
                band = jnp.asarray(0.2 * rng.randn(groups, length, width), dtype)
                inputs = q, k, v, band
                ids = jnp.asarray(np.broadcast_to(np.arange(length) // 37, (groups, length)), jnp.int32)
                if scenario.endswith("unpacked"):
                    ids = jnp.zeros_like(ids)
                ct = jnp.asarray(rng.randn(*shape), dtype)
                reference = saved if args.fused and length == 4096 else dense
                expected = jax.jit(functools.partial(evaluate, reference))(inputs, ids, ct)
                implementations = [("saved_xla", saved), ("saved_pallas", pallas)]
                if args.fused:
                    implementations.append(("fused_pallas", fused))
                if args.benchmark_source:
                    implementations.insert(0, ("source", dense))
                for name, kernel in implementations:
                    start = time.perf_counter()
                    compiled = jax.jit(functools.partial(evaluate, kernel)).lower(inputs, ids, ct).compile()
                    compile_time = time.perf_counter() - start
                    actual = jax.block_until_ready(compiled(inputs, ids, ct))
                    deviations = measure_errors(actual, expected, tolerance)
                    passed = all(np.isfinite(row["normalized_max"]) and row["normalized_max"] <= 1 for row in deviations)
                    if not passed:
                        failures.append((scenario, str(jnp.dtype(dtype)), name))
                        if length <= 256:
                            diagnostics = {}
                            if name == "saved_pallas":
                                diagnostics = {
                                    "source_scores": np.asarray(jax.jit(source_scores)(q, k, band, ids), np.float32),
                                    "pallas_scores": np.asarray(jax.jit(score_tiles)(q, k, band, ids, ids), np.float32),
                                }
                            np.savez_compressed(
                                output / f"attention-failure-{scenario}-{jnp.dtype(dtype)}-{name}.npz",
                                **{f"actual_{i}": np.asarray(x, np.float32) for i, x in enumerate(actual)},
                                **{f"expected_{i}": np.asarray(x, np.float32) for i, x in enumerate(expected)},
                                **{f"input_{i}": np.asarray(x, np.float32) for i, x in enumerate(inputs)},
                                ids=np.asarray(ids),
                                cotangent=np.asarray(ct, np.float32),
                                **diagnostics,
                            )
                    samples = []
                    for index in range(7):
                        start = time.perf_counter()
                        jax.block_until_ready(compiled(inputs, ids, ct))
                        if index >= 2:
                            samples.append(time.perf_counter() - start)
                    row = {
                        "stage": "kernel_pass" if passed else "kernel_numerical_failure",
                        "kernel": "inkling_attention",
                        "implementation": name,
                        "reference": "saved_xla" if reference is saved else "dense",
                        "rank": jax.process_index(),
                        "shape": shape,
                        "band_width": width,
                        "scenario": scenario,
                        "dtype": str(jnp.dtype(dtype)),
                        "backend": "tpu",
                        "device_type": device.device_kind,
                        "device_count": 1,
                        "block_sizes": {"query": 128, "score_key": 128},
                        "compile_time": compile_time,
                        "steady_state_time": statistics.median(samples),
                        "samples": samples,
                        "error": None if passed else "Frozen numerical bound exceeded; diagnostic timing only",
                        "errors": deviations,
                        "error_leaves": ["output", "dq", "dk", "dv", "dband"],
                        "tolerance": tolerance,
                        "temporary_bytes": compiled.memory_analysis().temp_size_in_bytes,
                        "git_sha": os.environ["BELAY_SOURCE_COMMIT"],
                        "xla_flags": os.environ.get("XLA_FLAGS", ""),
                        "backend_env": {"LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS", "")},
                    }
                    print(json.dumps(row), flush=True)
                    with (output / "attention-kernel-check.jsonl").open("a") as file:
                        file.write(json.dumps(row) + "\n")
    tracker.log_summary({"kernel_check/attention_pass": not failures})
    tracker.current_tracker().finish()
    if failures:
        raise AssertionError(f"Attention qualification failures retained without stopping remaining grid: {failures}")


if __name__ == "__main__":
    main()
