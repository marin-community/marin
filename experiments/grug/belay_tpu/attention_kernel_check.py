# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify saved-probability TPU attention without optimizer steps."""

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
from levanter.grug.attention import AttentionMask, reference_attention

from experiments.grug.belay_tpu.belay_ckpt1_tpu import DATA_ROOT, build_config, verified_tokenizer
from experiments.grug.belay_tpu.inkling_pallas import pallas_attention_local
from experiments.grug.belay_tpu.inkling_tpu import saved_attention_local
from experiments.grug.belay_tpu.kernel_check import errors


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
        raise RuntimeError("Attention kernel checks require TPU")
    tracker.log_summary({"kernel_check/optimizer_updates": 0})
    device = jax.local_devices()[0]
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    with jax.default_device(device), jax.default_matmul_precision("highest"):
        for groups, length, width, scenario in (
            (2, 128, 256, "packed"),
            (2, 256, 1152, "strong_logits"),
            (16, 4096, 1152, "target"),
        ):
            for dtype, tolerance in ((jnp.float32, 1e-4), (jnp.bfloat16, 2e-2)):
                if scenario == "target" and dtype == jnp.float32:
                    continue
                rng = np.random.RandomState(71)
                shape = (groups, length, 128)
                scale = 4 if scenario == "strong_logits" else 1
                q, k = (jnp.asarray(scale * rng.randn(*shape), dtype) for _ in range(2))
                v = jnp.asarray(rng.randn(*shape), dtype)
                band = jnp.asarray(0.2 * rng.randn(groups, length, width), dtype)
                inputs = q, k, v, band
                ids = jnp.asarray(np.broadcast_to(np.arange(length) // 37, (groups, length)), jnp.int32)
                ct = jnp.asarray(rng.randn(*shape), dtype)
                expected = jax.jit(functools.partial(evaluate, dense))(inputs, ids, ct)
                for name, kernel in (("source", dense), ("saved_xla", saved), ("saved_pallas", pallas)):
                    start = time.perf_counter()
                    compiled = jax.jit(functools.partial(evaluate, kernel)).lower(inputs, ids, ct).compile()
                    compile_time = time.perf_counter() - start
                    actual = jax.block_until_ready(compiled(inputs, ids, ct))
                    deviations = errors(actual, expected, tolerance)
                    samples = []
                    for index in range(7):
                        start = time.perf_counter()
                        jax.block_until_ready(compiled(inputs, ids, ct))
                        if index >= 2:
                            samples.append(time.perf_counter() - start)
                    row = {
                        "stage": "kernel_pass",
                        "kernel": "inkling_attention",
                        "implementation": name,
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
                        "error": None,
                        "errors": deviations,
                        "tolerance": tolerance,
                        "temporary_bytes": compiled.memory_analysis().temp_size_in_bytes,
                        "git_sha": os.environ["BELAY_SOURCE_COMMIT"],
                        "xla_flags": os.environ.get("XLA_FLAGS", ""),
                        "backend_env": {"LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS", "")},
                    }
                    print(json.dumps(row), flush=True)
                    with (output / "attention-kernel-check.jsonl").open("a") as file:
                        file.write(json.dumps(row) + "\n")
    tracker.log_summary({"kernel_check/attention_pass": True})
    tracker.current_tracker().finish()


if __name__ == "__main__":
    main()
