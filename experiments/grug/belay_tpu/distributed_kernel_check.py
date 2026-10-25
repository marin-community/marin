# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check Ckpt 1 kernel sharding and gradients on the actual TPU mesh."""

import functools
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import levanter.tracker as tracker
import numpy as np
from haliax.partitioning import set_mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug.attention import AttentionMask, reference_attention
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.belay_tpu.belay_ckpt1_tpu import DATA_ROOT, build_config, require_tpu_rounding, verified_tokenizer
from experiments.grug.belay_tpu.inkling_tpu import inkling_attention
from experiments.grug.belay_tpu.kda_tpu import chunk_kda_tpu
from experiments.grug.moe.kda import chunk_kda


def attention_forward(q, k, v, band, ids, *, implementation):
    mask = AttentionMask(is_causal=True, segment_ids=(ids, ids))
    if implementation == "source":
        return reference_attention(q, k, v, mask, logits_dtype=jnp.float32, rel_bias=band)
    return inkling_attention(q, k, v, mask, band, implementation=implementation)


def value_gradient(kernel, inputs, ids, cotangents):
    out, backward = jax.vjp(lambda *xs: kernel(*xs, ids), *inputs)
    return out, backward(cotangents)


@jax.jit
def normalized_errors(actual, expected, tolerance):
    rows = []
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        delta = jnp.abs(left.astype(jnp.float32) - right.astype(jnp.float32))
        rows.append(
            jnp.stack(
                (
                    jnp.max(delta),
                    jnp.mean(delta),
                    jnp.max(delta / (tolerance + tolerance * jnp.abs(right.astype(jnp.float32)))),
                )
            )
        )
    return jnp.stack(rows)


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
    if jax.default_backend() != "tpu" or jax.device_count() != 32:
        raise RuntimeError("Distributed kernel gate requires v4-64")
    mesh = compact_grug_mesh(expert_axis_size=16, replica_axis_size=1, model_axis_size=2)
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    tracker.log_summary({"kernel_check/optimizer_updates": 0})
    with set_mesh(mesh), jax.default_matmul_precision("highest"):
        rng = np.random.RandomState(81)

        def place(array, spec, dtype):
            return jax.make_array_from_callback(
                array.shape, NamedSharding(mesh, spec), lambda index: array[index].astype(dtype)
            )

        batch_axes = ("replica_dcn", "data", "expert")
        attn_spec = P(batch_axes, None, "model", None)
        band_spec = P(batch_axes, "model", None, None)
        ids = place(np.broadcast_to(np.arange(128) // 37, (128, 128)), P(batch_axes, None), np.int32)
        kda_spec = P(batch_axes, "model", None, None)
        beta_spec = P(batch_axes, "model", None)
        seg_spec = P(batch_axes, None, None)
        seg = ids[:, None, :]
        for dtype, tolerance in ((jnp.float32, 1e-4), (jnp.bfloat16, 2e-2)):
            shape = (128, 128, 4, 128)
            qkv = tuple(place(rng.randn(*shape), attn_spec, dtype) for _ in range(3))
            band = place(0.2 * rng.randn(128, 4, 128, 256), band_spec, dtype)
            ct = place(rng.randn(*shape), attn_spec, dtype)
            expected = jax.jit(
                functools.partial(value_gradient, functools.partial(attention_forward, implementation="source"))
            )((*qkv, band), ids, ct)
            checks = []
            for implementation in ("xla", "pallas"):
                fn = functools.partial(
                    value_gradient, functools.partial(attention_forward, implementation=implementation)
                )
                checks.append(("attention", implementation, fn, ((*qkv, band), ids, ct), expected))

            kshape = (128, 4, 128, 128)
            qkv_kda = tuple(place(rng.randn(*kshape), kda_spec, dtype) for _ in range(3))
            g = place(-5 * rng.rand(*kshape), kda_spec, np.float32)
            beta = place(rng.rand(*kshape[:-1]), beta_spec, np.float32)
            kinputs = (*qkv_kda, g, beta)
            kct = (
                place(rng.randn(*kshape), kda_spec, np.float32),
                place(rng.randn(128, 4, 128, 128), kda_spec, np.float32),
            )

            def local_kernel(kernel):
                return jax.shard_map(
                    kernel,
                    in_specs=(kda_spec, kda_spec, kda_spec, kda_spec, beta_spec, seg_spec),
                    out_specs=(kda_spec, kda_spec),
                    check_vma=False,
                )

            def source(q, k, v, g, beta, ids, *, matmul_dtype=dtype):
                return chunk_kda(q, k, v, g, beta, chunk_size=16, matmul_dtype=matmul_dtype, segment_ids=ids)

            def candidate(q, k, v, g, beta, ids, *, matmul_dtype=dtype):
                return chunk_kda_tpu(q, k, v, g, beta, matmul_dtype=matmul_dtype, segment_ids=ids)

            expected_kda = jax.jit(functools.partial(value_gradient, local_kernel(source)))(kinputs, seg, kct)
            fn = functools.partial(value_gradient, local_kernel(candidate))
            checks.append(("kda", "pallas", fn, (kinputs, seg, kct), expected_kda))
            for name, implementation, fn, arguments, expected_values in checks:
                jaxpr = str(jax.make_jaxpr(fn)(*arguments))
                if "shard_map" not in jaxpr:
                    raise AssertionError("Accelerator kernel lost its explicit shard_map boundary")
                compiled = jax.jit(fn).lower(*arguments).compile()
                actual = jax.block_until_ready(compiled(*arguments))
                deviations = np.asarray(normalized_errors(actual, expected_values, tolerance))
                if not np.isfinite(deviations).all() or deviations[:, 2].max() > 1:
                    raise AssertionError(f"Distributed {name}/{implementation} gradient mismatch: {deviations}")
                hlo = compiled.as_text()
                if "s32[128,4,128,128]" in hlo:
                    raise AssertionError("Global dense Inkling index materialized in the distributed kernel")
                row = {
                    "stage": "distributed_kernel_pass",
                    "kernel": name,
                    "implementation": implementation,
                    "dtype": str(jnp.dtype(dtype)),
                    "rank": jax.process_index(),
                    "tolerance": tolerance,
                    "errors_max_mean_normalized": deviations.tolist(),
                    "mesh": dict(mesh.shape),
                    "memory_analysis": str(compiled.memory_analysis()),
                }
                print(json.dumps(row), flush=True)
                with (output / "distributed-kernel-check.jsonl").open("a") as file:
                    file.write(json.dumps(row) + "\n")
    tracker.log_summary({"kernel_check/distributed_pass": True})
    tracker.current_tracker().finish()


if __name__ == "__main__":
    main()
