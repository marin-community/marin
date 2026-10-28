# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TPU-only Ckpt 1 access and numerical gates; never performs optimizer updates."""

import json
import math
import os
import time
from pathlib import Path

import draccus
import gcsfs
import jax
import jax.numpy as jnp
import levanter.tracker as tracker
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug.grug_moe import moe_mlp
from levanter.grug.sharding import compact_grug_mesh
from levanter.utils.activation import ActivationFunctionEnum
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.belay_tpu.belay_ckpt1_tpu import (
    DATA_ROOT,
    SOURCE_COMMIT,
    TOKENIZER_HASHES,
    build_config,
    require_tpu_rounding,
    validate_recipe,
    verified_tokenizer,
)
from experiments.grug.moe.kda import chunk_kda, recurrent_kda


def emit(stage: str, **values):
    record = {"stage": stage, "time": time.time(), **values}
    print(json.dumps(record), flush=True)
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "preflight.jsonl").open("a") as file:
        file.write(json.dumps(record) + "\n")


def check_kda():
    """Independent recurrence oracle; tolerances frozen in upstream test_kda.py."""
    rng = np.random.RandomState(0)
    shape = (2, 3, 57, 16)
    q, k, v = (jnp.asarray(rng.randn(*shape), jnp.float32) for _ in range(3))
    g = jnp.asarray(-0.1 * np.abs(rng.randn(*shape)), jnp.float32)
    beta = jnp.asarray(rng.rand(*shape[:-1]), jnp.float32)
    inputs = (q, k, v, g, beta)
    reference = recurrent_kda(*inputs)
    for chunk_size in (16, 32):
        got = chunk_kda(*inputs, chunk_size=chunk_size, matmul_dtype=jnp.float32)
        for actual, expected in zip(got, reference, strict=True):
            np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-4, atol=1e-4)
        actual_bf16, _ = chunk_kda(*inputs, chunk_size=chunk_size, matmul_dtype=jnp.bfloat16)
        np.testing.assert_allclose(np.asarray(actual_bf16), np.asarray(reference[0]), rtol=2e-2, atol=2e-2)

    def loss(kernel, *args):
        out, state = kernel(*args)
        return jnp.mean(out**2) + jnp.mean(state**2)

    grad_args = tuple(range(1, 6))
    ref_grad = jax.grad(loss, argnums=grad_args)(recurrent_kda, *inputs)
    actual_grad = jax.grad(loss, argnums=grad_args)(
        lambda *a: chunk_kda(*a, chunk_size=16, matmul_dtype=jnp.float32), *inputs
    )
    for actual, expected in zip(actual_grad, ref_grad, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-4, atol=1e-4)
    emit("kda_oracle_pass", rank=jax.process_index(), fp32_atol=1e-4, bf16_atol=2e-2)


def source_assignment_mask(selected, valid, factor):
    """Independent EP8/chunks2 capacity oracle in global token order."""
    tokens, topk = selected.shape
    experts, ranks, local_experts, chunks = 384, 8, 48, 2
    physical = max(local_experts, math.ceil(factor * (tokens // ranks) * topk))
    logical = min(physical, max(local_experts, math.ceil(int(valid.sum()) * topk * factor / ranks)))
    capacity = max(local_experts // chunks, (logical + chunks - 1) // chunks)
    flat = np.where(valid[:, None], selected, experts).reshape(-1)
    keep = np.zeros_like(flat, dtype=bool)
    for start in range(0, experts, local_experts // chunks):
        indices = np.flatnonzero((flat >= start) & (flat < start + local_experts // chunks))
        # Stable expert sort retains global token order (and hence the source's
        # sender-major order). This does not call any candidate routing helper.
        ordered = indices[np.argsort(flat[indices], kind="stable")]
        keep[ordered[:capacity]] = True
    return keep.reshape(selected.shape)


def check_ragged(mesh, *, chunks, concentrated=False):
    """Actual TPU collective vs dense ungated ReLU² oracle, including gradients."""
    rng = np.random.RandomState(1)
    tokens, hidden, intermediate, experts, topk = 128, 32, 16, 384, 10
    x_np = rng.randn(tokens, hidden).astype(np.float32)
    selected_np = np.argsort(rng.rand(tokens, experts), axis=-1)[:, :topk].astype(np.int32)
    if concentrated:
        selected_np[:] = np.arange(topk)
    weights_np = np.full((tokens, topk), 1 / topk, np.float32)
    valid_np = np.arange(tokens) % 5 != 3
    factor = 1.05
    accepted_np = source_assignment_mask(selected_np, valid_np, factor)
    up_np = (rng.randn(experts, hidden, intermediate) / np.sqrt(hidden)).astype(np.float32)
    down_np = (rng.randn(experts, intermediate, hidden) / np.sqrt(intermediate)).astype(np.float32)
    cotangent_np = rng.randn(tokens, hidden).astype(np.float32)

    def place(array, spec):
        return jax.make_array_from_callback(array.shape, NamedSharding(mesh, spec), lambda index: array[index])

    with jax.set_mesh(mesh):
        token_spec = P(("data", "expert"), None)
        x, weights, cotangent = (place(a, token_spec) for a in (x_np, weights_np, cotangent_np))
        selected = place(selected_np, token_spec)
        valid = place(valid_np, P(("data", "expert")))
        accepted = place(accepted_np, token_spec)
        up, down = (place(a, P("expert", None, None)) for a in (up_np, down_np))

        def dense(x, up, down, weights, sel, mask):
            # Replicate this small oracle so explicit-sharding gathers cannot
            # silently turn the reference into the candidate's dispatch path.
            x, up, down, weights, sel, mask = (jax.sharding.reshard(a, P()) for a in (x, up, down, weights, sel, mask))
            pre = jnp.einsum("th,tkhi->tki", x, up[sel])
            outputs = jnp.einsum("tki,tkih->tkh", jnp.square(jax.nn.relu(pre)), down[sel])
            return jnp.einsum("tkh,tk->th", outputs, weights * mask)

        def ragged(x, up, down, weights, sel, mask):
            return moe_mlp(
                x,
                sel,
                weights,
                up,
                down,
                token_valid=mask,
                activation=ActivationFunctionEnum.relu2,
                implementation="ragged_all_to_all",
                mesh=mesh,
                capacity_factor=factor,
                report_capacity_overflow=True,
                ragged_expert_chunks=chunks,
            )

        expected = jax.jit(dense)(x, up, down, weights, selected, accepted)
        actual, counts = jax.jit(ragged)(x, up, down, weights, selected, valid)
        jax.block_until_ready(actual)
        expected_dropped = int(valid_np.sum()) * topk - int(accepted_np.sum())
        if int(counts.dropped) != expected_dropped:
            raise AssertionError(f"Capacity mismatch: {int(counts.dropped)} != {expected_dropped}")

        def error(actual, expected):
            actual, expected = (jax.sharding.reshard(a, P()) for a in (actual, expected))
            tolerance = 1e-4 + 1e-4 * jnp.abs(expected)
            return jnp.max(jnp.abs(actual - expected) / tolerance)

        errors = [float(jax.jit(error)(actual, expected))]

        def dense_loss(x, up, down, weights, sel, mask, cot):
            cot = jax.sharding.reshard(cot, P())
            return jnp.sum(dense(x, up, down, weights, sel, mask) * cot)

        def ragged_loss(x, up, down, weights, sel, mask, cot):
            return jnp.sum(ragged(x, up, down, weights, sel, mask)[0] * cot)

        expected_grads = jax.jit(jax.grad(dense_loss, argnums=(0, 1, 2, 3)))(
            x, up, down, weights, selected, accepted, cotangent
        )
        actual_grads = jax.jit(jax.grad(ragged_loss, argnums=(0, 1, 2, 3)))(
            x, up, down, weights, selected, valid, cotangent
        )
        errors.extend(float(jax.jit(error)(a, e)) for a, e in zip(actual_grads, expected_grads, strict=True))
        if not all(np.isfinite(errors)) or max(errors) > 1:
            raise AssertionError(f"Ragged forward/gradient mismatch: normalized errors {errors}")
    emit(
        "ragged_dense_oracle_pass",
        rank=jax.process_index(),
        normalized_errors=errors,
        expert_ranks=mesh.shape["expert"],
        chunks=chunks,
        dropped=expected_dropped,
        concentrated=concentrated,
    )


def main():
    require_tpu_rounding()
    if os.environ.get("JAX_PLATFORMS") != "tpu":
        raise RuntimeError("This preflight requires real TPU execution")
    fs = gcsfs.GCSFileSystem(project="hai-gcp-models")
    tokenizer = verified_tokenizer()
    stats = json.loads(fs.cat(f"{DATA_ROOT}/datakit/hero_tok/v16384_shuf/train/.stats.json"))
    emit("worker_gcs_access_pass", train_stats=stats, tokenizer_sha256=TOKENIZER_HASHES)
    run_id = os.environ["BELAY_RUN_ID"]
    output_root = str(StoragePath(DATA_ROOT) / "kaiyuew/belay-ckpt1-20261025" / run_id)
    tpu = os.environ.get("BELAY_TPU", "v4-32")
    chips = {"v4-32": 16, "v4-64": 32}[tpu]
    config = build_config(run_id, output_root, str(tokenizer), tpu, 0)
    validate_recipe(config)
    config.trainer.trainer.initialize()
    jax.config.update("jax_default_matmul_precision", "highest")
    if jax.default_backend() != "tpu" or jax.device_count() != chips:
        raise AssertionError(f"Expected {chips} physical v4 chips, got {jax.devices()}")
    (Path(os.environ["IRIS_OUTPUT_DIR"]) / "resolved-config.json").write_text(
        json.dumps(draccus.encode(config), indent=2) + "\n"
    )
    tracker.log_configuration(config)
    tracker.log_summary({"preflight/optimizer_updates": 0, "source_commit": SOURCE_COMMIT})
    emit(
        "initialized",
        rank=jax.process_index(),
        devices=jax.device_count(),
        process_count=jax.process_count(),
        device_kind=jax.devices()[0].device_kind,
    )
    check_kda()
    for ranks in (8, 16):
        mesh = compact_grug_mesh(expert_axis_size=ranks, replica_axis_size=1, model_axis_size=chips // ranks)
        for concentrated in (False, True):
            check_ragged(mesh, chunks=16 // ranks, concentrated=concentrated)
    tracker.log_summary({"preflight/access_kda_ragged_pass": True})
    tracker.current_tracker().finish()
    emit("preflight_complete", optimizer_updates=0, rank=jax.process_index())


if __name__ == "__main__":
    main()
