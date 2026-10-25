# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TPU-only Ckpt 1 access and numerical gates; never performs optimizer updates."""

import hashlib
import json
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

from experiments.grug.fast_track.belay_ckpt1_tpu import DATA_ROOT, SOURCE_COMMIT, build_config, validate_recipe
from experiments.grug.moe.kda import chunk_kda, recurrent_kda

TOKENIZER_HASHES = {
    "tokenizer.json": "adaba3dc92dff6eed642f10101e4183e0525437f3432d2ce004cd14232b350ae",
    "tokenizer_config.json": "1ff46b7c04e1ef0335286b733db65c663261132d0a9866842fd48020255ea05e",
}


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


def check_ragged(mesh):
    """Actual TPU collective vs dense ungated ReLU² oracle, including gradients."""
    rng = np.random.RandomState(1)
    tokens, hidden, intermediate, experts, topk = 128, 32, 16, 384, 10
    x_np = rng.randn(tokens, hidden).astype(np.float32)
    selected_np = np.argsort(rng.rand(tokens, experts), axis=-1)[:, :topk].astype(np.int32)
    weights_np = np.full((tokens, topk), 1 / topk, np.float32)
    valid_np = np.arange(tokens) % 5 != 3
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
        up, down = (place(a, P("expert", None, None)) for a in (up_np, down_np))

        def dense(x, up, down, weights):
            # Replicate this small oracle so explicit-sharding gathers cannot
            # silently turn the reference into the candidate's dispatch path.
            x, up, down, weights, sel, mask = (
                jax.sharding.reshard(a, P()) for a in (x, up, down, weights, selected, valid)
            )
            pre = jnp.einsum("th,tkhi->tki", x, up[sel])
            outputs = jnp.einsum("tki,tkih->tkh", jnp.square(jax.nn.relu(pre)), down[sel])
            return jnp.einsum("tkh,tk->th", outputs, weights * mask[:, None])

        def ragged(x, up, down, weights):
            return moe_mlp(
                x,
                selected,
                weights,
                up,
                down,
                token_valid=valid,
                activation=ActivationFunctionEnum.relu2,
                implementation="ragged_all_to_all",
                mesh=mesh,
                capacity_factor=4.0,
                report_capacity_overflow=True,
                ragged_expert_chunks=2,
            )

        expected = jax.jit(dense)(x, up, down, weights)
        actual, counts = jax.jit(ragged)(x, up, down, weights)
        jax.block_until_ready(actual)
        if int(counts.dropped) != 0:
            raise AssertionError("No-drop oracle fixture overflowed")

        def error(actual, expected):
            actual, expected = (jax.sharding.reshard(a, P()) for a in (actual, expected))
            tolerance = 1e-4 + 1e-4 * jnp.abs(expected)
            return jnp.max(jnp.abs(actual - expected) / tolerance)

        errors = [float(jax.jit(error)(actual, expected))]
        args = (x, up, down, weights)
        expected_grads = jax.jit(jax.grad(lambda *a: jnp.sum(dense(*a) * cotangent), argnums=(0, 1, 2, 3)))(*args)
        actual_grads = jax.jit(jax.grad(lambda *a: jnp.sum(ragged(*a)[0] * cotangent), argnums=(0, 1, 2, 3)))(*args)
        errors.extend(float(jax.jit(error)(a, e)) for a, e in zip(actual_grads, expected_grads, strict=True))
        if not all(np.isfinite(errors)) or max(errors) > 1:
            raise AssertionError(f"Ragged forward/gradient mismatch: normalized errors {errors}")
    emit("ragged_dense_oracle_pass", rank=jax.process_index(), normalized_errors=errors)


def main():
    if os.environ.get("JAX_PLATFORMS") != "tpu":
        raise RuntimeError("This preflight requires real TPU execution")
    fs = gcsfs.GCSFileSystem(project="hai-gcp-models")
    tokenizer = Path("/tmp/belay-ckpt1-tokenizer")
    tokenizer.mkdir(exist_ok=True)
    for name, expected in TOKENIZER_HASHES.items():
        data = fs.cat(f"{DATA_ROOT}/tokenizers/hero-bpe-v16384/hf-hub-1.21.0/{name}")
        if hashlib.sha256(data).hexdigest() != expected:
            raise AssertionError(f"Tokenizer changed: {name}")
        (tokenizer / name).write_bytes(data)
    stats = json.loads(fs.cat(f"{DATA_ROOT}/datakit/hero_tok/v16384_shuf/train/.stats.json"))
    emit("worker_gcs_access_pass", train_stats=stats, tokenizer_sha256=TOKENIZER_HASHES)
    run_id = os.environ["BELAY_RUN_ID"]
    output_root = f"{DATA_ROOT}/kaiyuew/belay-ckpt1-20261025/{run_id}"
    config = build_config(run_id, output_root, str(tokenizer), "v4-32", 0)
    validate_recipe(config)
    config.trainer.trainer.initialize()
    jax.config.update("jax_default_matmul_precision", "highest")
    if jax.default_backend() != "tpu" or jax.device_count() != 16:
        raise AssertionError(f"Expected 16 physical v4 chips, got {jax.devices()}")
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
    mesh = compact_grug_mesh(expert_axis_size=8, replica_axis_size=1, model_axis_size=2)
    check_ragged(mesh)
    tracker.log_summary({"preflight/access_kda_ragged_pass": True})
    tracker.current_tracker().finish()
    emit("preflight_complete", optimizer_updates=0, rank=jax.process_index())


if __name__ == "__main__":
    main()
