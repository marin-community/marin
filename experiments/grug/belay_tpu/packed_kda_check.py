# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check packed-document KDA at the Ckpt 1 head width before full training."""

import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import levanter.tracker as tracker
import numpy as np

from experiments.grug.belay_tpu.kda_tpu import chunk_kda_tpu
from experiments.grug.moe.kda import chunk_kda, recurrent_kda


def check_packed_kda(*, state_precision=jax.lax.Precision.HIGHEST):
    if jax.default_backend() != "tpu":
        raise RuntimeError("Packed KDA numerical validation requires TPU")
    rng = np.random.RandomState(41)
    shape = (2, 4, 57, 128)
    inputs = tuple(jnp.asarray(rng.randn(*shape), jnp.float32) for _ in range(3))
    inputs += (
        jnp.asarray(-0.1 * np.abs(rng.randn(*shape)), jnp.float32),
        jnp.asarray(rng.rand(*shape[:-1]), jnp.float32),
    )
    # Boundaries inside chunks, on a boundary, and beside the padded tail.
    ids = np.searchsorted([7, 16, 17, 33, 56], np.arange(57), side="right")
    segments = jnp.asarray(np.broadcast_to(ids, (2, 1, 57)), jnp.int32)

    def reference(q, k, v, g, beta, seg):
        return recurrent_kda(q, k, v, g, beta, segment_ids=seg)

    def objective(kernel, *args):
        out, state = kernel(*args)
        return jnp.mean(out**2) + jnp.mean(state**2)

    # This local precision context does not change the subsequent training run.
    with jax.default_matmul_precision("highest"):
        expected = jax.jit(reference)(*inputs, segments)
        reference_grads = jax.jit(jax.grad(lambda *a: objective(reference, *a), argnums=(0, 1, 2, 3, 4)))(
            *inputs, segments
        )
        for implementation in ("source", "pallas"):
            for dtype, tolerance in ((jnp.float32, 1e-4), (jnp.bfloat16, 2e-2)):

                def candidate(q, k, v, g, beta, seg, *, matmul_dtype=dtype, implementation=implementation):
                    if implementation == "pallas":
                        return chunk_kda_tpu(
                            q, k, v, g, beta, segment_ids=seg, matmul_dtype=matmul_dtype, state_precision=state_precision
                        )
                    return chunk_kda(q, k, v, g, beta, segment_ids=seg, chunk_size=16, matmul_dtype=matmul_dtype)

                actual = jax.jit(candidate)(*inputs, segments)
                candidate_grads = jax.jit(jax.grad(lambda *a: objective(candidate, *a), argnums=(0, 1, 2, 3, 4)))(
                    *inputs, segments
                )
                errors = []
                for got, want in zip((*actual, *candidate_grads), (*expected, *reference_grads), strict=True):
                    got, want = np.asarray(got), np.asarray(want)
                    error = np.max(np.abs(got - want) / (tolerance + tolerance * np.abs(want)))
                    errors.append(float(error))
                    np.testing.assert_allclose(got, want, atol=tolerance, rtol=tolerance)
                record = {
                    "stage": "packed_kda_oracle_pass",
                    "implementation": implementation,
                    "rank": jax.process_index(),
                    "shape": list(shape),
                    "chunk_size": 16,
                    "matmul_dtype": str(jnp.dtype(dtype)),
                    "state_precision": state_precision.name if implementation == "pallas" else "source",
                    "atol": tolerance,
                    "rtol": tolerance,
                    "normalized_errors_output_state_five_gradients": errors,
                }
                print(json.dumps(record), flush=True)
                with (Path(os.environ["IRIS_OUTPUT_DIR"]) / "packed-kda-check.jsonl").open("a") as file:
                    file.write(json.dumps(record) + "\n")
    tracker.log_summary({"preflight/packed_kda_oracle_pass": True})
