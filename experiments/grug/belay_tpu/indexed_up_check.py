# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare fused token reads with materialized expert inputs on TPU."""

import json
import os
import time

import jax
import jax.numpy as jnp
import levanter.tracker as tracker
import numpy as np
from jax.experimental.pallas.ops.tpu.megablox.gmm import gmm

from experiments.grug.belay_tpu.expert_gradient_adapter import whole_width_tiling
from experiments.grug.belay_tpu.indexed_up_tpu import indexed_layout, indexed_up
from experiments.grug.belay_tpu.kernel_check import measure_errors
from experiments.grug.belay_tpu.operator_check import initialize_operator_check


def main():
    output, operator_topology, reference_topology = initialize_operator_check()
    tracker.log_summary({"kernel_check/input_seed": 421})
    with jax.default_device(jax.local_devices()[0]):
        tokens, topk, experts, width, hidden = 32768, 10, 384, 256, 384
        rng = np.random.default_rng(421)
        selected = np.stack([rng.choice(experts, topk, replace=False) for _ in range(tokens)])
        host_indices, host_sizes, host_tile_experts, host_restore = indexed_layout(selected, experts)
        indices, sizes, tile_experts, restore = jax.tree.map(
            jnp.asarray, (host_indices, host_sizes, host_tile_experts, host_restore)
        )
        order = jnp.asarray(np.argsort(selected.reshape(-1), stable=True))
        inverse = jnp.argsort(order)
        groups = jnp.asarray(np.bincount(selected.reshape(-1), minlength=experts), jnp.int32)
        x = jnp.asarray(rng.normal(scale=0.1, size=(tokens, width)), jnp.bfloat16)
        weights = jnp.asarray(rng.normal(scale=0.1, size=(experts, width, hidden)), jnp.bfloat16)

        def source(a, b):
            rows = a[order // topk]
            return gmm(rows, b, groups, preferred_element_type=a.dtype, tiling=whole_width_tiling)

        def materialized(a, b):
            slab_tokens = tokens // indices.shape[0]
            global_indices = jnp.maximum(indices, 0) + jnp.arange(indices.shape[0])[:, None] * slab_tokens
            rows = jnp.where(indices[..., None] >= 0, a[global_indices], 0)
            return jnp.stack(
                [
                    gmm(rows[slab], b, sizes[slab], preferred_element_type=a.dtype, tiling=whole_width_tiling)
                    for slab in range(indices.shape[0])
                ]
            )

        def fused(a, b):
            return indexed_up(a, b, indices, tile_experts)

        expected = None
        for name, function in (
            ("source_whole_width", source),
            ("aligned_materialized", materialized),
            ("indexed_fused", fused),
        ):
            start = time.perf_counter()
            compiled = jax.jit(function).lower(x, weights).compile()
            compile_seconds = time.perf_counter() - start
            actual = jax.block_until_ready(compiled(x, weights))
            if expected is None:
                expected = actual[inverse]
                compared = expected
            else:
                compared = actual.reshape(-1, hidden)[restore]
            errors = measure_errors([compared], [expected], 2e-2)
            passed = all(np.isfinite(row["normalized_max"]) and row["normalized_max"] <= 1 for row in errors)
            timings = []
            for _ in range(9):
                start = time.perf_counter()
                jax.block_until_ready(compiled(x, weights))
                timings.append(time.perf_counter() - start)
            record = {
                "stage": "indexed_up_comparison",
                "rank": jax.process_index(),
                "implementation": name,
                "operator_topology": operator_topology,
                "reference_topology": reference_topology,
                "shape": [tokens, topk, experts, width, hidden],
                "padded_rows": int(indices.size),
                "slab_input_bytes": int(tokens // indices.shape[0] * width * 2),
                "passed": passed,
                "errors": errors,
                "compile_seconds": compile_seconds,
                "median_seconds": float(np.median(timings[2:])),
                "timings": timings,
                "memory": str(compiled.memory_analysis()),
                "git_sha": os.environ["BELAY_SOURCE_COMMIT"],
            }
            print(json.dumps(record), flush=True)
            with (output / "indexed-up-check.jsonl").open("a") as file:
                file.write(json.dumps(record) + "\n")
            if not passed:
                raise AssertionError(f"Indexed projection mismatch: {name}")
    tracker.log_summary({"kernel_check/indexed_up_pass": True})
    tracker.current_tracker().finish()


if __name__ == "__main__":
    main()
