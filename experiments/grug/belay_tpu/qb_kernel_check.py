# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check exact QB thresholds and histogram timing on the Ckpt 1 TPU mesh."""

import functools
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import levanter.tracker as tracker
import numpy as np
from haliax.partitioning import set_mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.belay_tpu.belay_ckpt1_tpu import DATA_ROOT, build_config, require_tpu_rounding, verified_tokenizer
from experiments.grug.belay_tpu.qb_hist_tpu import bincount_upper_quantile_tpu, factorized_histogram
from experiments.grug.fast_track.qb_balancing import bincount_upper_quantile


def quantile(kernel, margins, ranks, batch_axes):
    lo = jax.lax.pmin(jax.lax.stop_gradient(jnp.min(margins)), batch_axes)
    hi = jax.lax.pmax(jax.lax.stop_gradient(jnp.max(margins)), batch_axes)
    return kernel(
        margins,
        num_experts=384,
        n_bins=10000,
        lo=lo,
        hi=jnp.maximum(hi, lo + 1e-6),
        target_rank=ranks,
        batch_axes=batch_axes,
    )


def count_mismatches(margins, batch_axes):
    indices = jnp.clip(((margins + 16) * 10000 / 32).astype(jnp.int32), 0, 9999)
    errors = []
    for bins, radix, values in ((157, 16, indices // 64), (64, 8, indices % 64)):
        flat = (jnp.arange(384)[None, :] * bins + values).reshape(-1)
        expected = jnp.bincount(flat, length=384 * bins).reshape(384, bins)
        actual = factorized_histogram(values, bins, 64, radix)
        errors.append(jnp.sum(actual != expected))
    return jax.lax.psum(jnp.stack(errors), batch_axes)


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
        raise RuntimeError("QB gate requires v4-64")
    mesh = compact_grug_mesh(expert_axis_size=16, replica_axis_size=1, model_axis_size=2)
    batch_axes = ("replica_dcn", "data", "expert")
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    tracker.log_summary({"kernel_check/optimizer_updates": 0})

    def emit(row):
        row["rank"] = jax.process_index()
        print(json.dumps(row), flush=True)
        with (output / "qb-kernel-check.jsonl").open("a") as file:
            file.write(json.dumps(row) + "\n")

    with set_mesh(mesh), jax.default_matmul_precision("highest"):
        spec = P(batch_axes, None)
        tokens = 524288
        ranks_host = np.full(384, tokens * 10 / 384, np.float32)
        ranks_host[:8] = [0, 1, 5.25, tokens / 2, tokens - 1, tokens, tokens + 1, 7.5]
        ranks = jax.make_array_from_callback((384,), NamedSharding(mesh, P()), lambda index: ranks_host[index])
        functions = {
            name: jax.jit(
                jax.shard_map(
                    functools.partial(quantile, kernel, batch_axes=batch_axes),
                    mesh=mesh,
                    in_specs=(spec, P()),
                    out_specs=P(),
                    check_vma=False,
                )
            )
            for name, kernel in (("source", bincount_upper_quantile), ("two_stage", bincount_upper_quantile_tpu))
        }
        counts = jax.jit(
            jax.shard_map(
                functools.partial(count_mismatches, batch_axes=batch_axes),
                mesh=mesh,
                in_specs=spec,
                out_specs=P(),
                check_vma=False,
            )
        )
        compiled = {}
        for case in ("random", "equal", "edges"):

            def shard(index, scenario=case):
                start, stop, _ = index[0].indices(tokens)
                rng = np.random.RandomState(871 + start)
                shape = (stop - start, 384)
                if scenario == "equal":
                    return np.full(shape, 0.5, np.float32)
                if scenario == "edges":
                    edge_values = np.array([-16, -15.7952, 0, 15.744, 15.9968, 16], np.float32)
                    return edge_values[rng.randint(0, len(edge_values), size=shape)]
                return rng.normal(size=shape).astype(np.float32)

            margins = jax.make_array_from_callback((tokens, 384), NamedSharding(mesh, spec), shard)
            jax.block_until_ready(margins)
            count_errors = np.asarray(counts(margins))
            if np.any(count_errors):
                raise AssertionError(f"Integer histogram mismatch: {case} {count_errors}")
            expected = None
            for name, function in functions.items():
                if name not in compiled:
                    start = time.perf_counter()
                    compiled[name] = function.lower(margins, ranks).compile()
                    emit(
                        {
                            "stage": "qb_compile",
                            "implementation": name,
                            "seconds": time.perf_counter() - start,
                            "memory": str(compiled[name].memory_analysis()),
                        }
                    )
                timings = []
                for _ in range(7):
                    start = time.perf_counter()
                    result = jax.block_until_ready(compiled[name](margins, ranks))
                    timings.append(time.perf_counter() - start)
                actual = np.asarray(result)
                if expected is None:
                    expected = actual
                np.testing.assert_array_equal(actual, expected)
                emit(
                    {
                        "stage": "qb_pass",
                        "case": case,
                        "implementation": name,
                        "count_mismatches": count_errors.tolist(),
                        "exact_thresholds": True,
                        "median_seconds": float(np.median(timings[2:])),
                        "timings": timings,
                        "shape": [tokens, 384],
                        "local_shape": [32768, 384],
                        "mesh": dict(mesh.shape),
                    }
                )
        tracker.log_summary({"kernel_check/qb_exact_pass": True})
    tracker.current_tracker().finish()


if __name__ == "__main__":
    main()
