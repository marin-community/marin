# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure full expert-gradient reductions on the actual TPU expert mesh."""

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
from experiments.grug.belay_tpu.expert_collectives import all_to_all_reduce_scatter, butterfly_reduce_scatter
from experiments.grug.belay_tpu.local_experts import reduce_expert_gradients


def reduce_compute_gradient(value, reduce):
    return reduce(value.astype(jnp.float32)).astype(value.dtype)


@jax.jit
def errors(actual, expected):
    actual, expected = actual.astype(jnp.float32), expected.astype(jnp.float32)
    return jnp.max(jnp.abs(actual - expected)), jnp.max(jnp.abs(expected))


def gradient_shard(index, *, shape, dtype):
    start, stop, _ = index[0].indices(shape[0])
    rng = np.random.default_rng(295 + start)
    return rng.standard_normal((stop - start, *shape[1:]), dtype=np.float32).astype(dtype)


def main():
    require_tpu_rounding()
    run_id = os.environ["BELAY_RUN_ID"]
    config = build_config(
        run_id,
        f"{DATA_ROOT}/kaiyuew/belay-ckpt1-20261025/{run_id}",
        verified_tokenizer(),
        os.environ["BELAY_TPU"],
        0,
        local_experts=True,
    )
    config.trainer.trainer.initialize()
    if jax.default_backend() != "tpu":
        raise RuntimeError("Expert collective qualification requires TPU")
    peers = config.trainer.expert_axis_size
    mesh = compact_grug_mesh(expert_axis_size=peers, replica_axis_size=1, model_axis_size=1)
    spec = P("expert", None, None)
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    tracker.log_summary({"kernel_check/optimizer_updates": 0})
    with set_mesh(mesh):
        for width, hidden in ((256, 384), (384, 256)):
            for dtype in (jnp.bfloat16, jnp.float32):
                shape = (peers * 384, width, hidden)

                values = jax.make_array_from_callback(
                    shape, NamedSharding(mesh, spec), functools.partial(gradient_shard, shape=shape, dtype=dtype)
                )
                jax.block_until_ready(values)
                expected = None
                for name, reduce in (
                    ("native", reduce_expert_gradients),
                    ("butterfly", butterfly_reduce_scatter),
                    ("all-to-all", all_to_all_reduce_scatter),
                ):
                    function = jax.jit(
                        jax.shard_map(
                            functools.partial(reduce_compute_gradient, reduce=reduce),
                            mesh=mesh,
                            in_specs=spec,
                            out_specs=spec,
                            check_vma=False,
                        )
                    )
                    start = time.perf_counter()
                    compiled = function.lower(values).compile()
                    compile_seconds = time.perf_counter() - start
                    actual = jax.block_until_ready(compiled(values))
                    if expected is None:
                        expected = actual
                    maximum, scale = map(float, errors(actual, expected))
                    tolerance = (2e-2 if dtype == jnp.bfloat16 else 1e-5) * max(scale, 1e-6)
                    passed = np.isfinite(maximum) and maximum <= tolerance
                    timings = []
                    for _ in range(9):
                        start = time.perf_counter()
                        jax.block_until_ready(compiled(values))
                        timings.append(time.perf_counter() - start)
                    row = {
                        "stage": "expert_collective_comparison",
                        "rank": jax.process_index(),
                        "implementation": name,
                        "local_shape": [384, width, hidden],
                        "peers": peers,
                        "dtype": str(dtype),
                        "passed": bool(passed),
                        "max_absolute_error": maximum,
                        "tolerance": tolerance,
                        "median_seconds": float(np.median(timings[2:])),
                        "timings": timings,
                        "compile_seconds": compile_seconds,
                        "memory": str(compiled.memory_analysis()),
                    }
                    print(json.dumps(row), flush=True)
                    with (output / "expert-collective-check.jsonl").open("a") as file:
                        file.write(json.dumps(row) + "\n")
                    if not passed:
                        raise AssertionError(f"Expert reduction mismatch: {name}/{dtype}/{width}")
    tracker.log_summary({"kernel_check/expert_collectives_pass": True})
    tracker.current_tracker().finish()


if __name__ == "__main__":
    main()
