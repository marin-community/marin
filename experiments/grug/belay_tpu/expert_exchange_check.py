# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare expert-gradient exchange and overlap with independent TPU projection."""

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
from experiments.grug.belay_tpu.expert_collective_check import errors
from experiments.grug.belay_tpu.expert_collectives import all_to_all_reduce_scatter
from experiments.grug.belay_tpu.local_experts import reduce_expert_gradients


def reduce_and_project(value, left, right, *, variant, projection):
    if variant == "native":
        result = reduce_expert_gradients(value.astype(jnp.float32))
    else:
        transport = value.astype(jnp.float32) if variant == "all-to-all-fp32" else value
        result = all_to_all_reduce_scatter(transport)
    result = result.astype(value.dtype)
    if not projection:
        return (result,)
    projected = jax.lax.dot_general(left, right, (((1,), (0,)), ((), ())), precision=jax.lax.Precision.HIGH)
    return result, projected


def input_shard(index, *, shape, dtype, seed):
    start, stop, _ = index[0].indices(shape[0])
    rng = np.random.default_rng(seed + start)
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
        raise RuntimeError("Expert exchange qualification requires TPU")
    peers = config.trainer.expert_axis_size
    mesh = compact_grug_mesh(expert_axis_size=peers, replica_axis_size=1, model_axis_size=1)
    value_spec, matrix_spec = P("expert", None, None), P("expert", None)
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    tracker.log_summary({"kernel_check/optimizer_updates": 0, "kernel_check/seed": 433})
    with set_mesh(mesh):
        for width, hidden in ((256, 384), (384, 256)):
            for dtype in (jnp.bfloat16, jnp.float32):
                values = []
                for shape, spec, seed in (
                    ((peers * 384, width, hidden), value_spec, 433),
                    ((peers * 32768, 256), matrix_spec, 434),
                    ((peers * 256, 384), matrix_spec, 435),
                ):
                    values.append(
                        jax.make_array_from_callback(
                            shape,
                            NamedSharding(mesh, spec),
                            functools.partial(input_shard, shape=shape, dtype=dtype, seed=seed),
                        )
                    )
                jax.block_until_ready(values)
                for projection in (False, True):
                    expected = None
                    for variant, asynchronous in (
                        ("native", False),
                        ("all-to-all-fp32", False),
                        ("all-to-all-source", False),
                        ("all-to-all-source", True),
                    ):
                        function = jax.jit(
                            jax.shard_map(
                                functools.partial(reduce_and_project, variant=variant, projection=projection),
                                mesh=mesh,
                                in_specs=(value_spec, matrix_spec, matrix_spec),
                                out_specs=(value_spec, matrix_spec) if projection else (value_spec,),
                                check_vma=False,
                            )
                        )
                        started = time.perf_counter()
                        compiled = function.lower(*values).compile(
                            compiler_options={"xla_tpu_enable_async_all_to_all": asynchronous}
                        )
                        compile_seconds = time.perf_counter() - started
                        actual = jax.block_until_ready(compiled(*values))
                        if expected is None:
                            expected = actual
                        discrepancies = []
                        for result, reference in zip(actual, expected, strict=True):
                            maximum, scale = map(float, errors(result, reference))
                            tolerance = (2e-2 if dtype == jnp.bfloat16 else 1e-5) * max(scale, 1e-6)
                            discrepancies.append(
                                {
                                    "max_absolute_error": maximum,
                                    "tolerance": tolerance,
                                    "passed": bool(np.isfinite(maximum) and maximum <= tolerance),
                                }
                            )
                        passed = all(row["passed"] for row in discrepancies)
                        timings = []
                        for _ in range(9):
                            started = time.perf_counter()
                            jax.block_until_ready(compiled(*values))
                            timings.append(time.perf_counter() - started)
                        row = {
                            "stage": "expert_exchange_comparison",
                            "rank": jax.process_index(),
                            "implementation": variant,
                            "async_all_to_all": asynchronous,
                            "workload": "reduction-plus-projection" if projection else "reduction",
                            "local_shape": [384, width, hidden],
                            "peers": peers,
                            "dtype": str(dtype),
                            "passed": passed,
                            "errors": discrepancies,
                            "median_seconds": float(np.median(timings[2:])),
                            "timings": timings,
                            "compile_seconds": compile_seconds,
                            "memory": str(compiled.memory_analysis()),
                            "seed": 433,
                            "source_commit": os.environ["BELAY_SOURCE_COMMIT"],
                        }
                        print(json.dumps(row), flush=True)
                        with (output / "expert-exchange-check.jsonl").open("a") as file:
                            file.write(json.dumps(row) + "\n")
                        if not passed:
                            raise AssertionError(f"Expert exchange mismatch: {variant}/{dtype}/{width}")
    tracker.log_summary({"kernel_check/expert_exchange_pass": True})
    tracker.current_tracker().finish()


if __name__ == "__main__":
    main()
