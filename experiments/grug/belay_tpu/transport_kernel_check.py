# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare native and coalesced MoE transfers on the Ckpt 1 TPU mesh."""

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
from levanter.grug._moe.ep_common import _clip_receiver_group_sizes, _expert_granular_a2a_params
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.belay_tpu.belay_ckpt1_tpu import DATA_ROOT, build_config, require_tpu_rounding, verified_tokenizer
from experiments.grug.belay_tpu.ragged_transport_tpu import coalesced_ragged_all_to_all, gather_rows


def transfer(values, initial, group_sizes, *, direction, implementation):
    clipped = _clip_receiver_group_sizes(group_sizes, local_expert_size=24, receiver_capacity=344064)
    params = _expert_granular_a2a_params(group_sizes, clipped, jax.lax.axis_index("expert"), local_expert_size=24)
    selected = params[0 if direction == "dispatch" else 1]
    if implementation == "source":
        return jax.lax.ragged_all_to_all(values, initial, *selected, axis_name="expert")
    return coalesced_ragged_all_to_all(values, initial, *selected, axis_name="expert", peers=16)


@jax.jit
def mismatch_count(actual, expected):
    bits_dtype = jnp.uint16 if actual.dtype == jnp.bfloat16 else jnp.uint32
    return jnp.sum(
        jax.lax.bitcast_convert_type(actual, bits_dtype) != jax.lax.bitcast_convert_type(expected, bits_dtype)
    )


def check_row_gathers(emit):
    """Measure the BF16 row permutations used by dispatch and combine on one chip."""
    with jax.default_device(jax.local_devices()[0]):
        for case, input_rows in (("dispatch", 32768), ("combine", 327680)):
            rng = np.random.RandomState(74)
            values = jnp.asarray(rng.normal(size=(input_rows, 256)), jnp.bfloat16)
            indices = jnp.asarray(rng.permutation(327680) // (10 if case == "dispatch" else 1), jnp.int32)
            expected = None
            for name, function in (("source", lambda x, i: x[i]), ("packed", gather_rows)):
                compiled = jax.jit(function).lower(values, indices).compile()
                timings = []
                for _ in range(7):
                    start = time.perf_counter()
                    actual = jax.block_until_ready(compiled(values, indices))
                    timings.append(time.perf_counter() - start)
                if expected is None:
                    expected = actual
                errors = int(mismatch_count(actual, expected))
                emit(
                    {
                        "stage": "row_gather_comparison",
                        "case": case,
                        "implementation": name,
                        "scope": "single_chip",
                        "bit_mismatches": errors,
                        "median_seconds": float(np.median(timings[2:])),
                        "timings": timings,
                        "memory": str(compiled.memory_analysis()),
                    }
                )
                if errors:
                    raise AssertionError(f"Row gather changed {errors} BF16 elements: {case}/{name}")


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
        raise RuntimeError("Transport gate requires v4-64")
    mesh = compact_grug_mesh(expert_axis_size=16, replica_axis_size=1, model_axis_size=2)
    batch_axes = ("replica_dcn", "data", "expert")
    spec = P(batch_axes, None)
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    tracker.log_summary({"kernel_check/optimizer_updates": 0})

    def emit(row):
        row["rank"] = jax.process_index()
        print(json.dumps(row), flush=True)
        with (output / "transport-kernel-check.jsonl").open("a") as file:
            file.write(json.dumps(row) + "\n")

    with set_mesh(mesh), jax.default_matmul_precision("highest"):
        compiled = {}
        for case in ("balanced", "skewed", "empty"):
            rng = np.random.RandomState(71)
            host_sizes = np.stack([rng.multinomial(327680, np.full(384, 1 / 384)) for _ in range(16)]).astype(np.int32)
            if case == "skewed":
                host_sizes.fill(0)
                host_sizes[:, 3] = 327680
            elif case == "empty":
                host_sizes.fill(0)
            sizes = jax.make_array_from_callback(
                host_sizes.shape, NamedSharding(mesh, P()), lambda index, array=host_sizes: array[index]
            )
            for direction in ("dispatch", "return"):
                input_rows, output_rows = (327680, 344064) if direction == "dispatch" else (344064, 327680)
                for width, dtype in ((256, jnp.bfloat16), (1, jnp.float32)):

                    def shard(index, rows=input_rows, columns=width, value_dtype=dtype):
                        start, stop, _ = index[0].indices(rows * 16)
                        return np.random.RandomState(73 + start).normal(size=(stop - start, columns)).astype(value_dtype)

                    values = jax.make_array_from_callback((input_rows * 16, width), NamedSharding(mesh, spec), shard)
                    initial = jax.make_array_from_callback(
                        (output_rows * 16, width),
                        NamedSharding(mesh, spec),
                        lambda index, rows=output_rows, columns=width, value_dtype=dtype: np.full(
                            (len(range(*index[0].indices(rows * 16))), columns), -7, value_dtype
                        ),
                    )
                    jax.block_until_ready((values, initial, sizes))
                    expected = None
                    for implementation in ("source", "coalesced"):
                        key = (implementation, direction, width)
                        if key not in compiled:
                            function = jax.jit(
                                jax.shard_map(
                                    functools.partial(transfer, direction=direction, implementation=implementation),
                                    mesh=mesh,
                                    in_specs=(spec, spec, P()),
                                    out_specs=spec,
                                    check_vma=False,
                                )
                            )
                            start = time.perf_counter()
                            compiled[key] = function.lower(values, initial, sizes).compile()
                            emit(
                                {
                                    "stage": "transport_compile",
                                    "key": key,
                                    "seconds": time.perf_counter() - start,
                                    "memory": str(compiled[key].memory_analysis()),
                                }
                            )
                        timings = []
                        for _ in range(7):
                            start = time.perf_counter()
                            actual = jax.block_until_ready(compiled[key](values, initial, sizes))
                            timings.append(time.perf_counter() - start)
                        if expected is None:
                            expected = actual
                        errors = int(mismatch_count(actual, expected))
                        emit(
                            {
                                "stage": "transport_comparison",
                                "case": case,
                                "key": key,
                                "bit_mismatches": errors,
                                "median_seconds": float(np.median(timings[2:])),
                                "timings": timings,
                            }
                        )
                        if errors:
                            raise AssertionError(f"Transport changed {errors} elements: {case}/{key}")
                        del actual
                    del expected, values, initial
        tracker.log_summary({"kernel_check/transport_exact_pass": True})
    check_row_gathers(emit)
    tracker.log_summary({"kernel_check/row_gather_exact_pass": True})
    tracker.current_tracker().finish()


if __name__ == "__main__":
    main()
