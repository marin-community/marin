# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay Shuttle-derived collective completion on a real local GPU mesh."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tile_lifetime.collective_transport import (
    CollectiveCompletionPlan,
    CollectiveFoldPlan,
    CollectiveReduction,
    PlacementTransitionPlan,
    ReplicaGroupDomain,
)
from tile_lifetime.event_dataflow import EventSchedulingMode
from tile_lifetime.ir import DType
from tile_lifetime.jax_collective_transport import (
    build_jax_collective_execution_plan,
    execute_jax_collective_completion,
)
from tile_lifetime.plan import NumericalPolicy

_AXIS_NAME = "collective"
_FEATURE_COUNT = 8
_NVIDIA_SMI_FIELDS = (
    "index",
    "name",
    "uuid",
    "driver_version",
    "compute_cap",
    "clocks.sm",
    "clocks.mem",
    "power.limit",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--shuttle-revision", required=True)
    return parser.parse_args()


def _completion(
    reduction: CollectiveReduction,
    groups: tuple[tuple[int, ...], ...],
) -> CollectiveCompletionPlan:
    return CollectiveCompletionPlan(
        shape=f"bf16[1,{_FEATURE_COUNT}]",
        fold=CollectiveFoldPlan(
            reduction=reduction,
            dtype=DType.BF16,
            numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        ),
        transport=PlacementTransitionPlan(
            source_value="partial",
            destination_value="complete",
            replica_domain=ReplicaGroupDomain(groups=groups, use_global_device_ids=True),
            channel_id=1,
        ),
    )


def _nvidia_smi() -> tuple[dict[str, str], ...]:
    command = (
        "nvidia-smi",
        f"--query-gpu={','.join(_NVIDIA_SMI_FIELDS)}",
        "--format=csv,noheader,nounits",
    )
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in completed.stdout.splitlines():
        values = tuple(value.strip() for value in line.split(","))
        if len(values) != len(_NVIDIA_SMI_FIELDS):
            raise ValueError(f"unexpected nvidia-smi row: {line!r}")
        rows.append(dict(zip(_NVIDIA_SMI_FIELDS, values, strict=True)))
    return tuple(rows)


def _global_reference(value: np.ndarray, reduction: CollectiveReduction) -> np.ndarray:
    if reduction is CollectiveReduction.SUM:
        reduced = np.sum(value.astype(np.float32), axis=0).astype(jnp.bfloat16)
    elif reduction is CollectiveReduction.MAXIMUM:
        reduced = np.max(value, axis=0)
    else:
        raise ValueError(f"unsupported reference reduction {reduction.value}")
    return np.broadcast_to(reduced, value.shape)


def main() -> None:
    arguments = _arguments()
    devices = tuple(device for device in jax.devices() if device.platform == "gpu")
    if len(devices) < 2:
        raise RuntimeError(f"the GPU collective replay requires at least two GPUs, found {devices}")
    device_ids = tuple(device.id for device in devices)
    groups = (device_ids,)
    mesh = Mesh(np.asarray(devices), (_AXIS_NAME,))

    def mapped_collective(reduction: CollectiveReduction):
        execution = build_jax_collective_execution_plan(
            _completion(reduction, groups),
            axis_name=_AXIS_NAME,
            device_id_by_axis_index=device_ids,
            scheduling_mode=EventSchedulingMode.STATIC,
        )
        mapped = jax.shard_map(
            lambda local: execute_jax_collective_completion(execution, local),
            mesh=mesh,
            in_specs=P(_AXIS_NAME),
            out_specs=P(_AXIS_NAME),
            check_vma=False,
        )
        return execution, jax.jit(mapped)

    sum_execution, sum_collective = mapped_collective(CollectiveReduction.SUM)
    maximum_execution, maximum_collective = mapped_collective(CollectiveReduction.MAXIMUM)
    host_value = np.arange(len(devices) * _FEATURE_COUNT, dtype=np.float32).reshape(len(devices), _FEATURE_COUNT)
    value = jnp.asarray(host_value, dtype=jnp.bfloat16)
    sum_output = sum_collective(value).block_until_ready()
    maximum_output = maximum_collective(value).block_until_ready()
    repeated_sum = sum_collective(value).block_until_ready()
    repeated_maximum = maximum_collective(value).block_until_ready()

    def loss(input_value: jax.Array) -> jax.Array:
        output = sum_collective(input_value)
        return jnp.sum(output.astype(jnp.float32) ** 2, dtype=jnp.float32) / 2

    gradient_function = jax.jit(jax.grad(loss))
    gradient = gradient_function(value).block_until_ready()
    repeated_gradient = gradient_function(value).block_until_ready()
    sum_reference = _global_reference(host_value, CollectiveReduction.SUM)
    maximum_reference = _global_reference(host_value, CollectiveReduction.MAXIMUM)
    gradient_reference = np.broadcast_to(
        (len(devices) * sum_reference[0].astype(np.float32)).astype(jnp.bfloat16),
        host_value.shape,
    )

    forward_hlo = str(sum_collective.lower(value).compiler_ir(dialect="stablehlo"))
    maximum_hlo = str(maximum_collective.lower(value).compiler_ir(dialect="stablehlo"))
    gradient_hlo = str(gradient_function.lower(value).compiler_ir(dialect="stablehlo"))
    output_directory = arguments.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "sum-forward-stablehlo.txt").write_text(forward_hlo)
    (output_directory / "maximum-forward-stablehlo.txt").write_text(maximum_hlo)
    (output_directory / "sum-gradient-stablehlo.txt").write_text(gradient_hlo)
    result = {
        "shuttle_revision": arguments.shuttle_revision,
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "jax_default_backend": jax.default_backend(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_count": len(devices),
        "devices": [
            {
                "id": device.id,
                "platform": device.platform,
                "device_kind": device.device_kind,
            }
            for device in devices
        ],
        "nvidia_smi": _nvidia_smi(),
        "dtype": "bf16",
        "shape": list(value.shape),
        "sum_axis_index_groups": sum_execution.axis_index_groups,
        "maximum_axis_index_groups": maximum_execution.axis_index_groups,
        "sum_event_initial_count": [
            {"coordinate": coordinate, "count": count}
            for coordinate, count in sum_execution.dataflow.program.event_plans[-1].initial_count.as_mapping().items()
        ],
        "completion_visibility": {
            "scope": sum_execution.dataflow.program.event_plans[-1].memory_scope.value,
            "release_on_notify": sum_execution.completion_visibility.release_on_notify,
            "acquire_before_consumer": sum_execution.completion_visibility.acquire_before_consumer,
            "physical_signal": "jax_array_result_data_dependency",
        },
        "sum_max_abs_error": float(np.max(np.abs(np.asarray(sum_output).astype(np.float32) - sum_reference))),
        "maximum_max_abs_error": float(
            np.max(np.abs(np.asarray(maximum_output).astype(np.float32) - maximum_reference))
        ),
        "gradient_max_abs_error": float(np.max(np.abs(np.asarray(gradient).astype(np.float32) - gradient_reference))),
        "sum_deterministic": bool(jnp.array_equal(sum_output, repeated_sum)),
        "maximum_deterministic": bool(jnp.array_equal(maximum_output, repeated_maximum)),
        "gradient_deterministic": bool(jnp.array_equal(gradient, repeated_gradient)),
        "sum_output_sha256": hashlib.sha256(np.asarray(sum_output).tobytes()).hexdigest(),
        "maximum_output_sha256": hashlib.sha256(np.asarray(maximum_output).tobytes()).hexdigest(),
        "gradient_sha256": hashlib.sha256(np.asarray(gradient).tobytes()).hexdigest(),
        "sum_forward_all_reduce_count": forward_hlo.count("stablehlo.all_reduce"),
        "maximum_forward_all_reduce_count": maximum_hlo.count("stablehlo.all_reduce"),
        "gradient_all_reduce_count": gradient_hlo.count("stablehlo.all_reduce"),
        "sum_forward_custom_call_count": forward_hlo.count("stablehlo.custom_call"),
        "maximum_forward_custom_call_count": maximum_hlo.count("stablehlo.custom_call"),
        "gradient_custom_call_count": gradient_hlo.count("stablehlo.custom_call"),
    }
    result_path = output_directory / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
