#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replace natural Grug group-batched weight adjoints with generated FFI."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import jax
import jaxlib
from haliax.partitioning import set_mesh

from lib.tile_lifetime.benchmarks.xla_grug_backward_multi_output_gpu_custom_call_smoke import (
    _compare_under_ordered_fp,
    _tree_hash,
)
from lib.tile_lifetime.benchmarks.xla_grug_pair_map_custom_call_smoke import _mesh, _natural_train_step
from lib.tile_lifetime.benchmarks.xla_pair_map_custom_call_smoke import (
    _compile_cuda_ffi_handler,
    write_gzip_text,
)
from tile_lifetime.xla_relation_program_recovery import plan_routed_weight_gradient_typed_ffi
from tile_lifetime.xla_routed_weight_gradient_ffi import (
    generate_cuda_group_batched_contract_ffi,
    replace_group_batched_contract_with_custom_call,
)

_PASS_NAME = "shuttle_grug_routed_weight_gradient_gpu_v1"
_TARGET_PREFIX = "shuttle.group_batched_contract.weight_adjoint.v1"


def _register_cuda_target(library: ctypes.CDLL, target: str) -> None:
    handler = getattr(library, target.replace(".", "_"))
    handler.restype = ctypes.c_void_p
    jax.ffi.register_ffi_target(
        target,
        jax.ffi.pycapsule(handler),
        platform="CUDA",
        api_version=1,
    )


def run_smoke(
    nvcc: Path,
    architecture: str,
    artifact_directory: Path | None,
    *,
    warmup: int = 4,
    repeats: int = 30,
) -> dict[str, Any]:
    """Compile, execute, and time both generated weight-adjoint Contracts."""
    if not jax.devices() or jax.devices()[0].platform != "gpu":
        raise RuntimeError("the group-batched Contract replacement requires a CUDA JAX device")
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if repeats <= 0 or repeats % 2:
        raise ValueError("repeats must be a positive even number")
    hlo = importlib.import_module("jaxlib._hlo")
    xla = importlib.import_module("jax.extend.xla")
    jax.config.update("jax_enable_compilation_cache", False)
    temporary = None
    if artifact_directory is None:
        temporary = tempfile.TemporaryDirectory(prefix="shuttle-grug-routed-weight-gradient-")
        directory = Path(temporary.name)
    else:
        artifact_directory.mkdir(parents=True, exist_ok=True)
        directory = artifact_directory

    original_modules: list[str] = []
    transformed_modules: list[str] = []
    recovered_plans: list[tuple[Any, ...]] = []
    generated_sources: list[tuple[Any, ...]] = []
    libraries: list[ctypes.CDLL] = []
    try:
        with set_mesh(_mesh()):
            train_step, state, batch = _natural_train_step()
            host_state = jax.device_get(state)
            host_batch = jax.device_get(batch)

            def fresh_inputs() -> tuple[Any, Any]:
                fresh_state = jax.tree.map(jax.numpy.array, host_state)
                fresh_batch = jax.tree.map(jax.numpy.array, host_batch)
                jax.block_until_ready((fresh_state, fresh_batch))
                return fresh_state, fresh_batch

            baseline = train_step.lower(state, batch, compute_watch=False).compile()
            expected = baseline(state, batch)
            jax.block_until_ready(expected)

            def replace(serialized_module: bytes) -> bytes | None:
                module = hlo.HloModule.from_serialized_hlo_module_proto(serialized_module)
                if module.name != "jit_train_step":
                    return None
                original = module.to_string()
                original_modules.append(original)
                if artifact_directory is not None:
                    write_gzip_text(directory / "original-gpu-pre-scheduler-hlo.txt.gz", original)
                plans = plan_routed_weight_gradient_typed_ffi(original)
                if len(plans) != 2:
                    raise RuntimeError(f"expected two group-batched weight adjoints, found {len(plans)}")
                generated = tuple(
                    generate_cuda_group_batched_contract_ffi(
                        plan,
                        target=f"{_TARGET_PREFIX}.{index}",
                    )
                    for index, plan in enumerate(plans)
                )
                transformed_text = original
                for index, (plan, source) in enumerate(zip(plans, generated, strict=True)):
                    target = f"{_TARGET_PREFIX}.{index}"
                    source_directory = directory / f"contract-{index}"
                    source_directory.mkdir(exist_ok=True)
                    (directory / f"generated_group_batched_contract_{index}.cu").write_text(source.source)
                    library = _compile_cuda_ffi_handler(
                        source.source,
                        source_directory,
                        nvcc,
                        architecture,
                    )
                    _register_cuda_target(library, target)
                    libraries.append(library)
                    transformed_text = replace_group_batched_contract_with_custom_call(
                        transformed_text,
                        plan,
                        target=target,
                    )
                recovered_plans.append(plans)
                generated_sources.append(generated)
                transformed_module = hlo.hlo_module_from_text(transformed_text)
                transformed_modules.append(transformed_module.to_string())
                return transformed_module.as_serialized_hlo_module_proto()

            xla.register_hlo_module_transformation(
                replace,
                name=_PASS_NAME,
                stage=xla.PipelineStage.PRE_SCHEDULER,
                platforms="cuda",
            )
            jax.clear_caches()
            transformed_state, transformed_batch = fresh_inputs()
            try:
                transformed = train_step.lower(
                    transformed_state,
                    transformed_batch,
                    compute_watch=False,
                ).compile()
            finally:
                xla.clear_hlo_module_transformation(
                    _PASS_NAME,
                    stage=xla.PipelineStage.PRE_SCHEDULER,
                    platforms="cuda",
                )
            actual = transformed(transformed_state, transformed_batch)
            jax.block_until_ready(actual)
            comparison = _compare_under_ordered_fp(expected, actual)

            def timed_execution(executable: Any) -> tuple[float, str]:
                timing_state, timing_batch = fresh_inputs()
                start = time.perf_counter()
                output = executable(timing_state, timing_batch)
                jax.block_until_ready(output)
                return (time.perf_counter() - start) * 1e3, _tree_hash(output)

            for index in range(warmup):
                order = (baseline, transformed) if index % 2 == 0 else (transformed, baseline)
                for executable in order:
                    timed_execution(executable)

            raw_samples: list[dict[str, Any]] = []
            baseline_samples: list[float] = []
            transformed_samples: list[float] = []
            baseline_hashes: list[str] = []
            transformed_hashes: list[str] = []
            for index in range(repeats):
                order = ("baseline", "transformed") if index % 2 == 0 else ("transformed", "baseline")
                sample: dict[str, Any] = {"order": order}
                for name in order:
                    executable = baseline if name == "baseline" else transformed
                    latency, output_hash = timed_execution(executable)
                    sample[name] = {"latency_ms": latency, "output_hash": output_hash}
                    if name == "baseline":
                        baseline_samples.append(latency)
                        baseline_hashes.append(output_hash)
                    else:
                        transformed_samples.append(latency)
                        transformed_hashes.append(output_hash)
                raw_samples.append(sample)

            call_counts: list[int] = []
            for library in libraries:
                call_count_function = library.shuttle_routed_weight_gradient_call_count
                call_count_function.restype = ctypes.c_int
                call_counts.append(int(call_count_function()))
    finally:
        if artifact_directory is not None:
            for index in range(2):
                compiled_library = directory / f"contract-{index}" / "generated_pair_map_handler.so"
                if compiled_library.exists():
                    compiled_library.unlink()
        if temporary is not None:
            temporary.cleanup()

    if len(original_modules) != 1 or len(transformed_modules) != 1 or len(recovered_plans) != 1:
        raise RuntimeError("expected one natural Grug weight-gradient replacement pass")
    transformed_hlo = transformed_modules[0]
    plans = recovered_plans[0]
    generated = generated_sources[0]
    occurrences = transformed_hlo.count(_TARGET_PREFIX)
    minimum_calls = repeats + warmup + 1
    if occurrences != 2 or any(count < minimum_calls for count in call_counts):
        raise RuntimeError(f"custom-call execution evidence mismatch: occurrences={occurrences}, calls={call_counts}")
    baseline_median = statistics.median(baseline_samples)
    transformed_median = statistics.median(transformed_samples)
    if artifact_directory is not None:
        write_gzip_text(directory / "transformed-gpu-pre-scheduler-hlo.txt.gz", transformed_hlo)
    return {
        "kind": "xla_grug_generated_group_batched_weight_adjoint_ffi",
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "platform": "cuda",
        "device_kind": jax.devices()[0].device_kind,
        "architecture": architecture,
        "natural_frontend": "ordinary one-layer Grug train step with JAX-owned differentiation",
        "recovered_structure": "two independent group-batched Contracts before external all-reduces",
        "external_collectives": [plan.region.external_collectives for plan in plans],
        "numerical_contracts": [
            {
                "input_dtype": plan.numerical_contract.input_dtype,
                "accumulation_dtype": plan.numerical_contract.accumulation_dtype,
                "output_dtype": plan.numerical_contract.output_dtype,
                "output_rounding": plan.numerical_contract.output_rounding,
                "numerical_policy": plan.numerical_contract.numerical_policy.value,
                "effect": plan.numerical_contract.effect,
            }
            for plan in plans
        ],
        "uses_atomic_accumulation": False,
        "output_alias_operands": [plan.output_alias_operand for plan in plans],
        "operand_bindings": [
            [
                {
                    "role": operand.role.value,
                    "instruction": operand.value.instruction,
                    "shape": operand.value.shape,
                    "parameter_ancestors": operand.parameter_ancestors,
                }
                for operand in plan.operands
            ]
            for plan in plans
        ],
        "generated_semantic_sha256": [source.semantic_digest for source in generated],
        "generated_source_sha256": [source.source_digest for source in generated],
        "custom_call_target_prefix": _TARGET_PREFIX,
        "custom_call_occurrences_in_transformed_hlo": occurrences,
        "custom_call_handler_executions": call_counts,
        "baseline_median_ms": baseline_median,
        "generated_median_ms": transformed_median,
        "generated_over_baseline": transformed_median / baseline_median,
        "baseline_unique_output_hashes": sorted(set(baseline_hashes)),
        "generated_unique_output_hashes": sorted(set(transformed_hashes)),
        "raw_samples": raw_samples,
        **comparison,
        "outputs_match": True,
        "remaining_ownership_gap": (
            "The input-adjoint replacement, forward routed chain, relation index plane, and all placement "
            "collectives are separate boundaries; this proof replaces only the two recovered weight Contracts."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--artifact-directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    result = run_smoke(
        args.nvcc,
        args.architecture,
        args.artifact_directory,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
