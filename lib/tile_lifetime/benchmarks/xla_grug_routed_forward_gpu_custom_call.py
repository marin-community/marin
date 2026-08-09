#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replace one natural Grug routed forward region with generated CUDA FFI."""

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
from lib.tile_lifetime.benchmarks.xla_grug_pair_map_custom_call_smoke import (
    _mesh,
    _natural_train_step,
)
from lib.tile_lifetime.benchmarks.xla_pair_map_custom_call_smoke import (
    _compile_cuda_ffi_handler,
    write_gzip_text,
)
from tile_lifetime.xla_hlo_recovery import parse_hlo_module_text
from tile_lifetime.xla_relation_program_recovery import (
    RoutedForwardCodegenDisposition,
    plan_routed_forward_typed_ffi,
)
from tile_lifetime.xla_routed_forward_ffi import (
    generate_cuda_routed_forward_ffi,
    replace_routed_forward_region_with_custom_call,
)

_PASS_NAME = "shuttle_grug_routed_forward_gpu_v1"
_TARGET_NAME = "shuttle.routed_forward_region.v1"


def _register_cuda_target(library: ctypes.CDLL) -> None:
    handler = getattr(library, _TARGET_NAME.replace(".", "_"))
    handler.restype = ctypes.c_void_p
    jax.ffi.register_ffi_target(
        _TARGET_NAME,
        jax.ffi.pycapsule(handler),
        platform="CUDA",
        api_version=1,
    )


def _parameter_ancestor_audit(hlo_text: str, operands: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    instructions = {instruction.name: instruction for instruction in entry.instructions}

    def ancestors(name: str) -> tuple[str, ...]:
        pending = [name]
        seen: set[str] = set()
        parameters: set[str] = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            instruction = instructions[current]
            if instruction.opcode == "parameter":
                parameters.add(current)
            else:
                pending.extend(instruction.operands)
        return tuple(sorted(parameters))

    return {operand: ancestors(operand) for operand in operands}


def run_smoke(
    nvcc: Path,
    architecture: str,
    artifact_directory: Path | None,
    *,
    warmup: int = 4,
    repeats: int = 30,
) -> dict[str, Any]:
    """Compile, execute, and time the natural and generated train-step paths."""
    if not jax.devices() or jax.devices()[0].platform != "gpu":
        raise RuntimeError("the routed Grug replacement requires a CUDA JAX device")
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if repeats <= 0 or repeats % 2:
        raise ValueError("repeats must be a positive even number")
    hlo = importlib.import_module("jaxlib._hlo")
    xla = importlib.import_module("jax.extend.xla")
    jax.config.update("jax_enable_compilation_cache", False)
    temporary = None
    if artifact_directory is None:
        temporary = tempfile.TemporaryDirectory(prefix="shuttle-grug-routed-forward-")
        directory = Path(temporary.name)
    else:
        artifact_directory.mkdir(parents=True, exist_ok=True)
        directory = artifact_directory

    original_modules: list[str] = []
    transformed_modules: list[str] = []
    plans: list[Any] = []
    generated_sources: list[Any] = []
    holder: dict[str, Any] = {}
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
                plan = plan_routed_forward_typed_ffi(original)
                if plan.disposition is not RoutedForwardCodegenDisposition.READY:
                    raise RuntimeError(f"routed forward plan is not executable: {plan.missing_segmented_layout}")
                generated = generate_cuda_routed_forward_ffi(plan, target=_TARGET_NAME)
                (directory / "generated_routed_forward_ffi.cu").write_text(generated.source)
                library = _compile_cuda_ffi_handler(generated.source, directory, nvcc, architecture)
                _register_cuda_target(library)
                holder["library"] = library
                plans.append(plan)
                generated_sources.append(generated)
                transformed_text = replace_routed_forward_region_with_custom_call(
                    original,
                    plan,
                    target=_TARGET_NAME,
                )
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

            call_count_function = holder["library"].shuttle_routed_forward_call_count
            call_count_function.restype = ctypes.c_int
            call_count = int(call_count_function())
    finally:
        if artifact_directory is not None:
            compiled_library = directory / "generated_pair_map_handler.so"
            if compiled_library.exists():
                compiled_library.unlink()
        if temporary is not None:
            temporary.cleanup()

    if len(original_modules) != 1 or len(transformed_modules) != 1 or len(plans) != 1:
        raise RuntimeError("expected exactly one natural Grug routed-region replacement")
    original = original_modules[0]
    transformed_hlo = transformed_modules[0]
    plan = plans[0]
    generated = generated_sources[0]
    occurrences = transformed_hlo.count(_TARGET_NAME)
    if occurrences != 1 or call_count < repeats + warmup + 1:
        raise RuntimeError(f"custom-call execution evidence mismatch: occurrences={occurrences}, calls={call_count}")
    parameter_ancestors = _parameter_ancestor_audit(
        original,
        tuple(operand.value.instruction for operand in plan.operands),
    )
    roles_without_parameters = tuple(
        operand.role.value for operand in plan.operands if not parameter_ancestors[operand.value.instruction]
    )
    if roles_without_parameters:
        raise RuntimeError(f"unexpected closed-over routed operands: {roles_without_parameters}")
    baseline_median = statistics.median(baseline_samples)
    transformed_median = statistics.median(transformed_samples)
    if artifact_directory is not None:
        write_gzip_text(directory / "transformed-gpu-pre-scheduler-hlo.txt.gz", transformed_hlo)
    return {
        "kind": "xla_grug_routed_forward_generated_ffi",
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "platform": "cuda",
        "device_kind": jax.devices()[0].device_kind,
        "architecture": architecture,
        "natural_frontend": "ordinary one-layer Grug train step with JAX-owned differentiation",
        "recovered_structure": "Contract -> generated Map -> Contract -> generated deterministic source Fold",
        "numerical_policy": plan.region.numerical_policy.value,
        "fold_iteration_order": "destination-major compact edge row, single writer per source-feature",
        "uses_atomic_accumulation": False,
        "operand_roles": [
            {
                "role": operand.role.value,
                "instruction": operand.value.instruction,
                "shape": operand.value.shape,
                "parameter_ancestors": parameter_ancestors[operand.value.instruction],
            }
            for operand in plan.operands
        ],
        "static_operand_roles": roles_without_parameters,
        "generated_semantic_sha256": generated.semantic_digest,
        "generated_source_sha256": generated.source_digest,
        "custom_call_target": _TARGET_NAME,
        "custom_call_occurrences_in_transformed_hlo": occurrences,
        "custom_call_handler_executions": call_count,
        "baseline_median_ms": baseline_median,
        "generated_median_ms": transformed_median,
        "generated_over_baseline": transformed_median / baseline_median,
        "baseline_unique_output_hashes": sorted(set(baseline_hashes)),
        "generated_unique_output_hashes": sorted(set(transformed_hashes)),
        "raw_samples": raw_samples,
        **comparison,
        "outputs_match": True,
        "remaining_ownership_gap": (
            "The recovered rematerialized forward chain, input adjoint, grouped weight adjoints, relation index plane, "
            "and external collectives remain under XLA in this one-region proof."
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
