#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute four generated routed regions in one natural Grug train step."""

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
from tile_lifetime.xla_relation_program_recovery import RoutedInputAdjointFfiOperandRole
from tile_lifetime.xla_routed_forward_ffi import generate_cuda_routed_forward_ffi
from tile_lifetime.xla_routed_input_adjoint_ffi import generate_cuda_routed_input_adjoint_ffi
from tile_lifetime.xla_routed_training_ffi import (
    RoutedTrainingFfiTargets,
    audit_routed_training_replacement,
    entry_parameter_ancestors,
    plan_routed_training_typed_ffi,
    replace_routed_training_regions_with_custom_calls,
)
from tile_lifetime.xla_routed_weight_gradient_ffi import generate_cuda_group_batched_contract_ffi

_PASS_NAME = "shuttle_grug_routed_training_gpu_v1"
_TARGETS = RoutedTrainingFfiTargets(
    forward="shuttle.routed_training.forward.v1",
    input_adjoint="shuttle.routed_training.input_adjoint.v1",
    weight_gradients=(
        "shuttle.routed_training.weight_gradient.0.v1",
        "shuttle.routed_training.weight_gradient.1.v1",
    ),
)


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
    """Compile, execute, and time the combined routed training transform."""
    if not jax.devices() or jax.devices()[0].platform != "gpu":
        raise RuntimeError("the combined routed training replacement requires a CUDA JAX device")
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if repeats <= 0 or repeats % 2:
        raise ValueError("repeats must be a positive even number")
    hlo = importlib.import_module("jaxlib._hlo")
    xla = importlib.import_module("jax.extend.xla")
    jax.config.update("jax_enable_compilation_cache", False)
    temporary = None
    if artifact_directory is None:
        temporary = tempfile.TemporaryDirectory(prefix="shuttle-grug-routed-training-")
        directory = Path(temporary.name)
    else:
        artifact_directory.mkdir(parents=True, exist_ok=True)
        directory = artifact_directory

    original_modules: list[str] = []
    transformed_modules: list[str] = []
    recovered_plans: list[Any] = []
    generated_programs: list[tuple[Any, Any, tuple[Any, Any]]] = []
    replacement_audits: list[Any] = []
    libraries: dict[str, ctypes.CDLL] = {}
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

            def compile_target(source: str, target: str, name: str) -> ctypes.CDLL:
                source_directory = directory / name
                source_directory.mkdir(exist_ok=True)
                (directory / f"generated_{name}.cu").write_text(source)
                library = _compile_cuda_ffi_handler(source, source_directory, nvcc, architecture)
                _register_cuda_target(library, target)
                libraries[name] = library
                return library

            def replace(serialized_module: bytes) -> bytes | None:
                module = hlo.HloModule.from_serialized_hlo_module_proto(serialized_module)
                if module.name != "jit_train_step":
                    return None
                original = module.to_string()
                original_modules.append(original)
                if artifact_directory is not None:
                    write_gzip_text(directory / "original-gpu-pre-scheduler-hlo.txt.gz", original)
                plan = plan_routed_training_typed_ffi(original)
                forward = generate_cuda_routed_forward_ffi(plan.forward, target=_TARGETS.forward)
                input_adjoint = generate_cuda_routed_input_adjoint_ffi(
                    plan.input_adjoint,
                    target=_TARGETS.input_adjoint,
                )
                weights = tuple(
                    generate_cuda_group_batched_contract_ffi(weight, target=target)
                    for weight, target in zip(plan.weight_gradients, _TARGETS.weight_gradients, strict=True)
                )
                generated_sources = (forward.source, input_adjoint.source, *(weight.source for weight in weights))
                if any("atomicAdd(" in source for source in generated_sources):
                    raise RuntimeError("generated routed training source contains semantic atomic accumulation")
                compile_target(forward.source, _TARGETS.forward, "routed_forward")
                compile_target(input_adjoint.source, _TARGETS.input_adjoint, "routed_input_adjoint")
                for index, (weight, target) in enumerate(zip(weights, _TARGETS.weight_gradients, strict=True)):
                    compile_target(weight.source, target, f"group_batched_weight_gradient_{index}")
                rewritten = replace_routed_training_regions_with_custom_calls(
                    original,
                    plan,
                    targets=_TARGETS,
                )
                recovered_plans.append(plan)
                generated_programs.append((forward, input_adjoint, weights))
                transformed_module = hlo.hlo_module_from_text(rewritten)
                transformed_text = transformed_module.to_string()
                transformed_modules.append(transformed_text)
                replacement_audits.append(
                    audit_routed_training_replacement(original, transformed_text, plan, targets=_TARGETS)
                )
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

            call_counts = {
                "forward": _call_count(libraries["routed_forward"], "shuttle_routed_forward_call_count"),
                "input_adjoint": _call_count(
                    libraries["routed_input_adjoint"],
                    "shuttle_routed_input_adjoint_call_count",
                ),
                "weight_gradients": [
                    _call_count(
                        libraries[f"group_batched_weight_gradient_{index}"],
                        "shuttle_routed_weight_gradient_call_count",
                    )
                    for index in range(2)
                ],
            }
    finally:
        if artifact_directory is not None:
            for name in (
                "routed_forward",
                "routed_input_adjoint",
                "group_batched_weight_gradient_0",
                "group_batched_weight_gradient_1",
            ):
                compiled_library = directory / name / "generated_pair_map_handler.so"
                if compiled_library.exists():
                    compiled_library.unlink()
        if temporary is not None:
            temporary.cleanup()

    if (
        len(original_modules) != 1
        or len(transformed_modules) != 1
        or len(recovered_plans) != 1
        or len(replacement_audits) != 1
    ):
        raise RuntimeError("expected one combined routed training replacement pass")
    original = original_modules[0]
    transformed_hlo = transformed_modules[0]
    plan = recovered_plans[0]
    replacement_audit = replacement_audits[0]
    forward, input_adjoint, weights = generated_programs[0]
    target_occurrences = {
        "forward": transformed_hlo.count(_TARGETS.forward),
        "input_adjoint": transformed_hlo.count(_TARGETS.input_adjoint),
        "weight_gradients": [transformed_hlo.count(target) for target in _TARGETS.weight_gradients],
    }
    minimum_calls = repeats + warmup + 1
    observed_calls = [call_counts["forward"], call_counts["input_adjoint"], *call_counts["weight_gradients"]]
    observed_occurrences = [
        target_occurrences["forward"],
        target_occurrences["input_adjoint"],
        *target_occurrences["weight_gradients"],
    ]
    if observed_occurrences != [1, 1, 1, 1] or any(count < minimum_calls for count in observed_calls):
        raise RuntimeError(f"custom-call evidence mismatch: occurrences={observed_occurrences}, calls={observed_calls}")
    all_operands = tuple(
        dict.fromkeys(
            (
                *(operand.value.instruction for operand in plan.forward.operands),
                *(operand.value.instruction for operand in plan.input_adjoint.operands),
                *(operand.value.instruction for weight in plan.weight_gradients for operand in weight.operands),
            )
        )
    )
    parameter_ancestors = entry_parameter_ancestors(original, all_operands)
    static_roles = tuple(
        operand.role.value
        for operand in plan.input_adjoint.operands
        if not parameter_ancestors[operand.value.instruction]
    )
    if static_roles != (RoutedInputAdjointFfiOperandRole.FOLD_INITIAL.value,):
        raise RuntimeError(f"unexpected routed training static operands: {static_roles}")
    baseline_median = statistics.median(baseline_samples)
    transformed_median = statistics.median(transformed_samples)
    if artifact_directory is not None:
        write_gzip_text(directory / "transformed-gpu-pre-scheduler-hlo.txt.gz", transformed_hlo)
    return {
        "kind": "xla_grug_combined_routed_training_generated_ffi",
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "platform": "cuda",
        "device_kind": jax.devices()[0].device_kind,
        "architecture": architecture,
        "natural_frontend": "ordinary one-layer Grug train step with JAX-owned differentiation",
        "generated_regions": [
            "Contract -> Map -> Contract -> source Fold",
            "Contract -> reverse Map -> Contract -> source Fold",
            "group-batched weight Contract 0",
            "group-batched weight Contract 1",
        ],
        "numerical_policies": {
            "forward": plan.forward.region.numerical_policy.value,
            "input_adjoint": plan.input_adjoint.region.numerical_policy.value,
            "weight_gradients": [weight.numerical_contract.numerical_policy.value for weight in plan.weight_gradients],
        },
        "external_collectives": replacement_audit.weight_gradient_collectives,
        "uses_atomic_accumulation": False,
        "static_operand_roles": static_roles,
        "operand_ancestry": {operand: parameter_ancestors[operand] for operand in all_operands},
        "output_alias_operands": [weight.output_alias_operand for weight in plan.weight_gradients],
        "custom_call_targets": {
            "forward": _TARGETS.forward,
            "input_adjoint": _TARGETS.input_adjoint,
            "weight_gradients": _TARGETS.weight_gradients,
        },
        "custom_call_occurrences_in_transformed_hlo": target_occurrences,
        "custom_call_handler_executions": call_counts,
        "input_adjoint_auxiliary": replacement_audit.input_adjoint_auxiliary,
        "target_instruction_names": replacement_audit.target_instructions,
        "copy_count": dict(zip(("original", "transformed"), replacement_audit.copy_count, strict=True)),
        "transpose_count": dict(zip(("original", "transformed"), replacement_audit.transpose_count, strict=True)),
        "generated_semantic_sha256": {
            "forward": forward.semantic_digest,
            "input_adjoint": input_adjoint.semantic_digest,
            "weight_gradients": [weight.semantic_digest for weight in weights],
        },
        "generated_source_sha256": {
            "forward": forward.source_digest,
            "input_adjoint": input_adjoint.source_digest,
            "weight_gradients": [weight.source_digest for weight in weights],
        },
        "baseline_median_ms": baseline_median,
        "generated_median_ms": transformed_median,
        "generated_over_baseline": transformed_median / baseline_median,
        "baseline_unique_output_hashes": sorted(set(baseline_hashes)),
        "generated_unique_output_hashes": sorted(set(transformed_hashes)),
        "raw_samples": raw_samples,
        **comparison,
        "outputs_match": True,
        "remaining_ownership_gap": (
            "The rematerialized routed forward chain, relation index plane, and placement collectives remain under "
            "XLA; this checkpoint composes four existing generated physical regions without a megakernel."
        ),
    }


def _call_count(library: ctypes.CDLL, symbol: str) -> int:
    function = getattr(library, symbol)
    function.restype = ctypes.c_int
    return int(function())


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
