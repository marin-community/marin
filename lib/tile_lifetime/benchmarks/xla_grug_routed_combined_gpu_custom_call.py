#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute seven generic generated regions in one natural Grug train step."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import statistics
import subprocess
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
from tile_lifetime.cuda_axis_fold_codegen import generate_cuda_axis_fold_ffi
from tile_lifetime.jax_streaming_attention_backward_ffi import (
    compile_streaming_attention_backward_ffi,
    generate_streaming_attention_backward_ffi,
    register_streaming_attention_backward_ffi,
)
from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.stablehlo_import import import_stablehlo
from tile_lifetime.stablehlo_streaming_attention_backward import recover_stablehlo_streaming_attention_backward
from tile_lifetime.streaming_attention import StreamingTileSchedule
from tile_lifetime.streaming_attention_backward import (
    StreamingAttentionBackwardDomainTraversal,
    derive_streaming_attention_backward_tile_schedule,
    eliminate_normalized_exp_maximum_vjp,
)
from tile_lifetime.streaming_attention_backward_reference import (
    STREAMING_ATTENTION_BACKWARD_INPUT_NAMES,
    StreamingAttentionBackwardDebugConfig,
    export_debug_streaming_attention_backward,
)
from tile_lifetime.xla_axis_fold_ffi import recover_axis_fold_hlo_region_candidates
from tile_lifetime.xla_relation_program_recovery import RoutedInputAdjointFfiOperandRole
from tile_lifetime.xla_routed_forward_ffi import generate_cuda_routed_forward_ffi
from tile_lifetime.xla_routed_input_adjoint_ffi import generate_cuda_routed_input_adjoint_ffi
from tile_lifetime.xla_routed_training_ffi import (
    RoutedTrainingAndAttentionFfiTargets,
    RoutedTrainingAttentionAndAxisFoldFfiTargets,
    RoutedTrainingFfiTargets,
    audit_routed_training_attention_and_axis_fold_replacement,
    entry_parameter_ancestors,
    plan_routed_training_and_attention_typed_ffi,
    plan_routed_training_attention_and_axis_fold_typed_ffi,
    replace_routed_training_attention_and_axis_fold_regions_with_custom_calls,
)
from tile_lifetime.xla_routed_weight_gradient_ffi import generate_cuda_group_batched_contract_ffi
from tile_lifetime.xla_streaming_attention_backward_ffi import (
    audit_streaming_attention_backward_region_replacement,
    derive_streaming_attention_backward_ffi_output_layouts,
)

_PASS_NAME = "shuttle_grug_routed_training_gpu_v1"
_ROUTED_ATTENTION_TARGETS = RoutedTrainingAndAttentionFfiTargets(
    routed=RoutedTrainingFfiTargets(
        forward="shuttle.routed_training.forward.v2",
        input_adjoint="shuttle.routed_training.input_adjoint.v2",
        weight_gradients=(
            "shuttle.routed_training.weight_gradient.0.v2",
            "shuttle.routed_training.weight_gradient.1.v2",
        ),
    ),
    attention_backward="shuttle.routed_training.attention_backward.v2",
)
_TARGETS = RoutedTrainingAttentionAndAxisFoldFfiTargets(
    routed_attention=_ROUTED_ATTENTION_TARGETS,
    axis_folds=(
        "shuttle.routed_training.axis_fold.0.v1",
        "shuttle.routed_training.axis_fold.1.v1",
    ),
)


def _attention_reverse_program() -> tuple[Any, Any, bytes]:
    config = StreamingAttentionBackwardDebugConfig(
        batch=2,
        query_length=4,
        key_length=4,
        query_heads=2,
        key_value_heads=1,
        head_dimension=16,
        scale=0.32421875,
    )
    source_stablehlo = export_debug_streaming_attention_backward(config)
    graph = import_stablehlo(source_stablehlo, input_names=STREAMING_ATTENTION_BACKWARD_INPUT_NAMES)
    recovered = recover_stablehlo_streaming_attention_backward(
        graph,
        schedule=StreamingTileSchedule(query_tile_size=4, key_value_tile_size=4, pipeline_depth=2),
    )
    program = eliminate_normalized_exp_maximum_vjp(
        recovered.program,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    schedule = derive_streaming_attention_backward_tile_schedule(
        program,
        query_tile_size=4,
        key_value_tile_size=4,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR,
    )
    return program, schedule, source_stablehlo


def _generate_axis_fold_programs(hlo_text: str) -> tuple[Any, ...]:
    report = recover_axis_fold_hlo_region_candidates(
        hlo_text,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    selected = tuple(
        plan
        for plan in report.plans
        if plan.program.rows == 8 and plan.program.columns == 32 and plan.output_ffi_shape.startswith("bf16[")
    )
    if len(selected) != len(_TARGETS.axis_folds):
        raise RuntimeError(f"expected two Grug row-axis Folds, found {len(selected)}")
    return tuple(
        generate_cuda_axis_fold_ffi((plan.program,), target_name=target)
        for plan, target in zip(selected, _TARGETS.axis_folds, strict=True)
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


def _runtime_dependency_audit(library_path: Path, source: str) -> tuple[str, ...]:
    if "torch" in source.lower() or "triton" in source.lower():
        raise RuntimeError(f"generated runtime source contains a Torch/Triton reference: {library_path}")
    completed = subprocess.run(("ldd", str(library_path)), check=True, capture_output=True, text=True)
    dependencies = tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())
    forbidden = tuple(line for line in dependencies if "torch" in line.lower() or "triton" in line.lower())
    if forbidden:
        raise RuntimeError(f"generated runtime library links Torch/Triton: {forbidden}")
    return dependencies


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_smoke(
    nvcc: Path,
    architecture: str,
    artifact_directory: Path | None,
    *,
    repository: Path,
    triton_target: str | None,
    warmup: int = 4,
    repeats: int = 30,
) -> dict[str, Any]:
    """Compile, execute, and time the combined routed-plus-attention transform."""
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
    attention_program, attention_schedule, source_attention_stablehlo = _attention_reverse_program()
    default_attention = generate_streaming_attention_backward_ffi(
        attention_program,
        attention_schedule,
        target_name=_TARGETS.routed_attention.attention_backward,
    )
    if artifact_directory is not None:
        (directory / "source-attention-vjp-stablehlo.mlir.bc").write_bytes(source_attention_stablehlo)

    original_modules: list[str] = []
    transformed_modules: list[str] = []
    recovered_plans: list[Any] = []
    generated_programs: list[tuple[Any, Any, tuple[Any, Any], Any, tuple[Any, ...]]] = []
    replacement_audits: list[Any] = []
    attention_liveness_audits: list[Any] = []
    libraries: dict[str, ctypes.CDLL] = {}
    runtime_dependencies: dict[str, tuple[str, ...]] = {}
    attention_library_path: Path | None = None
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
                runtime_dependencies[name] = _runtime_dependency_audit(Path(library._name), source)
                return library

            def replace(serialized_module: bytes) -> bytes | None:
                nonlocal attention_library_path
                module = hlo.HloModule.from_serialized_hlo_module_proto(serialized_module)
                if module.name != "jit_train_step":
                    return None
                original = module.to_string()
                original_modules.append(original)
                if artifact_directory is not None:
                    write_gzip_text(directory / "original-gpu-pre-scheduler-hlo.txt.gz", original)
                default_plan = plan_routed_training_and_attention_typed_ffi(
                    original,
                    attention_program,
                    default_attention,
                )
                attention = generate_streaming_attention_backward_ffi(
                    attention_program,
                    attention_schedule,
                    target_name=_TARGETS.routed_attention.attention_backward,
                    output_layouts=derive_streaming_attention_backward_ffi_output_layouts(
                        default_plan.attention_backward
                    ),
                )
                generated_axis_folds = _generate_axis_fold_programs(original)
                plan = plan_routed_training_attention_and_axis_fold_typed_ffi(
                    original,
                    attention_program,
                    attention,
                    generated_axis_folds,
                    axis_fold_numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
                )
                forward = generate_cuda_routed_forward_ffi(
                    plan.routed_attention.routed.forward,
                    target=_TARGETS.routed_attention.routed.forward,
                )
                input_adjoint = generate_cuda_routed_input_adjoint_ffi(
                    plan.routed_attention.routed.input_adjoint,
                    target=_TARGETS.routed_attention.routed.input_adjoint,
                )
                weights = tuple(
                    generate_cuda_group_batched_contract_ffi(weight, target=target)
                    for weight, target in zip(
                        plan.routed_attention.routed.weight_gradients,
                        _TARGETS.routed_attention.routed.weight_gradients,
                        strict=True,
                    )
                )
                generated_sources = (
                    forward.source,
                    input_adjoint.source,
                    *(weight.source for weight in weights),
                    *(axis_fold.source for axis_fold in generated_axis_folds),
                )
                if any("atomicAdd(" in source for source in generated_sources):
                    raise RuntimeError("generated routed training source contains semantic atomic accumulation")
                compile_target(forward.source, _TARGETS.routed_attention.routed.forward, "routed_forward")
                compile_target(
                    input_adjoint.source,
                    _TARGETS.routed_attention.routed.input_adjoint,
                    "routed_input_adjoint",
                )
                for index, (weight, target) in enumerate(
                    zip(weights, _TARGETS.routed_attention.routed.weight_gradients, strict=True)
                ):
                    compile_target(weight.source, target, f"group_batched_weight_gradient_{index}")
                for index, (axis_fold, target) in enumerate(zip(generated_axis_folds, _TARGETS.axis_folds, strict=True)):
                    compile_target(axis_fold.source, target, f"axis_fold_{index}")
                compiled_attention = compile_streaming_attention_backward_ffi(
                    attention,
                    repository=repository,
                    directory=directory / "attention_backward",
                    nvcc=nvcc,
                    architecture=architecture,
                    triton_target=triton_target,
                )
                register_streaming_attention_backward_ffi(compiled_attention)
                libraries["attention_backward"] = compiled_attention.library
                runtime_dependencies["attention_backward"] = _runtime_dependency_audit(
                    compiled_attention.library_path,
                    compiled_attention.source_path.read_text(),
                )
                attention_library_path = compiled_attention.library_path
                rewritten = replace_routed_training_attention_and_axis_fold_regions_with_custom_calls(
                    original,
                    plan,
                    targets=_TARGETS,
                )
                recovered_plans.append(plan)
                generated_programs.append((forward, input_adjoint, weights, compiled_attention, generated_axis_folds))
                transformed_module = hlo.hlo_module_from_text(rewritten)
                transformed_text = transformed_module.to_string()
                transformed_modules.append(transformed_text)
                replacement_audits.append(
                    audit_routed_training_attention_and_axis_fold_replacement(
                        original,
                        transformed_text,
                        plan,
                        targets=_TARGETS,
                    )
                )
                attention_liveness_audits.append(
                    audit_streaming_attention_backward_region_replacement(
                        original,
                        transformed_text,
                        plan.routed_attention.attention_backward,
                        target=_TARGETS.routed_attention.attention_backward,
                    )
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
                "attention_backward": _call_count(
                    libraries["attention_backward"],
                    "shuttle_streaming_attention_backward_ffi_call_count",
                ),
                "axis_folds": [
                    _call_count(libraries[f"axis_fold_{index}"], "shuttle_axis_fold_ffi_call_count")
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
                "axis_fold_0",
                "axis_fold_1",
            ):
                compiled_library = directory / name / "generated_pair_map_handler.so"
                if compiled_library.exists():
                    compiled_library.unlink()
            if attention_library_path is not None and attention_library_path.exists():
                attention_library_path.unlink()
        if temporary is not None:
            temporary.cleanup()

    if (
        len(original_modules) != 1
        or len(transformed_modules) != 1
        or len(recovered_plans) != 1
        or len(replacement_audits) != 1
        or len(attention_liveness_audits) != 1
    ):
        raise RuntimeError("expected one combined routed-plus-attention-plus-Fold replacement pass")
    original = original_modules[0]
    transformed_hlo = transformed_modules[0]
    plan = recovered_plans[0]
    replacement_audit = replacement_audits[0]
    attention_liveness = attention_liveness_audits[0]
    forward, input_adjoint, weights, compiled_attention, generated_axis_folds = generated_programs[0]
    target_occurrences = {
        "forward": transformed_hlo.count(_TARGETS.routed_attention.routed.forward),
        "input_adjoint": transformed_hlo.count(_TARGETS.routed_attention.routed.input_adjoint),
        "weight_gradients": [
            transformed_hlo.count(target) for target in _TARGETS.routed_attention.routed.weight_gradients
        ],
        "attention_backward": transformed_hlo.count(_TARGETS.routed_attention.attention_backward),
        "axis_folds": [transformed_hlo.count(target) for target in _TARGETS.axis_folds],
    }
    minimum_calls = repeats + warmup + 1
    observed_calls = [
        call_counts["forward"],
        call_counts["input_adjoint"],
        *call_counts["weight_gradients"],
        call_counts["attention_backward"],
        *call_counts["axis_folds"],
    ]
    observed_occurrences = [
        target_occurrences["forward"],
        target_occurrences["input_adjoint"],
        *target_occurrences["weight_gradients"],
        target_occurrences["attention_backward"],
        *target_occurrences["axis_folds"],
    ]
    if observed_occurrences != [1, 1, 1, 1, 1, 1, 1] or any(count < minimum_calls for count in observed_calls):
        raise RuntimeError(f"custom-call evidence mismatch: occurrences={observed_occurrences}, calls={observed_calls}")
    all_operands = tuple(
        dict.fromkeys(
            (
                *(operand.value.instruction for operand in plan.routed_attention.routed.forward.operands),
                *(operand.value.instruction for operand in plan.routed_attention.routed.input_adjoint.operands),
                *(
                    operand.value.instruction
                    for weight in plan.routed_attention.routed.weight_gradients
                    for operand in weight.operands
                ),
                *(value.instruction for value in plan.routed_attention.attention_backward.inputs),
                *(value.instruction for fold in plan.axis_folds for value in fold.inputs),
            )
        )
    )
    parameter_ancestors = entry_parameter_ancestors(original, all_operands)
    static_roles = tuple(
        operand.role.value
        for operand in plan.routed_attention.routed.input_adjoint.operands
        if not parameter_ancestors[operand.value.instruction]
    )
    if static_roles != (RoutedInputAdjointFfiOperandRole.FOLD_INITIAL.value,):
        raise RuntimeError(f"unexpected routed training static operands: {static_roles}")
    baseline_median = statistics.median(baseline_samples)
    transformed_median = statistics.median(transformed_samples)
    if len(set(transformed_hashes)) != 1:
        raise RuntimeError("generated seven-call result is not bitwise deterministic")
    if artifact_directory is not None:
        write_gzip_text(directory / "transformed-gpu-pre-scheduler-hlo.txt.gz", transformed_hlo)
    return {
        "kind": "xla_grug_combined_routed_training_attention_and_axis_fold_generated_ffi",
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
            "causal GQA attention reverse Contract/Fold/DomainRestriction region",
            "generic 8x32 row-axis Fold 0",
            "generic 8x32 row-axis Fold 1",
        ],
        "numerical_policies": {
            "forward": plan.routed_attention.routed.forward.region.numerical_policy.value,
            "input_adjoint": plan.routed_attention.routed.input_adjoint.region.numerical_policy.value,
            "weight_gradients": [
                weight.numerical_contract.numerical_policy.value
                for weight in plan.routed_attention.routed.weight_gradients
            ],
            "attention_backward": plan.routed_attention.attention_backward.reassociation,
            "axis_folds": [fold.program.numerical_policy.value for fold in plan.axis_folds],
        },
        "external_collectives": replacement_audit.routed_attention.routed.weight_gradient_collectives,
        "uses_atomic_accumulation": False,
        "static_operand_roles": static_roles,
        "operand_ancestry": {operand: parameter_ancestors[operand] for operand in all_operands},
        "output_alias_operands": [
            weight.output_alias_operand for weight in plan.routed_attention.routed.weight_gradients
        ],
        "custom_call_targets": {
            "forward": _TARGETS.routed_attention.routed.forward,
            "input_adjoint": _TARGETS.routed_attention.routed.input_adjoint,
            "weight_gradients": _TARGETS.routed_attention.routed.weight_gradients,
            "attention_backward": _TARGETS.routed_attention.attention_backward,
            "axis_folds": _TARGETS.axis_folds,
        },
        "custom_call_occurrences_in_transformed_hlo": target_occurrences,
        "custom_call_handler_executions": call_counts,
        "input_adjoint_auxiliary": replacement_audit.routed_attention.routed.input_adjoint_auxiliary,
        "target_instruction_names": (
            *replacement_audit.routed_attention.routed.target_instructions,
            replacement_audit.routed_attention.attention_backward_instruction,
            *(axis_fold.call_instruction for axis_fold in replacement_audit.axis_folds),
        ),
        "copy_count": dict(
            zip(("original", "transformed"), replacement_audit.routed_attention.routed.copy_count, strict=True)
        ),
        "transpose_count": dict(
            zip(
                ("original", "transformed"),
                replacement_audit.routed_attention.routed.transpose_count,
                strict=True,
            )
        ),
        "attention_dead_reverse_closure": attention_liveness.dead_reverse_closure,
        "attention_preserved_shared_users": dict(attention_liveness.preserved_shared_users),
        "generated_semantic_sha256": {
            "forward": forward.semantic_digest,
            "input_adjoint": input_adjoint.semantic_digest,
            "weight_gradients": [weight.semantic_digest for weight in weights],
            "attention_backward": compiled_attention.generated.semantic_fingerprint,
            "axis_folds": [axis_fold.semantic_fingerprints for axis_fold in generated_axis_folds],
        },
        "generated_source_sha256": {
            "forward": forward.source_digest,
            "input_adjoint": input_adjoint.source_digest,
            "weight_gradients": [weight.source_digest for weight in weights],
            "attention_backward": {
                "handler": _sha256(compiled_attention.source_path),
                "aot_sources": [_sha256(path) for path in compiled_attention.aot_sources],
            },
            "axis_folds": [axis_fold.source_sha256 for axis_fold in generated_axis_folds],
        },
        "baseline_median_ms": baseline_median,
        "generated_median_ms": transformed_median,
        "generated_over_baseline": transformed_median / baseline_median,
        "generated_minus_baseline_ms": transformed_median - baseline_median,
        "independent_custom_call_count": 7,
        "latency_delta_per_custom_call_us": (transformed_median - baseline_median) * 1e3 / 7,
        "latency_delta_attribution": (
            "Whole-step delta includes seven fixed typed-FFI dispatches and changed kernels; the per-call quotient is "
            "reported only to expose the fixed-call scale, not as causal attribution."
        ),
        "baseline_unique_output_hashes": sorted(set(baseline_hashes)),
        "generated_unique_output_hashes": sorted(set(transformed_hashes)),
        "generated_runtime_dependencies": runtime_dependencies,
        "generated_runtime_torch_triton_free": True,
        "raw_samples": raw_samples,
        **comparison,
        "outputs_match": True,
        "remaining_ownership_gap": (
            "The rematerialized routed forward chain, relation index plane, and placement collectives remain under "
            "XLA; this checkpoint composes seven generic generated physical regions without a megakernel."
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
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--triton-target")
    parser.add_argument("--artifact-directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    result = run_smoke(
        args.nvcc,
        args.architecture,
        args.artifact_directory,
        repository=args.repository,
        triton_target=args.triton_target,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
