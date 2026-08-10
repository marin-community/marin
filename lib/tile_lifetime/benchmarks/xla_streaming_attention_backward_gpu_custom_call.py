#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replace one natural JAX reverse entry at XLA PRE_SCHEDULER."""

from __future__ import annotations

import argparse
import ctypes
import gzip
import hashlib
import importlib
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from benchmark_metadata import (  # pyrefly: ignore[missing-import]
    command_record,
    file_sha256,
    nvidia_smi_snapshot,
    toolchain_snapshot,
)

from shuttle.experimental.stablehlo_import import import_stablehlo
from tile_lifetime.jax_streaming_attention_backward_ffi import (
    call_streaming_attention_backward_ffi,
    compile_streaming_attention_backward_ffi,
    generate_streaming_attention_backward_ffi,
    register_streaming_attention_backward_ffi,
)
from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.stablehlo_streaming_attention_backward import (
    recover_experimental_whole_pattern_streaming_attention_backward,
)
from tile_lifetime.streaming_attention import StreamingTileSchedule
from tile_lifetime.streaming_attention_backward import (
    StreamingAttentionBackwardDomainTraversal,
    derive_streaming_attention_backward_tile_schedule,
    eliminate_normalized_exp_maximum_vjp,
)
from tile_lifetime.streaming_attention_backward_reference import (
    STREAMING_ATTENTION_BACKWARD_INPUT_NAMES,
    StreamingAttentionBackwardDebugConfig,
    causal_gqa_attention_vjp,
    export_debug_streaming_attention_backward,
)
from tile_lifetime.xla_streaming_attention_backward_ffi import (
    derive_streaming_attention_backward_ffi_output_layouts,
    plan_streaming_attention_backward_hlo_replacement,
    replace_streaming_attention_backward_entry_with_custom_call,
)

_PASS_NAME = "shuttle_streaming_reverse_entry_v1"
_CAPTURE_PASS_NAME = "shuttle_streaming_reverse_layout_capture_v1"


def _error(actual: jax.Array, expected: jax.Array) -> dict[str, float]:
    difference = np.abs(np.asarray(actual, dtype=np.float32) - np.asarray(expected, dtype=np.float32))
    return {
        "maximum_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
    }


def _hash(values: tuple[jax.Array, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def _timed_execution(executable: Any, arguments: tuple[jax.Array, ...], *, iterations: int) -> tuple[float, str]:
    start = time.perf_counter()
    result = None
    for _ in range(iterations):
        result = executable(*arguments)
    if result is None:
        raise RuntimeError("timed streaming reverse produced no result")
    jax.block_until_ready(result)
    return (time.perf_counter() - start) * 1e3 / iterations, _hash(tuple(result))


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Compile and execute baseline and transformed natural JAX VJPs."""
    if not jax.devices() or jax.devices()[0].platform != "gpu":
        raise RuntimeError("streaming reverse HLO replacement requires a CUDA JAX device")
    if args.repeats <= 0 or args.repeats % 6:
        raise ValueError("three-path counterbalanced replay requires a positive repeat count divisible by six")
    if args.iterations <= 0 or args.warmups < 0 or args.determinism_repeats <= 0:
        raise ValueError("iteration counts must be positive and warmups must be nonnegative")
    scale = args.head_dimension**-0.5
    config = StreamingAttentionBackwardDebugConfig(
        batch=1,
        query_length=args.sequence,
        key_length=args.sequence,
        query_heads=args.query_heads,
        key_value_heads=args.key_value_heads,
        head_dimension=args.head_dimension,
        scale=scale,
    )
    source_stablehlo = export_debug_streaming_attention_backward(config)
    args.artifact_directory.mkdir(parents=True, exist_ok=True)
    (args.artifact_directory / "source-vjp-stablehlo.mlir.bc").write_bytes(source_stablehlo)
    graph = import_stablehlo(source_stablehlo, input_names=STREAMING_ATTENTION_BACKWARD_INPUT_NAMES)
    recovered = recover_experimental_whole_pattern_streaming_attention_backward(
        graph,
        schedule=StreamingTileSchedule(
            query_tile_size=args.block_m,
            key_value_tile_size=args.block_n,
            pipeline_depth=args.num_stages,
        ),
    )
    program = eliminate_normalized_exp_maximum_vjp(
        recovered.program,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    schedule = derive_streaming_attention_backward_tile_schedule(
        program,
        query_tile_size=args.block_m,
        key_value_tile_size=args.block_n,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR,
    )
    target = (
        f"shuttle.streaming_reverse.entry_s{args.sequence}_d{args.head_dimension}_"
        f"bm{args.block_m}_bn{args.block_n}_layout_v2"
    )
    default_generated = generate_streaming_attention_backward_ffi(
        program,
        schedule,
        target_name=target,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )
    key = jax.random.key(20260809)
    arguments = tuple(
        jax.random.normal(fold_key, specification.shape, dtype=jnp.bfloat16)
        for fold_key, specification in zip(
            jax.random.split(key, len(default_generated.inputs)),
            default_generated.inputs,
            strict=True,
        )
    )
    jax.config.update("jax_enable_compilation_cache", False)
    hlo = importlib.import_module("jaxlib._hlo")
    xla = importlib.import_module("jax.extend.xla")
    captured_modules: list[str] = []
    captured_protos: list[bytes] = []

    def capture(serialized_module: bytes) -> bytes:
        module = hlo.HloModule.from_serialized_hlo_module_proto(serialized_module)
        captured_modules.append(module.to_string())
        captured_protos.append(serialized_module)
        return serialized_module

    xla.register_hlo_module_transformation(
        capture,
        name=_CAPTURE_PASS_NAME,
        stage=xla.PipelineStage.PRE_SCHEDULER,
        platforms="cuda",
    )
    jax.clear_caches()
    try:
        baseline = jax.jit(causal_gqa_attention_vjp(config)).lower(*arguments).compile()
    finally:
        xla.clear_hlo_module_transformation(
            _CAPTURE_PASS_NAME,
            stage=xla.PipelineStage.PRE_SCHEDULER,
            platforms="cuda",
        )
    if len(captured_modules) != 1 or len(captured_protos) != 1:
        raise RuntimeError("expected exactly one natural reverse layout-capture callback")
    default_plan = plan_streaming_attention_backward_hlo_replacement(
        captured_modules[0],
        program,
        default_generated,
    )
    generated = generate_streaming_attention_backward_ffi(
        program,
        schedule,
        target_name=target,
        output_layouts=derive_streaming_attention_backward_ffi_output_layouts(default_plan),
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )
    compiled = compile_streaming_attention_backward_ffi(
        generated,
        repository=args.repository,
        directory=args.build_directory,
        nvcc=args.nvcc,
        architecture=args.architecture,
        triton_target=args.triton_target,
    )
    register_streaming_attention_backward_ffi(compiled)
    expected = baseline(*arguments)
    jax.block_until_ready(expected)

    def direct_call(*values: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        return call_streaming_attention_backward_ffi(
            generated,
            query=values[0],
            key=values[1],
            value=values[2],
            output_cotangent=values[3],
        )

    direct = jax.jit(direct_call).lower(*arguments).compile()
    direct_actual = direct(*arguments)
    jax.block_until_ready(direct_actual)

    original_modules: list[str] = []
    transformed_modules: list[str] = []
    original_protos: list[bytes] = []
    transformed_protos: list[bytes] = []
    plans: list[Any] = []

    def replace(serialized_module: bytes) -> bytes:
        module = hlo.HloModule.from_serialized_hlo_module_proto(serialized_module)
        original = module.to_string()
        # Persist the actual natural-JAX GPU module before planning so a
        # fail-closed structural mismatch still leaves an exact repro artifact.
        (args.artifact_directory / "original-pre-scheduler-hlo.pb").write_bytes(serialized_module)
        (args.artifact_directory / "original-pre-scheduler-hlo.txt.gz").write_bytes(
            gzip.compress(original.encode(), mtime=0)
        )
        plan = plan_streaming_attention_backward_hlo_replacement(original, program, generated)
        transformed_text = replace_streaming_attention_backward_entry_with_custom_call(
            original,
            plan,
            target=target,
        )
        transformed = hlo.hlo_module_from_text(transformed_text)
        transformed_proto = transformed.as_serialized_hlo_module_proto()
        original_modules.append(original)
        transformed_modules.append(transformed.to_string())
        original_protos.append(serialized_module)
        transformed_protos.append(transformed_proto)
        plans.append(plan)
        return transformed_proto

    xla.register_hlo_module_transformation(
        replace,
        name=_PASS_NAME,
        stage=xla.PipelineStage.PRE_SCHEDULER,
        platforms="cuda",
    )
    jax.clear_caches()
    try:
        transformed = jax.jit(causal_gqa_attention_vjp(config)).lower(*arguments).compile()
    finally:
        xla.clear_hlo_module_transformation(
            _PASS_NAME,
            stage=xla.PipelineStage.PRE_SCHEDULER,
            platforms="cuda",
        )
    actual = transformed(*arguments)
    jax.block_until_ready(actual)
    if not (
        len(original_modules)
        == len(transformed_modules)
        == len(original_protos)
        == len(transformed_protos)
        == len(plans)
        == 1
    ):
        raise RuntimeError("expected exactly one whole-entry streaming reverse transformation")
    call_count = compiled.library.shuttle_streaming_attention_backward_ffi_call_count
    call_count.restype = ctypes.c_int
    executions = int(call_count())
    if executions < 1 or transformed_modules[0].count(target) != 1:
        raise RuntimeError("generated streaming reverse handler did not execute")

    plan = plans[0]
    if any(value.physical_shape != value.ffi_shape for value in (*plan.inputs, *plan.outputs)):
        raise RuntimeError("layout-native replacement unexpectedly retained a boundary copy")
    if tuple(value.physical_shape for value in default_plan.outputs) != tuple(
        value.physical_shape for value in plan.outputs
    ):
        raise RuntimeError("captured and replacement output layouts disagree")
    shuttle_copy_count = len(re.findall(r"^\s*%shuttle\.[^\n]*\scopy\(", transformed_modules[0], flags=re.MULTILINE))
    if shuttle_copy_count:
        raise RuntimeError(f"layout-native replacement emitted {shuttle_copy_count} Shuttle copies")
    errors = {
        implementation: {
            role: _error(output, reference)
            for role, output, reference in zip(
                ("query", "key", "value"),
                outputs,
                expected,
                strict=True,
            )
        }
        for implementation, outputs in (("transformed", actual), ("direct", direct_actual))
    }
    if any(
        error["maximum_absolute_error"] > 0.03125
        for implementation_errors in errors.values()
        for error in implementation_errors.values()
    ):
        raise RuntimeError(f"streaming reverse maximum error exceeds the accepted BF16 bound: {errors}")
    if any(
        error["mean_absolute_error"] > 2e-4
        for implementation_errors in errors.values()
        for error in implementation_errors.values()
    ):
        raise RuntimeError(f"streaming reverse mean error exceeds the accepted BF16 bound: {errors}")

    executables = {"baseline": baseline, "transformed": transformed, "direct": direct}
    counterbalanced_orders = (
        ("baseline", "transformed", "direct"),
        ("baseline", "direct", "transformed"),
        ("transformed", "baseline", "direct"),
        ("transformed", "direct", "baseline"),
        ("direct", "baseline", "transformed"),
        ("direct", "transformed", "baseline"),
    )
    for warmup in range(args.warmups):
        order = counterbalanced_orders[warmup % len(counterbalanced_orders)]
        for name in order:
            _timed_execution(executables[name], arguments, iterations=1)
    raw_samples: list[dict[str, Any]] = []
    samples: dict[str, list[float]] = {name: [] for name in executables}
    output_hashes: dict[str, list[str]] = {name: [] for name in executables}
    telemetry_before = nvidia_smi_snapshot()
    for repeat in range(args.repeats):
        order = counterbalanced_orders[repeat % len(counterbalanced_orders)]
        sample: dict[str, Any] = {"order": order}
        for name in order:
            latency, output_hash = _timed_execution(
                executables[name],
                arguments,
                iterations=args.iterations,
            )
            sample[name] = {"latency_ms": latency, "output_hash": output_hash}
            samples[name].append(latency)
            output_hashes[name].append(output_hash)
        raw_samples.append(sample)
    telemetry_after = nvidia_smi_snapshot()
    determinism = {
        name: [_timed_execution(executable, arguments, iterations=1)[1] for _ in range(args.determinism_repeats)]
        for name, executable in executables.items()
    }
    executions = int(call_count())
    minimum_handler_executions = 2 + 2 * args.warmups + 2 * args.repeats * args.iterations + 2 * args.determinism_repeats
    if executions < minimum_handler_executions:
        raise RuntimeError(
            f"typed-FFI handler executed {executions} times; expected at least {minimum_handler_executions}"
        )
    if any(len(set(hashes)) != 1 for hashes in determinism.values()):
        raise RuntimeError(f"repeated executions were not deterministic: {determinism}")
    baseline_median = statistics.median(samples["baseline"])
    transformed_median = statistics.median(samples["transformed"])
    direct_median = statistics.median(samples["direct"])
    observed_revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = {
        "kind": "xla_streaming_attention_backward_pre_scheduler_replacement",
        "command": command_record(),
        "requested_shuttle_revision": args.shuttle_revision,
        "observed_shuttle_revision": observed_revision,
        "holder_revision": args.holder_revision,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "platform": "cuda",
        "device_kind": jax.devices()[0].device_kind,
        "architecture": args.architecture,
        "allocation": {
            "gpu_variant": "H100",
            "gpu_count": 1,
            "cpu": args.allocation_cpu,
            "memory": args.allocation_memory,
            "disk": args.allocation_disk,
            "priority": args.allocation_priority,
        },
        "toolchain": toolchain_snapshot(str(args.nvcc)),
        "telemetry_before": telemetry_before,
        "telemetry_after": telemetry_after,
        "natural_frontend": "ordinary JAX tensor algebra differentiated by JAX",
        "pipeline_stage": "PRE_SCHEDULER",
        "state_policy": plan.state_policy.value,
        "maximum_vjp": plan.maximum_vjp,
        "reassociation": plan.reassociation,
        "semantic_fingerprint": plan.semantic_fingerprint,
        "score_scale": plan.provenance.score_scale,
        "contract_count": 1 + len(plan.provenance.reverse_contracts),
        "additive_fold_count": len(plan.provenance.additive_folds),
        "domain_restriction_recovered": plan.provenance.domain_restriction is not None,
        "input_roles": [value.role.value for value in plan.inputs],
        "output_roles": [value.role.value for value in plan.outputs],
        "output_layouts": {
            output.name: {
                "minor_to_major": output.layout,
                "logical_strides": output.strides,
            }
            for output in generated.outputs
        },
        "layout_binding": {
            "default_output_copy_count": sum(value.physical_shape != value.ffi_shape for value in default_plan.outputs),
            "layout_native_output_copy_count": sum(value.physical_shape != value.ffi_shape for value in plan.outputs),
            "transformed_shuttle_copy_count": shuttle_copy_count,
        },
        "custom_call_occurrences": transformed_modules[0].count(target),
        "handler_executions": executions,
        "errors": errors,
        "baseline_hash": _hash(expected),
        "transformed_hash": _hash(actual),
        "direct_hash": _hash(direct_actual),
        "determinism": {
            name: {"hashes": hashes, "bitwise_stable": len(set(hashes)) == 1} for name, hashes in determinism.items()
        },
        "timing": {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "iterations_per_sample": args.iterations,
            "baseline_median_ms": baseline_median,
            "transformed_median_ms": transformed_median,
            "direct_median_ms": direct_median,
            "transformed_over_baseline": transformed_median / baseline_median,
            "transformed_over_direct": transformed_median / direct_median,
            "raw_samples": raw_samples,
            "unique_output_hashes": {name: sorted(set(hashes)) for name, hashes in output_hashes.items()},
        },
        "hlo": {
            "captured_proto_sha256": hashlib.sha256(captured_protos[0]).hexdigest(),
            "captured_text_sha256": hashlib.sha256(captured_modules[0].encode()).hexdigest(),
            "original_proto_sha256": hashlib.sha256(original_protos[0]).hexdigest(),
            "transformed_proto_sha256": hashlib.sha256(transformed_protos[0]).hexdigest(),
            "original_text_sha256": hashlib.sha256(original_modules[0].encode()).hexdigest(),
            "transformed_text_sha256": hashlib.sha256(transformed_modules[0].encode()).hexdigest(),
        },
        "generated_handler": {
            "source_sha256": file_sha256(compiled.source_path),
            "library_sha256": file_sha256(compiled.library_path),
            "aot_source_sha256": {source.name: file_sha256(source) for source in compiled.aot_sources},
        },
        "runtime_imports": {
            "torch": "torch" in sys.modules,
            "triton": "triton" in sys.modules,
        },
    }
    (args.artifact_directory / "captured-pre-scheduler-hlo.pb").write_bytes(captured_protos[0])
    (args.artifact_directory / "captured-pre-scheduler-hlo.txt.gz").write_bytes(
        gzip.compress(captured_modules[0].encode(), mtime=0)
    )
    (args.artifact_directory / "original-pre-scheduler-hlo.pb").write_bytes(original_protos[0])
    (args.artifact_directory / "transformed-pre-scheduler-hlo.pb").write_bytes(transformed_protos[0])
    (args.artifact_directory / "original-pre-scheduler-hlo.txt.gz").write_bytes(
        gzip.compress(original_modules[0].encode(), mtime=0)
    )
    (args.artifact_directory / "transformed-pre-scheduler-hlo.txt.gz").write_bytes(
        gzip.compress(transformed_modules[0].encode(), mtime=0)
    )
    (args.artifact_directory / "generated-handler.cu").write_bytes(compiled.source_path.read_bytes())
    (args.artifact_directory / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    artifact_files = sorted(path for path in args.artifact_directory.iterdir() if path.is_file())
    checksum_lines = [f"{file_sha256(path)}  {path.name}" for path in artifact_files]
    (args.artifact_directory / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--architecture", default="sm_90a")
    parser.add_argument("--triton-target")
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--key-value-heads", type=int, default=8)
    parser.add_argument("--head-dimension", type=int, choices=(64, 128), default=128)
    parser.add_argument("--block-m", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--block-n", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--num-warps", type=int, choices=(4, 8), default=8)
    parser.add_argument("--num-stages", type=int, choices=(2, 3, 4), default=3)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--determinism-repeats", type=int, default=5)
    parser.add_argument("--shuttle-revision", required=True)
    parser.add_argument("--holder-revision", required=True)
    parser.add_argument("--allocation-cpu", type=float, required=True)
    parser.add_argument("--allocation-memory", required=True)
    parser.add_argument("--allocation-disk", required=True)
    parser.add_argument("--allocation-priority", required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run_smoke(_parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
