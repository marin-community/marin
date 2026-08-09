#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute JAX-owned row-statistic reverse algebra through generated CUDA FFI."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import statistics
import subprocess
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from tile_lifetime.cuda_axis_fold_codegen import AxisFoldPipelineSchedule, generate_cuda_axis_fold_ffi
from tile_lifetime.cuda_toolchain import cuda_toolkit_link_flags, cuda_toolkit_shared_library_link_flags
from tile_lifetime.jax_axis_fold_ffi import call_cuda_axis_fold_ffi, register_cuda_axis_fold_ffi
from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.stablehlo_import import import_stablehlo
from tile_lifetime.stablehlo_row_normalization_backward import (
    compile_stablehlo_row_normalization_backward,
    compile_stablehlo_row_normalization_backward_ffi,
)

_TARGET_NAME = "shuttle.axis_fold_reverse_v1"
_INPUT_TARGET_NAME = "shuttle.axis_fold_reverse_input_v1"
_FEATURE_TARGET_NAME = "shuttle.axis_fold_reverse_feature_v1"


def _normalization(x: jax.Array, feature_scale: jax.Array) -> jax.Array:
    local = x.astype(jnp.float32)
    inverse = jax.lax.rsqrt(jnp.mean(jnp.square(local), axis=-1, keepdims=True) + 1e-5)
    return (local * inverse * feature_scale.astype(jnp.float32)).astype(jnp.bfloat16)


def _natural_reverse(
    x: jax.Array,
    feature_scale: jax.Array,
    cotangent: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    _, pullback = jax.vjp(_normalization, x, feature_scale)
    return pullback(cotangent)


def _compile_generated_source(source: str, directory: Path, nvcc: Path, architecture: str) -> ctypes.CDLL:
    directory.mkdir(parents=True, exist_ok=True)
    source_path = directory / "generated_axis_fold_ffi.cu"
    library_path = directory / "generated_axis_fold_ffi.so"
    include_directory = Path(jaxlib.__file__).resolve().parent / "include"
    source_path.write_text(source + "\n")
    subprocess.run(
        (
            str(nvcc),
            "-std=c++17",
            "-O3",
            f"-arch={architecture}",
            "-shared",
            "-Xcompiler",
            "-fPIC",
            "-I",
            str(include_directory),
            str(source_path),
            "-o",
            str(library_path),
            "-cudart=none",
            *cuda_toolkit_link_flags(nvcc, runtime_search_path=True),
            *cuda_toolkit_shared_library_link_flags(nvcc, ("cudart",)),
        ),
        check=True,
    )
    return ctypes.CDLL(str(library_path))


def _error(actual: jax.Array, expected: jax.Array) -> dict[str, float]:
    difference = np.abs(np.asarray(actual, dtype=np.float32) - np.asarray(expected, dtype=np.float32))
    return {
        "maximum_absolute_error": float(difference.max(initial=0.0)),
        "mean_absolute_error": float(difference.mean()),
    }


def _hash(value: jax.Array) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _measure(
    variants: tuple[tuple[str, Any], ...],
    *,
    warmups: int,
    repeats: int,
    iterations: int,
) -> tuple[dict[str, dict[str, Any]], list[list[str]]]:
    for _ in range(warmups):
        for _, function in variants:
            jax.block_until_ready(function())
    samples: dict[str, list[float]] = {name: [] for name, _ in variants}
    orders: list[list[str]] = []
    for repeat in range(repeats):
        order = variants if repeat % 2 == 0 else tuple(reversed(variants))
        orders.append([name for name, _ in order])
        for name, function in order:
            started = time.perf_counter()
            result = None
            for _ in range(iterations):
                result = function()
            jax.block_until_ready(result)
            samples[name].append((time.perf_counter() - started) * 1e3 / iterations)
    return (
        {
            name: {
                "samples_ms": values,
                "median_ms": statistics.median(values),
                "minimum_ms": min(values),
            }
            for name, values in samples.items()
        },
        orders,
    )


def _handler_call_count(library: ctypes.CDLL) -> int:
    function = library.shuttle_axis_fold_ffi_call_count
    function.restype = ctypes.c_int
    return int(function())


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Compile ordinary JAX AD, register generated Fold code, and benchmark it."""
    if not jax.devices() or jax.devices()[0].platform != "gpu":
        raise RuntimeError("the JAX axis-Fold benchmark requires a CUDA device")
    if args.repeats % 2:
        raise ValueError("counterbalanced benchmark requires an even repeat count")
    arguments = (
        jax.ShapeDtypeStruct((args.rows, args.hidden), jnp.bfloat16),
        jax.ShapeDtypeStruct((args.hidden,), jnp.bfloat16),
        jax.ShapeDtypeStruct((args.rows, args.hidden), jnp.bfloat16),
    )
    exported = jax.export.export(jax.jit(_natural_reverse))(*arguments)
    serialized = exported.mlir_module_serialized
    graph = import_stablehlo(serialized, input_names=("matrix_a", "feature_vector", "matrix_b"))
    compilation = compile_stablehlo_row_normalization_backward_ffi(
        graph,
        target_name=_TARGET_NAME,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        threads=args.threads,
        feature_groups_per_block=args.column_groups_per_block,
        pipeline_schedule=AxisFoldPipelineSchedule(args.pipeline_schedule),
    )
    generated = compilation.generated
    library = _compile_generated_source(generated.source, args.artifact_directory, args.nvcc, args.architecture)
    register_cuda_axis_fold_ffi(generated, library)

    input_generated = None
    input_library = None
    feature_generated = None
    feature_library = None
    if args.profile_components:
        component_compilation = compile_stablehlo_row_normalization_backward(graph, threads=args.threads)
        programs = (
            component_compilation.programs.input_cotangent,
            replace(
                component_compilation.programs.feature_scale_cotangent,
                groups_per_block=args.column_groups_per_block,
            ),
        )
        input_generated = generate_cuda_axis_fold_ffi((programs[0],), target_name=_INPUT_TARGET_NAME)
        input_library = _compile_generated_source(
            input_generated.source,
            args.artifact_directory / "input_cotangent",
            args.nvcc,
            args.architecture,
        )
        register_cuda_axis_fold_ffi(input_generated, input_library)
        feature_generated = generate_cuda_axis_fold_ffi((programs[1],), target_name=_FEATURE_TARGET_NAME)
        feature_library = _compile_generated_source(
            feature_generated.source,
            args.artifact_directory / "feature_scale_cotangent",
            args.nvcc,
            args.architecture,
        )
        register_cuda_axis_fold_ffi(feature_generated, feature_library)

    key = jax.random.key(args.seed)
    x_key, scale_key, cotangent_key = jax.random.split(key, 3)
    x = jax.random.normal(x_key, (args.rows, args.hidden), dtype=jnp.bfloat16)
    feature_scale = jax.random.normal(scale_key, (args.hidden,), dtype=jnp.bfloat16)
    cotangent = jax.random.normal(cotangent_key, (args.rows, args.hidden), dtype=jnp.bfloat16)
    local = x.astype(jnp.float32)
    inverse_scale = jax.lax.rsqrt(jnp.mean(jnp.square(local), axis=-1) + 1e-5)
    standardized = (local * inverse_scale[:, None]).astype(jnp.bfloat16)
    projected = cotangent.astype(jnp.float32)

    @jax.jit
    def generated_reverse(
        primal_argument: jax.Array,
        feature_scale_argument: jax.Array,
        output_cotangent_argument: jax.Array,
    ) -> tuple[jax.Array, ...]:
        return call_cuda_axis_fold_ffi(
            generated,
            {
                "primal": primal_argument,
                "feature_scale": feature_scale_argument,
                "output_cotangent": output_cotangent_argument,
            },
        )

    @jax.jit
    def matched_xla_algebra(
        primal_argument: jax.Array,
        feature_scale_argument: jax.Array,
        output_cotangent_argument: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return _natural_reverse(primal_argument, feature_scale_argument, output_cotangent_argument)

    @jax.jit
    def matched_xla_input_cotangent(
        projected_argument: jax.Array,
        feature_scale_argument: jax.Array,
        standardized_argument: jax.Array,
        inverse_scale_argument: jax.Array,
    ) -> jax.Array:
        scaled = projected_argument * feature_scale_argument.astype(jnp.float32)
        correlation = jnp.sum(scaled * standardized_argument.astype(jnp.float32), axis=1, keepdims=True) / args.hidden
        return inverse_scale_argument[:, None] * (scaled - standardized_argument.astype(jnp.float32) * correlation)

    @jax.jit
    def matched_xla_feature_scale_cotangent(
        projected_argument: jax.Array,
        standardized_argument: jax.Array,
    ) -> jax.Array:
        return jnp.sum(projected_argument * standardized_argument.astype(jnp.float32), axis=0)

    def execute_generated_reverse() -> tuple[jax.Array, ...]:
        return generated_reverse(x, feature_scale, cotangent)

    def execute_matched_xla_algebra() -> tuple[jax.Array, jax.Array]:
        return matched_xla_algebra(x, feature_scale, cotangent)

    if args.xla_dump_directory is not None:
        args.xla_dump_directory.mkdir(parents=True, exist_ok=True)
        (args.xla_dump_directory / "matched_full_optimized_hlo.txt").write_text(
            matched_xla_algebra.lower(x, feature_scale, cotangent).compile().as_text()
        )
        (args.xla_dump_directory / "matched_input_cotangent_optimized_hlo.txt").write_text(
            matched_xla_input_cotangent.lower(projected, feature_scale, standardized, inverse_scale).compile().as_text()
        )
        (args.xla_dump_directory / "matched_feature_scale_cotangent_optimized_hlo.txt").write_text(
            matched_xla_feature_scale_cotangent.lower(projected, standardized).compile().as_text()
        )

    generated_outputs = execute_generated_reverse()
    xla_outputs = execute_matched_xla_algebra()
    natural_outputs = jax.jit(_natural_reverse)(x, feature_scale, cotangent)
    jax.block_until_ready((generated_outputs, xla_outputs, natural_outputs))
    correctness = {
        "natural_jax_vjp": {
            "input_cotangent": _error(generated_outputs[0], xla_outputs[0]),
            "feature_scale_cotangent": _error(generated_outputs[1], xla_outputs[1]),
        },
        "independent_natural_jax_vjp": {
            "input_cotangent": _error(generated_outputs[0], natural_outputs[0]),
            "feature_scale_cotangent": _error(generated_outputs[1], natural_outputs[1]),
        },
    }
    first_hashes = tuple(_hash(value) for value in generated_outputs)
    second_outputs = execute_generated_reverse()
    jax.block_until_ready(second_outputs)
    second_hashes = tuple(_hash(value) for value in second_outputs)
    if first_hashes != second_hashes:
        raise AssertionError("generated axis-Fold FFI is not deterministic")
    measurements, execution_order = _measure(
        (("generated_ffi", execute_generated_reverse), ("matched_xla", execute_matched_xla_algebra)),
        warmups=args.warmups,
        repeats=args.repeats,
        iterations=args.iterations,
    )
    component_profile = None
    if input_generated is not None and feature_generated is not None:

        @jax.jit
        def generated_input_cotangent(
            projected_argument: jax.Array,
            feature_scale_argument: jax.Array,
            standardized_argument: jax.Array,
            inverse_scale_argument: jax.Array,
        ) -> jax.Array:
            return call_cuda_axis_fold_ffi(
                input_generated,
                {
                    "projected": projected_argument,
                    "feature_scale": feature_scale_argument,
                    "standardized": standardized_argument,
                    "inverse_scale": inverse_scale_argument,
                },
            )[0]

        @jax.jit
        def generated_feature_scale_cotangent(
            projected_argument: jax.Array,
            standardized_argument: jax.Array,
        ) -> jax.Array:
            return call_cuda_axis_fold_ffi(
                feature_generated,
                {"projected": projected_argument, "standardized": standardized_argument},
            )[0]

        def execute_generated_input_cotangent() -> jax.Array:
            return generated_input_cotangent(projected, feature_scale, standardized, inverse_scale)

        def execute_generated_feature_scale_cotangent() -> jax.Array:
            return generated_feature_scale_cotangent(projected, standardized)

        def execute_matched_xla_input_cotangent() -> jax.Array:
            return matched_xla_input_cotangent(projected, feature_scale, standardized, inverse_scale)

        def execute_matched_xla_feature_scale_cotangent() -> jax.Array:
            return matched_xla_feature_scale_cotangent(projected, standardized)

        component_outputs = (
            execute_generated_input_cotangent(),
            execute_generated_feature_scale_cotangent(),
        )
        jax.block_until_ready(component_outputs)
        component_hashes = tuple(_hash(value) for value in component_outputs)
        repeated_component_outputs = (
            execute_generated_input_cotangent(),
            execute_generated_feature_scale_cotangent(),
        )
        jax.block_until_ready(repeated_component_outputs)
        repeated_component_hashes = tuple(_hash(value) for value in repeated_component_outputs)
        if component_hashes != repeated_component_hashes:
            raise AssertionError("separately generated axis-Fold FFI components are not deterministic")

        input_measurements, input_execution_order = _measure(
            (
                ("generated_ffi", execute_generated_input_cotangent),
                ("matched_xla", execute_matched_xla_input_cotangent),
            ),
            warmups=args.warmups,
            repeats=args.repeats,
            iterations=args.iterations,
        )
        feature_measurements, feature_execution_order = _measure(
            (
                ("generated_ffi", execute_generated_feature_scale_cotangent),
                ("matched_xla", execute_matched_xla_feature_scale_cotangent),
            ),
            warmups=args.warmups,
            repeats=args.repeats,
            iterations=args.iterations,
        )
        assert input_library is not None
        assert feature_library is not None
        component_profile = {
            "correctness": {
                "against_full_generated": {
                    "input_cotangent": _error(component_outputs[0], generated_outputs[0]),
                    "feature_scale_cotangent": _error(component_outputs[1], generated_outputs[1]),
                },
                "against_matched_xla": {
                    "input_cotangent": _error(component_outputs[0], xla_outputs[0]),
                    "feature_scale_cotangent": _error(component_outputs[1], xla_outputs[1]),
                },
                "deterministic_hashes": list(component_hashes),
            },
            "input_cotangent": {
                "measurements": input_measurements,
                "execution_order": input_execution_order,
                "ratio_generated_to_matched_xla": (
                    input_measurements["generated_ffi"]["median_ms"] / input_measurements["matched_xla"]["median_ms"]
                ),
                "generated_handler_executions": _handler_call_count(input_library),
            },
            "feature_scale_cotangent": {
                "measurements": feature_measurements,
                "execution_order": feature_execution_order,
                "ratio_generated_to_matched_xla": (
                    feature_measurements["generated_ffi"]["median_ms"] / feature_measurements["matched_xla"]["median_ms"]
                ),
                "generated_handler_executions": _handler_call_count(feature_library),
            },
        }
    generated_ms = measurements["generated_ffi"]["median_ms"]
    xla_ms = measurements["matched_xla"]["median_ms"]
    telemetry = subprocess.check_output(
        (
            "nvidia-smi",
            "--query-gpu=name,driver_version,power.limit,clocks.current.sm,clocks.current.memory",
            "--format=csv,noheader,nounits",
            "--id=0",
        ),
        text=True,
    ).strip()
    return {
        "schema_version": 1,
        "workload": {"rows": args.rows, "hidden": args.hidden, "statistic": "uncentered_second_moment"},
        "frontend": {
            "source": "ordinary JAX function differentiated by jax.vjp and exported to StableHLO",
            "jax_owns_ad": True,
            "stablehlo_sha256": hashlib.sha256(serialized).hexdigest(),
            "operation_kinds": dict(sorted(Counter(operation.kind for operation in graph.operations).items())),
            "named_semantics_erased": True,
        },
        "generated": {
            "runtime": "JAX CUDA typed FFI; no Torch dependency",
            "target": generated.target_name,
            "source_sha256": generated.source_sha256,
            "semantic_fingerprints": list(generated.semantic_fingerprints),
            "pipeline_schedule": generated.pipeline_schedule.value,
            "handler_executions": _handler_call_count(library),
        },
        "numerical_contract": {
            "policy": compilation.numerical_policy.value,
            "generated_reassociation": [stage.program.reassociation.value for stage in compilation.pipeline.stages],
            "accepted_reference": "natural JAX VJP with BF16 inputs and outputs",
            "natural_jax_vjp": (
                "source-ordered equivalence is not established because generated and XLA reductions may "
                "select different deterministic trees"
            ),
        },
        "correctness": {**correctness, "deterministic_hashes": list(first_hashes)},
        "measurements": measurements,
        "component_profile": component_profile,
        "execution_order": execution_order,
        "ratio_generated_to_matched_xla": generated_ms / xla_ms,
        "benchmark": {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "iterations_per_sample": args.iterations,
            "counterbalanced": True,
            "timing": "host enqueue interval followed by jax.block_until_ready",
        },
        "environment": {
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "device": jax.devices()[0].device_kind,
            "telemetry": telemetry,
        },
        "revisions": {"shuttle": args.shuttle_revision},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--column-groups-per-block", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--profile-components", action="store_true")
    parser.add_argument(
        "--pipeline-schedule",
        choices=tuple(schedule.value for schedule in AxisFoldPipelineSchedule),
        default=AxisFoldPipelineSchedule.SEPARATE_STAGES.value,
    )
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--architecture", choices=("sm_90a", "sm_100a"), required=True)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--xla-dump-directory", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--shuttle-revision", required=True)
    args = parser.parse_args()
    result = run(args)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
