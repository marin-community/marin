#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute one recovered Grug reverse Contract+Map region through GPU XLA FFI."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import tempfile
from pathlib import Path
from typing import Any

import jax
import jaxlib
import numpy as np
from haliax.partitioning import set_mesh

from lib.tile_lifetime.benchmarks.xla_grug_backward_multi_output_custom_call_smoke import (
    _multi_output_region_index,
)
from lib.tile_lifetime.benchmarks.xla_grug_pair_map_custom_call_smoke import (
    _mesh,
    _natural_train_step,
)
from lib.tile_lifetime.benchmarks.xla_pair_map_custom_call_smoke import (
    _PASS_NAME,
    _TARGET_NAME,
    _compile_cuda_ffi_handler,
    _register_cuda_ffi_custom_call,
    generate_cuda_multi_output_ffi_handler,
    pair_map_recovery_diagnostic,
    recover_multi_output_region_rewrite,
    replace_multi_output_region_with_custom_call,
    write_gzip_text,
)


def _compare_under_ordered_fp(expected: Any, actual: Any) -> dict[str, Any]:
    """Compare a source-ordered Map around implementation-ordered Contracts."""
    expected_leaves = jax.tree.leaves(expected)
    actual_leaves = jax.tree.leaves(actual)
    if len(expected_leaves) != len(actual_leaves):
        raise RuntimeError("transformed train step changed the result tree")
    maximum = 0.0
    total = 0.0
    count = 0
    exact_leaves = 0
    for expected_leaf, actual_leaf in zip(expected_leaves, actual_leaves, strict=True):
        expected_array = np.asarray(expected_leaf)
        actual_array = np.asarray(actual_leaf)
        if expected_array.shape != actual_array.shape or expected_array.dtype != actual_array.dtype:
            raise RuntimeError("transformed train step changed a result leaf type")
        if expected_array.tobytes() == actual_array.tobytes():
            exact_leaves += 1
        if not np.issubdtype(expected_array.dtype, np.inexact):
            if not np.array_equal(actual_array, expected_array):
                raise RuntimeError("integer or Boolean result leaf changed")
            continue
        expected_nan = np.isnan(expected_array)
        actual_nan = np.isnan(actual_array)
        if not np.array_equal(expected_nan, actual_nan):
            raise RuntimeError("transformed train step changed NaN positions")
        finite = np.isfinite(expected_array) & np.isfinite(actual_array)
        difference = np.abs(actual_array[finite].astype(np.float64) - expected_array[finite].astype(np.float64))
        maximum = max(maximum, float(difference.max(initial=0.0)))
        total += float(difference.sum())
        count += difference.size
        if not np.allclose(actual_array, expected_array, rtol=3e-3, atol=3e-4, equal_nan=True):
            raise RuntimeError(f"ordered-FP result mismatch: max_abs={float(difference.max(initial=0.0))}")
    return {
        "result_leaf_count": len(expected_leaves),
        "bitwise_equal_leaf_count": exact_leaves,
        "maximum_absolute_error": maximum,
        "mean_absolute_error": total / count if count else 0.0,
    }


def run_smoke(nvcc: Path, architecture: str, artifact_directory: Path | None) -> dict[str, Any]:
    """Compile and execute a GPU replacement from the recovered multi-output AST."""
    if not jax.devices() or jax.devices()[0].platform != "gpu":
        raise RuntimeError("the GPU XLA replacement smoke requires a CUDA JAX device")
    hlo = importlib.import_module("jaxlib._hlo")
    xla = importlib.import_module("jax.extend.xla")
    jax.config.update("jax_enable_compilation_cache", False)
    original_modules: list[str] = []
    transformed_modules: list[str] = []
    rewrites: list[Any] = []
    temporary = None
    if artifact_directory is None:
        temporary = tempfile.TemporaryDirectory(prefix="shuttle-grug-backward-gpu-region-")
        directory = Path(temporary.name)
    else:
        artifact_directory.mkdir(parents=True, exist_ok=True)
        directory = artifact_directory
    try:
        with set_mesh(_mesh()):
            train_step, state, batch = _natural_train_step()
            baseline = train_step.lower(state, batch, compute_watch=False).compile()
            expected = baseline(state, batch)
            jax.block_until_ready(expected)
            holder: dict[str, Any] = {}

            def replace(serialized_module: bytes) -> bytes | None:
                module = hlo.HloModule.from_serialized_hlo_module_proto(serialized_module)
                if module.name != "jit_train_step":
                    return None
                original = module.to_string()
                original_modules.append(original)
                if artifact_directory is not None:
                    write_gzip_text(directory / "original-gpu-pre-scheduler-hlo.txt.gz", original)
                    (directory / "gpu-recovery-diagnostic.json").write_text(
                        json.dumps(pair_map_recovery_diagnostic(original), indent=2, sort_keys=True) + "\n"
                    )
                rewrite = recover_multi_output_region_rewrite(original, _multi_output_region_index(original))
                source = generate_cuda_multi_output_ffi_handler(rewrite.program)
                library = _compile_cuda_ffi_handler(source, directory, nvcc, architecture)
                _register_cuda_ffi_custom_call(library)
                holder["library"] = library
                rewrites.append(rewrite)
                transformed_text = replace_multi_output_region_with_custom_call(
                    original,
                    rewrite,
                    _TARGET_NAME,
                    typed_ffi=True,
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
            try:
                transformed = train_step.lower(state, batch, compute_watch=False).compile()
            finally:
                xla.clear_hlo_module_transformation(
                    _PASS_NAME,
                    stage=xla.PipelineStage.PRE_SCHEDULER,
                    platforms="cuda",
                )
            actual = transformed(state, batch)
            jax.block_until_ready(actual)
            comparison = _compare_under_ordered_fp(expected, actual)
            library = holder["library"]
            call_count_function = library.shuttle_pair_map_smoke_call_count
            call_count_function.restype = ctypes.c_int
            call_count = int(call_count_function())
        if artifact_directory is not None:
            (directory / "generated_pair_map_handler.so").unlink()
    finally:
        if temporary is not None:
            temporary.cleanup()

    if len(original_modules) != 1 or len(transformed_modules) != 1 or len(rewrites) != 1:
        raise RuntimeError("expected exactly one GPU Grug backward-region replacement")
    rewrite = rewrites[0]
    transformed_hlo = transformed_modules[0]
    target_occurrences = transformed_hlo.count(_TARGET_NAME)
    tuple_gets = transformed_hlo.count("get-tuple-element(%shuttle_generated_multi_output_region)")
    if target_occurrences != 1 or tuple_gets != len(rewrite.boundary.outputs) or call_count != 1:
        raise RuntimeError(
            "GPU custom-call evidence mismatch: "
            f"targets={target_occurrences}, tuple_gets={tuple_gets}, executions={call_count}"
        )
    source = generate_cuda_multi_output_ffi_handler(rewrite.program)
    if artifact_directory is not None:
        write_gzip_text(artifact_directory / "original-pre-scheduler-hlo.txt.gz", original_modules[0])
        write_gzip_text(artifact_directory / "transformed-pre-scheduler-hlo.txt.gz", transformed_hlo)
    return {
        "kind": "xla_pre_scheduler_grug_backward_multi_output_gpu_custom_call_smoke",
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "platform": "cuda",
        "device_kind": jax.devices()[0].device_kind,
        "cuda_architecture": architecture,
        "natural_frontend": "ordinary one-layer Grug train step",
        "selection": "unique generic pair-Map region with several externally used pointwise outputs",
        "region_inputs": tuple((value.instruction, value.shape) for value in rewrite.boundary.inputs),
        "region_outputs": tuple((value.instruction, value.shape) for value in rewrite.boundary.outputs),
        "generated_scalar_expressions": rewrite.program.scalar_expressions,
        "generated_handler_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "custom_call_target": _TARGET_NAME,
        "custom_call_occurrences_in_transformed_hlo": target_occurrences,
        "tuple_get_element_count": tuple_gets,
        "custom_call_handler_executions": call_count,
        "numerical_contract": {
            "scalar_map": "source_ordered including recovered BF16 round trips",
            "contracts": "ordered_fp using CUBLAS_COMPUTE_32F_PEDANTIC generic Contract primitives",
        },
        **comparison,
        "outputs_match": True,
        "ffi_api": "XLA typed FFI api_version=1 with CUDA platform stream",
        "explicit_warning": (
            "Execution proof only: two generic cuBLAS Contracts feed the generated source-ordered Map. "
            "Competitive lowering still needs the reusable tiled Contract mainloop and fusion policy."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--cuda-architecture", choices=("sm_90a", "sm_100a"), required=True)
    parser.add_argument("--artifact-directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_smoke(args.nvcc, args.cuda_architecture, args.artifact_directory)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    elif args.artifact_directory is not None:
        (args.artifact_directory / "summary.json").write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
