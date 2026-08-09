# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile and benchmark the generated normalized-exp Contract forward family."""

from __future__ import annotations

import argparse
import ctypes
import gzip
import hashlib
import itertools
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from tile_lifetime.cuda_normalized_exp_contract_forward_codegen import (
    GeneratedCudaNormalizedExpContractForwardFfi,
    generate_cuda_normalized_exp_contract_forward_ffi,
)
from tile_lifetime.cuda_toolchain import cuda_toolkit_link_flags, cuda_toolkit_shared_library_link_flags
from tile_lifetime.jax_normalized_exp_contract_forward_ffi import (
    call_cuda_normalized_exp_contract_forward_ffi,
    register_cuda_normalized_exp_contract_forward_ffi,
)
from tile_lifetime.tensor_program import ScalarExpressionKind, scalar_binary, scalar_constant, scalar_input, scalar_unary
from tile_lifetime.xla_normalized_exp_contract_forward import plan_normalized_exp_contract_forward_hlo_replacement

_IDENTITY_TARGET = "shuttle.generic.normalized_exp_contract_forward.h100_identity"
_SOFTCAP_TARGET = "shuttle.generic.normalized_exp_contract_forward.preflight_softcap"
_MAX_ABSOLUTE_ERROR = 2e-5
_MAX_MEAN_ABSOLUTE_ERROR = 2e-6


def _score_contract(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    score = jax.lax.dot_general(
        lhs,
        rhs,
        (((1,), (0,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    return score.astype(jnp.bfloat16)


def _row_validity(fold_validity: jax.Array, selected_indices: jax.Array) -> tuple[jax.Array, jax.Array]:
    safe_indices = jnp.clip(selected_indices, 0, fold_validity.shape[0] - 1)
    in_bounds = (selected_indices >= 0) & (selected_indices < fold_validity.shape[0])
    return in_bounds & fold_validity[safe_indices], safe_indices


def _matched_forward(
    lhs: jax.Array,
    rhs: jax.Array,
    fold_validity: jax.Array,
    selected_indices: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    score = _score_contract(lhs, rhs).astype(jnp.float32)
    restricted_score = jnp.where(fold_validity[None, :], score, -jnp.inf)
    maximum = jnp.max(restricted_score, axis=1)
    sum_exp = jnp.sum(jnp.exp(restricted_score - maximum[:, None]), axis=1)
    saved_state = jnp.log(sum_exp) + maximum
    row_validity, safe_indices = _row_validity(fold_validity, selected_indices)
    selected = jnp.take_along_axis(score, safe_indices[:, None], axis=1)[:, 0]
    loss = saved_state - selected
    return jnp.where(row_validity, loss, 0.0), jnp.where(row_validity, saved_state, 0.0)


def _independent_natural_forward(
    lhs: jax.Array,
    rhs: jax.Array,
    fold_validity: jax.Array,
    selected_indices: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Natural normalized-exponential objective with the exported BF16 score boundary."""
    score = _score_contract(lhs, rhs).astype(jnp.float32)
    restricted_score = jnp.where(fold_validity[None, :], score, -jnp.inf)
    saved_state = jax.scipy.special.logsumexp(restricted_score, axis=1)
    row_validity, safe_indices = _row_validity(fold_validity, selected_indices)
    selected = jnp.take_along_axis(score, safe_indices[:, None], axis=1)[:, 0]
    loss = saved_state - selected
    return jnp.where(row_validity, loss, 0.0), jnp.where(row_validity, saved_state, 0.0)


def _relaxed_forward_without_bf16_score(
    lhs: jax.Array,
    rhs: jax.Array,
    fold_validity: jax.Array,
    selected_indices: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    score = jax.lax.dot_general(
        lhs,
        rhs,
        (((1,), (0,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    restricted_score = jnp.where(fold_validity[None, :], score, -jnp.inf)
    saved_state = jax.scipy.special.logsumexp(restricted_score, axis=1)
    row_validity, safe_indices = _row_validity(fold_validity, selected_indices)
    selected = jnp.take_along_axis(score, safe_indices[:, None], axis=1)[:, 0]
    loss = saved_state - selected
    return jnp.where(row_validity, loss, 0.0), jnp.where(row_validity, saved_state, 0.0)


def _softcap_expression() -> Any:
    raw_score = scalar_input("raw_score")
    cap = scalar_constant(6.0)
    return scalar_binary(
        ScalarExpressionKind.MULTIPLY,
        cap,
        scalar_unary(
            ScalarExpressionKind.TANH,
            scalar_binary(ScalarExpressionKind.DIVIDE, raw_score, cap),
        ),
    )


def _compile_generated_source(
    generated: GeneratedCudaNormalizedExpContractForwardFfi,
    directory: Path,
    nvcc: Path,
    architecture: str,
) -> ctypes.CDLL:
    directory.mkdir(parents=True, exist_ok=True)
    source_path = directory / "generated_normalized_exp_contract_forward.cu"
    library_path = directory / "generated_normalized_exp_contract_forward.so"
    include_directory = Path(jaxlib.__file__).resolve().parent / "include"
    source_path.write_text(generated.source + "\n")
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
    library = ctypes.CDLL(str(library_path))
    getattr(library, generated.handler_symbol)
    call_count_symbol = library.shuttle_normalized_exp_contract_forward_call_count
    if not call_count_symbol:
        raise RuntimeError("generated library does not export its call-count symbol")
    return library


def _hash(value: jax.Array) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _error(actual: jax.Array, expected: jax.Array) -> dict[str, float]:
    difference = np.abs(np.asarray(actual, dtype=np.float32) - np.asarray(expected, dtype=np.float32))
    return {
        "maximum_absolute_error": float(difference.max(initial=0.0)),
        "mean_absolute_error": float(difference.mean()),
    }


def _guard_error(name: str, error: dict[str, float]) -> None:
    if error["maximum_absolute_error"] > _MAX_ABSOLUTE_ERROR:
        raise AssertionError(f"{name} maximum absolute error exceeds {_MAX_ABSOLUTE_ERROR}: {error}")
    if error["mean_absolute_error"] > _MAX_MEAN_ABSOLUTE_ERROR:
        raise AssertionError(f"{name} mean absolute error exceeds {_MAX_MEAN_ABSOLUTE_ERROR}: {error}")


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
    orderings = tuple(itertools.permutations(variants))
    for repeat in range(repeats):
        ordering = orderings[repeat % len(orderings)]
        orders.append([name for name, _ in ordering])
        for name, function in ordering:
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
    function = library.shuttle_normalized_exp_contract_forward_call_count
    function.restype = ctypes.c_int
    return int(function())


def _plan(hlo_fixture: Path) -> Any:
    return plan_normalized_exp_contract_forward_hlo_replacement(gzip.decompress(hlo_fixture.read_bytes()).decode())


def _preflight(args: argparse.Namespace, plan: Any) -> dict[str, Any]:
    identity = generate_cuda_normalized_exp_contract_forward_ffi(plan, target=_IDENTITY_TARGET, threads=args.threads)
    mutation = generate_cuda_normalized_exp_contract_forward_ffi(
        plan,
        target=_SOFTCAP_TARGET,
        score_expression=_softcap_expression(),
        threads=args.threads,
    )
    identity_library = _compile_generated_source(
        identity, args.artifact_directory / "identity", args.nvcc, args.architecture
    )
    mutation_library = _compile_generated_source(
        mutation, args.artifact_directory / "tanh_softcap", args.nvcc, args.architecture
    )
    return {
        "mode": "compile_and_load_only",
        "gpu_execution": False,
        "identity": {
            "target": identity.target,
            "handler_symbol": identity.handler_symbol,
            "semantic_digest": identity.semantic_digest,
            "source_digest": identity.source_digest,
            "handler_symbol_resolved": bool(getattr(identity_library, identity.handler_symbol)),
        },
        "tanh_softcap_mutation": {
            "target": mutation.target,
            "handler_symbol": mutation.handler_symbol,
            "semantic_digest": mutation.semantic_digest,
            "source_digest": mutation.source_digest,
            "handler_symbol_resolved": bool(getattr(mutation_library, mutation.handler_symbol)),
            "same_physical_extents": (
                (identity.rows, identity.reduction, identity.fold_extent, identity.threads, identity.shared_bytes)
                == (mutation.rows, mutation.reduction, mutation.fold_extent, mutation.threads, mutation.shared_bytes)
            ),
            "source_contains_tanh": "tanhf" in mutation.source,
        },
        "shape": {"rows": identity.rows, "reduction": identity.reduction, "fold_extent": identity.fold_extent},
        "environment": {"jax": jax.__version__, "jaxlib": jaxlib.__version__},
        "revision": args.shuttle_revision,
    }


def _gpu_run(args: argparse.Namespace, plan: Any) -> dict[str, Any]:
    if not jax.devices() or jax.devices()[0].platform != "gpu":
        raise RuntimeError("the normalized-exp Contract forward benchmark requires one CUDA device")
    if args.repeats % 2:
        raise ValueError("counterbalanced two-path benchmark requires an even repeat count")
    generated = generate_cuda_normalized_exp_contract_forward_ffi(plan, target=_IDENTITY_TARGET, threads=args.threads)
    library = _compile_generated_source(generated, args.artifact_directory / "identity", args.nvcc, args.architecture)
    register_cuda_normalized_exp_contract_forward_ffi(generated, library)

    key = jax.random.key(args.seed)
    lhs_key, rhs_key = jax.random.split(key)
    lhs = (jax.random.normal(lhs_key, (generated.rows, generated.reduction), dtype=jnp.float32) * 0.25).astype(
        jnp.bfloat16
    )
    rhs = (jax.random.normal(rhs_key, (generated.reduction, generated.fold_extent), dtype=jnp.float32) * 0.125).astype(
        jnp.bfloat16
    )
    fold_index = jnp.arange(generated.fold_extent, dtype=jnp.int32)
    fold_validity = ((fold_index % 7) != 0) & ((fold_index % 11) != 0)
    selected_indices = jnp.asarray((1, -1, 31, 47, 0, 79, 101, 127), dtype=jnp.int32)
    expected_row_validity, _ = _row_validity(fold_validity, selected_indices)

    @jax.jit
    def generated_forward() -> tuple[jax.Array, jax.Array]:
        return call_cuda_normalized_exp_contract_forward_ffi(
            generated,
            {
                "lhs": lhs,
                "rhs": rhs,
                "fold_validity": fold_validity,
                "selected_indices": selected_indices,
            },
        )

    matched_forward = jax.jit(_matched_forward)
    natural_forward = jax.jit(_independent_natural_forward)
    relaxed_forward = jax.jit(_relaxed_forward_without_bf16_score)

    def execute_generated() -> tuple[jax.Array, jax.Array]:
        return generated_forward()

    def execute_matched() -> tuple[jax.Array, jax.Array]:
        return matched_forward(lhs, rhs, fold_validity, selected_indices)

    generated_outputs = execute_generated()
    matched_outputs = execute_matched()
    natural_outputs = natural_forward(lhs, rhs, fold_validity, selected_indices)
    relaxed_outputs = relaxed_forward(lhs, rhs, fold_validity, selected_indices)
    jax.block_until_ready((generated_outputs, matched_outputs, natural_outputs, relaxed_outputs))
    correctness: dict[str, Any] = {}
    for label, reference in (
        ("matched_explicit_jax_forward", matched_outputs),
        ("independent_natural_jax_forward", natural_outputs),
    ):
        errors = {
            "loss": _error(generated_outputs[0], reference[0]),
            "saved_state": _error(generated_outputs[1], reference[1]),
        }
        _guard_error(f"{label} loss", errors["loss"])
        _guard_error(f"{label} saved state", errors["saved_state"])
        correctness[label] = errors

    generated_hashes = tuple(_hash(value) for value in generated_outputs)
    deterministic_hashes = [generated_hashes]
    for _ in range(2):
        repeated = execute_generated()
        jax.block_until_ready(repeated)
        repeated_hashes = tuple(_hash(value) for value in repeated)
        if repeated_hashes != generated_hashes:
            raise AssertionError("generated normalized-exp Contract forward is not deterministic")
        deterministic_hashes.append(repeated_hashes)

    measurements, execution_order = _measure(
        (("generated_ffi", execute_generated), ("matched_natural_jax_forward", execute_matched)),
        warmups=args.warmups,
        repeats=args.repeats,
        iterations=args.iterations,
    )
    expected_handler_count = 3 + args.warmups + args.repeats * args.iterations
    handler_count = _handler_call_count(library)
    if handler_count != expected_handler_count:
        raise AssertionError(f"generated handler count {handler_count} does not match {expected_handler_count}")

    args.artifact_directory.mkdir(parents=True, exist_ok=True)
    natural_arguments = (
        jax.ShapeDtypeStruct(lhs.shape, lhs.dtype),
        jax.ShapeDtypeStruct(rhs.shape, rhs.dtype),
        jax.ShapeDtypeStruct(fold_validity.shape, fold_validity.dtype),
        jax.ShapeDtypeStruct(selected_indices.shape, selected_indices.dtype),
    )
    exported = jax.export.export(jax.jit(_independent_natural_forward))(*natural_arguments)
    stablehlo_path = args.artifact_directory / "source-natural-forward-stablehlo.mlir.bc"
    stablehlo_path.write_bytes(exported.mlir_module_serialized)
    (args.artifact_directory / "matched-natural-forward-optimized-hlo.txt").write_text(
        matched_forward.lower(lhs, rhs, fold_validity, selected_indices).compile().as_text()
    )

    generated_ms = measurements["generated_ffi"]["median_ms"]
    matched_ms = measurements["matched_natural_jax_forward"]["median_ms"]
    telemetry = subprocess.check_output(
        (
            "nvidia-smi",
            "--query-gpu=name,driver_version,power.limit,clocks.current.sm,clocks.current.memory",
            "--format=csv,noheader,nounits",
            "--id=0",
        ),
        text=True,
    ).strip()
    input_values = {
        "lhs": lhs,
        "rhs": rhs,
        "fold_validity": fold_validity,
        "selected_indices": selected_indices,
    }
    return {
        "schema_version": 1,
        "mode": "one_h100_proof_of_execution",
        "shape": {"rows": generated.rows, "reduction": generated.reduction, "fold_extent": generated.fold_extent},
        "frontend": {
            "reference": "ordinary JAX BF16 Contract and normalized-exponential indexed objective",
            "stablehlo_sha256": hashlib.sha256(exported.mlir_module_serialized).hexdigest(),
            "jax_owns_ad": True,
        },
        "fixture": {
            "seed": args.seed,
            "fold_valid_count": int(np.count_nonzero(np.asarray(fold_validity))),
            "fold_invalid_count": int(generated.fold_extent - np.count_nonzero(np.asarray(fold_validity))),
            "selected_indices": np.asarray(selected_indices).tolist(),
            "row_validity": np.asarray(expected_row_validity).tolist(),
            "invalid_row_count": int(generated.rows - np.count_nonzero(np.asarray(expected_row_validity))),
            "input_hashes": {name: _hash(value) for name, value in input_values.items()},
        },
        "generated": {
            "target": generated.target,
            "handler_symbol": generated.handler_symbol,
            "semantic_digest": generated.semantic_digest,
            "source_digest": generated.source_digest,
            "threads": generated.threads,
            "shared_bytes": generated.shared_bytes,
            "handler_count": handler_count,
            "expected_handler_count": expected_handler_count,
            "runtime": "JAX CUDA typed FFI; no Torch dependency",
        },
        "numerical_contract": {
            "score_contract_output": "FP32 accumulation rounded to BF16 RNE before normalized-exp Map/Fold",
            "fold_order": "generated ordered FP32 loops; XLA may choose another FP32 reduction tree",
            "invalid_row_output": "zero loss and zero saved state",
            "error_guard": {
                "maximum_absolute_error": _MAX_ABSOLUTE_ERROR,
                "mean_absolute_error": _MAX_MEAN_ABSOLUTE_ERROR,
            },
            "relaxed_boundary_hashes_differ": (
                tuple(_hash(value) for value in relaxed_outputs) != tuple(_hash(value) for value in matched_outputs)
            ),
            "matched_vs_natural_error": {
                "loss": _error(matched_outputs[0], natural_outputs[0]),
                "saved_state": _error(matched_outputs[1], natural_outputs[1]),
            },
        },
        "correctness": correctness,
        "determinism": {
            "trials": len(deterministic_hashes),
            "output_hashes": [list(hashes) for hashes in deterministic_hashes],
        },
        "measurements": measurements,
        "execution_order": execution_order,
        "ratio_generated_to_matched_jax": generated_ms / matched_ms,
        "benchmark": {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "iterations_per_sample": args.iterations,
            "counterbalanced": True,
            "timing": "host enqueue interval followed by jax.block_until_ready; compilation excluded",
        },
        "environment": {
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "device": jax.devices()[0].device_kind,
            "telemetry": telemetry,
        },
        "revision": args.shuttle_revision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hlo-fixture", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--architecture", choices=("sm_90a", "sm_100a"), required=True)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--shuttle-revision", required=True)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    plan = _plan(args.hlo_fixture)
    result = _preflight(args, plan) if args.preflight_only else _gpu_run(args, plan)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
