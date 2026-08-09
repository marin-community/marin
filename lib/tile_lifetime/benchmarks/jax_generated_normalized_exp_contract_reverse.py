# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile and benchmark the generated normalized-exp Contract reverse family."""

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

from tile_lifetime.cuda_normalized_exp_contract_reverse_codegen import (
    GeneratedCudaNormalizedExpContractReverseFfi,
    generate_cuda_normalized_exp_contract_reverse_ffi,
)
from tile_lifetime.cuda_toolchain import cuda_toolkit_link_flags, cuda_toolkit_shared_library_link_flags
from tile_lifetime.jax_normalized_exp_contract_reverse_ffi import (
    call_cuda_normalized_exp_contract_reverse_ffi,
    register_cuda_normalized_exp_contract_reverse_ffi,
)
from tile_lifetime.tensor_program import ScalarExpressionKind, scalar_binary, scalar_constant, scalar_input, scalar_unary
from tile_lifetime.xla_normalized_exp_contract_reverse import plan_normalized_exp_contract_reverse_hlo_replacement

_IDENTITY_TARGET = "shuttle.generic.normalized_exp_contract_reverse.h100_identity"
_SOFTCAP_TARGET = "shuttle.generic.normalized_exp_contract_reverse.preflight_softcap"
_MAX_ABSOLUTE_ERROR = 0.0625
_MAX_MEAN_ABSOLUTE_ERROR = 5e-4


def _score_contract(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    score = jax.lax.dot_general(
        lhs,
        rhs,
        (((1,), (0,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    return score.astype(jnp.bfloat16)


def _forward_state(
    lhs: jax.Array,
    rhs: jax.Array,
    fold_validity: jax.Array,
) -> jax.Array:
    score = _score_contract(lhs, rhs).astype(jnp.float32)
    restricted_score = jnp.where(fold_validity[None, :], score, -jnp.inf)
    return jax.scipy.special.logsumexp(restricted_score, axis=1)


def _natural_row_values(
    lhs: jax.Array,
    rhs: jax.Array,
    fold_validity: jax.Array,
    selected_indices: jax.Array,
    row_validity: jax.Array,
) -> jax.Array:
    score = _score_contract(lhs, rhs).astype(jnp.float32)
    restricted_score = jnp.where(fold_validity[None, :], score, -jnp.inf)
    normalizer = jax.scipy.special.logsumexp(restricted_score, axis=1)
    selected = jnp.take_along_axis(score, selected_indices[:, None], axis=1)[:, 0]
    return normalizer - jnp.where(row_validity, selected, 0.0)


def _independent_natural_jax_vjp(
    lhs: jax.Array,
    rhs: jax.Array,
    fold_validity: jax.Array,
    row_cotangent: jax.Array,
    selected_indices: jax.Array,
    row_validity: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    _, pullback = jax.vjp(
        lambda lhs_argument, rhs_argument: _natural_row_values(
            lhs_argument,
            rhs_argument,
            fold_validity,
            selected_indices,
            row_validity,
        ),
        lhs,
        rhs,
    )
    return pullback(row_cotangent)


def _matched_reverse(
    lhs: jax.Array,
    rhs: jax.Array,
    saved_state: jax.Array,
    fold_validity: jax.Array,
    row_cotangent: jax.Array,
    selected_indices: jax.Array,
    row_validity: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    raw_score = _score_contract(lhs, rhs)
    score = raw_score.astype(jnp.float32)
    probability = jnp.where(fold_validity[None, :], jnp.exp(score - saved_state[:, None]), 0.0)
    selected = row_validity[:, None] & (selected_indices[:, None] == jnp.arange(rhs.shape[1])[None, :])
    base_cotangent = probability * row_cotangent[:, None] - jnp.where(selected, row_cotangent[:, None], 0.0)
    score_cotangent = base_cotangent.astype(jnp.bfloat16)
    input_cotangent = jax.lax.dot_general(
        score_cotangent,
        rhs,
        (((1,), (1,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    operand_cotangent = jax.lax.dot_general(
        lhs,
        score_cotangent,
        (((0,), (0,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    return input_cotangent, operand_cotangent


def _relaxed_reverse_without_bf16_intermediates(
    lhs: jax.Array,
    rhs: jax.Array,
    saved_state: jax.Array,
    fold_validity: jax.Array,
    row_cotangent: jax.Array,
    selected_indices: jax.Array,
    row_validity: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    score = jax.lax.dot_general(
        lhs,
        rhs,
        (((1,), (0,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    probability = jnp.where(fold_validity[None, :], jnp.exp(score - saved_state[:, None]), 0.0)
    selected = row_validity[:, None] & (selected_indices[:, None] == jnp.arange(rhs.shape[1])[None, :])
    score_cotangent = probability * row_cotangent[:, None] - jnp.where(selected, row_cotangent[:, None], 0.0)
    input_cotangent = jax.lax.dot_general(
        score_cotangent,
        rhs.astype(jnp.float32),
        (((1,), (1,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
    ).astype(jnp.bfloat16)
    operand_cotangent = jax.lax.dot_general(
        lhs.astype(jnp.float32),
        score_cotangent,
        (((0,), (0,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
    ).astype(jnp.bfloat16)
    return input_cotangent, operand_cotangent


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
    generated: GeneratedCudaNormalizedExpContractReverseFfi,
    directory: Path,
    nvcc: Path,
    architecture: str,
) -> ctypes.CDLL:
    directory.mkdir(parents=True, exist_ok=True)
    source_path = directory / "generated_normalized_exp_contract_reverse.cu"
    library_path = directory / "generated_normalized_exp_contract_reverse.so"
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
    call_count_symbol = library.shuttle_normalized_exp_contract_reverse_call_count
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
    function = library.shuttle_normalized_exp_contract_reverse_call_count
    function.restype = ctypes.c_int
    return int(function())


def _plan(hlo_fixture: Path) -> Any:
    return plan_normalized_exp_contract_reverse_hlo_replacement(gzip.decompress(hlo_fixture.read_bytes()).decode())


def _preflight(args: argparse.Namespace, plan: Any) -> dict[str, Any]:
    identity = generate_cuda_normalized_exp_contract_reverse_ffi(plan, target=_IDENTITY_TARGET, threads=args.threads)
    mutation = generate_cuda_normalized_exp_contract_reverse_ffi(
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
                (
                    identity.rows,
                    identity.reduction,
                    identity.fold_extent,
                    identity.threads,
                    identity.shared_bytes,
                )
                == (
                    mutation.rows,
                    mutation.reduction,
                    mutation.fold_extent,
                    mutation.threads,
                    mutation.shared_bytes,
                )
            ),
            "source_contains_tanh": "tanhf" in mutation.source,
            "source_contains_generated_derivative": "generated_score_derivative(raw_score)" in mutation.source,
        },
        "shape": {"rows": identity.rows, "reduction": identity.reduction, "fold_extent": identity.fold_extent},
        "environment": {"jax": jax.__version__, "jaxlib": jaxlib.__version__},
        "revision": args.shuttle_revision,
    }


def _gpu_run(args: argparse.Namespace, plan: Any) -> dict[str, Any]:
    if not jax.devices() or jax.devices()[0].platform != "gpu":
        raise RuntimeError("the normalized-exp Contract reverse benchmark requires one CUDA device")
    if args.repeats % 2:
        raise ValueError("counterbalanced two-path benchmark requires an even repeat count")
    generated = generate_cuda_normalized_exp_contract_reverse_ffi(plan, target=_IDENTITY_TARGET, threads=args.threads)
    library = _compile_generated_source(generated, args.artifact_directory / "identity", args.nvcc, args.architecture)
    register_cuda_normalized_exp_contract_reverse_ffi(generated, library)

    key = jax.random.key(args.seed)
    lhs_key, rhs_key, cotangent_key = jax.random.split(key, 3)
    lhs = (jax.random.normal(lhs_key, (generated.rows, generated.reduction), dtype=jnp.float32) * 0.25).astype(
        jnp.bfloat16
    )
    rhs = (jax.random.normal(rhs_key, (generated.reduction, generated.fold_extent), dtype=jnp.float32) * 0.125).astype(
        jnp.bfloat16
    )
    fold_index = jnp.arange(generated.fold_extent, dtype=jnp.int32)
    fold_validity = ((fold_index % 7) != 0) & ((fold_index % 11) != 0)
    selected_indices = jnp.asarray((1, 17, 31, 47, 61, 79, 101, 127), dtype=jnp.int32)
    row_validity = jnp.asarray((True, False, True, True, False, True, True, False), dtype=jnp.bool_)
    row_cotangent = jax.random.normal(cotangent_key, (generated.rows,), dtype=jnp.float32)
    saved_state = jax.jit(_forward_state)(lhs, rhs, fold_validity)
    jax.block_until_ready(saved_state)
    if not bool(np.all(np.asarray(fold_validity)[np.asarray(selected_indices)])):
        raise AssertionError("selected-index fixture must address valid Fold positions")

    @jax.jit
    def generated_reverse() -> tuple[jax.Array, jax.Array]:
        return call_cuda_normalized_exp_contract_reverse_ffi(
            generated,
            {
                "lhs": lhs,
                "rhs": rhs,
                "saved_state": saved_state,
                "fold_validity": fold_validity,
                "row_cotangent": row_cotangent,
                "selected_indices": selected_indices,
                "row_validity": row_validity,
            },
        )

    matched_reverse = jax.jit(_matched_reverse)
    natural_vjp = jax.jit(_independent_natural_jax_vjp)
    relaxed_reverse = jax.jit(_relaxed_reverse_without_bf16_intermediates)

    def execute_generated() -> tuple[jax.Array, jax.Array]:
        return generated_reverse()

    def execute_matched() -> tuple[jax.Array, jax.Array]:
        return matched_reverse(
            lhs,
            rhs,
            saved_state,
            fold_validity,
            row_cotangent,
            selected_indices,
            row_validity,
        )

    generated_outputs = execute_generated()
    matched_outputs = execute_matched()
    natural_outputs = natural_vjp(lhs, rhs, fold_validity, row_cotangent, selected_indices, row_validity)
    relaxed_outputs = relaxed_reverse(
        lhs,
        rhs,
        saved_state,
        fold_validity,
        row_cotangent,
        selected_indices,
        row_validity,
    )
    jax.block_until_ready((generated_outputs, matched_outputs, natural_outputs, relaxed_outputs))
    correctness: dict[str, Any] = {}
    for label, reference in (
        ("matched_explicit_jax_reverse", matched_outputs),
        ("independent_natural_jax_vjp", natural_outputs),
    ):
        errors = {
            "input_cotangent": _error(generated_outputs[0], reference[0]),
            "operand_cotangent": _error(generated_outputs[1], reference[1]),
        }
        _guard_error(f"{label} input cotangent", errors["input_cotangent"])
        _guard_error(f"{label} operand cotangent", errors["operand_cotangent"])
        correctness[label] = errors

    generated_hashes = tuple(_hash(value) for value in generated_outputs)
    deterministic_hashes = [generated_hashes]
    for _ in range(2):
        repeated = execute_generated()
        jax.block_until_ready(repeated)
        repeated_hashes = tuple(_hash(value) for value in repeated)
        if repeated_hashes != generated_hashes:
            raise AssertionError("generated normalized-exp Contract reverse is not deterministic")
        deterministic_hashes.append(repeated_hashes)

    measurements, execution_order = _measure(
        (("generated_ffi", execute_generated), ("matched_natural_jax_reverse", execute_matched)),
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
        jax.ShapeDtypeStruct(row_cotangent.shape, row_cotangent.dtype),
        jax.ShapeDtypeStruct(selected_indices.shape, selected_indices.dtype),
        jax.ShapeDtypeStruct(row_validity.shape, row_validity.dtype),
    )
    exported = jax.export.export(jax.jit(_independent_natural_jax_vjp))(*natural_arguments)
    stablehlo_path = args.artifact_directory / "source-natural-reverse-stablehlo.mlir.bc"
    stablehlo_path.write_bytes(exported.mlir_module_serialized)
    (args.artifact_directory / "matched-natural-reverse-optimized-hlo.txt").write_text(
        matched_reverse.lower(lhs, rhs, saved_state, fold_validity, row_cotangent, selected_indices, row_validity)
        .compile()
        .as_text()
    )

    generated_ms = measurements["generated_ffi"]["median_ms"]
    matched_ms = measurements["matched_natural_jax_reverse"]["median_ms"]
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
        "saved_state": saved_state,
        "fold_validity": fold_validity,
        "row_cotangent": row_cotangent,
        "selected_indices": selected_indices,
        "row_validity": row_validity,
    }
    return {
        "schema_version": 1,
        "mode": "one_h100_proof_of_execution",
        "shape": {"rows": generated.rows, "reduction": generated.reduction, "fold_extent": generated.fold_extent},
        "frontend": {
            "reference": "ordinary JAX BF16 Contract and normalized-exp row objective differentiated by jax.vjp",
            "stablehlo_sha256": hashlib.sha256(exported.mlir_module_serialized).hexdigest(),
            "jax_owns_ad_for_independent_reference": True,
        },
        "fixture": {
            "seed": args.seed,
            "fold_valid_count": int(np.count_nonzero(np.asarray(fold_validity))),
            "fold_invalid_count": int(generated.fold_extent - np.count_nonzero(np.asarray(fold_validity))),
            "selected_indices": np.asarray(selected_indices).tolist(),
            "selected_indices_all_valid": True,
            "row_validity": np.asarray(row_validity).tolist(),
            "invalid_row_count": int(generated.rows - np.count_nonzero(np.asarray(row_validity))),
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
            "score_cotangent": "FP32 Map/Fold reverse rounded to BF16 RNE before both reverse Contracts",
            "reverse_contract_output": "FP32 accumulation rounded to BF16 RNE",
            "reduction_order": "generated ordered FP32 loops; XLA may choose another FP32 reduction tree",
            "error_guard": {
                "maximum_absolute_error": _MAX_ABSOLUTE_ERROR,
                "mean_absolute_error": _MAX_MEAN_ABSOLUTE_ERROR,
            },
            "relaxed_boundary_hashes_differ": (
                tuple(_hash(value) for value in relaxed_outputs) != tuple(_hash(value) for value in matched_outputs)
            ),
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
            "timing": (
                "host enqueue interval followed by jax.block_until_ready; compilation and saved-state forward excluded"
            ),
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
