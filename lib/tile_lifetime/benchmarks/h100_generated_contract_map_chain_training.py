# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile and benchmark a generated two-Contract scalar-Map training chain."""

from __future__ import annotations

import argparse
import ctypes
import gzip
import hashlib
import importlib.metadata
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from tile_lifetime.command_buffer_capture import (
    CallbackCheckpoint,
    CaptureAcceptanceError,
    CaptureAcceptancePolicy,
    CaptureSiteManifest,
    CaptureStabilization,
    CounterbalancedVariant,
    assess_command_buffer_capture,
    derive_capture_site_manifest,
    measure_counterbalanced_variants,
    serialize_then_assess_capture,
    stabilize_counterbalanced_variants,
)
from tile_lifetime.contract_map_chain import (
    TwoContractMapTrainingProgram,
    execute_two_contract_map_forward,
    execute_two_contract_map_reverse,
    form_two_contract_map_training_program,
)
from tile_lifetime.cuda_contract_map_chain_codegen import (
    ContractMapChainFfiPhysicalCandidate,
    GeneratedCudaContractMapChainFfi,
    audit_cuda_contract_map_chain_source,
    generate_cuda_contract_map_chain_ffi,
)
from tile_lifetime.cuda_toolchain import cuda_toolkit_link_flags, cuda_toolkit_shared_library_link_flags
from tile_lifetime.ffi_command_buffer import require_custom_call_command_buffers_enabled
from tile_lifetime.jax_contract_map_chain_ffi import (
    call_cuda_contract_map_chain_forward_ffi,
    call_cuda_contract_map_chain_reverse_ffi,
    register_cuda_contract_map_chain_ffi,
)
from tile_lifetime.xla_low_rank_gated_product import recover_low_rank_gated_product_training
from tile_lifetime.xla_normalized_exp_contract_forward import (
    plan_normalized_exp_contract_forward_hlo_replacement,
    replace_normalized_exp_contract_forward_hlo_region_with_custom_call,
)
from tile_lifetime.xla_normalized_exp_contract_reverse import (
    plan_normalized_exp_contract_reverse_hlo_replacement,
    replace_normalized_exp_contract_reverse_hlo_region_with_custom_call,
)

_FORWARD_TARGET = "shuttle.generic.contract_map_chain.forward.h100"
_REVERSE_TARGET = "shuttle.generic.contract_map_chain.reverse.h100"
_MAX_ABSOLUTE_ERROR = 0.0078125
_MAX_MEAN_ABSOLUTE_ERROR = 0.0005


def _natural_forward(
    activation: jax.Array,
    first_weight: jax.Array,
    second_weight: jax.Array,
) -> jax.Array:
    one = jnp.asarray(1.0, dtype=jnp.bfloat16)
    first = jax.lax.dot_general(
        activation,
        first_weight,
        (((1,), (0,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    hidden = (first * (one / (one + jnp.exp(-first)))).astype(jnp.bfloat16)
    second = jax.lax.dot_general(
        hidden,
        second_weight,
        (((1,), (0,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    return (activation * (one / (one + jnp.exp(-second)))).astype(jnp.bfloat16)


def _natural_training_step(
    activation: jax.Array,
    first_weight: jax.Array,
    second_weight: jax.Array,
    output_cotangent: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    output, pullback = jax.vjp(_natural_forward, activation, first_weight, second_weight)
    input_adjoint, first_weight_adjoint, second_weight_adjoint = pullback(output_cotangent)
    return output, input_adjoint, first_weight_adjoint, second_weight_adjoint


def _program(hlo_fixture: Path) -> TwoContractMapTrainingProgram:
    hlo = gzip.decompress(hlo_fixture.read_bytes()).decode()
    forward = plan_normalized_exp_contract_forward_hlo_replacement(hlo)
    hlo = replace_normalized_exp_contract_forward_hlo_region_with_custom_call(
        hlo,
        forward,
        target="shuttle.preflight.normalized_exp.forward",
    )
    reverse = plan_normalized_exp_contract_reverse_hlo_replacement(hlo)
    hlo = replace_normalized_exp_contract_reverse_hlo_region_with_custom_call(
        hlo,
        reverse,
        target="shuttle.preflight.normalized_exp.reverse",
    )
    report = recover_low_rank_gated_product_training(hlo)
    if not report.reverse_families:
        raise ValueError("HLO fixture contains no recovered two-Contract Map reverse family")
    selected = report.reverse_families[0]
    return form_two_contract_map_training_program(selected.primal, selected)


def _compile_generated_source(
    generated: GeneratedCudaContractMapChainFfi,
    directory: Path,
    nvcc: Path,
    architecture: str,
) -> tuple[ctypes.CDLL, tuple[str, ...]]:
    directory.mkdir(parents=True, exist_ok=True)
    source_path = directory / "generated_contract_map_chain.cu"
    library_path = directory / "generated_contract_map_chain.so"
    include_directory = Path(jaxlib.__file__).resolve().parent / "include"
    ffi_header = include_directory / "xla/ffi/api/ffi.h"
    if not ffi_header.is_file():
        raise ValueError(f"JAX typed-FFI header does not exist: {ffi_header}")
    source_path.write_text(generated.source + "\n")
    command = (
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
    )
    subprocess.run(command, check=True)
    library = ctypes.CDLL(str(library_path))
    for symbol in (
        generated.forward_handler_symbol,
        generated.reverse_handler_symbol,
        "shuttle_contract_map_chain_forward_call_count",
        "shuttle_contract_map_chain_reverse_call_count",
    ):
        getattr(library, symbol)
    return library, command


def _handler_call_counts(library: ctypes.CDLL) -> tuple[int, int]:
    forward = library.shuttle_contract_map_chain_forward_call_count
    reverse = library.shuttle_contract_map_chain_reverse_call_count
    forward.restype = ctypes.c_uint64
    reverse.restype = ctypes.c_uint64
    return int(forward()), int(reverse())


def _hash(value: jax.Array | np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _errors(actual: jax.Array, expected: jax.Array | np.ndarray) -> dict[str, float]:
    difference = np.abs(np.asarray(actual, dtype=np.float32) - np.asarray(expected, dtype=np.float32))
    return {
        "maximum_absolute_error": float(difference.max(initial=0.0)),
        "mean_absolute_error": float(difference.mean()),
    }


def _guard_errors(name: str, errors: dict[str, float]) -> None:
    if errors["maximum_absolute_error"] > _MAX_ABSOLUTE_ERROR:
        raise AssertionError(f"{name} maximum absolute error exceeds {_MAX_ABSOLUTE_ERROR}: {errors}")
    if errors["mean_absolute_error"] > _MAX_MEAN_ABSOLUTE_ERROR:
        raise AssertionError(f"{name} mean absolute error exceeds {_MAX_MEAN_ABSOLUTE_ERROR}: {errors}")


def _warm_up(
    variants: tuple[tuple[str, Any], ...],
    *,
    warmups: int,
) -> None:
    for _ in range(warmups):
        for _, function in variants:
            jax.block_until_ready(function())


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment_versions() -> dict[str, str | None]:
    return {
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "jax-cuda13-plugin": _installed_version("jax-cuda13-plugin"),
        "jax-cuda13-pjrt": _installed_version("jax-cuda13-pjrt"),
    }


def _require_toolchain(args: argparse.Namespace, *, require_gpu: bool) -> None:
    if jax.__version__ != args.require_jax_version or jaxlib.__version__ != args.require_jax_version:
        raise RuntimeError(
            f"benchmark requires matched jax/jaxlib {args.require_jax_version}; "
            f"found jax {jax.__version__}, jaxlib {jaxlib.__version__}"
        )
    if not args.nvcc.is_file():
        raise ValueError(f"CUDA compiler does not exist: {args.nvcc}")
    if args.repeats <= 0 or args.repeats % 2:
        raise ValueError("counterbalanced benchmark requires a positive even repeat count")
    if min(args.warmups, args.iterations) <= 0:
        raise ValueError("warmups and iterations must be positive")
    if not require_gpu:
        return
    environment = _environment_versions()
    for distribution in ("jax-cuda13-plugin", "jax-cuda13-pjrt"):
        if environment[distribution] != args.require_jax_version:
            raise RuntimeError(
                f"benchmark requires {distribution} {args.require_jax_version}; " f"found {environment[distribution]}"
            )
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu" or "H100" not in devices[0].device_kind:
        raise RuntimeError(f"benchmark requires exactly one visible H100, found {devices}")


def _preflight(args: argparse.Namespace, program: TwoContractMapTrainingProgram) -> dict[str, Any]:
    _require_toolchain(args, require_gpu=False)
    generated = generate_cuda_contract_map_chain_ffi(
        program,
        forward_target=_FORWARD_TARGET,
        reverse_target=_REVERSE_TARGET,
        threads=args.threads,
        physical_candidate=args.physical_candidate,
    )
    audit = audit_cuda_contract_map_chain_source(generated)
    if generated.command_buffer_compatible and not audit.command_buffer_eligible:
        raise AssertionError(f"capture-safe source failed command-buffer audit: {audit}")
    library, command = _compile_generated_source(generated, args.artifact_directory, args.nvcc, args.architecture)
    counts = _handler_call_counts(library)
    if counts != (0, 0):
        raise AssertionError(f"freshly loaded handlers have nonzero call counts: {counts}")
    return {
        "schema_version": 2,
        "mode": "compile_link_load_only",
        "gpu_execution": False,
        "shape": {
            "rows": generated.rows,
            "input_features": generated.input_features,
            "rank": generated.rank,
        },
        "generated": {
            "forward_target": generated.forward_target,
            "reverse_target": generated.reverse_target,
            "forward_handler_symbol": generated.forward_handler_symbol,
            "reverse_handler_symbol": generated.reverse_handler_symbol,
            "semantic_digest": generated.semantic_digest,
            "source_digest": generated.source_digest,
            "source_audit": asdict(audit),
            "physical_candidate": generated.physical_candidate.value,
            "command_buffer_compatible": generated.command_buffer_compatible,
            "handler_counts_after_load": counts,
        },
        "compile_command": command,
        "environment": _environment_versions(),
        "revision": args.shuttle_revision,
    }


def _gpu_run(
    args: argparse.Namespace,
    program: TwoContractMapTrainingProgram,
) -> tuple[dict[str, Any], CaptureStabilization | None, tuple[CallbackCheckpoint, ...]]:
    _require_toolchain(args, require_gpu=True)
    command_buffer_flag_audit = (
        require_custom_call_command_buffers_enabled(os.environ.get("XLA_FLAGS", ""))
        if args.physical_candidate.command_buffer_compatible
        else None
    )
    generated = generate_cuda_contract_map_chain_ffi(
        program,
        forward_target=_FORWARD_TARGET,
        reverse_target=_REVERSE_TARGET,
        threads=args.threads,
        physical_candidate=args.physical_candidate,
    )
    audit = audit_cuda_contract_map_chain_source(generated)
    if generated.command_buffer_compatible and not audit.command_buffer_eligible:
        raise AssertionError(f"capture-safe source failed command-buffer audit: {audit}")
    library, command = _compile_generated_source(generated, args.artifact_directory, args.nvcc, args.architecture)
    if _handler_call_counts(library) != (0, 0):
        raise AssertionError("generated handler counts must start at zero")
    register_cuda_contract_map_chain_ffi(generated, library)

    key = jax.random.key(args.seed)
    activation_key, first_weight_key, second_weight_key, cotangent_key = jax.random.split(key, 4)
    activation = (jax.random.normal(activation_key, (generated.rows, generated.input_features)) * 0.2).astype(
        jnp.bfloat16
    )
    first_weight = (jax.random.normal(first_weight_key, (generated.input_features, generated.rank)) * 0.2).astype(
        jnp.bfloat16
    )
    second_weight = (jax.random.normal(second_weight_key, (generated.rank, generated.input_features)) * 0.2).astype(
        jnp.bfloat16
    )
    output_cotangent = (jax.random.normal(cotangent_key, (generated.rows, generated.input_features)) * 0.2).astype(
        jnp.bfloat16
    )

    @jax.jit
    def generated_step() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        output, first_contract, hidden, second_contract = call_cuda_contract_map_chain_forward_ffi(
            generated,
            activation,
            first_weight,
            second_weight,
        )
        input_adjoint, first_weight_adjoint, second_weight_adjoint = call_cuda_contract_map_chain_reverse_ffi(
            generated,
            activation,
            first_weight,
            second_weight,
            first_contract,
            hidden,
            second_contract,
            output_cotangent,
        )
        return output, input_adjoint, first_weight_adjoint, second_weight_adjoint

    matched_natural_step = jax.jit(_natural_training_step)
    compiled_generated_step = generated_step.lower().compile()
    capture_sites = derive_capture_site_manifest(
        "generated_two_ffi_calls",
        compiled_generated_step.as_text(),
        {
            generated.forward_target: "forward",
            generated.reverse_target: "reverse",
        },
    )

    def execute_generated() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        return compiled_generated_step()

    def execute_natural() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        return matched_natural_step(activation, first_weight, second_weight, output_cotangent)

    generated_outputs = execute_generated()
    natural_outputs = execute_natural()
    jax.block_until_ready((generated_outputs, natural_outputs))
    handler_counts_after_correctness = _handler_call_counts(library)

    activation_numpy = np.asarray(activation, dtype=np.float32)
    first_weight_numpy = np.asarray(first_weight, dtype=np.float32)
    second_weight_numpy = np.asarray(second_weight, dtype=np.float32)
    cotangent_numpy = np.asarray(output_cotangent, dtype=np.float32)
    reference_forward = execute_two_contract_map_forward(
        program,
        activation_numpy,
        first_weight_numpy,
        second_weight_numpy,
    )
    reference_reverse = execute_two_contract_map_reverse(
        program,
        activation_numpy,
        first_weight_numpy,
        second_weight_numpy,
        reference_forward,
        cotangent_numpy,
    )
    cpu_outputs = (
        reference_forward.output,
        reference_reverse.input_adjoint,
        reference_reverse.first_weight_adjoint,
        reference_reverse.second_weight_adjoint,
    )
    output_names = ("output", "input_adjoint", "first_weight_adjoint", "second_weight_adjoint")
    correctness: dict[str, dict[str, dict[str, float]]] = {
        "matched_natural_jax_forward_vjp": {},
        "ordered_cpu": {},
    }
    for name, generated_output, natural_output, cpu_output in zip(
        output_names,
        generated_outputs,
        natural_outputs,
        cpu_outputs,
        strict=True,
    ):
        natural_errors = _errors(generated_output, natural_output)
        cpu_errors = _errors(generated_output, cpu_output)
        _guard_errors(f"{name} versus natural JAX", natural_errors)
        _guard_errors(f"{name} versus ordered CPU", cpu_errors)
        correctness["matched_natural_jax_forward_vjp"][name] = natural_errors
        correctness["ordered_cpu"][name] = cpu_errors

    first_hashes = tuple(_hash(value) for value in generated_outputs)
    deterministic_hashes = [first_hashes]
    for _ in range(2):
        repeated = execute_generated()
        jax.block_until_ready(repeated)
        hashes = tuple(_hash(value) for value in repeated)
        if hashes != first_hashes:
            raise AssertionError("generated Contract/Map training step is not bitwise deterministic")
        deterministic_hashes.append(hashes)
    handler_counts_after_determinism = _handler_call_counts(library)

    variants = (
        CounterbalancedVariant(
            name="generated_two_ffi_calls",
            function=execute_generated,
            capture_sites=capture_sites,
        ),
        CounterbalancedVariant(
            name="matched_natural_jax_forward_vjp",
            function=execute_natural,
            capture_sites=CaptureSiteManifest.uninstrumented("matched_natural_jax_forward_vjp"),
        ),
    )
    warmup_variants = tuple((variant.name, variant.function) for variant in variants)
    _warm_up(warmup_variants, warmups=args.warmups)
    handler_counts_after_warmup = _handler_call_counts(library)
    stabilization = (
        stabilize_counterbalanced_variants(
            variants,
            iterations=args.iterations,
            synchronize=jax.block_until_ready,
            read_handler_counts=lambda: dict(zip(("forward", "reverse"), _handler_call_counts(library), strict=True)),
        )
        if generated.command_buffer_compatible
        else None
    )
    measurement = measure_counterbalanced_variants(
        variants,
        repeats=args.repeats,
        iterations=args.iterations,
        synchronize=jax.block_until_ready,
        read_handler_counts=lambda: dict(zip(("forward", "reverse"), _handler_call_counts(library), strict=True)),
    )
    expected_handler_count = 3 + args.warmups + args.repeats * args.iterations
    handler_counts = _handler_call_counts(library)
    handler_count_checkpoints = {
        "after_correctness": handler_counts_after_correctness,
        "after_determinism": handler_counts_after_determinism,
        "after_warmup": handler_counts_after_warmup,
        "after_measurement": handler_counts,
    }
    if stabilization is not None:
        handler_count_checkpoints["after_stabilization"] = tuple(
            stabilization.final_counts[name] for name in ("forward", "reverse")
        )
    stabilization_json = stabilization.to_json() if stabilization is not None else None
    if not generated.command_buffer_compatible and handler_counts != (expected_handler_count, expected_handler_count):
        raise AssertionError(
            f"generated handler counts {handler_counts} do not match "
            f"{(expected_handler_count, expected_handler_count)}"
        )

    specifications = tuple(
        jax.ShapeDtypeStruct(value.shape, value.dtype)
        for value in (activation, first_weight, second_weight, output_cotangent)
    )
    exported = jax.export.export(jax.jit(_natural_training_step))(*specifications)
    args.artifact_directory.mkdir(parents=True, exist_ok=True)
    (args.artifact_directory / "natural-training-step-stablehlo.mlir.bc").write_bytes(exported.mlir_module_serialized)
    (args.artifact_directory / "natural-training-step-optimized-hlo.txt").write_text(
        matched_natural_step.lower(activation, first_weight, second_weight, output_cotangent).compile().as_text()
    )
    generated_ms = measurement.measurements["generated_two_ffi_calls"]["median_ms"]
    natural_ms = measurement.measurements["matched_natural_jax_forward_vjp"]["median_ms"]
    telemetry = subprocess.check_output(
        (
            "nvidia-smi",
            "--query-gpu=name,driver_version,power.limit,clocks.current.sm,clocks.current.memory",
            "--format=csv,noheader,nounits",
            "--id=0",
        ),
        text=True,
    ).strip()
    result = {
        "schema_version": 3,
        "mode": "one_h100_contract_map_training_component",
        "shape": {
            "rows": generated.rows,
            "input_features": generated.input_features,
            "rank": generated.rank,
        },
        "frontend": {
            "reference": "ordinary JAX two-Contract scalar-Map forward differentiated by jax.vjp",
            "jax_owns_ad": True,
            "stablehlo_sha256": hashlib.sha256(exported.mlir_module_serialized).hexdigest(),
        },
        "fixture": {
            "seed": args.seed,
            "input_hashes": {
                "activation": _hash(activation),
                "first_weight": _hash(first_weight),
                "second_weight": _hash(second_weight),
                "output_cotangent": _hash(output_cotangent),
            },
        },
        "generated": {
            "forward_target": generated.forward_target,
            "reverse_target": generated.reverse_target,
            "semantic_digest": generated.semantic_digest,
            "source_digest": generated.source_digest,
            "source_audit": asdict(audit),
            "physical_candidate": generated.physical_candidate.value,
            "command_buffer_compatible": generated.command_buffer_compatible,
            "threads": generated.threads,
            "forward_shared_bytes": generated.forward_shared_bytes,
            "reverse_shared_bytes": generated.reverse_shared_bytes,
            "handler_counts": {"forward": handler_counts[0], "reverse": handler_counts[1]},
            "handler_count_checkpoints": {
                name: {"forward": counts[0], "reverse": counts[1]} for name, counts in handler_count_checkpoints.items()
            },
            "handler_count_contract": (
                {
                    "kind": "topology_matched_steady_state",
                    "acceptance_policy": asdict(CaptureAcceptancePolicy()),
                    "capture_sites": capture_sites.to_json(),
                    "logical_execution_count_not_expected": expected_handler_count,
                }
                if generated.command_buffer_compatible
                else {"kind": "logical_execution", "expected_each": expected_handler_count}
            ),
            "runtime": "JAX CUDA typed FFI; no Torch dependency",
        },
        "numerical_contract": {
            "Contracts": "fixed ordered FP32 multiply/add; BF16 RNE output",
            "Maps": "JAX-recovered source-ordered BF16 scalar ASTs",
            "weight_adjoints": "BF16 RNE before the surrounding FP32 optimizer conversion",
            "error_guard": {
                "maximum_absolute_error": _MAX_ABSOLUTE_ERROR,
                "mean_absolute_error": _MAX_MEAN_ABSOLUTE_ERROR,
            },
        },
        "correctness": correctness,
        "determinism": {
            "trials": len(deterministic_hashes),
            "output_names": output_names,
            "output_hashes": [list(hashes) for hashes in deterministic_hashes],
        },
        "measurements": measurement.measurements,
        "execution_order": [list(order) for order in measurement.execution_order],
        "ratio_generated_to_natural_jax": generated_ms / natural_ms,
        "benchmark": {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "iterations_per_sample": args.iterations,
            "counterbalanced": True,
            "generated_kernel_launches_per_step": 2,
            "stabilization": "up to eight untimed topology-matched rounds; two consecutive quiescent rounds required",
            "timing": "host enqueue interval followed by jax.block_until_ready; compilation excluded",
        },
        "command_buffer": (
            {
                "xla_flags": os.environ.get("XLA_FLAGS", ""),
                "uses_xla_default": command_buffer_flag_audit.uses_xla_default,
                "selected_entries": command_buffer_flag_audit.selected_entries,
                "evidence": "host-handler counts are attributed to every timed variant and counterbalanced order",
                "callback_checkpoints": [asdict(checkpoint) for checkpoint in measurement.callback_checkpoints],
                "capture_acceptance_policy": asdict(CaptureAcceptancePolicy()),
                "capture_sites": capture_sites.to_json(),
                "stabilization": stabilization_json,
                "capture_acceptance_evaluated_after_raw_serialization": True,
                "profiler_evidence": None,
            }
            if command_buffer_flag_audit is not None
            else None
        ),
        "compile_command": command,
        "environment": {
            **_environment_versions(),
            "device": jax.devices()[0].device_kind,
            "telemetry": telemetry,
        },
        "revision": args.shuttle_revision,
    }
    return result, stabilization, measurement.callback_checkpoints


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hlo-fixture", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--architecture", choices=("sm_90a",), default="sm_90a")
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--shuttle-revision", required=True)
    parser.add_argument("--require-jax-version", default="0.11.0")
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument(
        "--physical-candidate",
        type=ContractMapChainFfiPhysicalCandidate,
        choices=tuple(ContractMapChainFfiPhysicalCandidate),
        default=ContractMapChainFfiPhysicalCandidate.LAUNCH_CHECKED,
    )
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    program = _program(args.hlo_fixture)
    if args.preflight_only:
        result = _preflight(args, program)
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    raw_result, stabilization, checkpoints = _gpu_run(args, program)
    if not args.physical_candidate.command_buffer_compatible:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(raw_result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(raw_result, indent=2, sort_keys=True))
        return
    if stabilization is None:
        raise AssertionError("command-buffer-compatible execution did not produce stabilization evidence")

    try:
        result = serialize_then_assess_capture(
            args.json_output,
            raw_result,
            lambda: assess_command_buffer_capture(stabilization, checkpoints),
        )
    except CaptureAcceptanceError as error:
        print(json.dumps(error.result, indent=2, sort_keys=True))
        raise SystemExit(str(error)) from None
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
