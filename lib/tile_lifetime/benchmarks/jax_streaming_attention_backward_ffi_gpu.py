# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Historical component benchmark for the recompute/combined attention FFI.

This remains useful as isolated physical evidence, but it is not the accepted
public Shuttle attention source boundary. A separate unaccepted diagnostic
imports an ordinary JAX plus ``jax.vjp`` graph, regenerates a reverse, and
compares separate forward-state and saved-state targets against Grug FA4 in
``experimental_jax_streaming_attention_saved_state_fa4_gpu.py``.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import importlib.metadata
import json
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp

from tile_lifetime.benchmark_boundary import (
    BenchmarkRepeatabilityMode,
    BenchmarkRepeatabilityPolicy,
    BenchmarkRepeatabilityReport,
    DenseBufferContract,
    DTypeRepeatabilityTolerance,
    NumericalAcceptanceContract,
    benchmark_repeatability_report,
    verify_benchmark_repeatability,
    verify_dense_buffer_boundary,
)
from tile_lifetime.jax_streaming_attention_backward_ffi import (
    StreamingAttentionBackwardResultPolicy,
    StreamingAttentionBackwardStatePolicy,
    call_streaming_attention_backward_ffi,
    call_streaming_attention_training_ffi,
    compile_streaming_attention_backward_ffi,
    generate_streaming_attention_backward_ffi,
    register_streaming_attention_backward_ffi,
)
from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.stablehlo_import import import_stablehlo
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
    causal_gqa_attention_training,
    causal_gqa_attention_vjp,
    export_debug_streaming_attention_backward,
    export_debug_streaming_attention_training,
)


def _samples(
    generated,
    oracle,
    generated_block,
    oracle_block,
    *,
    warmups: int,
    repeats: int,
    iterations: int,
) -> dict[str, object]:
    for _ in range(warmups):
        generated_block(generated())
        oracle_block(oracle())
    measurements = {"generated": [], "oracle": []}
    orders: list[list[str]] = []
    functions = {"generated": generated, "oracle": oracle}
    blockers = {"generated": generated_block, "oracle": oracle_block}
    for repeat in range(repeats):
        order = ("generated", "oracle") if repeat % 2 == 0 else ("oracle", "generated")
        orders.append(list(order))
        for name in order:
            start = time.perf_counter()
            result = None
            for _ in range(iterations):
                result = functions[name]()
            blockers[name](result)
            measurements[name].append((time.perf_counter() - start) * 1_000 / iterations)
    summaries = {
        name: {
            "samples_ms": values,
            "median_ms": statistics.median(values),
            "minimum_ms": min(values),
        }
        for name, values in measurements.items()
    }
    return {
        "variants": summaries,
        "execution_order": orders,
        "ratio_generated_to_oracle": summaries["generated"]["median_ms"] / summaries["oracle"]["median_ms"],
    }


def _generated_buffer_contracts(specifications) -> tuple[DenseBufferContract, ...]:
    return tuple(
        DenseBufferContract(
            name=specification.name,
            dtype=specification.dtype.value,
            shape=specification.shape,
            strides=specification.strides,
            minor_to_major=specification.layout,
        )
        for specification in specifications
    )


def _torch_buffer_contracts(names, tensors) -> tuple[DenseBufferContract, ...]:
    return tuple(
        DenseBufferContract.from_strides(
            name=name,
            dtype={"torch.bfloat16": "bf16"}.get(str(tensor.dtype), str(tensor.dtype)),
            shape=tuple(tensor.shape),
            strides=tuple(tensor.stride()),
        )
        for name, tensor in zip(names, tensors, strict=True)
    )


def _embedded_cubin(source: Path) -> dict[str, object]:
    match = re.search(r"unsigned char CUBIN_NAME\[\d+\] = \{(.*?)\};", source.read_text(), re.DOTALL)
    if match is None:
        raise ValueError(f"Triton AOT source has no embedded CUBIN: {source}")
    cubin = bytes(int(value, 16) for value in re.findall(r"0x([0-9a-fA-F]{2})", match.group(1)))
    return {
        "bytes": len(cubin),
        "sha256": hashlib.sha256(cubin).hexdigest(),
    }


def _torch_flash_oracle(arguments, *, scale: float, return_forward_output: bool):
    """Build the benchmark-only expert with the matched training boundary."""
    torch = importlib.import_module("torch")
    functional = importlib.import_module("torch.nn.functional")
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    logical_inputs = tuple(torch.utils.dlpack.from_dlpack(argument).detach() for argument in arguments)

    def call():
        query, key, value, output_cotangent = (tensor.transpose(1, 2) for tensor in logical_inputs)
        query.requires_grad_(True)
        key.requires_grad_(True)
        value.requires_grad_(True)
        output = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            scale=scale,
            enable_gqa=query.shape[1] != key.shape[1],
        )
        gradients = torch.autograd.grad(output, (query, key, value), output_cotangent)
        flash_results = (output, *gradients) if return_forward_output else gradients
        return tuple(tensor.transpose(1, 2) for tensor in flash_results)

    def block(_result):
        torch.cuda.synchronize()

    def numpy_outputs(result):
        return tuple(tensor.detach().float().cpu().numpy() for tensor in result)

    metadata = {
        "name": "torch_flash_sdpa_training" if return_forward_output else "torch_flash_sdpa_recompute",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "saved_forward_state": "internal to the timed forward/backward call",
        "returns_forward_output": return_forward_output,
        "backend_policy": "flash enabled; cuDNN, memory-efficient, and math disabled",
        "logical_interface": "B,S,H,D inputs and zero-copy B,S,H,D result views",
        "timed_views": "B,S,H,D to B,H,S,D inputs and B,H,S,D to B,S,H,D results; no explicit copies",
    }
    return call, block, numpy_outputs, logical_inputs, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--architecture", default="sm_90a")
    parser.add_argument("--triton-target")
    parser.add_argument("--sequence", type=int, default=64)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--key-value-heads", type=int, default=8)
    parser.add_argument("--head-dimension", type=int, choices=(64, 128), default=128)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--block-m", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--block-n", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--num-warps", type=int, choices=(4, 8), default=8)
    parser.add_argument("--num-stages", type=int, choices=(2, 3, 4), default=3)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-absolute-error-threshold", type=float, default=0.125)
    parser.add_argument("--mean-absolute-error-threshold", type=float, default=0.01)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--shuttle-revision", default="working-tree")
    parser.add_argument("--oracle", choices=("jax", "torch_flash"), default="jax")
    parser.add_argument(
        "--boundary",
        choices=("reverse_recompute", "training_forward_backward"),
        default="reverse_recompute",
    )
    args = parser.parse_args()
    if args.repeats % 2:
        raise ValueError("counterbalanced benchmark requires an even repeat count")
    numerical_acceptance = NumericalAcceptanceContract(
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        maximum_absolute_error=args.max_absolute_error_threshold,
        mean_absolute_error=args.mean_absolute_error_threshold,
    )
    generated_repeatability_policy = BenchmarkRepeatabilityPolicy(
        mode=BenchmarkRepeatabilityMode.BITWISE,
        minimum_repeats=2,
    )
    oracle_repeatability_policy = BenchmarkRepeatabilityPolicy(
        mode=BenchmarkRepeatabilityMode.BOUNDED_DRIFT,
        minimum_repeats=3,
        dtype_tolerances=(
            DTypeRepeatabilityTolerance(
                dtype="bf16",
                maximum_absolute_error=args.max_absolute_error_threshold,
                mean_absolute_error=args.mean_absolute_error_threshold,
            ),
        ),
    )
    pre_timing_audit_path = args.json_output.with_suffix(".pre-timing.json")
    pre_timing_audit: dict[str, object] = {
        "schema_version": 1,
        "status": "collecting",
        "note": "Oracle admissibility is independent of generated implementation correctness.",
    }

    def persist_pre_timing_audit() -> None:
        pre_timing_audit_path.parent.mkdir(parents=True, exist_ok=True)
        pre_timing_audit_path.write_text(json.dumps(pre_timing_audit, allow_nan=False, indent=2, sort_keys=True) + "\n")

    def verify_and_persist_repeatability(report: BenchmarkRepeatabilityReport, *, boundary_name: str) -> None:
        audit_key = "generated" if boundary_name == "generated Shuttle" else "benchmark_oracle"
        pre_timing_audit[audit_key] = asdict(report)
        persist_pre_timing_audit()
        try:
            verify_benchmark_repeatability(
                report,
                numerical_acceptance=numerical_acceptance,
                boundary_name=boundary_name,
            )
        except ValueError as error:
            pre_timing_audit["status"] = "failed"
            pre_timing_audit["failure"] = str(error)
            persist_pre_timing_audit()
            raise
        pre_timing_audit[f"{audit_key}_admissible"] = True
        persist_pre_timing_audit()

    scale = args.scale if args.scale is not None else args.head_dimension**-0.5
    config = StreamingAttentionBackwardDebugConfig(
        batch=1,
        query_length=args.sequence,
        key_length=args.sequence,
        query_heads=args.query_heads,
        key_value_heads=args.key_value_heads,
        head_dimension=args.head_dimension,
        scale=scale,
    )
    training_boundary = args.boundary == "training_forward_backward"
    hlo = (
        export_debug_streaming_attention_training(config)
        if training_boundary
        else export_debug_streaming_attention_backward(config)
    )
    args.build_directory.mkdir(parents=True, exist_ok=True)
    stablehlo_path = args.build_directory / "source_vjp_stablehlo.mlir.bc"
    stablehlo_path.write_bytes(hlo)
    graph = import_stablehlo(hlo, input_names=STREAMING_ATTENTION_BACKWARD_INPUT_NAMES)
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
    result_policy = (
        StreamingAttentionBackwardResultPolicy.FORWARD_OUTPUT_AND_GRADIENTS
        if training_boundary
        else StreamingAttentionBackwardResultPolicy.GRADIENTS_ONLY
    )
    output_names = (
        ("forward_output", "query_cotangent", "key_cotangent", "value_cotangent")
        if training_boundary
        else ("query_cotangent", "key_cotangent", "value_cotangent")
    )
    generated = generate_streaming_attention_backward_ffi(
        program,
        schedule,
        target_name=(
            f"shuttle.streaming_{'training' if training_boundary else 'reverse'}."
            f"recompute_s{args.sequence}_d{args.head_dimension}_"
            f"bm{args.block_m}_bn{args.block_n}_v1"
        ),
        state_policy=StreamingAttentionBackwardStatePolicy.RECOMPUTE,
        result_policy=result_policy,
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
    key = jax.random.key(20260809)
    shapes = tuple(specification.shape for specification in generated.inputs)
    arguments = tuple(
        jax.random.normal(fold_key, shape, dtype=jnp.bfloat16)
        for fold_key, shape in zip(jax.random.split(key, len(shapes)), shapes, strict=True)
    )

    @jax.jit
    def generated_call(query, key_tensor, value, output_cotangent):
        arguments = {
            "query": query,
            "key": key_tensor,
            "value": value,
            "output_cotangent": output_cotangent,
        }
        if training_boundary:
            return call_streaming_attention_training_ffi(generated, **arguments)
        return call_streaming_attention_backward_ffi(generated, **arguments)

    semantic_oracle_call = jax.jit(
        causal_gqa_attention_training(config) if training_boundary else causal_gqa_attention_vjp(config)
    )

    def generated_bound():
        return generated_call(*arguments)

    def semantic_oracle_bound():
        return semantic_oracle_call(*arguments)

    generated_outputs = generated_bound()
    semantic_oracle_outputs = semantic_oracle_bound()
    jax.block_until_ready((generated_outputs, semantic_oracle_outputs))
    second_generated_outputs = generated_bound()
    jax.block_until_ready(second_generated_outputs)
    output_dtypes = {specification.name: specification.dtype.value for specification in generated.outputs}
    generated_repeatability = benchmark_repeatability_report(
        output_names,
        (generated_outputs, second_generated_outputs),
        semantic_oracle_outputs,
        output_dtypes=output_dtypes,
        policy=generated_repeatability_policy,
    )
    verify_and_persist_repeatability(generated_repeatability, boundary_name="generated Shuttle")
    generated_first_hash = generated_repeatability.repeats[0].combined_sha256
    generated_deterministic = generated_repeatability.bitwise_repeat
    generated_errors = {
        output.output_name: output.semantic_error for output in generated_repeatability.repeats[0].outputs
    }
    runtime_imports_before_expert_oracle = {
        "torch": "torch" in sys.modules,
        "triton": "triton" in sys.modules,
    }
    if args.oracle == "torch_flash":
        oracle_bound, oracle_block, oracle_numpy_outputs, oracle_inputs, oracle_metadata = _torch_flash_oracle(
            arguments,
            scale=scale,
            return_forward_output=training_boundary,
        )
        expected_input_contracts = _generated_buffer_contracts(generated.inputs)
        observed_input_contracts = _torch_buffer_contracts(
            tuple(specification.name for specification in generated.inputs),
            oracle_inputs,
        )
        verify_dense_buffer_boundary(
            expected_input_contracts,
            observed_input_contracts,
            boundary_name="Flash-SDPA logical inputs",
        )
        expert_output_repeats = tuple(oracle_bound() for _ in range(oracle_repeatability_policy.minimum_repeats))
        for outputs in expert_output_repeats:
            oracle_block(outputs)
        expert_outputs = expert_output_repeats[0]
        expected_output_contracts = _generated_buffer_contracts(generated.outputs)
        observed_output_contracts = _torch_buffer_contracts(output_names, expert_outputs)
        verify_dense_buffer_boundary(
            expected_output_contracts,
            observed_output_contracts,
            boundary_name="Flash-SDPA logical results",
        )
        expert_numpy_repeats = tuple(oracle_numpy_outputs(outputs) for outputs in expert_output_repeats)
        oracle_repeatability = benchmark_repeatability_report(
            output_names,
            expert_numpy_repeats,
            semantic_oracle_outputs,
            output_dtypes=output_dtypes,
            policy=oracle_repeatability_policy,
        )
        pre_timing_audit["input_boundary_bshd"] = [asdict(contract) for contract in observed_input_contracts]
        pre_timing_audit["result_boundary_bshd"] = [asdict(contract) for contract in observed_output_contracts]
        verify_and_persist_repeatability(oracle_repeatability, boundary_name="Flash-SDPA oracle")
        oracle_metadata["input_boundary_bshd"] = [asdict(contract) for contract in observed_input_contracts]
        oracle_metadata["result_boundary_bshd"] = [asdict(contract) for contract in observed_output_contracts]
    else:
        oracle_bound = semantic_oracle_bound
        oracle_block = jax.block_until_ready
        expert_output_repeats = (
            semantic_oracle_outputs,
            *(semantic_oracle_bound() for _ in range(oracle_repeatability_policy.minimum_repeats - 1)),
        )
        jax.block_until_ready(expert_output_repeats)
        expert_outputs = expert_output_repeats[0]
        oracle_repeatability = benchmark_repeatability_report(
            output_names,
            expert_output_repeats,
            semantic_oracle_outputs,
            output_dtypes=output_dtypes,
            policy=oracle_repeatability_policy,
        )
        verify_and_persist_repeatability(oracle_repeatability, boundary_name="natural JAX oracle")
        oracle_metadata = {
            "name": "natural_jax_vjp",
            "saved_forward_state": "owned by natural JAX boundary",
            "returns_forward_output": training_boundary,
        }

    oracle_first_hash = oracle_repeatability.repeats[0].combined_sha256
    oracle_deterministic = oracle_repeatability.bitwise_repeat
    expert_errors = {output.output_name: output.semantic_error for output in oracle_repeatability.repeats[0].outputs}
    pre_timing_audit["status"] = "passed"
    persist_pre_timing_audit()

    correctness: dict[str, object] = {name: asdict(error) for name, error in generated_errors.items()}
    correctness["deterministic"] = generated_deterministic
    correctness["benchmark_oracle_against_semantic_reference"] = {
        name: asdict(error) for name, error in expert_errors.items()
    }
    correctness["benchmark_oracle_deterministic"] = oracle_deterministic
    correctness["generated_repeatability"] = asdict(generated_repeatability)
    correctness["benchmark_oracle_repeatability"] = asdict(oracle_repeatability)
    measurements = _samples(
        generated_bound,
        oracle_bound,
        jax.block_until_ready,
        oracle_block,
        warmups=args.warmups,
        repeats=args.repeats,
        iterations=args.iterations,
    )
    call_count = compiled.library.shuttle_streaming_attention_backward_ffi_call_count
    call_count.restype = ctypes.c_int
    aot_commands = [
        kernel.compile_argv(
            repository=args.repository,
            output_directory=args.build_directory,
            target=args.triton_target,
            python=Path(sys.executable),
        )
        for kernel in generated.aot_kernels
    ]
    result = {
        "schema_version": 1,
        "claim_status": "historical_component_only_not_accepted_public_frontend",
        "boundary": (
            "natural JAX Q/K/V/output-cotangent to output plus Q/K/V cotangents; "
            "forward state produced and consumed inside one timed call"
            if training_boundary
            else "natural JAX VJP Q/K/V/output-cotangent to Q/K/V cotangents; forward state recomputed"
        ),
        "shape": {
            "batch": 1,
            "sequence": args.sequence,
            "query_heads": args.query_heads,
            "key_value_heads": args.key_value_heads,
            "head_dimension": args.head_dimension,
            "dtype": "bfloat16",
        },
        "semantic": {
            "provenance": program.provenance.value,
            "fingerprint": generated.semantic_fingerprint,
            "state_policy": generated.state_policy.value,
            "result_policy": generated.result_policy.value,
            "result_layouts_minor_to_major": {output.name: output.layout for output in generated.outputs},
            "maximum_vjp": program.maximum_vjp.value,
            "numerical_policy": NumericalPolicy.ALLOW_ROUNDING_REORDER.value,
            "numerical_acceptance": asdict(numerical_acceptance),
            "source_operation_count": len(recovered.source_operation_ids),
            "contract_operation_count": len(recovered.contract_operation_ids),
            "fold_operation_count": (
                len(recovered.normalized_exponential_fold_operation_ids)
                + len(recovered.broadcast_vjp_fold_operation_ids)
                + 1
            ),
        },
        "schedule": {
            "block_m": args.block_m,
            "block_n": args.block_n,
            "num_warps": args.num_warps,
            "num_stages": args.num_stages,
            "standalone_output_dot": False,
            "atomic_accumulation": False,
        },
        "correctness": correctness,
        "deterministic_hash": generated_first_hash,
        "benchmark_oracle_deterministic_hash": oracle_first_hash,
        "measurements": measurements,
        "benchmark_oracle": oracle_metadata,
        "build": {
            "triton_aot_commands": aot_commands,
            "ffi_compile_command": compiled.compile_argv,
            "handler_source": str(compiled.source_path),
            "handler_source_sha256": hashlib.sha256(compiled.source_path.read_bytes()).hexdigest(),
            "library": str(compiled.library_path),
            "library_sha256": hashlib.sha256(compiled.library_path.read_bytes()).hexdigest(),
            "aot_sources": [
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size,
                    "embedded_cubin": _embedded_cubin(path),
                }
                for path in compiled.aot_sources
            ],
            "aot_headers": [
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size,
                }
                for path in sorted(args.build_directory.glob("shuttle_streaming_*.h"))
            ],
            "aot_input_sources": [
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size,
                }
                for path in sorted({args.build_directory / kernel.source for kernel in generated.aot_kernels})
            ],
            "stablehlo_fixture": {
                "path": str(stablehlo_path),
                "sha256": hashlib.sha256(hlo).hexdigest(),
                "bytes": len(hlo),
            },
        },
        "runtime_dependency_audit": {
            "generated_path_imports_before_expert_oracle": runtime_imports_before_expert_oracle,
            "benchmark_process_imports_after_expert_oracle": {
                "torch": "torch" in sys.modules,
                "triton": "triton" in sys.modules,
            },
            "handler_contains_torch": "torch" in compiled.source_path.read_text().lower(),
            "handler_contains_triton": "triton" in compiled.source_path.read_text().lower(),
            "library_dependencies": subprocess.check_output(("ldd", str(compiled.library_path)), text=True).splitlines(),
            "ffi_handler_calls": call_count(),
            "note": (
                "Triton is a build-time AOT compiler; the registered DSO embeds CUBIN launchers. "
                "Torch is imported only when explicitly selected as the benchmark oracle."
            ),
        },
        "environment": {
            "jax": jax.__version__,
            "jaxlib": importlib.metadata.version("jaxlib"),
            "triton_build_time": importlib.metadata.version("triton"),
            "python": platform.python_version(),
            "device": str(jax.devices()[0]),
            "gpu_telemetry": (
                subprocess.check_output(
                    (
                        "nvidia-smi",
                        "--query-gpu=name,uuid,compute_cap,driver_version,power.limit,clocks.current.sm,clocks.current.memory",
                        "--format=csv,noheader,nounits",
                        "--id=0",
                    ),
                    text=True,
                ).strip()
            ),
            "nvcc_version": subprocess.check_output((str(args.nvcc), "--version"), text=True).strip(),
        },
        "revisions": {"shuttle": args.shuttle_revision},
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
