# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile and benchmark the natural-signature JAX attention reverse FFI."""

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
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tile_lifetime.jax_streaming_attention_backward_ffi import (
    StreamingAttentionBackwardStatePolicy,
    call_streaming_attention_backward_ffi,
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
    causal_gqa_attention_vjp,
    export_debug_streaming_attention_backward,
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


def _hash(tensors: tuple[jax.Array, ...]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(np.asarray(tensor).tobytes())
    return digest.hexdigest()


def _error(actual: jax.Array, expected: jax.Array) -> dict[str, float]:
    difference = np.abs(np.asarray(actual, dtype=np.float32) - np.asarray(expected, dtype=np.float32))
    return {
        "maximum_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
    }


def _embedded_cubin(source: Path) -> dict[str, object]:
    match = re.search(r"unsigned char CUBIN_NAME\[\d+\] = \{(.*?)\};", source.read_text(), re.DOTALL)
    if match is None:
        raise ValueError(f"Triton AOT source has no embedded CUBIN: {source}")
    cubin = bytes(int(value, 16) for value in re.findall(r"0x([0-9a-fA-F]{2})", match.group(1)))
    return {
        "bytes": len(cubin),
        "sha256": hashlib.sha256(cubin).hexdigest(),
    }


def _torch_flash_recompute_oracle(arguments, *, scale: float):
    """Build the benchmark-only expert with the same state-recompute boundary."""
    torch = importlib.import_module("torch")
    functional = importlib.import_module("torch.nn.functional")
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    query, key, value, output_cotangent = (torch.utils.dlpack.from_dlpack(argument).detach() for argument in arguments)
    query = query.transpose(1, 2).requires_grad_(True)
    key = key.transpose(1, 2).requires_grad_(True)
    value = value.transpose(1, 2).requires_grad_(True)
    output_cotangent = output_cotangent.transpose(1, 2)

    def call():
        output = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            scale=scale,
            enable_gqa=query.shape[1] != key.shape[1],
        )
        return torch.autograd.grad(output, (query, key, value), output_cotangent)

    def block(_result):
        torch.cuda.synchronize()

    def numpy_outputs(result):
        return tuple(tensor.detach().transpose(1, 2).float().cpu().numpy() for tensor in result)

    metadata = {
        "name": "torch_flash_sdpa_recompute",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "saved_forward_state": False,
        "backend_policy": "flash enabled; cuDNN, memory-efficient, and math disabled",
    }
    return call, block, numpy_outputs, metadata


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
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--shuttle-revision", default="working-tree")
    parser.add_argument("--oracle", choices=("jax", "torch_flash_recompute"), default="jax")
    args = parser.parse_args()
    if args.repeats % 2:
        raise ValueError("counterbalanced benchmark requires an even repeat count")
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
    hlo = export_debug_streaming_attention_backward(config)
    args.build_directory.mkdir(parents=True, exist_ok=True)
    stablehlo_path = args.build_directory / "source_vjp_stablehlo.mlir.bc"
    stablehlo_path.write_bytes(hlo)
    graph = import_stablehlo(hlo, input_names=STREAMING_ATTENTION_BACKWARD_INPUT_NAMES)
    recovered = recover_stablehlo_streaming_attention_backward(
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
    generated = generate_streaming_attention_backward_ffi(
        program,
        schedule,
        target_name=(
            f"shuttle.streaming_reverse.recompute_s{args.sequence}_d{args.head_dimension}_"
            f"bm{args.block_m}_bn{args.block_n}_v1"
        ),
        state_policy=StreamingAttentionBackwardStatePolicy.RECOMPUTE,
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
        return call_streaming_attention_backward_ffi(
            generated,
            query=query,
            key=key_tensor,
            value=value,
            output_cotangent=output_cotangent,
        )

    semantic_oracle_call = jax.jit(causal_gqa_attention_vjp(config))

    def generated_bound():
        return generated_call(*arguments)

    def semantic_oracle_bound():
        return semantic_oracle_call(*arguments)

    generated_outputs = generated_bound()
    semantic_oracle_outputs = semantic_oracle_bound()
    jax.block_until_ready((generated_outputs, semantic_oracle_outputs))
    runtime_imports_before_expert_oracle = {
        "torch": "torch" in sys.modules,
        "triton": "triton" in sys.modules,
    }
    if args.oracle == "torch_flash_recompute":
        oracle_bound, oracle_block, oracle_numpy_outputs, oracle_metadata = _torch_flash_recompute_oracle(
            arguments,
            scale=scale,
        )
        expert_outputs = oracle_bound()
        oracle_block(expert_outputs)
        expert_correctness = {
            name: _error(actual, expected)
            for name, actual, expected in zip(
                ("query", "key", "value"),
                oracle_numpy_outputs(expert_outputs),
                semantic_oracle_outputs,
                strict=True,
            )
        }
    else:
        oracle_bound = semantic_oracle_bound
        oracle_block = jax.block_until_ready
        oracle_metadata = {
            "name": "natural_jax_vjp",
            "saved_forward_state": False,
        }
        expert_correctness = None

    first_hash = _hash(generated_outputs)
    second_hash = _hash(generated_bound())
    correctness = {
        name: _error(actual, expected)
        for name, actual, expected in zip(
            ("query", "key", "value"),
            generated_outputs,
            semantic_oracle_outputs,
            strict=True,
        )
    }
    correctness["deterministic"] = first_hash == second_hash
    correctness["benchmark_oracle_against_semantic_reference"] = expert_correctness
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
        "boundary": "natural JAX VJP Q/K/V/output-cotangent to Q/K/V cotangents; forward state recomputed",
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
            "maximum_vjp": program.maximum_vjp.value,
            "numerical_policy": NumericalPolicy.ALLOW_ROUNDING_REORDER.value,
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
        "deterministic_hash": first_hash,
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
