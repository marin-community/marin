# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Shuttle's generated saved-state attention split against Grug FA4.

The generated path is recovered from ordinary JAX plus ``jax.vjp``. Grug FA4
is imported only after both generated typed-FFI targets have been built and
registered, and remains an oracle rather than a generated runtime dependency.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax

from tile_lifetime.benchmark_boundary import (
    BenchmarkRepeatabilityMode,
    BenchmarkRepeatabilityPolicy,
    BenchmarkRepeatabilityReport,
    DTypeRepeatabilityTolerance,
    NumericalAcceptanceContract,
    benchmark_repeatability_report,
    verify_benchmark_repeatability,
)
from tile_lifetime.jax_streaming_attention_backward_ffi import (
    StreamingAttentionBackwardStatePolicy,
    call_streaming_attention_backward_ffi,
    compile_streaming_attention_backward_ffi,
    generate_streaming_attention_backward_ffi,
    register_streaming_attention_backward_ffi,
)
from tile_lifetime.jax_streaming_attention_forward_ffi import (
    call_streaming_attention_forward_ffi,
    compile_streaming_attention_forward_ffi,
    generate_streaming_attention_forward_ffi,
    register_streaming_attention_forward_ffi,
)
from tile_lifetime.jax_streaming_attention_training_frontend import (
    recover_jax_vjp_streaming_attention_training,
)
from tile_lifetime.plan import NumericalPolicy
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
    causal_gqa_attention_with_log_sum_exp,
    streaming_attention_training_input_specifications,
)

GRUG_FA4_SOURCE_PATHS = (
    Path("lib/levanter/src/levanter/grug/attention/_core.py"),
    Path("lib/levanter/src/levanter/grug/attention/_fa4_cute.py"),
    Path("lib/levanter/src/levanter/grug/attention/_fa4_cute_backend.py"),
    Path("lib/levanter/src/levanter/grug/attention/_fa4_cute_config.py"),
    Path("lib/levanter/src/levanter/grug/attention/_fa4_cute_kernels.py"),
    Path("lib/levanter/src/levanter/grug/attention/_fa4_cute_segmented_bwd.py"),
    Path("experiments/grug/base/model.py"),
    Path("experiments/grug/moe/model.py"),
)
OUTPUT_NAMES = ("output", "query_cotangent", "key_cotangent", "value_cotangent")
FORWARD_OUTPUT_NAMES = ("output", "log_sum_exp")
GRADIENT_OUTPUT_NAMES = ("query_cotangent", "key_cotangent", "value_cotangent")
GENERATED_FORWARD_TARGET = "shuttle.normalized_weighted_reduction.forward_saved_bshd_s2048_d128_v1"
GENERATED_REVERSE_TARGET = "shuttle.normalized_weighted_reduction.reverse_saved_bshd_s2048_d128_v1"
FORBIDDEN_GENERATED_DEPENDENCIES = ("fa4", "flash_attn", "torch")
GRUG_MODULES = (
    "levanter.grug.attention",
    "levanter.grug.attention._fa4_cute",
    "levanter.grug.attention._fa4_cute_backend",
    "levanter.grug.attention._fa4_cute_config",
    "levanter.grug.attention._fa4_cute_kernels",
)


def _counterbalanced_samples(
    generated: Callable[[], object],
    oracle: Callable[[], object],
    *,
    warmups: int,
    repeats: int,
    iterations: int,
) -> dict[str, object]:
    if repeats % 2:
        raise ValueError("counterbalanced timing requires an even repeat count")
    for _ in range(warmups):
        jax.block_until_ready(generated())
        jax.block_until_ready(oracle())
    measurements: dict[str, list[float]] = {"generated": [], "grug_fa4": []}
    orders: list[tuple[str, str]] = []
    functions = {"generated": generated, "grug_fa4": oracle}
    for repeat in range(repeats):
        order = ("generated", "grug_fa4") if repeat % 2 == 0 else ("grug_fa4", "generated")
        orders.append(order)
        for name in order:
            start = time.perf_counter()
            result = None
            for _ in range(iterations):
                result = functions[name]()
            jax.block_until_ready(result)
            measurements[name].append((time.perf_counter() - start) * 1_000 / iterations)
    summaries = {
        name: {
            "samples_ms": samples,
            "median_ms": statistics.median(samples),
            "minimum_ms": min(samples),
        }
        for name, samples in measurements.items()
    }
    return {
        "variants": summaries,
        "execution_order": orders,
        "ratio_generated_to_grug_fa4": summaries["generated"]["median_ms"] / summaries["grug_fa4"]["median_ms"],
    }


def _embedded_cubin(source: Path) -> dict[str, object]:
    match = re.search(r"unsigned char CUBIN_NAME\[\d+\] = \{(.*?)\};", source.read_text(), re.DOTALL)
    if match is None:
        raise ValueError(f"Triton AOT source has no embedded CUBIN: {source}")
    cubin = bytes(int(value, 16) for value in re.findall(r"0x([0-9a-fA-F]{2})", match.group(1)))
    return {"bytes": len(cubin), "sha256": hashlib.sha256(cubin).hexdigest()}


def _source_audit(repository: Path) -> dict[str, object]:
    sources: list[dict[str, object]] = []
    combined = ""
    for relative_path in GRUG_FA4_SOURCE_PATHS:
        path = repository / relative_path
        source_bytes = path.read_bytes()
        source = source_bytes.decode()
        sources.append(
            {
                "path": str(relative_path),
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
                "bytes": len(source_bytes),
            }
        )
        combined += source
    required_fragments = (
        'if implementation == "gpu_fa4_cute"',
        "def gpu_fa4_cute_attention(",
        "def segmented_flash_attention_forward(",
        "def segmented_flash_attention_backward(",
        "return out, (q, k, v, out, lse, lower_bounds, valid)",
        "def flash4_cute_kernel_config(",
    )
    missing = tuple(fragment for fragment in required_fragments if fragment not in combined)
    if missing:
        raise ValueError(f"Grug FA4 source audit is missing required saved-state evidence: {missing}")
    return {
        "implementation": "gpu_fa4_cute",
        "interface": {
            "forward": "Q/K/V BSHD BF16 -> O BSHD BF16 + natural-log LSE BHS FP32",
            "reverse": "Q/K/V/O/LSE/dO -> dQ/dK/dV",
            "causal_metadata": "lower_bounds[B,S] int32 and valid[B,S] bool",
        },
        "sources": sources,
        "checked_in_grug_default": (
            "available implementation, but base model does not explicitly select it and MoE config defaults to None"
        ),
        "rematerialization_scope": (
            "the matched benchmark saves O/LSE; checked-in MoE training may checkpoint the whole block"
        ),
    }


def _compiled_hlo_audit(compiled: Any) -> dict[str, object]:
    hlo = compiled.as_text()
    custom_call_targets = tuple(sorted(set(re.findall(r'custom_call_target="([^"]+)"', hlo))))
    return {
        "sha256": hashlib.sha256(hlo.encode()).hexdigest(),
        "bytes": len(hlo.encode()),
        "contains_custom_call": "custom-call" in hlo,
        "contains_copy": " copy(" in hlo or "copy(" in hlo,
        "custom_call_targets": custom_call_targets,
        "entry_layout": hlo.splitlines()[0] if hlo else "",
    }


def _verify_compiled_hlo_audit(
    audit: dict[str, object],
    *,
    boundary_name: str,
    expected_target: str | None = None,
) -> None:
    if not audit["entry_layout"]:
        raise ValueError(f"{boundary_name} has no compiled entry layout")
    if not audit["contains_custom_call"]:
        raise ValueError(f"{boundary_name} has no compiled custom call")
    targets = audit["custom_call_targets"]
    if expected_target is not None and expected_target not in targets:
        raise ValueError(f"{boundary_name} does not contain generated target {expected_target!r}: {targets}")


def _require_fresh_directory(path: Path, *, label: str) -> None:
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise ValueError(f"{label} must be a fresh empty directory: {path}")
        return
    path.mkdir(parents=True)


def _repository_audit(repository: Path, *, expected_revision: str) -> dict[str, str]:
    repository = repository.resolve()
    root = Path(
        subprocess.check_output(("git", "-C", str(repository), "rev-parse", "--show-toplevel"), text=True).strip()
    ).resolve()
    revision = subprocess.check_output(("git", "-C", str(repository), "rev-parse", "HEAD"), text=True).strip()
    if root != repository:
        raise ValueError(f"--repository is not the exact Git worktree root: {repository} != {root}")
    if revision != expected_revision:
        raise ValueError(f"--repository HEAD {revision} does not match --shuttle-revision {expected_revision}")
    return {"root": str(root), "head": revision}


def _grug_module_audit(repository: Path) -> tuple[dict[str, str], ...]:
    levanter_root = (repository / "lib/levanter/src").resolve()
    records = []
    for module_name in GRUG_MODULES:
        module = importlib.import_module(module_name)
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise ValueError(f"Grug module has no source file: {module_name}")
        path = Path(module_file).resolve()
        if not path.is_relative_to(levanter_root):
            raise ValueError(f"Grug module {module_name} imported outside --repository: {path}")
        records.append(
            {"module": module_name, "file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    return tuple(records)


def _generated_dependency_audit(forward: Any, reverse: Any) -> dict[str, object]:
    forward_source = forward.source_path.read_text().lower()
    reverse_source = reverse.source_path.read_text().lower()
    forward_dependencies = subprocess.check_output(("ldd", str(forward.library_path)), text=True).splitlines()
    reverse_dependencies = subprocess.check_output(("ldd", str(reverse.library_path)), text=True).splitlines()
    combined = "\n".join((forward_source, reverse_source, *forward_dependencies, *reverse_dependencies)).lower()
    forbidden = tuple(name for name in FORBIDDEN_GENERATED_DEPENDENCIES if name in combined)
    if forbidden:
        raise ValueError(f"generated runtime depends on forbidden expert implementation names: {forbidden}")
    return {
        "forbidden_names": forbidden,
        "forward_library_dependencies": forward_dependencies,
        "reverse_library_dependencies": reverse_dependencies,
    }


def _generated_saved_state_abi(forward: Any, reverse: Any) -> dict[str, object]:
    forward_inputs = tuple((value.name, value.shape, value.dtype.value, value.layout) for value in forward.inputs)
    forward_outputs = tuple((value.name, value.shape, value.dtype.value, value.layout) for value in forward.outputs)
    reverse_inputs = tuple((value.name, value.shape, value.dtype.value, value.layout) for value in reverse.inputs)
    reverse_outputs = tuple((value.name, value.shape, value.dtype.value, value.layout) for value in reverse.outputs)
    expected_forward_names = ("query", "key", "value")
    expected_forward_output_names = ("output", "log_sum_exp")
    expected_reverse_names = ("query", "key", "value", "output", "log_sum_exp", "output_cotangent")
    if tuple(value[0] for value in forward_inputs) != expected_forward_names:
        raise ValueError(f"generated forward inputs do not match the reviewed ABI: {forward_inputs}")
    if tuple(value[0] for value in forward_outputs) != expected_forward_output_names:
        raise ValueError(f"generated forward outputs do not match the reviewed ABI: {forward_outputs}")
    if tuple(value[0] for value in reverse_inputs) != expected_reverse_names:
        raise ValueError(f"generated saved reverse inputs do not match the reviewed ABI: {reverse_inputs}")
    if tuple(value[0] for value in reverse_outputs) != GRADIENT_OUTPUT_NAMES:
        raise ValueError(f"generated saved reverse outputs do not match the reviewed ABI: {reverse_outputs}")
    if forward_outputs != tuple(reverse_inputs[3:5]):
        raise ValueError("generated forward state and saved reverse state differ in shape, dtype, or layout")
    return {
        "forward_inputs": forward_inputs,
        "forward_outputs": forward_outputs,
        "reverse_inputs": reverse_inputs,
        "reverse_outputs": reverse_outputs,
    }


def _repeatability_policy(*, bounded: bool, maximum: float, mean: float) -> BenchmarkRepeatabilityPolicy:
    if not bounded:
        return BenchmarkRepeatabilityPolicy(mode=BenchmarkRepeatabilityMode.BITWISE, minimum_repeats=2)
    return BenchmarkRepeatabilityPolicy(
        mode=BenchmarkRepeatabilityMode.BOUNDED_DRIFT,
        minimum_repeats=3,
        dtype_tolerances=(
            DTypeRepeatabilityTolerance(dtype="bf16", maximum_absolute_error=maximum, mean_absolute_error=mean),
            DTypeRepeatabilityTolerance(dtype="fp32", maximum_absolute_error=maximum, mean_absolute_error=mean),
        ),
    )


def _repeatability_report(
    names: Sequence[str],
    repeats: Sequence[Sequence[object]],
    reference: Sequence[object],
    *,
    dtypes: dict[str, str],
    policy: BenchmarkRepeatabilityPolicy,
) -> BenchmarkRepeatabilityReport:
    return benchmark_repeatability_report(names, repeats, reference, output_dtypes=dtypes, policy=policy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--architecture", default="sm_90a")
    parser.add_argument("--triton-target")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--key-value-heads", type=int, default=8)
    parser.add_argument("--head-dimension", type=int, default=128)
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--block-n", type=int, default=32)
    parser.add_argument("--num-warps", type=int, default=8)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-absolute-error-threshold", type=float, default=0.125)
    parser.add_argument("--mean-absolute-error-threshold", type=float, default=0.01)
    parser.add_argument("--shuttle-revision", required=True)
    args = parser.parse_args()
    if (
        args.batch,
        args.sequence,
        args.query_heads,
        args.key_value_heads,
        args.head_dimension,
        args.block_m,
        args.block_n,
        args.num_warps,
        args.num_stages,
    ) != (1, 2048, 32, 8, 128, 32, 32, 8, 3):
        raise ValueError("this reviewed one-run gate accepts only fixed B1 S2048 Hq32 Hkv8 D128 BM/BN32 W8 S3")
    if args.repeats % 2:
        raise ValueError("counterbalanced benchmark requires an even repeat count")
    if "torch" in sys.modules:
        raise RuntimeError("Torch must not be imported by the generated or Grug-JAX benchmark path")

    _require_fresh_directory(args.artifact_directory, label="artifact directory")
    _require_fresh_directory(args.build_directory, label="build directory")
    repository_audit = _repository_audit(args.repository, expected_revision=args.shuttle_revision)
    source_audit = _source_audit(args.repository)
    numerical_acceptance = NumericalAcceptanceContract(
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        maximum_absolute_error=args.max_absolute_error_threshold,
        mean_absolute_error=args.mean_absolute_error_threshold,
    )
    generated_repeat_policy = _repeatability_policy(
        bounded=False,
        maximum=args.max_absolute_error_threshold,
        mean=args.mean_absolute_error_threshold,
    )
    oracle_repeat_policy = _repeatability_policy(
        bounded=True,
        maximum=args.max_absolute_error_threshold,
        mean=args.mean_absolute_error_threshold,
    )
    pre_timing_path = args.artifact_directory / "pre_timing.json"
    pre_timing: dict[str, object] = {
        "schema_version": 1,
        "status": "collecting",
        "oracle": source_audit,
        "repository": repository_audit,
        "note": "Grug FA4 admissibility is independent of generated Shuttle correctness.",
    }

    def persist_pre_timing() -> None:
        pre_timing_path.write_text(json.dumps(pre_timing, allow_nan=False, indent=2, sort_keys=True) + "\n")

    def verify_and_record(key: str, report: BenchmarkRepeatabilityReport, *, boundary_name: str) -> None:
        pre_timing[key] = asdict(report)
        persist_pre_timing()
        try:
            verify_benchmark_repeatability(
                report,
                numerical_acceptance=numerical_acceptance,
                boundary_name=boundary_name,
            )
        except ValueError as error:
            pre_timing["status"] = "failed"
            pre_timing["failure"] = str(error)
            persist_pre_timing()
            raise
        pre_timing[f"{key}_admissible"] = True
        persist_pre_timing()

    persist_pre_timing()
    scale = 1.0 / math.sqrt(args.head_dimension)
    config = StreamingAttentionBackwardDebugConfig(
        batch=args.batch,
        query_length=args.sequence,
        key_length=args.sequence,
        query_heads=args.query_heads,
        key_value_heads=args.key_value_heads,
        head_dimension=args.head_dimension,
        scale=scale,
    )
    input_specifications = streaming_attention_training_input_specifications(config)
    frontend = recover_jax_vjp_streaming_attention_training(
        causal_gqa_attention_training(config),
        input_specifications,
        input_names=STREAMING_ATTENTION_BACKWARD_INPUT_NAMES,
        schedule=StreamingTileSchedule(
            query_tile_size=args.block_m,
            key_value_tile_size=args.block_n,
            pipeline_depth=args.num_stages,
        ),
    )
    jaxpr_path = args.artifact_directory / "natural_training.jaxpr.txt"
    stablehlo_path = args.artifact_directory / "natural_training.stablehlo.bc"
    jaxpr_path.parent.mkdir(parents=True, exist_ok=True)
    jaxpr_path.write_text(frontend.jaxpr + "\n")
    stablehlo_path.write_bytes(frontend.stablehlo)
    program = eliminate_normalized_exp_maximum_vjp(
        frontend.recovered.program,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    schedule = derive_streaming_attention_backward_tile_schedule(
        program,
        query_tile_size=args.block_m,
        key_value_tile_size=args.block_n,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR,
    )
    generated_forward = generate_streaming_attention_forward_ffi(
        program,
        schedule,
        target_name=GENERATED_FORWARD_TARGET,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )
    generated_reverse = generate_streaming_attention_backward_ffi(
        program,
        schedule,
        target_name=GENERATED_REVERSE_TARGET,
        state_policy=StreamingAttentionBackwardStatePolicy.SAVED_OUTPUT_AND_LOG_SUM_EXP,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )
    if generated_forward.saved_state_encoding is not generated_reverse.saved_state_encoding:
        raise ValueError("generated forward and reverse disagree on saved LSE encoding")
    logical_abi = _generated_saved_state_abi(generated_forward, generated_reverse)
    pre_timing["logical_saved_state_abi"] = logical_abi
    persist_pre_timing()
    forward_build = args.build_directory / "forward"
    reverse_build = args.build_directory / "reverse"
    compiled_forward = compile_streaming_attention_forward_ffi(
        generated_forward,
        repository=args.repository,
        directory=forward_build,
        nvcc=args.nvcc,
        architecture=args.architecture,
        triton_target=args.triton_target,
    )
    compiled_reverse = compile_streaming_attention_backward_ffi(
        generated_reverse,
        repository=args.repository,
        directory=reverse_build,
        nvcc=args.nvcc,
        architecture=args.architecture,
        triton_target=args.triton_target,
    )
    register_streaming_attention_forward_ffi(compiled_forward)
    register_streaming_attention_backward_ffi(compiled_reverse)
    generated_runtime_imports = {"torch": "torch" in sys.modules, "flash_attn": "flash_attn" in sys.modules}
    if any(generated_runtime_imports.values()):
        raise RuntimeError(f"generated registration imported an expert runtime: {generated_runtime_imports}")
    generated_dependency_audit = _generated_dependency_audit(compiled_forward, compiled_reverse)
    pre_timing["generated_runtime_dependency_audit"] = generated_dependency_audit
    persist_pre_timing()

    keys = jax.random.split(jax.random.key(20260809), len(input_specifications))
    arguments = tuple(
        jax.random.normal(key, specification.shape, dtype=specification.dtype)
        for key, specification in zip(keys, input_specifications, strict=True)
    )
    query, key, value, output_cotangent = arguments
    logical_boundary = {
        "comparison": "logical BSHD, adapter-inclusive",
        "inputs": tuple(
            {
                "name": name,
                "shape": specification.shape,
                "dtype": str(specification.dtype),
            }
            for name, specification in zip(STREAMING_ATTENTION_BACKWARD_INPUT_NAMES, input_specifications, strict=True)
        ),
        "forward_outputs": {
            "output": {"shape": generated_forward.outputs[0].shape, "dtype": "bf16", "logical_axes": "BSHD"},
            "log_sum_exp": {
                "shape": generated_forward.outputs[1].shape,
                "dtype": "fp32",
                "logical_axes": "BHS",
            },
        },
        "reverse_outputs": tuple(
            {"name": output.name, "shape": output.shape, "dtype": output.dtype.value, "logical_axes": "BSHD"}
            for output in generated_reverse.outputs
        ),
        "timed_adapter_policy": (
            "all JAX/XLA input/output layout adapters, copies, and Grug causal-metadata construction execute "
            "inside the lowered timed call; no host or device copy is timed outside either callable"
        ),
    }
    pre_timing["logical_comparison_boundary"] = logical_boundary
    persist_pre_timing()

    @jax.jit
    def generated_forward_call(q, k, v):
        return call_streaming_attention_forward_ffi(generated_forward, query=q, key=k, value=v)

    @jax.jit
    def generated_reverse_call(q, k, v, output, log_sum_exp, dout):
        return call_streaming_attention_backward_ffi(
            generated_reverse,
            query=q,
            key=k,
            value=v,
            output=output,
            log_sum_exp=log_sum_exp,
            output_cotangent=dout,
        )

    @jax.jit
    def generated_composed_call(q, k, v, dout):
        output, log_sum_exp = call_streaming_attention_forward_ffi(generated_forward, query=q, key=k, value=v)
        gradients = call_streaming_attention_backward_ffi(
            generated_reverse,
            query=q,
            key=k,
            value=v,
            output=output,
            log_sum_exp=log_sum_exp,
            output_cotangent=dout,
        )
        return output, *gradients

    semantic_forward_call = jax.jit(causal_gqa_attention_with_log_sum_exp(config))
    semantic_training_call = jax.jit(causal_gqa_attention_training(config))
    semantic_forward = semantic_forward_call(query, key, value)
    semantic_training = semantic_training_call(query, key, value, output_cotangent)
    generated_forward_repeats = tuple(
        generated_forward_call(query, key, value) for _ in range(generated_repeat_policy.minimum_repeats)
    )
    generated_composed_repeats = tuple(
        generated_composed_call(query, key, value, output_cotangent)
        for _ in range(generated_repeat_policy.minimum_repeats)
    )
    jax.block_until_ready((semantic_forward, semantic_training, generated_forward_repeats, generated_composed_repeats))
    generated_forward_report = _repeatability_report(
        FORWARD_OUTPUT_NAMES,
        generated_forward_repeats,
        semantic_forward,
        dtypes={"output": "bf16", "log_sum_exp": "fp32"},
        policy=generated_repeat_policy,
    )
    generated_training_report = _repeatability_report(
        OUTPUT_NAMES,
        generated_composed_repeats,
        semantic_training,
        dtypes={name: "bf16" for name in OUTPUT_NAMES},
        policy=generated_repeat_policy,
    )
    verify_and_record("generated_forward", generated_forward_report, boundary_name="generated Shuttle forward")
    verify_and_record(
        "generated_training",
        generated_training_report,
        boundary_name="generated Shuttle saved-state training split",
    )

    from levanter.grug.attention import AttentionMask, gpu_fa4_cute_attention  # noqa: PLC0415
    from levanter.grug.attention._fa4_cute import (  # noqa: PLC0415
        _segmented_kernel_config,
        _simple_causal_lower_bounds,
    )
    from levanter.grug.attention._fa4_cute_backend import (  # noqa: PLC0415
        segmented_flash_attention_backward,
        segmented_flash_attention_forward,
    )

    grug_module_audit = _grug_module_audit(args.repository)
    pre_timing["grug_imported_modules"] = grug_module_audit
    persist_pre_timing()

    causal_mask = AttentionMask.causal()
    kernel_config = _segmented_kernel_config(args.head_dimension)

    @jax.jit
    def grug_forward_call(q, k, v):
        lower_bounds, valid = _simple_causal_lower_bounds(
            batch_size=q.shape[0],
            seq_len=q.shape[1],
            sliding_window=None,
        )
        return segmented_flash_attention_forward(
            q,
            k,
            v,
            lower_bounds,
            valid,
            softmax_scale=scale,
            kernel_config=kernel_config,
        )

    @jax.jit
    def grug_reverse_call(q, k, v, output, log_sum_exp, dout):
        lower_bounds, valid = _simple_causal_lower_bounds(
            batch_size=q.shape[0],
            seq_len=q.shape[1],
            sliding_window=None,
        )
        return segmented_flash_attention_backward(
            q,
            k,
            v,
            output,
            dout,
            log_sum_exp,
            lower_bounds,
            valid,
            softmax_scale=scale,
            kernel_config=kernel_config,
        )

    @jax.jit
    def grug_composed_call(q, k, v, dout):
        output, pullback = jax.vjp(lambda q_, k_, v_: gpu_fa4_cute_attention(q_, k_, v_, causal_mask), q, k, v)
        return output, *pullback(dout)

    grug_forward_repeats = tuple(
        grug_forward_call(query, key, value) for _ in range(oracle_repeat_policy.minimum_repeats)
    )
    grug_composed_repeats = tuple(
        grug_composed_call(query, key, value, output_cotangent) for _ in range(oracle_repeat_policy.minimum_repeats)
    )
    jax.block_until_ready((grug_forward_repeats, grug_composed_repeats))
    grug_forward_report = _repeatability_report(
        FORWARD_OUTPUT_NAMES,
        grug_forward_repeats,
        semantic_forward,
        dtypes={"output": "bf16", "log_sum_exp": "fp32"},
        policy=oracle_repeat_policy,
    )
    grug_training_report = _repeatability_report(
        OUTPUT_NAMES,
        grug_composed_repeats,
        semantic_training,
        dtypes={name: "bf16" for name in OUTPUT_NAMES},
        policy=oracle_repeat_policy,
    )
    oracle_output, oracle_lse = grug_forward_repeats[0]
    semantic_gradients = semantic_training[1:]
    grug_reverse_repeats = tuple(
        grug_reverse_call(query, key, value, oracle_output, oracle_lse, output_cotangent)
        for _ in range(oracle_repeat_policy.minimum_repeats)
    )
    jax.block_until_ready(grug_reverse_repeats)
    grug_reverse_report = _repeatability_report(
        GRADIENT_OUTPUT_NAMES,
        grug_reverse_repeats,
        semantic_gradients,
        dtypes={name: "bf16" for name in GRADIENT_OUTPUT_NAMES},
        policy=oracle_repeat_policy,
    )
    verify_and_record("grug_forward", grug_forward_report, boundary_name="Grug FA4 forward oracle")
    verify_and_record("grug_reverse", grug_reverse_report, boundary_name="Grug FA4 saved-state reverse oracle")
    verify_and_record(
        "grug_training",
        grug_training_report,
        boundary_name="Grug FA4 saved-state training oracle",
    )

    generated_output, generated_lse = generated_forward_repeats[0]
    generated_forward_compiled = generated_forward_call.lower(query, key, value).compile()
    generated_reverse_compiled = generated_reverse_call.lower(
        query, key, value, generated_output, generated_lse, output_cotangent
    ).compile()
    generated_composed_compiled = generated_composed_call.lower(query, key, value, output_cotangent).compile()
    grug_forward_compiled = grug_forward_call.lower(query, key, value).compile()
    grug_reverse_compiled = grug_reverse_call.lower(
        query, key, value, oracle_output, oracle_lse, output_cotangent
    ).compile()
    grug_composed_compiled = grug_composed_call.lower(query, key, value, output_cotangent).compile()
    compiled_hlo = {
        "generated_forward": _compiled_hlo_audit(generated_forward_compiled),
        "generated_reverse": _compiled_hlo_audit(generated_reverse_compiled),
        "generated_composed": _compiled_hlo_audit(generated_composed_compiled),
        "grug_forward": _compiled_hlo_audit(grug_forward_compiled),
        "grug_reverse": _compiled_hlo_audit(grug_reverse_compiled),
        "grug_composed": _compiled_hlo_audit(grug_composed_compiled),
        "note": "all layout adapters, copies, and causal metadata construction remain inside these compiled boundaries",
    }
    _verify_compiled_hlo_audit(
        compiled_hlo["generated_forward"],
        boundary_name="generated forward",
        expected_target=GENERATED_FORWARD_TARGET,
    )
    _verify_compiled_hlo_audit(
        compiled_hlo["generated_reverse"],
        boundary_name="generated saved reverse",
        expected_target=GENERATED_REVERSE_TARGET,
    )
    _verify_compiled_hlo_audit(compiled_hlo["generated_composed"], boundary_name="generated composed training")
    _verify_compiled_hlo_audit(compiled_hlo["grug_forward"], boundary_name="Grug FA4 forward")
    _verify_compiled_hlo_audit(compiled_hlo["grug_reverse"], boundary_name="Grug FA4 saved reverse")
    _verify_compiled_hlo_audit(compiled_hlo["grug_composed"], boundary_name="Grug FA4 composed training")
    generated_composed_targets = set(compiled_hlo["generated_composed"]["custom_call_targets"])
    if not {GENERATED_FORWARD_TARGET, GENERATED_REVERSE_TARGET}.issubset(generated_composed_targets):
        raise ValueError(f"generated composed boundary omits a generated target: {generated_composed_targets}")
    pre_timing["compiled_hlo"] = compiled_hlo
    pre_timing["status"] = "passed"
    persist_pre_timing()

    measurements = {
        "forward": _counterbalanced_samples(
            lambda: generated_forward_call(query, key, value),
            lambda: grug_forward_call(query, key, value),
            warmups=args.warmups,
            repeats=args.repeats,
            iterations=args.iterations,
        ),
        "reverse_saved_state": _counterbalanced_samples(
            lambda: generated_reverse_call(
                query,
                key,
                value,
                generated_output,
                generated_lse,
                output_cotangent,
            ),
            lambda: grug_reverse_call(query, key, value, oracle_output, oracle_lse, output_cotangent),
            warmups=args.warmups,
            repeats=args.repeats,
            iterations=args.iterations,
        ),
        "composed_forward_reverse": _counterbalanced_samples(
            lambda: generated_composed_call(query, key, value, output_cotangent),
            lambda: grug_composed_call(query, key, value, output_cotangent),
            warmups=args.warmups,
            repeats=args.repeats,
            iterations=args.iterations,
        ),
    }
    forward_call_count = compiled_forward.library.shuttle_streaming_attention_forward_ffi_call_count
    forward_call_count.restype = ctypes.c_int
    reverse_call_count = compiled_reverse.library.shuttle_streaming_attention_backward_ffi_call_count
    reverse_call_count.restype = ctypes.c_int
    result = {
        "schema_version": 1,
        "claim_scope": (
            "single-device attention component only: generated saved-state forward+reverse versus Grug "
            "gpu_fa4_cute at fixed B1 S2048 Hq32 Hkv8 D128 BF16 causal GQA"
        ),
        "shape": {
            "batch": args.batch,
            "sequence": args.sequence,
            "query_heads": args.query_heads,
            "key_value_heads": args.key_value_heads,
            "head_dimension": args.head_dimension,
            "dtype": "bf16",
            "scale": scale,
            "causal": True,
        },
        "frontend_provenance": asdict(frontend.audit),
        "saved_state_abi": {
            "output": {"shape": generated_forward.outputs[0].shape, "dtype": "bf16"},
            "log_sum_exp": {
                "shape": generated_forward.outputs[1].shape,
                "dtype": "fp32",
                "encoding": generated_forward.saved_state_encoding.value,
            },
            "reverse_state_policy": generated_reverse.state_policy.value,
            "generated_forward_fingerprint": generated_forward.semantic_fingerprint,
            "generated_reverse_fingerprint": generated_reverse.semantic_fingerprint,
            "logical_buffers": logical_abi,
        },
        "oracle": {**source_audit, "kernel_config": asdict(kernel_config)},
        "correctness": {
            "numerical_acceptance": asdict(numerical_acceptance),
            "generated_forward": asdict(generated_forward_report),
            "generated_training": asdict(generated_training_report),
            "grug_forward": asdict(grug_forward_report),
            "grug_reverse": asdict(grug_reverse_report),
            "grug_training": asdict(grug_training_report),
        },
        "measurements": measurements,
        "logical_comparison_boundary": logical_boundary,
        "compiled_hlo": compiled_hlo,
        "generated_build": {
            "forward_handler_sha256": hashlib.sha256(compiled_forward.source_path.read_bytes()).hexdigest(),
            "forward_library_sha256": hashlib.sha256(compiled_forward.library_path.read_bytes()).hexdigest(),
            "forward_cubin": _embedded_cubin(compiled_forward.aot_source),
            "reverse_handler_sha256": hashlib.sha256(compiled_reverse.source_path.read_bytes()).hexdigest(),
            "reverse_library_sha256": hashlib.sha256(compiled_reverse.library_path.read_bytes()).hexdigest(),
            "reverse_cubins": [_embedded_cubin(path) for path in compiled_reverse.aot_sources],
            "forward_compile_argv": compiled_forward.compile_argv,
            "reverse_compile_argv": compiled_reverse.compile_argv,
            "ffi_handler_calls": {"forward": forward_call_count(), "reverse": reverse_call_count()},
        },
        "runtime_dependency_audit": {
            "before_grug_import": generated_runtime_imports,
            "generated": generated_dependency_audit,
            "torch_imported": "torch" in sys.modules,
            "grug_imported_modules": grug_module_audit,
        },
        "environment": {
            "jax": jax.__version__,
            "jaxlib": importlib.metadata.version("jaxlib"),
            "python": platform.python_version(),
            "device": str(jax.devices()[0]),
            "nvcc": subprocess.check_output((str(args.nvcc), "--version"), text=True).strip(),
            "gpu": (
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
        },
        "revision": args.shuttle_revision,
        "repository": repository_audit,
    }
    output = args.artifact_directory / "result.json"
    output.write_text(json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
