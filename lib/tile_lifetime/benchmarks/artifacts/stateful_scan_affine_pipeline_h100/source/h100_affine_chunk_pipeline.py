# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate and benchmark the generic three-stage affine chunk pipeline."""

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from triton_affine_chunk_pipeline import (
    UPSTREAM_INSPIRATION_FILES,
    UPSTREAM_INSPIRATION_REVISION,
    affine_intra_chunk_prepare,
    affine_readout,
    affine_state_scan,
    allocate_affine_chunk_buffers,
    canonicalize_affine_chunk_inputs,
)
from triton_affine_scan import execute_recurrent_affine_scan

from tile_lifetime.delta_rule_reference import delta_rule_update_expression
from tile_lifetime.stablehlo_scan_recovery import compile_stablehlo_stateful_scan
from tile_lifetime.stateful_scan_recovery import RecoveredAffineStateUpdate, recover_affine_state_update
from tile_lifetime.stateful_scan_reference import (
    STATEFUL_SCAN_INPUT_NAMES,
    ScanDecayAxes,
    StatefulScanDebugConfig,
    export_debug_stateful_scan,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--key-dimension", type=int, default=128)
    parser.add_argument("--value-dimension", type=int, default=128)
    parser.add_argument("--update-rank", type=int, choices=(1, 2), default=1)
    parser.add_argument("--decay-axes", choices=("scalar", "key"), default="scalar")
    parser.add_argument("--chunk-size", type=int, choices=(16, 32, 64), default=64)
    parser.add_argument("--block-v", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--mutation-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--shuttle-revision", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args()


def _inputs(
    *,
    batch_size: int,
    sequence_length: int,
    heads: int,
    key_dimension: int,
    value_dimension: int,
    update_rank: int,
    decay_axes: str,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    prefix = (batch_size, sequence_length, heads)
    read = torch.randn((*prefix, key_dimension), device="cuda", dtype=torch.bfloat16, generator=generator) * (
        key_dimension**-0.5
    )
    diagonal_width = 1 if decay_axes == "scalar" else key_dimension
    log_decay = -torch.rand((*prefix, diagonal_width), device="cuda", dtype=torch.float32, generator=generator) * 0.01
    diagonal = torch.exp(log_decay).expand(*prefix, key_dimension).contiguous()
    vector_shape = (*prefix, update_rank, key_dimension)
    left = torch.randn(vector_shape, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.03
    right = torch.randn(vector_shape, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.03
    additive = (
        torch.randn((*prefix, update_rank, value_dimension), device="cuda", dtype=torch.bfloat16, generator=generator)
        * 0.1
    )
    residual_scale = torch.rand((*prefix, update_rank), device="cuda", dtype=torch.float32, generator=generator)
    state = (
        torch.randn(
            (batch_size, heads, key_dimension, value_dimension),
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.01
    )
    return {
        "read": read,
        "diagonal": diagonal,
        "left": left,
        "right": right,
        "additive": additive,
        "residual_scale": residual_scale,
        "state": state,
    }


def _recovery(
    *,
    batch_size: int,
    sequence_length: int,
    heads: int,
    key_dimension: int,
    value_dimension: int,
    update_rank: int,
    decay_axes: str,
) -> tuple[RecoveredAffineStateUpdate, bytes, dict[str, Any]]:
    config = StatefulScanDebugConfig(
        batch=batch_size,
        sequence=sequence_length,
        heads=heads,
        key_dimension=key_dimension,
        value_dimension=value_dimension,
        update_rank=update_rank,
        decay_axes=ScanDecayAxes(decay_axes),
    )
    stablehlo = export_debug_stateful_scan(config)
    compilation = compile_stablehlo_stateful_scan(
        stablehlo,
        input_names=STATEFUL_SCAN_INPUT_NAMES,
        chunk_sizes=(16, 32),
    )
    recovered = compilation.recovered_update
    return (
        recovered,
        stablehlo,
        {
            "source_operation_count": compilation.source_operation_count,
            "transition_structure": recovered.transition_structure.value,
            "maximum_low_rank": recovered.maximum_low_rank,
            "diagonal_scale_extents": [axis.extent for axis in recovered.diagonal_scale_axes],
            "term_signatures": recovered.term_signatures,
            "candidate_execution_forms": [candidate.execution_form.value for candidate in compilation.candidates],
        },
    )


def _expression_recovery(
    *,
    batch_size: int,
    heads: int,
    key_dimension: int,
    value_dimension: int,
    update_rank: int,
    decay_axes: str,
) -> RecoveredAffineStateUpdate:
    fixture = delta_rule_update_expression(
        batch_size=batch_size,
        heads=heads,
        key_dimension=key_dimension,
        value_dimension=value_dimension,
        decay_axes=decay_axes,
        gate_operation="exp",
        update_rank=update_rank,
    )
    return recover_affine_state_update(fixture.update, fixture.state_name)


def _timings(action: Callable[[], Any], warmups: int, repeats: int) -> list[float]:
    for _ in range(warmups):
        action()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        action()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]


def _summary(samples: list[float]) -> dict[str, Any]:
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(samples),
        "minimum_ms": ordered[0],
        "maximum_ms": ordered[-1],
        "mean_ms": statistics.mean(samples),
        "samples_ms": samples,
    }


def _tensor_hash(tensor: torch.Tensor) -> str:
    data = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    absolute = (actual.float() - expected.float()).abs()
    relative = absolute / expected.float().abs().clamp_min(1e-6)
    return {
        "maximum_absolute_error": float(absolute.max().item()),
        "mean_absolute_error": float(absolute.mean().item()),
        "maximum_relative_error": float(relative.max().item()),
        "finite": bool(torch.isfinite(actual).all().item()),
    }


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _environment() -> dict[str, Any]:
    gpu = torch.cuda.get_device_properties(0)
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version,clocks.sm,clocks.mem,power.limit", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": __import__("triton").__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu.name,
        "compute_capability": [gpu.major, gpu.minor],
        "driver_and_clocks": driver,
    }


def _execute_case(
    recovery: RecoveredAffineStateUpdate,
    inputs: dict[str, torch.Tensor],
    *,
    chunk_size: int,
    block_v: int,
) -> dict[str, Any]:
    physical_inputs = canonicalize_affine_chunk_inputs(
        recovery,
        inputs["read"],
        inputs["diagonal"],
        inputs["left"],
        inputs["right"],
        inputs["additive"],
        inputs["residual_scale"],
        chunk_size=chunk_size,
    )
    buffers = allocate_affine_chunk_buffers(physical_inputs, inputs["state"])
    expected_output, expected_state = execute_recurrent_affine_scan(
        recovery,
        inputs["read"],
        inputs["diagonal"],
        inputs["left"],
        inputs["right"],
        inputs["additive"],
        inputs["residual_scale"],
        inputs["state"].clone(),
        block_v=16,
    )
    actual_state = inputs["state"].clone()
    affine_intra_chunk_prepare(physical_inputs, buffers, block_v=block_v)
    affine_state_scan(physical_inputs, buffers, actual_state, block_v=block_v)
    actual_output = affine_readout(physical_inputs, buffers, block_v=block_v)
    repeat_state = inputs["state"].clone()
    affine_intra_chunk_prepare(physical_inputs, buffers, block_v=block_v)
    affine_state_scan(physical_inputs, buffers, repeat_state, block_v=block_v)
    repeat_output = affine_readout(physical_inputs, buffers, block_v=block_v).clone()
    torch.cuda.synchronize()
    return {
        "output": _error(actual_output, expected_output),
        "state": _error(actual_state, expected_state),
        "output_bitwise_repeat": bool(torch.equal(actual_output, repeat_output)),
        "state_bitwise_repeat": bool(torch.equal(actual_state, repeat_state)),
        "output_sha256": _tensor_hash(actual_output),
        "state_sha256": _tensor_hash(actual_state),
        "preparation_bytes": buffers.preparation_bytes,
        "forwarded_bytes": buffers.forwarded_bytes,
    }


def _mutation_matrix(args: argparse.Namespace) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for decay_axes in ("scalar", "key"):
        for update_rank in (1, 2):
            recovery = _expression_recovery(
                batch_size=args.batch_size,
                heads=args.heads,
                key_dimension=args.key_dimension,
                value_dimension=args.value_dimension,
                update_rank=update_rank,
                decay_axes=decay_axes,
            )
            inputs = _inputs(
                batch_size=args.batch_size,
                sequence_length=args.mutation_length,
                heads=args.heads,
                key_dimension=args.key_dimension,
                value_dimension=args.value_dimension,
                update_rank=update_rank,
                decay_axes=decay_axes,
                seed=args.seed + update_rank + (10 if decay_axes == "key" else 0),
            )
            results[f"{decay_axes}_rank_{update_rank}"] = _execute_case(
                recovery,
                inputs,
                chunk_size=16,
                block_v=args.block_v,
            )
    return results


def main() -> None:
    args = _arguments()
    torch.set_float32_matmul_precision("high")
    recovery, stablehlo, stablehlo_recovery = _recovery(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        heads=args.heads,
        key_dimension=args.key_dimension,
        value_dimension=args.value_dimension,
        update_rank=args.update_rank,
        decay_axes=args.decay_axes,
    )
    inputs = _inputs(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        heads=args.heads,
        key_dimension=args.key_dimension,
        value_dimension=args.value_dimension,
        update_rank=args.update_rank,
        decay_axes=args.decay_axes,
        seed=args.seed,
    )
    physical_inputs = canonicalize_affine_chunk_inputs(
        recovery,
        inputs["read"],
        inputs["diagonal"],
        inputs["left"],
        inputs["right"],
        inputs["additive"],
        inputs["residual_scale"],
        chunk_size=args.chunk_size,
    )
    buffers = allocate_affine_chunk_buffers(physical_inputs, inputs["state"])

    correctness = _execute_case(recovery, inputs, chunk_size=args.chunk_size, block_v=args.block_v)
    working_state = inputs["state"].clone()

    def prepare_action() -> None:
        affine_intra_chunk_prepare(physical_inputs, buffers, block_v=args.block_v)

    prepare_action()

    def scan_action() -> None:
        working_state.copy_(inputs["state"])
        affine_state_scan(physical_inputs, buffers, working_state, block_v=args.block_v)

    scan_action()

    def readout_action() -> torch.Tensor:
        return affine_readout(physical_inputs, buffers, block_v=args.block_v)

    def combined_action() -> torch.Tensor:
        working_state.copy_(inputs["state"])
        affine_intra_chunk_prepare(physical_inputs, buffers, block_v=args.block_v)
        affine_state_scan(physical_inputs, buffers, working_state, block_v=args.block_v)
        return affine_readout(physical_inputs, buffers, block_v=args.block_v)

    timings = {
        "affine_intra_chunk_prepare": _summary(_timings(prepare_action, args.warmups, args.repeats)),
        "affine_state_scan": _summary(_timings(scan_action, args.warmups, args.repeats)),
        "affine_readout": _summary(_timings(readout_action, args.warmups, args.repeats)),
        "combined": _summary(_timings(combined_action, args.warmups, args.repeats)),
    }
    benchmark_directory = Path(__file__).parent
    sources = (
        benchmark_directory / "h100_affine_chunk_pipeline.py",
        benchmark_directory / "triton_affine_chunk_pipeline.py",
        benchmark_directory / "triton_affine_scan.py",
        benchmark_directory.parent / "src/tile_lifetime/stablehlo_scan_recovery.py",
        benchmark_directory.parent / "src/tile_lifetime/stateful_scan_recovery.py",
    )
    result = {
        "schema_version": 1,
        "shuttle_revision": args.shuttle_revision,
        "environment": _environment(),
        "configuration": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "heads": args.heads,
            "key_dimension": args.key_dimension,
            "value_dimension": args.value_dimension,
            "update_rank": args.update_rank,
            "decay_axes": args.decay_axes,
            "chunk_size": args.chunk_size,
            "block_v": args.block_v,
            "seed": args.seed,
        },
        "stablehlo": {
            "sha256": hashlib.sha256(stablehlo).hexdigest(),
            "bytes": len(stablehlo),
            "recovery": stablehlo_recovery,
        },
        "physical_stages": [
            "affine_intra_chunk_prepare",
            "affine_state_scan",
            "affine_readout",
        ],
        "physical_buffers": {
            "preparation_bytes": buffers.preparation_bytes,
            "forwarded_bytes": buffers.forwarded_bytes,
            "summary_rank": args.chunk_size * args.update_rank,
            "chunk_count": physical_inputs.chunk_count,
            "numerical_contract": buffers.numerical_contract,
        },
        "correctness": correctness,
        "mutation_matrix": _mutation_matrix(args),
        "timing": timings,
        "oracle": {
            "name": "pinned_external_chunk_oracle",
            "median_ms": 0.5106240212917328,
            "target_1_2x_ms": 0.6127488255500793,
            "combined_ratio": timings["combined"]["median_ms"] / 0.5106240212917328,
            "called_by_compiler_path": False,
        },
        "provenance": {
            "upstream_inspiration_revision": UPSTREAM_INSPIRATION_REVISION,
            "upstream_inspiration_files": UPSTREAM_INSPIRATION_FILES,
            "external_kernel_imported": False,
            "source_sha256": {str(path.relative_to(benchmark_directory.parent)): _source_hash(path) for path in sources},
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.json_output.with_suffix(".stablehlo.mlir.bc").write_bytes(stablehlo)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
