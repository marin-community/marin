# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate and benchmark Shuttle's compiler-owned factored chunk scan."""

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
from triton_affine_scan import execute_recurrent_affine_scan
from triton_factored_chunk_scan import (
    execute_ordered_factored_chunks,
    prepare_factored_affine_chunks,
)

from tile_lifetime.experimental_stablehlo_scan_recovery import compile_experimental_natural_affine_scan
from tile_lifetime.stateful_scan_recovery import RecoveredAffineStateUpdate
from tile_lifetime.stateful_scan_reference import NaturalAffineScanConfig, ScanDecayAxes


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--key-dimension", type=int, default=128)
    parser.add_argument("--value-dimension", type=int, default=128)
    parser.add_argument("--update-rank", type=int, default=1)
    parser.add_argument("--decay-axes", choices=("scalar", "key"), default="scalar")
    parser.add_argument("--chunk-size", type=int, choices=(16, 32, 64), default=64)
    parser.add_argument("--block-v", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--preparation-backend", choices=("eager", "compile"), default="compile")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--shuttle-revision", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args()


def _inputs(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)
    prefix = (args.batch_size, args.sequence_length, args.heads)
    read = torch.randn(
        (*prefix, args.key_dimension),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ) * (args.key_dimension**-0.5)
    diagonal_width = 1 if args.decay_axes == "scalar" else args.key_dimension
    log_decay = (
        -torch.rand(
            (*prefix, diagonal_width),
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.01
    )
    diagonal = torch.exp(log_decay).expand(*prefix, args.key_dimension).contiguous()
    vector_shape = (*prefix, args.update_rank, args.key_dimension)
    left = torch.randn(vector_shape, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.03
    right = left
    additive = (
        torch.randn(
            (*prefix, args.update_rank, args.value_dimension),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    )
    residual_scale = torch.rand(
        (*prefix, args.update_rank),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    state = (
        torch.randn(
            (args.batch_size, args.heads, args.key_dimension, args.value_dimension),
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


def _prepare(
    recovery: RecoveredAffineStateUpdate,
    inputs: dict[str, torch.Tensor],
    chunk_size: int,
):
    return prepare_factored_affine_chunks(
        recovery,
        inputs["read"],
        inputs["diagonal"],
        inputs["left"],
        inputs["right"],
        inputs["additive"],
        inputs["residual_scale"],
        chunk_size=chunk_size,
    )


def _timings(action: Callable[[], Any], warmups: int, repeats: int) -> list[float]:
    for _ in range(warmups):
        action()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        action()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


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


def _environment() -> dict[str, Any]:
    gpu = torch.cuda.get_device_properties(0)
    driver = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,clocks.sm,clocks.mem,power.limit",
            "--format=csv,noheader",
        ],
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


def main() -> None:
    args = _arguments()
    torch.set_float32_matmul_precision("high")
    compilation = compile_experimental_natural_affine_scan(
        NaturalAffineScanConfig(
            batch=args.batch_size,
            sequence=args.sequence_length,
            heads=args.heads,
            key_dimension=args.key_dimension,
            value_dimension=args.value_dimension,
            update_rank=args.update_rank,
            decay_axes=ScanDecayAxes(args.decay_axes),
        ),
        chunk_sizes=(args.chunk_size,),
    )
    recovery = compilation.recovered_update
    inputs = _inputs(args)

    def eager_prepare():
        return _prepare(recovery, inputs, args.chunk_size)

    prepare = eager_prepare
    if args.preparation_backend == "compile":
        prepare = torch.compile(eager_prepare, fullgraph=True)

    summary = prepare()
    expected_output, expected_state = execute_recurrent_affine_scan(
        recovery,
        inputs["read"],
        inputs["diagonal"],
        inputs["left"],
        inputs["right"],
        inputs["additive"],
        inputs["residual_scale"],
        inputs["state"].clone(),
        block_v=32,
    )
    actual_output, actual_state = execute_ordered_factored_chunks(
        recovery,
        summary,
        inputs["state"].clone(),
        block_v=args.block_v,
    )
    repeat_output, repeat_state = execute_ordered_factored_chunks(
        recovery,
        summary,
        inputs["state"].clone(),
        block_v=args.block_v,
    )
    torch.cuda.synchronize()

    working_state = inputs["state"].clone()

    def execute_action() -> tuple[torch.Tensor, torch.Tensor]:
        working_state.copy_(inputs["state"])
        return execute_ordered_factored_chunks(recovery, summary, working_state, block_v=args.block_v)

    def combined_action() -> tuple[torch.Tensor, torch.Tensor]:
        prepared = prepare()
        working_state.copy_(inputs["state"])
        return execute_ordered_factored_chunks(recovery, prepared, working_state, block_v=args.block_v)

    preparation_samples = _timings(prepare, args.warmups, args.repeats)
    execution_samples = _timings(execute_action, args.warmups, args.repeats)
    combined_samples = _timings(combined_action, args.warmups, args.repeats)
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
            "preparation_backend": args.preparation_backend,
            "seed": args.seed,
        },
        "recovery": {
            "source_kind": compilation.provenance.source_kind.value,
            "source_artifact_sha256": compilation.provenance.artifact_sha256,
            "transition_structure": recovery.transition_structure.value,
            "maximum_low_rank": recovery.maximum_low_rank,
            "diagonal_scale_axes": [axis.label for axis in recovery.diagonal_scale_axes],
            "term_signatures": recovery.term_signatures,
        },
        "physical_summary": {
            "numerical_contract": summary.numerical_contract,
            "materialized_bytes": summary.materialized_bytes,
            "chunk_count": summary.diagonal.shape[1],
            "summary_rank": summary.chunk_size * summary.update_rank,
        },
        "correctness": {
            "reference": "compiler_owned_source_ordered_recurrent_skeleton",
            "output": _error(actual_output, expected_output),
            "state": _error(actual_state, expected_state),
            "output_bitwise_repeat": bool(torch.equal(actual_output, repeat_output)),
            "state_bitwise_repeat": bool(torch.equal(actual_state, repeat_state)),
            "output_sha256": _tensor_hash(actual_output),
            "state_sha256": _tensor_hash(actual_state),
        },
        "timing": {
            "preparation": _summary(preparation_samples),
            "execution": _summary(execution_samples),
            "combined": _summary(combined_samples),
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
