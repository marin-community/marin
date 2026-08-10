# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark recurrent and chunkwise GDN backends for a Shuttle StatefulScan."""

import argparse
import hashlib
import importlib
import json
import platform
import statistics
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tile_lifetime.gated_delta_scan import compile_experimental_gated_delta_scan, recurrent_gated_delta_reference

FLA_REVISION = "9c8e42e762fce087c27b673af4922795d9edb85e"
FLASH_QLA_REVISION = "050c6bbee9e03efbbfe41063fe4e33742c4a87cb"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("correctness", "decode", "prefill"), required=True)
    parser.add_argument("--backend", choices=("fla_recurrent", "fla_chunk", "flash_qla"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--query-heads", type=int, default=16)
    parser.add_argument("--value-heads", type=int, default=32)
    parser.add_argument("--key-dimension", type=int, default=128)
    parser.add_argument("--value-dimension", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--decay-scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--fla-root", type=Path)
    parser.add_argument("--flash-qla-root", type=Path)
    parser.add_argument("--shuttle-revision", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args()


def _load_backend(args: argparse.Namespace) -> Callable[..., tuple[torch.Tensor, torch.Tensor | None]]:
    if args.fla_root is not None:
        sys.path.insert(0, str(args.fla_root))
    if args.flash_qla_root is not None:
        sys.path.insert(0, str(args.flash_qla_root))

    if args.backend == "fla_recurrent":
        module = importlib.import_module("fla.ops.gated_delta_rule")
        return module.fused_recurrent_gated_delta_rule
    if args.backend == "fla_chunk":
        module = importlib.import_module("fla.ops.gated_delta_rule")
        return module.chunk_gated_delta_rule
    module = importlib.import_module("flash_qla")
    return module.chunk_gated_delta_rule


def _inputs(
    *,
    batch_size: int,
    sequence_length: int,
    query_heads: int,
    value_heads: int,
    key_dimension: int,
    value_dimension: int,
    seed: int,
    decay_scale: float,
) -> dict[str, torch.Tensor]:
    if value_heads % query_heads != 0:
        raise ValueError("value heads must be divisible by query heads")
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    q = torch.randn(
        (batch_size, sequence_length, query_heads, key_dimension),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k = torch.randn(
        (batch_size, sequence_length, query_heads, key_dimension),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    v = torch.randn(
        (batch_size, sequence_length, value_heads, value_dimension),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    log_decay = (
        -torch.rand(
            (batch_size, sequence_length, value_heads),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * decay_scale
    )
    beta = torch.rand(
        (batch_size, sequence_length, value_heads),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    state = (
        torch.randn(
            (batch_size, value_heads, key_dimension, value_dimension),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * 0.01
    )
    return {"q": q, "k": k, "v": v, "g": log_decay, "beta": beta, "initial_state": state}


def _invoke(
    backend: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    backend_name: str,
    inputs: dict[str, torch.Tensor],
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    kwargs: dict[str, Any] = {
        **inputs,
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
    }
    if backend_name == "fla_chunk":
        kwargs["chunk_size"] = chunk_size
    if backend_name == "flash_qla":
        kwargs.pop("use_qk_l2norm_in_kernel")
        q = inputs["q"].float()
        k = inputs["k"].float()
        epsilon = torch.tensor(1e-6, device="cuda", dtype=torch.float32)
        kwargs["q"] = (q * torch.rsqrt(torch.sum(q * q, dim=-1, keepdim=True) + epsilon)).to(torch.bfloat16)
        kwargs["k"] = (k * torch.rsqrt(torch.sum(k * k, dim=-1, keepdim=True) + epsilon)).to(torch.bfloat16)
    output, final_state = backend(**kwargs)
    if final_state is None:
        raise RuntimeError(f"backend {backend_name} did not return the requested final state")
    return output, final_state


def _timings(operation: Callable[[], tuple[torch.Tensor, torch.Tensor]], warmups: int, repeats: int) -> list[float]:
    for _ in range(warmups):
        operation()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        operation()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]


def _statistics(samples: list[float]) -> dict[str, Any]:
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(samples),
        "minimum_ms": ordered[0],
        "maximum_ms": ordered[-1],
        "mean_ms": statistics.mean(samples),
        "samples_ms": samples,
    }


def _tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def _revision(root: Path | None) -> str | None:
    if root is None:
        return None
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu.name,
        "compute_capability": [gpu.major, gpu.minor],
        "driver_and_clocks": driver,
    }


def _correctness(
    backend: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    inputs = _inputs(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        query_heads=args.query_heads,
        value_heads=args.value_heads,
        key_dimension=args.key_dimension,
        value_dimension=args.value_dimension,
        seed=args.seed,
        decay_scale=args.decay_scale,
    )
    output, final_state = _invoke(backend, args.backend, inputs, args.chunk_size)
    torch.cuda.synchronize()

    group_size = args.value_heads // args.query_heads
    q = inputs["q"].float().repeat_interleave(group_size, dim=2).cpu().numpy()
    k = inputs["k"].float().repeat_interleave(group_size, dim=2).cpu().numpy()
    value = inputs["v"].float().cpu().numpy()
    log_decay = inputs["g"].cpu().numpy()
    beta = inputs["beta"].cpu().numpy()
    state = inputs["initial_state"].cpu().numpy()
    expected_output, expected_state = recurrent_gated_delta_reference(
        q,
        k,
        value,
        log_decay,
        beta,
        initial_state=state,
    )
    actual_output = output.float().cpu().numpy()
    actual_state = final_state.float().cpu().numpy()
    output_error = np.abs(actual_output - expected_output)
    state_error = np.abs(actual_state - expected_state)

    output_repeat, state_repeat = _invoke(backend, args.backend, inputs, args.chunk_size)
    torch.cuda.synchronize()
    return {
        "output": {
            "maximum_absolute_error": float(output_error.max()),
            "mean_absolute_error": float(output_error.mean()),
            "p99_absolute_error": float(np.quantile(output_error, 0.99)),
            "finite": bool(np.isfinite(actual_output).all()),
            "sha256": _tensor_hash(output),
            "bitwise_repeat": bool(torch.equal(output, output_repeat)),
        },
        "final_state": {
            "maximum_absolute_error": float(state_error.max()),
            "mean_absolute_error": float(state_error.mean()),
            "p99_absolute_error": float(np.quantile(state_error, 0.99)),
            "finite": bool(np.isfinite(actual_state).all()),
            "sha256": _tensor_hash(final_state),
            "bitwise_repeat": bool(torch.equal(final_state, state_repeat)),
        },
    }


def main() -> None:
    args = _arguments()
    if args.mode == "decode" and args.sequence_length != 1:
        raise ValueError("decode mode requires --sequence-length 1")
    if args.mode == "correctness" and args.sequence_length > 256:
        raise ValueError("correctness mode is intentionally bounded to sequence length <= 256")
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be nonnegative and repeats must be positive")
    if args.decay_scale < 0:
        raise ValueError("decay scale must be nonnegative")

    backend = _load_backend(args)
    compilation = compile_experimental_gated_delta_scan(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        heads=args.value_heads,
        key_dimension=args.key_dimension,
        value_dimension=args.value_dimension,
        chunk_sizes=(16, 32, 64),
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "mode": args.mode,
        "backend": args.backend,
        "config": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "query_heads": args.query_heads,
            "value_heads": args.value_heads,
            "key_dimension": args.key_dimension,
            "value_dimension": args.value_dimension,
            "chunk_size": args.chunk_size,
            "input_dtype": "bf16",
            "state_dtype": "fp32",
            "seed": args.seed,
            "decay_scale": args.decay_scale,
        },
        "revisions": {
            "shuttle": args.shuttle_revision,
            "fla_expected": FLA_REVISION,
            "fla_actual": _revision(args.fla_root),
            "flash_qla_expected": FLASH_QLA_REVISION,
            "flash_qla_actual": _revision(args.flash_qla_root),
        },
        "candidate_set": [asdict(candidate) for candidate in compilation.candidates],
        "environment": _environment(),
    }

    if args.mode == "correctness":
        result["correctness"] = _correctness(backend, args)
    else:
        inputs = _inputs(
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            query_heads=args.query_heads,
            value_heads=args.value_heads,
            key_dimension=args.key_dimension,
            value_dimension=args.value_dimension,
            seed=args.seed,
            decay_scale=args.decay_scale,
        )

        def operation() -> tuple[torch.Tensor, torch.Tensor]:
            return _invoke(backend, args.backend, inputs, args.chunk_size)

        samples = _timings(operation, args.warmups, args.repeats)
        output, final_state = operation()
        output_repeat, state_repeat = operation()
        torch.cuda.synchronize()
        result["timing"] = _statistics(samples)
        result["output_sha256"] = _tensor_hash(output)
        result["state_sha256"] = _tensor_hash(final_state)
        result["output_bitwise_repeat"] = bool(torch.equal(output, output_repeat))
        result["state_bitwise_repeat"] = bool(torch.equal(final_state, state_repeat))

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
