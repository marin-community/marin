# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate and benchmark Shuttle's generated bounded-rank affine scan."""

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
from pathlib import Path
from typing import Any

import torch
from triton_affine_scan import execute_recurrent_affine_scan

from tile_lifetime.delta_rule_reference import delta_rule_update_expression
from tile_lifetime.stateful_scan_recovery import recover_affine_state_update


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--key-dimension", type=int, default=128)
    parser.add_argument("--value-dimension", type=int, default=128)
    parser.add_argument("--update-rank", type=int, default=1)
    parser.add_argument("--decay-axes", choices=("scalar", "key"), default="scalar")
    parser.add_argument(
        "--gate-operation",
        choices=("exp", "sigmoid", "clamped_softplus"),
        default="exp",
    )
    parser.add_argument("--block-v", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--shuttle-revision", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args()


def _gate(log_decay: torch.Tensor, operation: str) -> torch.Tensor:
    if operation == "exp":
        return torch.exp(log_decay)
    if operation == "sigmoid":
        return torch.sigmoid(log_decay)
    return torch.clamp(torch.nn.functional.softplus(log_decay), min=0.05, max=0.99)


def _inputs(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)
    factor_dtype = torch.bfloat16
    prefix = (args.batch_size, args.sequence_length, args.heads)
    read = torch.randn(
        (*prefix, args.key_dimension),
        device="cuda",
        dtype=factor_dtype,
        generator=generator,
    ) * (args.key_dimension**-0.5)
    diagonal_shape = (*prefix, 1 if args.decay_axes == "scalar" else args.key_dimension)
    log_decay = -torch.rand(diagonal_shape, device="cuda", dtype=torch.float32, generator=generator) * 0.1
    diagonal = _gate(log_decay, args.gate_operation).expand(*prefix, args.key_dimension).contiguous()
    vector_shape = (*prefix, args.update_rank, args.key_dimension)
    left = torch.randn(vector_shape, device="cuda", dtype=factor_dtype, generator=generator) * 0.05
    right = torch.randn(vector_shape, device="cuda", dtype=factor_dtype, generator=generator) * 0.05
    additive = (
        torch.randn(
            (*prefix, args.update_rank, args.value_dimension),
            device="cuda",
            dtype=factor_dtype,
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


def _torch_reference(inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    read = inputs["read"].float()
    diagonal = inputs["diagonal"].float()
    left = inputs["left"].float()
    right = inputs["right"].float()
    additive = inputs["additive"].float()
    residual_scale = inputs["residual_scale"].float()
    state = inputs["state"].clone()
    outputs = []
    for position in range(read.shape[1]):
        state *= diagonal[:, position, :, :, None]
        prediction = torch.einsum("bhkv,bhrk->bhrv", state, right[:, position])
        residual = residual_scale[:, position, :, :, None] * (additive[:, position] - prediction)
        state += torch.einsum("bhrk,bhrv->bhkv", left[:, position], residual)
        outputs.append(torch.einsum("bhkv,bhk->bhv", state, read[:, position]))
    return torch.stack(outputs, dim=1).to(inputs["read"].dtype), state


def _invoke(
    recovery: Any,
    inputs: dict[str, torch.Tensor],
    state: torch.Tensor,
    block_v: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return execute_recurrent_affine_scan(
        recovery,
        inputs["read"],
        inputs["diagonal"],
        inputs["left"],
        inputs["right"],
        inputs["additive"],
        inputs["residual_scale"],
        state,
        block_v=block_v,
    )


def _timings(
    recovery: Any,
    inputs: dict[str, torch.Tensor],
    block_v: int,
    warmups: int,
    repeats: int,
) -> list[float]:
    initial_state = inputs["state"]
    working_state = initial_state.clone()
    for _ in range(warmups):
        working_state.copy_(initial_state)
        _invoke(recovery, inputs, working_state, block_v)
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        working_state.copy_(initial_state)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _invoke(recovery, inputs, working_state, block_v)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def _tensor_hash(tensor: torch.Tensor) -> str:
    data = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    absolute = (actual.float() - expected.float()).abs()
    return {
        "maximum_absolute_error": float(absolute.max().item()),
        "mean_absolute_error": float(absolute.mean().item()),
        "finite": bool(torch.isfinite(actual).all().item()),
    }


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


def main() -> None:
    args = _arguments()
    fixture = delta_rule_update_expression(
        batch_size=args.batch_size,
        heads=args.heads,
        key_dimension=args.key_dimension,
        value_dimension=args.value_dimension,
        decay_axes=args.decay_axes,
        gate_operation=args.gate_operation,
        update_rank=args.update_rank,
    )
    recovery = recover_affine_state_update(fixture.update, fixture.state_name)
    inputs = _inputs(args)
    expected_output, expected_state = _torch_reference(inputs)
    actual_output, actual_state = _invoke(recovery, inputs, inputs["state"].clone(), args.block_v)
    repeat_output, repeat_state = _invoke(recovery, inputs, inputs["state"].clone(), args.block_v)
    torch.cuda.synchronize()
    samples = _timings(recovery, inputs, args.block_v, args.warmups, args.repeats)
    ordered = sorted(samples)
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
            "gate_operation": args.gate_operation,
            "block_v": args.block_v,
            "seed": args.seed,
        },
        "recovery": {
            "transition_structure": recovery.transition_structure.value,
            "maximum_low_rank": recovery.maximum_low_rank,
            "diagonal_scale_axes": [axis.label for axis in recovery.diagonal_scale_axes],
            "term_signatures": recovery.term_signatures,
        },
        "correctness": {
            "output": _error(actual_output, expected_output),
            "state": _error(actual_state, expected_state),
            "output_bitwise_repeat": bool(torch.equal(actual_output, repeat_output)),
            "state_bitwise_repeat": bool(torch.equal(actual_state, repeat_state)),
            "output_sha256": _tensor_hash(actual_output),
            "state_sha256": _tensor_hash(actual_state),
        },
        "timing": {
            "median_ms": statistics.median(samples),
            "minimum_ms": ordered[0],
            "maximum_ms": ordered[-1],
            "mean_ms": statistics.mean(samples),
            "samples_ms": samples,
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
