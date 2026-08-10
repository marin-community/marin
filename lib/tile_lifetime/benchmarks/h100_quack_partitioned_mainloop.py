# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile and execute one generic partitioned SM90 Contract correctness gate."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import ml_dtypes
import numpy as np
import torch
from cutlass.cute.runtime import from_dlpack

from tile_lifetime.partitioned_gemm_reference import evaluate_partitioned_gemm_reference
from tile_lifetime.quack_partitioned_mainloop import generate_quack_partitioned_mainloop
from tile_lifetime.xla_partitioned_contract_map import plan_attached_partitioned_contract_maps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hlo", type=Path, required=True)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 2:
        raise ValueError("determinism gate requires at least two repetitions")
    args.artifact_directory.mkdir(parents=True, exist_ok=True)

    hlo = gzip.decompress(args.hlo.read_bytes()).decode() if args.hlo.suffix == ".gz" else args.hlo.read_text()
    semantic_plan = plan_attached_partitioned_contract_maps(
        hlo, target_prefix="shuttle.generic.partitioned_contract_map.h100"
    )
    if len(semantic_plan.families) != 1:
        raise RuntimeError(f"expected one partitioned Contract family, found {len(semantic_plan.families)}")
    program = semantic_plan.families[0].program
    generated = generate_quack_partitioned_mainloop(program)
    generated_path = args.artifact_directory / f"{generated.module_name}.py"
    generated_path.write_text(generated.source)
    generated_module = _import_generated_module(generated.module_name, generated_path)

    rng = np.random.default_rng(args.seed)
    host_operands = tuple(
        np.asarray(rng.normal(size=_physical_shape(shape)), dtype=ml_dtypes.bfloat16) for shape in program.operand_shapes
    )
    expected = evaluate_partitioned_gemm_reference(program, host_operands)
    device_operands = tuple(
        torch.as_tensor(np.asarray(operand, dtype=np.float32), device="cuda").to(torch.bfloat16)
        for operand in host_operands
    )
    lhs = device_operands[0].reshape(program.shape[0], program.shape[2])
    rhs = device_operands[1:]
    device_outputs = tuple(
        torch.empty((program.shape[0], shape[-1]), dtype=torch.bfloat16, device="cuda")
        for shape in (output.shape for output in expected)
    )
    cute_operands = tuple(_cute_bf16_tensor(tensor) for tensor in (lhs, *rhs, *device_outputs))
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(generated_module.run, *cute_operands, stream)

    output_hashes: list[tuple[str, ...]] = []
    last_outputs: tuple[np.ndarray, ...] = ()
    for _ in range(args.repetitions):
        compiled(*cute_operands, stream)
        torch.cuda.synchronize()
        last_outputs = tuple(
            np.asarray(output.float().cpu(), dtype=np.float32).reshape(reference.shape)
            for output, reference in zip(device_outputs, expected, strict=True)
        )
        output_hashes.append(tuple(hashlib.sha256(output.tobytes()).hexdigest() for output in last_outputs))

    errors = tuple(
        {
            "maximum_absolute_error": float(np.max(np.abs(actual - np.asarray(reference, dtype=np.float32)))),
            "mean_absolute_error": float(np.mean(np.abs(actual - np.asarray(reference, dtype=np.float32)))),
        }
        for actual, reference in zip(last_outputs, expected, strict=True)
    )
    if any(error["maximum_absolute_error"] != 0.0 for error in errors):
        raise AssertionError(f"partitioned SM90 output differs from the source-ordered reference: {errors}")
    if any(hashes != output_hashes[0] for hashes in output_hashes[1:]):
        raise AssertionError(f"partitioned SM90 output is nondeterministic: {output_hashes}")

    report = {
        "device": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "semantic_digest": program.semantic_digest,
        "generated_module": generated.module_name,
        "generated_source_sha256": generated.source_digest,
        "rhs_mma_ns": generated.rhs_mma_ns,
        "output_count": generated.output_count,
        "correctness": errors,
        "deterministic_hashes": output_hashes,
        "repetitions": args.repetitions,
        "latency_samples_ms": [],
        "throughput_claimed": False,
    }
    (args.artifact_directory / "result.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


def _cute_bf16_tensor(tensor: torch.Tensor) -> cute.Tensor:
    cute_tensor = from_dlpack(tensor, assumed_align=16)
    cute_tensor.element_type = cutlass.BFloat16
    return cute_tensor.mark_layout_dynamic(leading_dim=1)


def _import_generated_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load generated module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _physical_shape(shape: str) -> tuple[int, ...]:
    dimensions = shape.split("[", 1)[1].split("]", 1)[0]
    return tuple(int(value) for value in dimensions.split(",") if value)


if __name__ == "__main__":
    main()
