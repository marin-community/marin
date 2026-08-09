# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark generated generic Map/Fold row-normalization backward kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load

from tile_lifetime import (
    DType,
    RowNormalizationSavePolicy,
    RowStatisticKind,
    RowStatisticScalePlacement,
    build_row_normalized_contract_program,
    compile_row_normalization_training,
    lower_row_normalization_axis_folds,
)
from tile_lifetime.cuda_axis_fold_codegen import GeneratedCudaAxisFold, generate_cuda_axis_fold


def _load_generated_axis_fold(
    generated: GeneratedCudaAxisFold,
    source_directory: Path,
    *,
    label: str,
) -> tuple[ModuleType, Path]:
    source_directory.mkdir(parents=True, exist_ok=True)
    module_name = f"shuttle_axis_fold_{label}_{generated.source_sha256[:16]}"
    source_path = source_directory / f"{module_name}.cu"
    build_directory = source_directory / f"build_{module_name}"
    build_directory.mkdir(parents=True, exist_ok=True)
    source_path.write_text(generated.source + "\n")
    module = load(
        name=module_name,
        sources=[str(source_path)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--threads=4"],
        build_directory=str(build_directory),
        with_cuda=True,
        verbose=True,
    )
    return module, source_path


def _benchmark_variants(
    variants: tuple[tuple[str, Callable[[], None]], ...],
    *,
    warmups: int,
    repeats: int,
    iterations: int,
) -> dict[str, object]:
    for _ in range(warmups):
        for _, function in variants:
            function()
    torch.cuda.synchronize()
    samples: dict[str, list[float]] = {name: [] for name, _ in variants}
    orders: list[list[str]] = []
    for repeat in range(repeats):
        order = variants if repeat % 2 == 0 else tuple(reversed(variants))
        orders.append([name for name, _ in order])
        for name, function in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                function()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / iterations)
    records = {
        name: {
            "samples_ms": values,
            "median_ms": statistics.median(values),
            "minimum_ms": min(values),
        }
        for name, values in samples.items()
    }
    generated_ms = float(records["generated"]["median_ms"])
    oracle_ms = float(records["torch_compile"]["median_ms"])
    return {
        "variants": records,
        "execution_order": orders,
        "ratio_generated_to_torch_compile": generated_ms / oracle_ms,
    }


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = (actual.float() - expected.float()).abs()
    return {
        "maximum_absolute_error": difference.max().item(),
        "mean_absolute_error": difference.mean().item(),
    }


def _sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _torch_reference(
    projected: torch.Tensor,
    feature_scale: torch.Tensor,
    standardized: torch.Tensor,
    inverse_scale: torch.Tensor,
    *,
    centered: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    local = projected * feature_scale.float()
    correlation = torch.sum(local * standardized.float(), dim=1, keepdim=True) / projected.shape[1]
    input_cotangent = local - standardized.float() * correlation
    if centered:
        input_cotangent = input_cotangent - torch.sum(local, dim=1, keepdim=True) / projected.shape[1]
    input_cotangent = input_cotangent * inverse_scale[:, None]
    feature_scale_cotangent = torch.sum(projected * standardized.float(), dim=0)
    return input_cotangent, feature_scale_cotangent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--column-groups-per-block", type=int, default=32)
    parser.add_argument("--statistic", choices=("rms", "layer"), default="rms")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--source-directory", type=Path, default=Path("/tmp/shuttle-axis-fold"))
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--shuttle-revision", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the generated axis-Fold benchmark requires CUDA")
    if args.repeats % 2:
        raise ValueError("counterbalanced benchmark requires an even repeat count")

    statistic_kind = (
        RowStatisticKind.UNCENTERED_SECOND_MOMENT if args.statistic == "rms" else RowStatisticKind.CENTERED_SECOND_MOMENT
    )
    source = build_row_normalized_contract_program(
        rows=args.rows,
        hidden=args.hidden,
        features=128,
        statistic_kind=statistic_kind,
        dtype=DType.BF16,
    )
    plan = compile_row_normalization_training(
        source,
        save_policy=RowNormalizationSavePolicy.SAVE_NORMALIZED,
        scale_placement=RowStatisticScalePlacement.SOURCE_ORDERED_PREPARATION,
    )
    programs = lower_row_normalization_axis_folds(plan, threads=args.threads)
    generated_input = generate_cuda_axis_fold(programs.input_cotangent)
    scale_program = replace(
        programs.feature_scale_cotangent,
        groups_per_block=args.column_groups_per_block,
    )
    generated_scale = generate_cuda_axis_fold(scale_program)
    input_module, input_source_path = _load_generated_axis_fold(
        generated_input,
        args.source_directory,
        label="input",
    )
    scale_module, scale_source_path = _load_generated_axis_fold(
        generated_scale,
        args.source_directory,
        label="scale",
    )

    generator = torch.Generator(device="cuda").manual_seed(20260809)
    projected = torch.randn(
        (args.rows, args.hidden),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    feature_scale = torch.randn((args.hidden,), dtype=torch.bfloat16, device="cuda", generator=generator)
    standardized = torch.randn(
        (args.rows, args.hidden),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    inverse_scale = torch.rand((args.rows,), dtype=torch.float32, device="cuda", generator=generator) + 0.25
    input_cotangent = torch.empty_like(projected)
    feature_scale_cotangent = torch.empty((args.hidden,), dtype=torch.float32, device="cuda")
    centered = statistic_kind is RowStatisticKind.CENTERED_SECOND_MOMENT

    @torch.compile(fullgraph=True, dynamic=False)
    def torch_reference() -> tuple[torch.Tensor, torch.Tensor]:
        return _torch_reference(
            projected,
            feature_scale,
            standardized,
            inverse_scale,
            centered=centered,
        )

    reference_outputs = torch_reference()

    def generated_input_only() -> None:
        input_module.run_out(projected, feature_scale, standardized, inverse_scale, input_cotangent)

    def generated_scale_only() -> None:
        scale_module.run_out(projected, standardized, feature_scale_cotangent)

    def generated_full() -> None:
        generated_input_only()
        generated_scale_only()

    def compiled_full() -> None:
        nonlocal reference_outputs
        reference_outputs = torch_reference()

    generated_full()
    torch.cuda.synchronize()
    reference_input, reference_scale = reference_outputs
    correctness = {
        "input_cotangent": _error(input_cotangent, reference_input),
        "feature_scale_cotangent": _error(feature_scale_cotangent, reference_scale),
    }
    first_hashes = {
        "input_cotangent": _sha256(input_cotangent),
        "feature_scale_cotangent": _sha256(feature_scale_cotangent),
    }
    generated_full()
    torch.cuda.synchronize()
    second_hashes = {
        "input_cotangent": _sha256(input_cotangent),
        "feature_scale_cotangent": _sha256(feature_scale_cotangent),
    }
    if first_hashes != second_hashes:
        raise AssertionError("generated row-normalization backward is not deterministic")
    correctness["deterministic_hashes"] = first_hashes

    measurements = {
        "full": _benchmark_variants(
            (("generated", generated_full), ("torch_compile", compiled_full)),
            warmups=args.warmups,
            repeats=args.repeats,
            iterations=args.iterations,
        )
    }
    telemetry = subprocess.check_output(
        (
            "nvidia-smi",
            "--query-gpu=name,driver_version,power.limit,clocks.current.sm,clocks.current.memory",
            "--format=csv,noheader,nounits",
            "--id=0",
        ),
        text=True,
    ).strip()
    properties = torch.cuda.get_device_properties(0)
    result = {
        "schema_version": 1,
        "workload": {
            "rows": args.rows,
            "hidden": args.hidden,
            "statistic": args.statistic,
            "inputs": {
                "projected": "float32",
                "feature_scale": "bfloat16",
                "standardized": "bfloat16",
                "inverse_scale": "float32",
            },
            "outputs": {"input_cotangent": "float32", "feature_scale_cotangent": "float32"},
        },
        "semantic_lowering": {
            "named_semantics_erased": True,
            "input_fold_state_count": len(programs.input_cotangent.reductions),
            "input_fold_fingerprint": programs.input_cotangent.semantic_fingerprint,
            "scale_fold_fingerprint": programs.feature_scale_cotangent.semantic_fingerprint,
            "column_groups_per_block": args.column_groups_per_block,
            "reassociation": programs.input_cotangent.reassociation.value,
        },
        "correctness": correctness,
        "measurements": measurements,
        "acceptance": {
            "comparison": "generated generic axis Folds versus torch.compile of identical scalar/reduction algebra",
            "ratio": measurements["full"]["ratio_generated_to_torch_compile"],
            "threshold": 1.2,
            "passes": measurements["full"]["ratio_generated_to_torch_compile"] <= 1.2,
        },
        "benchmark": {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "iterations_per_sample": args.iterations,
            "counterbalanced_order": True,
        },
        "generated_source": {
            "input": str(input_source_path),
            "input_sha256": generated_input.source_sha256,
            "feature_scale": str(scale_source_path),
            "feature_scale_sha256": generated_scale.source_sha256,
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "telemetry": telemetry,
        },
        "revisions": {"shuttle": args.shuttle_revision},
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
