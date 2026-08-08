# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark MoK's standalone BF16 routed-expert GEMM primitive on SM100.

This builds a small extension against a pinned Mixture-of-Kittens checkout. It
does not invoke MoK dispatch, communication, SwiGLU, combine, or the complete
MoE megakernel.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import struct
import subprocess
import sys
import sysconfig
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

PACKAGE_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(PACKAGE_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE_ROOT))

from tile_lifetime.cuda_map_fold_codegen import (  # noqa: E402
    shuttle_map_fold_program,
    verify_cuda_map_fold_include,
)

MOK_COMMIT = "3e1cf43ab93ad040afed52a45ab03cb490ffe4be"
THUNDERKITTENS_COMMIT = "1c3920d993404dd49a6d4c7267ea11d583bd5c68"
DEFAULT_EXPERTS = 96
DEFAULT_HIDDEN_SIZE = 7_168
DEFAULT_INTERMEDIATE_SIZE = 3_072
DEFAULT_PADDED_ROWS_PER_EXPERT = 256
TILE_ROWS = 256
TILE_COLUMNS = 256
TILE_REDUCTION = 64


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mok-root",
        type=Path,
        default=Path(os.environ["MOK_ROOT"]) if "MOK_ROOT" in os.environ else None,
        help="Pinned Mixture-of-Kittens checkout (or set MOK_ROOT).",
    )
    parser.add_argument("--component", choices=("w2", "w13"), default="w2")
    parser.add_argument("--experts", type=int, default=DEFAULT_EXPERTS)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--intermediate-size", type=int, default=DEFAULT_INTERMEDIATE_SIZE)
    parser.add_argument("--rows-per-expert", type=int, default=DEFAULT_PADDED_ROWS_PER_EXPERT)
    parser.add_argument(
        "--expert-counts",
        type=str,
        help="Comma-separated 256-padded row counts; overrides --experts and --rows-per-expert.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--correctness-atol", type=float, default=0.2)
    parser.add_argument("--correctness-rtol", type=float, default=0.1)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--nvcc", default=os.environ.get("MOK_NVCC", "nvcc"))
    parser.add_argument("--json-output", type=Path)
    return parser


def _git_output(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _validate_checkout(mok_root: Path) -> dict[str, Any]:
    header = mok_root / "csrc" / "mok_megakernel.cuh"
    thunderkittens_root = mok_root / "third_party" / "ThunderKittens"
    kittens_header = thunderkittens_root / "include" / "kittens.cuh"
    if not header.is_file():
        raise FileNotFoundError(f"MoK megakernel header not found: {header}")
    if not kittens_header.is_file():
        raise FileNotFoundError(f"ThunderKittens submodule is missing: {kittens_header}")

    mok_commit = _git_output(mok_root, "rev-parse", "HEAD")
    thunderkittens_commit = _git_output(thunderkittens_root, "rev-parse", "HEAD")
    if mok_commit != MOK_COMMIT:
        raise ValueError(f"MoK checkout is {mok_commit}; expected pinned commit {MOK_COMMIT}")
    if thunderkittens_commit != THUNDERKITTENS_COMMIT:
        raise ValueError(
            f"ThunderKittens checkout is {thunderkittens_commit}; expected pinned commit {THUNDERKITTENS_COMMIT}"
        )
    mok_status = _git_output(mok_root, "status", "--porcelain").splitlines()
    thunderkittens_status = _git_output(thunderkittens_root, "status", "--porcelain").splitlines()
    return {
        "mok_commit": mok_commit,
        "mok_dirty": bool(mok_status),
        "mok_local_modifications": mok_status,
        "thunderkittens_commit": thunderkittens_commit,
        "thunderkittens_dirty": bool(thunderkittens_status),
        "thunderkittens_local_modifications": thunderkittens_status,
    }


def _probe_root() -> Path:
    return Path(__file__).resolve().parents[1] / "backends" / "sm100" / "mok_gmm_probe"


def _extension_path(build_dir: Path) -> Path:
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not extension_suffix:
        raise RuntimeError("Python did not report an extension suffix")
    return build_dir / f"_mok_gmm_probe{extension_suffix}"


def _build_extension(mok_root: Path, build_dir: Path, nvcc: str) -> Path:
    verify_cuda_map_fold_include(
        _probe_root() / "generated_map_fold.inc",
        shuttle_map_fold_program(),
    )
    output = _extension_path(build_dir)
    subprocess.run(
        [
            "make",
            "-C",
            str(_probe_root()),
            f"MOK_ROOT={mok_root}",
            f"NVCC={nvcc}",
            f"PYTHON={sys.executable}",
            f"OUT={output}",
        ],
        check=True,
    )
    return output


def _load_extension(path: Path) -> ModuleType:
    module_name = "_mok_gmm_probe"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _counts(args: argparse.Namespace) -> list[int]:
    if args.expert_counts:
        counts = [int(value) for value in args.expert_counts.split(",")]
    else:
        counts = [args.rows_per_expert] * args.experts
    if not counts:
        raise ValueError("at least one expert count is required")
    if any(count < 0 or count % TILE_ROWS for count in counts):
        raise ValueError(f"every expert count must be non-negative and divisible by {TILE_ROWS}: {counts}")
    if sum(counts) == 0:
        raise ValueError("at least one expert must have rows")
    return counts


def _validate_arguments(args: argparse.Namespace, counts: list[int]) -> None:
    if args.mok_root is None:
        raise ValueError("--mok-root or MOK_ROOT is required")
    if args.hidden_size <= 0 or args.hidden_size % TILE_COLUMNS:
        raise ValueError(f"hidden-size must be positive and divisible by {TILE_COLUMNS}")
    if args.intermediate_size <= 0 or args.intermediate_size % TILE_COLUMNS:
        raise ValueError(f"intermediate-size must be positive and divisible by {TILE_COLUMNS}")
    reduction_size = args.intermediate_size if args.component == "w2" else args.hidden_size
    if reduction_size % TILE_REDUCTION:
        raise ValueError(f"the component reduction dimension must be divisible by {TILE_REDUCTION}")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if args.correctness_atol < 0 or args.correctness_rtol < 0:
        raise ValueError("correctness tolerances must be non-negative")


def _counts_sha256(counts: list[int]) -> str:
    payload = b"".join(struct.pack("<i", count) for count in counts)
    return hashlib.sha256(payload).hexdigest()


def _fill_bf16(shape: tuple[int, ...], device: torch.device, standard_deviation: float = 1.0) -> torch.Tensor:
    tensor = torch.empty(shape, device=device, dtype=torch.bfloat16)
    tensor.normal_(mean=0.0, std=standard_deviation)
    return tensor


def _reference_grouped_gemm(
    activations: torch.Tensor,
    weights: torch.Tensor,
    counts: list[int],
) -> torch.Tensor:
    expected = torch.empty(
        (activations.shape[0], weights.shape[1]),
        device=activations.device,
        dtype=torch.float32,
    )
    row_start = 0
    for expert, count in enumerate(counts):
        row_end = row_start + count
        if count:
            expected[row_start:row_end] = activations[row_start:row_end].float() @ weights[expert].float().T
        row_start = row_end
    return expected


def _error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | int | bool]:
    absolute_error = (actual.float() - expected).abs()
    relative_error = absolute_error / expected.abs().clamp_min(1e-6)
    return {
        "maximum_absolute_error": float(absolute_error.max().item()),
        "mean_absolute_error": float(absolute_error.mean().item()),
        "p99_absolute_error": float(torch.quantile(absolute_error, 0.99).item()),
        "maximum_relative_error": float(relative_error.max().item()),
        "mean_relative_error": float(relative_error.mean().item()),
        "nan_count": int(torch.isnan(actual).sum().item()),
        "infinity_count": int(torch.isinf(actual).sum().item()),
    }


def _correctness_check(
    module: ModuleType,
    component: str,
    device: torch.device,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    torch.manual_seed(1234)
    counts = [TILE_ROWS, 2 * TILE_ROWS]
    rows = sum(counts)
    hidden_size = TILE_COLUMNS
    intermediate_size = TILE_COLUMNS
    reduction_size = intermediate_size if component == "w2" else hidden_size
    output_size = hidden_size if component == "w2" else intermediate_size
    activations = _fill_bf16((rows, reduction_size), device)
    padded_counts = torch.tensor(counts, dtype=torch.int32, device=device)
    weight_scale = 1.0 / math.sqrt(reduction_size)
    weight_count = 1 if component == "w2" else 2
    weights = [_fill_bf16((len(counts), output_size, reduction_size), device, weight_scale) for _ in range(weight_count)]
    outputs = [torch.empty((rows, output_size), device=device, dtype=torch.bfloat16) for _ in weights]

    results = []
    for weight, output in zip(weights, outputs, strict=True):
        module.grouped_gemm_out(activations, weight, padded_counts, output)
        expected = _reference_grouped_gemm(activations, weight, counts)
        torch.cuda.synchronize(device)
        metrics = _error_metrics(output, expected)
        metrics["passed"] = bool(torch.allclose(output.float(), expected, atol=atol, rtol=rtol))
        results.append(metrics)
    passed = all(result["passed"] for result in results)
    return {
        "passed": passed,
        "atol": atol,
        "rtol": rtol,
        "shape": {
            "experts": len(counts),
            "padded_counts": counts,
            "K": reduction_size,
            "N": output_size,
        },
        "projections": results,
    }


def _timings(function: Callable[[], None], warmup: int, iterations: int, device: torch.device) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize(device)

    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iterations)]
    for start, end in events:
        start.record()
        function()
        end.record()
    torch.cuda.synchronize(device)
    return [start.elapsed_time(end) for start, end in events]


def _benchmark(
    args: argparse.Namespace,
    module: ModuleType,
    counts: list[int],
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    rows = sum(counts)
    reduction_size = args.intermediate_size if args.component == "w2" else args.hidden_size
    output_size = args.hidden_size if args.component == "w2" else args.intermediate_size
    projection_count = 1 if args.component == "w2" else 2
    activations = _fill_bf16((rows, reduction_size), device)
    padded_counts = torch.tensor(counts, dtype=torch.int32, device=device)
    weight_scale = 1.0 / math.sqrt(reduction_size)
    weights = [
        _fill_bf16((len(counts), output_size, reduction_size), device, weight_scale) for _ in range(projection_count)
    ]
    outputs = [torch.empty((rows, output_size), device=device, dtype=torch.bfloat16) for _ in range(projection_count)]

    def run() -> None:
        for weight, output in zip(weights, outputs, strict=True):
            module.grouped_gemm_out(activations, weight, padded_counts, output)

    samples = _timings(run, args.warmup, args.iterations, device)
    median_ms = statistics.median(samples)
    physical_flops = 2 * projection_count * rows * reduction_size * output_size
    return {
        "samples_ms": samples,
        "median_ms": median_ms,
        "mean_ms": statistics.mean(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "physical_tflops": physical_flops / (median_ms * 1e9),
        "physical_flops": physical_flops,
    }


def _write_result(result: dict[str, Any], output: Path | None) -> None:
    serialized = json.dumps(result, sort_keys=True)
    print(serialized)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n")


def main() -> None:
    args = _parser().parse_args()
    counts = _counts(args)
    _validate_arguments(args, counts)
    assert args.mok_root is not None
    mok_root = args.mok_root.resolve()
    source = _validate_checkout(mok_root)
    build_dir = (
        args.build_dir.resolve()
        if args.build_dir is not None
        else Path("/tmp") / "tile_lifetime_mok_gmm_probe" / MOK_COMMIT[:12]
    )
    extension_path = _build_extension(mok_root, build_dir, args.nvcc)

    result: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "mok_sm100_grouped_gemm_primitive_probe",
        "status": "built" if args.build_only else "running",
        "scope": "primitive_probe_not_full_mok_forward",
        "source": source,
        "build": {
            "extension": str(extension_path),
            "nvcc": args.nvcc,
            "architecture": "sm100a",
        },
    }
    if args.build_only:
        _write_result(result, args.json_output)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the MoK grouped-GEMM probe")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    module = _load_extension(extension_path)
    properties = torch.cuda.get_device_properties(device)
    correctness = None
    if not args.skip_correctness:
        correctness = _correctness_check(
            module,
            args.component,
            device,
            args.correctness_atol,
            args.correctness_rtol,
        )
        if not correctness["passed"]:
            result.update({"status": "correctness_failed", "correctness": correctness})
            _write_result(result, args.json_output)
            raise RuntimeError("MoK grouped-GEMM primitive failed the Torch correctness check")

    timing = _benchmark(args, module, counts, device)
    result.update(
        {
            "status": "ok",
            "component": args.component,
            "shape": {
                "experts": len(counts),
                "total_padded_rows": sum(counts),
                "minimum_padded_rows_per_expert": min(counts),
                "maximum_padded_rows_per_expert": max(counts),
                "counts_sha256": _counts_sha256(counts),
                "hidden_size": args.hidden_size,
                "intermediate_size": args.intermediate_size,
                "projection_launches": 1 if args.component == "w2" else 2,
            },
            "configuration": {
                "warmup": args.warmup,
                "iterations": args.iterations,
                "seed": args.seed,
                "schedule": "direct_one_cluster_per_output_tile",
                "cross_task_events": "all_null",
            },
            "environment": {
                "device": str(device),
                "gpu_name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
            },
            "correctness": correctness,
            "timing": timing,
        }
    )
    _write_result(result, args.json_output)


if __name__ == "__main__":
    main()
