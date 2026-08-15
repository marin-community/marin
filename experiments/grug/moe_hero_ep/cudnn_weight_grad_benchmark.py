# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark cuDNN Frontend grouped Wgrad on the exact EP hero shapes."""

import dataclasses
import importlib
import json
import logging
import os
import site
import subprocess
import tempfile
import time
from typing import Any

import click
import fsspec
from fray.cluster import ResourceConfig
from fray.types import ANY_REGION
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name
from pydantic import BaseModel
from rigging.filesystem.storage_path import prefix_join
from rigging.provenance import launch_provenance

logger = logging.getLogger(__name__)

CUDNN_FRONTEND_VERSION = "1.27.0"
ROUTED_ROWS = 348_672
ACTIVE_GROUP_SIZES = (116_218, 116_217, 116_217)
PADDED_GROUP_SIZE = 116_224
PADDED_GROUP_SIZES = (PADDED_GROUP_SIZE,) * 3
WARMUP_RUNS = 2
TIMED_RUNS = 5
BENCHMARK_RESOURCES = ResourceConfig.with_gpu(
    "GB200",
    count=1,
    cpu=16,
    ram="128g",
    disk="128g",
    regions=[ANY_REGION],
)


@dataclasses.dataclass(frozen=True, slots=True)
class WeightGradientShape:
    name: str
    m: int
    n: int


@dataclasses.dataclass(frozen=True, slots=True)
class KernelConfig:
    mma_m: int
    mma_n: int
    cluster_m: int
    cluster_n: int


TARGET_SHAPES = (
    WeightGradientShape("dw13", m=3_072, n=12_544),
    WeightGradientShape("dw2", m=6_272, n=3_072),
)
KERNEL_CONFIGS = (
    KernelConfig(mma_m=256, mma_n=256, cluster_m=2, cluster_n=1),
    KernelConfig(mma_m=128, mma_n=128, cluster_m=1, cluster_n=1),
)


class BenchmarkRow(BaseModel):
    kernel: str
    implementation: str
    package_version: str
    shape: str
    dtype: str
    device_type: str
    device_count: int
    active_group_sizes: tuple[int, ...]
    padded_group_sizes: tuple[int, ...]
    kernel_config: dict[str, int]
    install_time: float
    compile_time: float
    samples: list[float]
    median_time: float
    mean_time: float
    tflops: float
    max_abs_error: float | None
    mean_abs_error: float | None
    error: str | None
    git_sha: str
    torch_version: str
    torch_cuda_version: str
    platform_machine: str
    backend_env: dict[str, str]


class CudnnWeightGradBenchmarkResult(Artifact):
    """Machine-readable cuDNN grouped-Wgrad benchmark rows."""


@dataclasses.dataclass(frozen=True)
class CudnnWeightGradBenchmarkConfig:
    output_path: str


def _install_cudnn_frontend() -> tuple[Any, Any, float]:
    """Install the pinned experimental frontend without changing the locked environment."""
    target = tempfile.mkdtemp(prefix="ra2a-cudnn-frontend-")
    env = dict(os.environ)
    env["UV_CACHE_DIR"] = "/tmp/ra2a-cudnn-frontend-uv-cache"
    start = time.perf_counter()
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--target",
            target,
            "--no-deps",
            f"nvidia-cudnn-frontend=={CUDNN_FRONTEND_VERSION}",
        ],
        check=True,
        env=env,
    )
    install_time = time.perf_counter() - start
    site.addsitedir(target)

    # Torch is part of Marin's GPU environment. cuDNN Frontend stays experiment-local
    # until this benchmark establishes that its kernel is worth a JAX FFI adapter.
    torch = importlib.import_module("torch")
    cudnn = importlib.import_module("cudnn")
    return torch, cudnn, install_time


def _problem(torch: Any, shape: WeightGradientShape) -> tuple[Any, Any, Any, Any]:
    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(20260810 + shape.m + shape.n)

    # Allocate the transpose view used by the production contraction: lhs.T @ rhs.
    # Per-expert padding consumes the existing 20-row physical tail exactly.
    a_storage = torch.empty((ROUTED_ROWS, shape.m), dtype=torch.bfloat16, device=device)
    a_storage.uniform_(-0.03125, 0.03125, generator=generator)
    a = a_storage.t()
    b = torch.empty((ROUTED_ROWS, shape.n), dtype=torch.bfloat16, device=device)
    b.uniform_(-0.03125, 0.03125, generator=generator)

    logical_start = 0
    padded_start = 0
    for active_size, padded_size in zip(ACTIVE_GROUP_SIZES, PADDED_GROUP_SIZES, strict=True):
        padding_start = padded_start + active_size
        padding_end = padded_start + padded_size
        a_storage[padding_start:padding_end].zero_()
        b[padding_start:padding_end].zero_()
        logical_start += active_size
        padded_start = padding_end
    assert logical_start == sum(ACTIVE_GROUP_SIZES)
    assert padded_start == ROUTED_ROWS

    offsets = torch.tensor(
        [sum(PADDED_GROUP_SIZES[: index + 1]) for index in range(len(PADDED_GROUP_SIZES))],
        dtype=torch.int32,
        device=device,
    )
    output = torch.empty(
        (len(PADDED_GROUP_SIZES), shape.m, shape.n),
        dtype=torch.bfloat16,
        device=device,
    )
    return a, b, offsets, output


def _reference(torch: Any, a: Any, b: Any, output: Any) -> Any:
    expected = torch.empty_like(output)
    begin = 0
    for expert, group_size in enumerate(PADDED_GROUP_SIZES):
        end = begin + group_size
        torch.matmul(a[:, begin:end], b[begin:end], out=expected[expert])
        begin = end
    return expected


def _timed_samples(torch: Any, function) -> list[float]:
    for _ in range(WARMUP_RUNS):
        function()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(TIMED_RUNS)]
    for start, end in events:
        start.record()
        function()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) / 1_000.0 for start, end in events]


def _benchmark_config(
    torch: Any,
    cudnn: Any,
    shape: WeightGradientShape,
    kernel_config: KernelConfig,
    install_time: float,
) -> BenchmarkRow:
    a, b, offsets, output = _problem(torch, shape)
    error = None
    compile_time = 0.0
    samples: list[float] = []
    max_abs_error = None
    mean_abs_error = None
    try:
        op = cudnn.GroupedGemmWgradSm100(
            sample_a=a,
            sample_b=b,
            sample_sfa=None,
            sample_sfb=None,
            sample_offsets=offsets,
            sample_wgrad=output,
            acc_dtype=torch.float32,
            mma_tiler_mn=(kernel_config.mma_m, kernel_config.mma_n),
            cluster_shape_mn=(kernel_config.cluster_m, kernel_config.cluster_n),
            input_order="tensor2d",
        )
        op.check_support()
        start = time.perf_counter()
        op.compile()
        torch.cuda.synchronize()
        compile_time = time.perf_counter() - start

        def run() -> None:
            op.execute(
                a_tensor=a,
                b_tensor=b,
                sfa_tensor=None,
                sfb_tensor=None,
                offsets_tensor=offsets,
                wgrad_tensor=output,
            )

        samples = _timed_samples(torch, run)
        expected = _reference(torch, a, b, output)
        difference = (output.float() - expected.float()).abs()
        max_abs_error = float(difference.max().item())
        mean_abs_error = float(difference.mean().item())
    except (RuntimeError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"

    median_time = sorted(samples)[len(samples) // 2] if samples else 0.0
    mean_time = sum(samples) / len(samples) if samples else 0.0
    logical_flops = 2 * sum(ACTIVE_GROUP_SIZES) * shape.m * shape.n
    tflops = logical_flops / median_time / 1e12 if median_time else 0.0
    row = BenchmarkRow(
        kernel="grouped_wgrad",
        implementation="cudnn_frontend_bf16_sm100",
        package_version=CUDNN_FRONTEND_VERSION,
        shape=f"{shape.name}:({ROUTED_ROWS},{shape.m})T@({ROUTED_ROWS},{shape.n})->(3,{shape.m},{shape.n})",
        dtype="bfloat16",
        device_type=torch.cuda.get_device_name(0),
        device_count=torch.cuda.device_count(),
        active_group_sizes=ACTIVE_GROUP_SIZES,
        padded_group_sizes=PADDED_GROUP_SIZES,
        kernel_config=dataclasses.asdict(kernel_config),
        install_time=install_time,
        compile_time=compile_time,
        samples=samples,
        median_time=median_time,
        mean_time=mean_time,
        tflops=tflops,
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        error=error,
        git_sha=launch_provenance().tree_hash,
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,
        platform_machine=os.uname().machine,
        backend_env={
            "CUDNNFE_CLUSTER_OVERLAP_MARGIN": os.environ.get("CUDNNFE_CLUSTER_OVERLAP_MARGIN", ""),
        },
    )
    logger.info("cudnn_weight_grad_benchmark %s", row.model_dump_json())
    return row


def run_benchmark(config: CudnnWeightGradBenchmarkConfig) -> None:
    torch, cudnn, install_time = _install_cudnn_frontend()
    rows = [
        _benchmark_config(torch, cudnn, shape, kernel_config, install_time)
        for shape in TARGET_SHAPES
        for kernel_config in KERNEL_CONFIGS
    ]
    if all(row.error is not None for row in rows):
        raise RuntimeError("Every cuDNN Frontend grouped-Wgrad configuration failed")
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fs.open(prefix_join(config.output_path, "results.json"), "w") as handle:
        json.dump([row.model_dump(mode="json") for row in rows], handle, indent=2)


def build_benchmark(*, version: str | None = None) -> ArtifactStep[CudnnWeightGradBenchmarkResult]:
    name = "benchmarks/cudnn-ragged-weight-grad-gb200"
    version = resolve_version(name, version)

    def build_config(ctx: StepContext) -> CudnnWeightGradBenchmarkConfig:
        return CudnnWeightGradBenchmarkConfig(output_path=ctx.output_path)

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=CudnnWeightGradBenchmarkResult,
        run=remote(
            run_benchmark,
            name="cudnn-ragged-weight-grad-gb200",
            resources=BENCHMARK_RESOURCES,
        ),
        build_config=build_config,
    )


@click.command()
@build_options
def main() -> ArtifactStep[CudnnWeightGradBenchmarkResult]:
    return build_benchmark()


if __name__ == "__main__":
    main()
