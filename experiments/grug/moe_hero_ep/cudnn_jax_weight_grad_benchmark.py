# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark a JAX bridge to cuDNN Frontend grouped Wgrad on hero shapes."""

import dataclasses
import importlib
import importlib.metadata
import json
import logging
import time

import click
import fsspec
import jax
import jax.numpy as jnp
from fray.cluster import ResourceConfig
from fray.types import ANY_REGION
from levanter.grug._moe.cudnn_wgrad_cute import cudnn_grouped_wgrad
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name
from pydantic import BaseModel
from rigging.filesystem import prefix_join
from rigging.provenance import launch_provenance

logger = logging.getLogger(__name__)

CUDNN_FRONTEND_VERSION = "1.27.0"
ROUTED_ROWS = 348_672
ACTIVE_GROUP_SIZES = (116_218, 116_217, 116_217)
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


TARGET_SHAPES = (
    WeightGradientShape("dw13", m=3_072, n=12_544),
    WeightGradientShape("dw2", m=6_272, n=3_072),
)


class BenchmarkRow(BaseModel):
    shape: str
    package_version: str
    device_type: str
    compile_time: float
    samples: list[float]
    median_time: float
    mean_time: float
    tflops: float
    max_abs_error: float | None
    error: str | None
    git_sha: str
    jax_version: str
    jaxlib_version: str


class CudnnJaxWeightGradBenchmarkResult(Artifact):
    """Machine-readable JAX cuDNN grouped-Wgrad benchmark rows."""


@dataclasses.dataclass(frozen=True)
class CudnnJaxWeightGradBenchmarkConfig:
    output_path: str


def _load_cudnn_frontend() -> float:
    start = time.perf_counter()
    importlib.import_module("cudnn")
    package_version = importlib.metadata.version("nvidia-cudnn-frontend")
    if package_version != CUDNN_FRONTEND_VERSION:
        raise RuntimeError(f"expected cuDNN Frontend {CUDNN_FRONTEND_VERSION}, found {package_version}")
    return time.perf_counter() - start


def _benchmark_shape(shape: WeightGradientShape) -> BenchmarkRow:
    row_ids = jnp.arange(ROUTED_ROWS, dtype=jnp.int32)
    group_sizes = jnp.asarray(ACTIVE_GROUP_SIZES, dtype=jnp.int32)
    active_offsets = jnp.cumsum(group_sizes)
    expert_ids = jnp.sum(row_ids[:, None] >= active_offsets[None, :], axis=1, dtype=jnp.int32)
    safe_expert_ids = jnp.minimum(expert_ids, len(ACTIVE_GROUP_SIZES) - 1)
    active_mask = expert_ids < len(ACTIVE_GROUP_SIZES)
    expert_factors = (safe_expert_ids + 1).astype(jnp.bfloat16)
    lhs_values = (jnp.arange(shape.m, dtype=jnp.int32) % 17 + 1).astype(jnp.bfloat16) / 1_024
    rhs_values = (jnp.arange(shape.n, dtype=jnp.int32) % 19 + 1).astype(jnp.bfloat16) / 512
    lhs = active_mask[:, None].astype(jnp.bfloat16) * expert_factors[:, None] * lhs_values[None, :]
    rhs = active_mask[:, None].astype(jnp.bfloat16) * rhs_values[None, :]

    error = None
    compile_time = 0.0
    samples: list[float] = []
    max_abs_error = None
    try:
        function = jax.jit(cudnn_grouped_wgrad)
        start = time.perf_counter()
        compiled = function.lower(lhs, rhs, group_sizes).compile()
        compile_time = time.perf_counter() - start
        for _ in range(WARMUP_RUNS):
            compiled(lhs, rhs, group_sizes).block_until_ready()
        for _ in range(TIMED_RUNS):
            start = time.perf_counter()
            output = compiled(lhs, rhs, group_sizes)
            output.block_until_ready()
            samples.append(time.perf_counter() - start)
        expected = (
            group_sizes[:, None, None].astype(jnp.float32)
            * jnp.arange(1, len(ACTIVE_GROUP_SIZES) + 1, dtype=jnp.float32)[:, None, None]
            * lhs_values[None, :, None].astype(jnp.float32)
            * rhs_values[None, None, :].astype(jnp.float32)
        ).astype(jnp.bfloat16)
        max_abs_error = float(jnp.max(jnp.abs(output.astype(jnp.float32) - expected.astype(jnp.float32))).item())
    except (AssertionError, RuntimeError, TypeError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"

    median_time = sorted(samples)[len(samples) // 2] if samples else 0.0
    mean_time = sum(samples) / len(samples) if samples else 0.0
    logical_flops = 2 * sum(ACTIVE_GROUP_SIZES) * shape.m * shape.n
    return BenchmarkRow(
        shape=f"{shape.name}:({ROUTED_ROWS},{shape.m})T@({ROUTED_ROWS},{shape.n})->(3,{shape.m},{shape.n})",
        package_version=CUDNN_FRONTEND_VERSION,
        device_type=jax.devices()[0].device_kind,
        compile_time=compile_time,
        samples=samples,
        median_time=median_time,
        mean_time=mean_time,
        tflops=logical_flops / median_time / 1e12 if median_time else 0.0,
        max_abs_error=max_abs_error,
        error=error,
        git_sha=launch_provenance().tree_hash,
        jax_version=importlib.metadata.version("jax"),
        jaxlib_version=importlib.metadata.version("jaxlib"),
    )


def run_benchmark(config: CudnnJaxWeightGradBenchmarkConfig) -> None:
    install_time = _load_cudnn_frontend()
    rows = [_benchmark_shape(shape) for shape in TARGET_SHAPES]
    logger.info(
        "cudnn_jax_weight_grad_result install_time=%s rows=%s",
        install_time,
        json.dumps([row.model_dump(mode="json") for row in rows]),
    )
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fs.open(prefix_join(config.output_path, "results.json"), "w") as handle:
        json.dump(
            {"install_time": install_time, "rows": [row.model_dump(mode="json") for row in rows]},
            handle,
            indent=2,
        )


def build_benchmark(*, version: str | None = None) -> ArtifactStep[CudnnJaxWeightGradBenchmarkResult]:
    name = "benchmarks/cudnn-jax-ragged-weight-grad-gb200"
    version = resolve_version(name, version)

    def build_config(ctx: StepContext) -> CudnnJaxWeightGradBenchmarkConfig:
        return CudnnJaxWeightGradBenchmarkConfig(output_path=ctx.output_path)

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=CudnnJaxWeightGradBenchmarkResult,
        run=remote(
            run_benchmark,
            name="cudnn-jax-ragged-weight-grad-gb200",
            resources=BENCHMARK_RESOURCES,
        ),
        build_config=build_config,
    )


@click.command()
@build_options
def main() -> ArtifactStep[CudnnJaxWeightGradBenchmarkResult]:
    return build_benchmark()


if __name__ == "__main__":
    main()
