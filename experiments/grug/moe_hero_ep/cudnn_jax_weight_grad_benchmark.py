# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark a JAX bridge to cuDNN Frontend grouped Wgrad on hero shapes."""

import dataclasses
import importlib
import importlib.metadata
import json
import logging
import os
import site
import subprocess
import tempfile
import time

import click
import fsspec
import jax
import jax.numpy as jnp
from fray.cluster import ResourceConfig
from fray.types import ANY_REGION
from levanter.cutlass_kernel_cache import cute_launcher_factory, cutlass_call
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


def _install_cudnn_frontend() -> float:
    target = tempfile.mkdtemp(prefix="ra2a-cudnn-jax-")
    env = dict(os.environ)
    env["UV_CACHE_DIR"] = "/tmp/ra2a-cudnn-jax-uv-cache"
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
    return install_time


@cute_launcher_factory
def _build_launcher(
    modules,
    *,
    expert_count: int,
    max_active_clusters: int,
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
):
    cutlass, cute, _cjax, kernel_type, weight_mode, input_order = modules

    @cute.jit
    def launcher(stream, mat_a, mat_b, offsets, output, workspace):
        kernel = kernel_type(
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=mma_tiler_mn[0] == 256,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            accumulate_on_output=False,
            expert_cnt=expert_count,
            weight_mode=weight_mode.DENSE,
            input_order=input_order.Tensor2D,
        )
        kernel(mat_a, mat_b, output, offsets, workspace, max_active_clusters, stream, None)

    return launcher


def _grouped_wgrad(shape: WeightGradientShape, modules, lhs: jax.Array, rhs: jax.Array, offsets: jax.Array):
    cutlass, _cute, cjax, _kernel_type, _weight_mode, _input_order = modules
    cluster_shape_mn = (2, 1)
    max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1])
    launcher = _build_launcher(
        modules,
        expert_count=len(PADDED_GROUP_SIZES),
        max_active_clusters=max_active_clusters,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=cluster_shape_mn,
    )
    tensor_spec = cjax.TensorSpec
    call = cutlass_call(
        launcher,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((len(PADDED_GROUP_SIZES), shape.m, shape.n), lhs.dtype),
            jax.ShapeDtypeStruct((1,), jnp.uint8),
        ),
        input_spec=(
            tensor_spec(mode=(1, 0), divisibility=(1, 8), static=False),
            tensor_spec(divisibility=(1, 8), static=False),
            tensor_spec(static=False),
        ),
        output_spec=(
            tensor_spec(divisibility=(1, 1, 8), static=False),
            tensor_spec(static=False),
        ),
        use_static_tensors=False,
    )
    output, _workspace = call(lhs, rhs, offsets)
    return output


def _benchmark_shape(shape: WeightGradientShape, modules) -> BenchmarkRow:
    row_ids = jnp.arange(ROUTED_ROWS, dtype=jnp.int32)
    expert_ids = row_ids // PADDED_GROUP_SIZE
    rows_within_expert = row_ids % PADDED_GROUP_SIZE
    active_limits = jnp.asarray(ACTIVE_GROUP_SIZES, dtype=jnp.int32)
    active_mask = rows_within_expert < active_limits[expert_ids]
    lhs = jnp.broadcast_to(
        active_mask[:, None].astype(jnp.bfloat16) * jnp.asarray(1 / 128, dtype=jnp.bfloat16),
        (ROUTED_ROWS, shape.m),
    )
    rhs = jnp.broadcast_to(
        active_mask[:, None].astype(jnp.bfloat16) * jnp.asarray(1 / 64, dtype=jnp.bfloat16),
        (ROUTED_ROWS, shape.n),
    )
    offsets = jnp.cumsum(jnp.asarray(PADDED_GROUP_SIZES, dtype=jnp.int32))

    error = None
    compile_time = 0.0
    samples: list[float] = []
    max_abs_error = None
    try:
        function = jax.jit(lambda x, y, group_offsets: _grouped_wgrad(shape, modules, x, y, group_offsets))
        start = time.perf_counter()
        compiled = function.lower(lhs, rhs, offsets).compile()
        compile_time = time.perf_counter() - start
        for _ in range(WARMUP_RUNS):
            compiled(lhs, rhs, offsets).block_until_ready()
        for _ in range(TIMED_RUNS):
            start = time.perf_counter()
            output = compiled(lhs, rhs, offsets)
            output.block_until_ready()
            samples.append(time.perf_counter() - start)
        expected = jnp.asarray(ACTIVE_GROUP_SIZES, dtype=jnp.bfloat16) / jnp.asarray(8_192, dtype=jnp.bfloat16)
        max_abs_error = float(jnp.max(jnp.abs(output.astype(jnp.float32) - expected[:, None, None])).item())
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
    install_time = _install_cudnn_frontend()
    cutlass = importlib.import_module("cutlass")
    cute = importlib.import_module("cutlass.cute")
    cjax = importlib.import_module("cutlass.jax")
    kernel_module = importlib.import_module("cudnn.gemm.cutedsl.grouped.wgrad.moe_grouped_gemm_wgrad")
    utility_module = importlib.import_module("cudnn.gemm.cutedsl.grouped.moe_utils")
    modules = (
        cutlass,
        cute,
        cjax,
        kernel_module.MoEGroupedGemmWgradBF16Kernel,
        utility_module.MoEWeightMode,
        utility_module.WGradInputOrder,
    )
    rows = [_benchmark_shape(shape, modules) for shape in TARGET_SHAPES]
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
