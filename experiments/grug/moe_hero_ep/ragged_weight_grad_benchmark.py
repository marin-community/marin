# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark exact EP hero ragged weight-gradient tiles on one GB200 GPU."""

import dataclasses
import json
import logging
import os
import time
from functools import partial

import click
import fsspec
import jax
import jax.numpy as jnp
from fray.cluster import ResourceConfig
from fray.types import ANY_REGION
from haliax.nn.ragged_dot import _triton_ragged_contracting_dim_dot_kernel
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
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

ROUTED_ROWS = 348_672
ACTIVE_ROUTED_ROWS = 348_652
NUM_LOCAL_EXPERTS = 3
WARMUP_RUNS = 2
TIMED_RUNS = 5
BENCHMARK_RESOURCES = ResourceConfig.with_gpu(
    "GB200",
    count=1,
    cpu=8,
    ram="64g",
    disk="128g",
    regions=[ANY_REGION],
)


@dataclasses.dataclass(frozen=True, slots=True)
class WeightGradientShape:
    name: str
    m: int
    n: int


@dataclasses.dataclass(frozen=True, slots=True)
class TileConfig:
    block_m: int
    block_n: int
    block_k: int = 32
    num_warps: int = 4
    num_stages: int = 4


TARGET_SHAPES = (
    WeightGradientShape("dw13", m=3_072, n=12_544),
    WeightGradientShape("dw2", m=6_272, n=3_072),
)
TILE_CONFIGS = (
    TileConfig(128, 128, num_stages=4),
    TileConfig(128, 128, num_stages=2),
    TileConfig(64, 128, num_stages=4),
    TileConfig(64, 128, num_stages=2),
    TileConfig(128, 64, num_stages=4),
    TileConfig(128, 64, num_stages=2),
    TileConfig(64, 64, num_stages=4),
    TileConfig(64, 64, num_stages=2),
    TileConfig(128, 128, num_warps=8, num_stages=4),
    TileConfig(128, 128, block_k=16, num_stages=4),
    TileConfig(128, 128, block_k=64, num_stages=4),
    TileConfig(128, 128, block_k=64, num_warps=8, num_stages=4),
    TileConfig(256, 128, num_warps=8, num_stages=4),
    TileConfig(128, 256, num_warps=8, num_stages=4),
    TileConfig(256, 64, num_warps=8, num_stages=4),
    TileConfig(64, 256, num_warps=8, num_stages=4),
)


class BenchmarkRow(BaseModel):
    kernel: str
    implementation: str
    shape: str
    dtype: str
    backend: str
    device_type: str
    device_count: int
    block_sizes: dict[str, int]
    compile_time: float
    steady_state_time: float
    tflops: float
    max_abs_error: float | None
    error: str | None
    git_sha: str
    xla_flags: str
    backend_env: dict[str, str]


class RaggedWeightGradBenchmarkResult(Artifact):
    """Machine-readable single-GPU weight-gradient benchmark rows."""


@dataclasses.dataclass(frozen=True)
class RaggedWeightGradBenchmarkConfig:
    output_path: str


@partial(jax.jit, static_argnames=("rows", "cols", "offset"))
def _benchmark_input(*, rows: int, cols: int, offset: int) -> jax.Array:
    row_ids = jnp.arange(rows, dtype=jnp.int32)[:, None]
    col_ids = jnp.arange(cols, dtype=jnp.int32)[None, :]
    values = (row_ids * 17 + col_ids * 13 + offset) % 251
    return ((values.astype(jnp.float32) - 125.0) / 256.0).astype(jnp.bfloat16)


def _cost_estimate(shape: WeightGradientShape) -> pl.CostEstimate:
    rows_per_group = (ACTIVE_ROUTED_ROWS + NUM_LOCAL_EXPERTS - 1) // NUM_LOCAL_EXPERTS
    input_bytes = rows_per_group * (shape.m + shape.n) * jnp.dtype(jnp.bfloat16).itemsize
    output_bytes = shape.m * shape.n * jnp.dtype(jnp.bfloat16).itemsize
    return pl.CostEstimate(
        flops=2 * rows_per_group * shape.m * shape.n,
        transcendentals=0,
        bytes_accessed=input_bytes + output_bytes,
    )


def _matmul(shape: WeightGradientShape, tile: TileConfig):
    one_group = pl.pallas_call(
        lambda a, b, lo, hi, out: _triton_ragged_contracting_dim_dot_kernel(
            a,
            b,
            lo,
            hi,
            out,
            block_m=tile.block_m,
            block_k=tile.block_k,
        ),
        out_shape=jax.ShapeDtypeStruct((shape.m, shape.n), jnp.bfloat16),
        in_specs=[
            pl.BlockSpec((ROUTED_ROWS, tile.block_m), lambda i, j: (0, i)),
            pl.BlockSpec((ROUTED_ROWS, tile.block_n), lambda i, j: (0, j)),
            pl.no_block_spec,
            pl.no_block_spec,
        ],
        out_specs=pl.BlockSpec((tile.block_m, tile.block_n), lambda i, j: (i, j)),
        grid=(pl.cdiv(shape.m, tile.block_m), pl.cdiv(shape.n, tile.block_n)),
        compiler_params=plgpu.CompilerParams(num_warps=tile.num_warps, num_stages=tile.num_stages),
        cost_estimate=_cost_estimate(shape),
    )
    return jax.jit(jax.vmap(one_group, in_axes=(None, None, 0, 0)))


def _group_bounds() -> tuple[jax.Array, jax.Array]:
    base, remainder = divmod(ACTIVE_ROUTED_ROWS, NUM_LOCAL_EXPERTS)
    group_sizes = jnp.asarray(
        [base + (index < remainder) for index in range(NUM_LOCAL_EXPERTS)],
        dtype=jnp.int32,
    )
    cumulative_rows = jnp.cumulative_sum(group_sizes, include_initial=True)
    return cumulative_rows[:-1], cumulative_rows[1:]


def _benchmark_shape(shape: WeightGradientShape) -> list[BenchmarkRow]:
    lhs = _benchmark_input(rows=ROUTED_ROWS, cols=shape.m, offset=1)
    rhs = _benchmark_input(rows=ROUTED_ROWS, cols=shape.n, offset=2)
    lhs.block_until_ready()
    rhs.block_until_ready()
    lo, hi = _group_bounds()
    rows: list[BenchmarkRow] = []
    baseline: jax.Array | None = None
    device_type = jax.devices()[0].device_kind
    git_sha = launch_provenance().tree_hash
    xla_flags = os.environ.get("XLA_FLAGS", "")
    backend_env = {"JAX_ENABLE_PGLE": os.environ.get("JAX_ENABLE_PGLE", "")}

    for tile_index, tile in enumerate(TILE_CONFIGS):
        block_sizes = dataclasses.asdict(tile)
        compile_time = 0.0
        steady_state_time = 0.0
        max_abs_error = None
        error = None
        try:
            matmul = _matmul(shape, tile)
            start = time.perf_counter()
            output = matmul(lhs, rhs, lo, hi)
            output.block_until_ready()
            compile_time = time.perf_counter() - start
            if baseline is None:
                baseline = output
            max_abs_error = float(jnp.max(jnp.abs(output.astype(jnp.float32) - baseline.astype(jnp.float32))))
            for _ in range(WARMUP_RUNS):
                matmul(lhs, rhs, lo, hi).block_until_ready()
            start = time.perf_counter()
            for _ in range(TIMED_RUNS):
                matmul(lhs, rhs, lo, hi).block_until_ready()
            steady_state_time = (time.perf_counter() - start) / TIMED_RUNS
        except (jax.errors.JaxRuntimeError, RuntimeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            if tile_index == 0:
                raise RuntimeError(f"Production ragged weight-gradient tile failed for {shape.name}") from exc

        tflops = 0.0
        if steady_state_time > 0.0:
            tflops = 2 * ACTIVE_ROUTED_ROWS * shape.m * shape.n / steady_state_time / 1e12
        row = BenchmarkRow(
            kernel="ragged_contracting_dim_dot",
            implementation="pallas_triton",
            shape=f"{shape.name}:({ROUTED_ROWS},{shape.m})x({ROUTED_ROWS},{shape.n})->(3,{shape.m},{shape.n})",
            dtype="bfloat16",
            backend=jax.default_backend(),
            device_type=device_type,
            device_count=1,
            block_sizes=block_sizes,
            compile_time=compile_time,
            steady_state_time=steady_state_time,
            tflops=tflops,
            max_abs_error=max_abs_error,
            error=error,
            git_sha=git_sha,
            xla_flags=xla_flags,
            backend_env=backend_env,
        )
        logger.info("ragged_weight_grad_benchmark %s", row.model_dump_json())
        rows.append(row)

    return rows


def run_benchmark(config: RaggedWeightGradBenchmarkConfig) -> None:
    rows = [row for shape in TARGET_SHAPES for row in _benchmark_shape(shape)]
    if all(row.error is not None for row in rows):
        raise RuntimeError("Every ragged weight-gradient tile failed")
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fs.open(prefix_join(config.output_path, "results.json"), "w") as handle:
        json.dump([row.model_dump(mode="json") for row in rows], handle, indent=2)


def build_benchmark(*, version: str | None = None) -> ArtifactStep[RaggedWeightGradBenchmarkResult]:
    name = "benchmarks/ragged-weight-grad-gb200"
    version = resolve_version(name, version)

    def build_config(ctx: StepContext) -> RaggedWeightGradBenchmarkConfig:
        return RaggedWeightGradBenchmarkConfig(output_path=ctx.output_path)

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=RaggedWeightGradBenchmarkResult,
        run=remote(
            run_benchmark,
            name="ragged-weight-grad-gb200",
            resources=BENCHMARK_RESOURCES,
            env_vars={
                "JAX_ENABLE_PGLE": "false",
                "XLA_FLAGS": "--xla_gpu_enable_command_buffer=",
            },
        ),
        build_config=build_config,
    )


@click.command()
@build_options
def main() -> ArtifactStep[RaggedWeightGradBenchmarkResult]:
    return build_benchmark()


if __name__ == "__main__":
    main()
