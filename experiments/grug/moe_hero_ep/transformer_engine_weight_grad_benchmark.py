# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Transformer Engine V2 grouped Wgrad on the exact EP hero shapes."""

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
from pathlib import Path
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
from rigging.filesystem import prefix_join
from rigging.provenance import launch_provenance

logger = logging.getLogger(__name__)

TRANSFORMER_ENGINE_VERSION = "2.17.1"
CUDNN_FRONTEND_VERSION = "1.27.0"
CUDA_CCCL_VERSION = "13.0.85"
ROUTED_ROWS = 348_672
ACTIVE_GROUP_SIZES = (116_218, 116_217, 116_217)
PADDED_GROUP_SIZE = 116_224
PADDED_GROUP_SIZES = (PADDED_GROUP_SIZE,) * 3
WARMUP_RUNS = 2
TIMED_RUNS = 5
INSTALL_LOG_TAIL = 8_000
BENCHMARK_RESOURCES = ResourceConfig.with_gpu(
    "GB200",
    count=1,
    cpu=32,
    ram="128g",
    disk="256g",
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
    kernel: str
    implementation: str
    package_version: str
    shape: str
    dtype: str
    device_type: str
    device_count: int
    active_group_sizes: tuple[int, ...]
    padded_group_sizes: tuple[int, ...]
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
    backend_env: dict[str, str]


class BenchmarkResult(BaseModel):
    package_version: str
    install_time: float
    install_error: str | None
    install_stdout_tail: str
    install_stderr_tail: str
    platform_machine: str
    cuda_home: str | None
    cccl_include: str | None
    cuda_version: int | None
    cublas_lt_version: int | None
    grouped_gemm_workspace_size: int | None
    capability_error: str | None
    rows: list[BenchmarkRow]


class TransformerEngineWeightGradBenchmarkResult(Artifact):
    """Machine-readable Transformer Engine grouped-Wgrad benchmark result."""


@dataclasses.dataclass(frozen=True)
class TransformerEngineWeightGradBenchmarkConfig:
    output_path: str


def _pip_cuda_home() -> Path | None:
    for package_root in site.getsitepackages():
        cuda_home = Path(package_root) / "nvidia" / "cu13"
        if (cuda_home / "include" / "cuda_runtime_api.h").is_file():
            return cuda_home
    return None


def _cccl_include(package_roots: tuple[Path, ...]) -> Path | None:
    for package_root in package_roots:
        for target_header in package_root.rglob("nv/target"):
            return target_header.parent.parent
    return None


def _installed_header_path(header: str) -> Path | None:
    for package_root in site.getsitepackages():
        nvidia_root = Path(package_root) / "nvidia"
        for header_path in nvidia_root.rglob(header):
            return header_path
    return None


def _installed_nvidia_include_paths() -> tuple[Path, ...]:
    include_paths = set()
    for package_root in site.getsitepackages():
        nvidia_root = Path(package_root) / "nvidia"
        include_paths.update(path for path in nvidia_root.rglob("include") if path.is_dir())
    return tuple(sorted(include_paths))


def _installed_nvidia_library_paths() -> tuple[Path, ...]:
    library_paths = set()
    for package_root in site.getsitepackages():
        nvidia_root = Path(package_root) / "nvidia"
        library_paths.update(path for path in nvidia_root.rglob("lib") if path.is_dir())
    return tuple(sorted(library_paths))


def _install_transformer_engine() -> tuple[Any | None, float, str, str, str | None, str | None, str | None]:
    """Build the pinned JAX extension against the job's CUDA 13 JAX runtime."""
    target = tempfile.mkdtemp(prefix="ra2a-transformer-engine-")
    env = dict(os.environ)
    env["UV_CACHE_DIR"] = "/tmp/ra2a-transformer-engine-uv-cache"
    cuda_home = _pip_cuda_home()
    if cuda_home is None:
        return None, 0.0, "", "", "CUDA 13 pip toolkit headers were not found", None, None
    env["CUDA_HOME"] = str(cuda_home)
    env["PATH"] = f"{target}/bin:{cuda_home}/bin:{env['PATH']}"
    env["PYTHONPATH"] = f"{target}:{env.get('PYTHONPATH', '')}"
    env["LIBRARY_PATH"] = f"{cuda_home}/lib:{env.get('LIBRARY_PATH', '')}"

    setup = subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--target",
            target,
            "--no-deps",
            "cmake>=3.21",
            "ninja",
            "pybind11>=3",
            f"nvidia-cudnn-frontend=={CUDNN_FRONTEND_VERSION}",
            f"nvidia-cuda-cccl=={CUDA_CCCL_VERSION}",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if setup.returncode != 0:
        error = f"build tool installation exited {setup.returncode}"
        return (
            None,
            0.0,
            setup.stdout[-INSTALL_LOG_TAIL:],
            setup.stderr[-INSTALL_LOG_TAIL:],
            error,
            str(cuda_home),
            None,
        )
    cccl_include = _cccl_include((Path(target),))
    if cccl_include is None:
        error = f"nvidia-cuda-cccl=={CUDA_CCCL_VERSION} did not contain nv/target"
        return None, 0.0, setup.stdout[-INSTALL_LOG_TAIL:], setup.stderr[-INSTALL_LOG_TAIL:], error, str(cuda_home), None
    required_headers = {
        "cudnn.h": _installed_header_path("cudnn.h"),
        "nccl.h": _installed_header_path("nccl.h"),
        "nvtx3/nvToolsExt.h": _installed_header_path("nvToolsExt.h"),
    }
    missing_headers = [name for name, path in required_headers.items() if path is None]
    if missing_headers:
        error = f"staged SDK headers missing: {missing_headers}"
        return (
            None,
            0.0,
            setup.stdout[-INSTALL_LOG_TAIL:],
            setup.stderr[-INSTALL_LOG_TAIL:],
            error,
            str(cuda_home),
            str(cccl_include),
        )
    include_paths = (Path(target) / "include", cccl_include, *_installed_nvidia_include_paths())
    env["CPLUS_INCLUDE_PATH"] = f"{':'.join(str(path) for path in include_paths)}:{env.get('CPLUS_INCLUDE_PATH', '')}"
    library_paths = (cuda_home / "lib", *_installed_nvidia_library_paths())
    joined_library_paths = ":".join(str(path) for path in library_paths)
    env["LIBRARY_PATH"] = f"{joined_library_paths}:{env.get('LIBRARY_PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{joined_library_paths}:{env.get('LD_LIBRARY_PATH', '')}"

    start = time.perf_counter()
    install = subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--target",
            target,
            "--no-build-isolation",
            "--no-deps",
            f"transformer-engine-cu13=={TRANSFORMER_ENGINE_VERSION}",
            f"transformer-engine-jax=={TRANSFORMER_ENGINE_VERSION}",
            f"transformer-engine=={TRANSFORMER_ENGINE_VERSION}",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    install_time = time.perf_counter() - start
    stdout = install.stdout[-INSTALL_LOG_TAIL:]
    stderr = install.stderr[-INSTALL_LOG_TAIL:]
    if install.returncode != 0:
        error = f"Transformer Engine installation exited {install.returncode}"
        return None, install_time, stdout, stderr, error, str(cuda_home), str(cccl_include)

    site.addsitedir(target)
    try:
        transformer_engine_jax = importlib.import_module("transformer_engine_jax")
        importlib.import_module("transformer_engine.jax")
    except (ImportError, OSError, RuntimeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        return None, install_time, stdout, stderr, error, str(cuda_home), str(cccl_include)
    return transformer_engine_jax, install_time, stdout, stderr, None, str(cuda_home), str(cccl_include)


def _benchmark_shape(shape: WeightGradientShape) -> BenchmarkRow:
    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    tex = importlib.import_module("transformer_engine.jax.cpp_extensions")
    quantize = importlib.import_module("transformer_engine.jax.quantize")

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
    group_sizes = jnp.asarray(PADDED_GROUP_SIZES, dtype=jnp.int32)

    def grouped_wgrad(lhs, rhs, group_sizes):
        casted_lhs = tex.grouped_quantize(
            lhs,
            quantize.noop_quantizer_set.x,
            group_sizes,
            flatten_axis=-1,
        )
        casted_rhs = tex.grouped_quantize(
            rhs,
            quantize.noop_quantizer_set.dgrad,
            group_sizes,
            flatten_axis=-1,
        )
        return tex.grouped_gemm(
            casted_lhs.get_tensor(usage=quantize.TensorUsage.LHS_TRANS),
            casted_rhs.get_tensor(usage=quantize.TensorUsage.RHS),
            contracting_dims=((0,), (0,)),
        )

    error = None
    compile_time = 0.0
    samples: list[float] = []
    max_abs_error = None
    try:
        start = time.perf_counter()
        compiled = jax.jit(grouped_wgrad).lower(lhs, rhs, group_sizes).compile()
        compile_time = time.perf_counter() - start
        for _ in range(WARMUP_RUNS):
            compiled(lhs, rhs, group_sizes).block_until_ready()
        for _ in range(TIMED_RUNS):
            start = time.perf_counter()
            output = compiled(lhs, rhs, group_sizes)
            output.block_until_ready()
            samples.append(time.perf_counter() - start)

        expected = jnp.asarray(ACTIVE_GROUP_SIZES, dtype=jnp.bfloat16) / jnp.asarray(
            8_192,
            dtype=jnp.bfloat16,
        )
        max_abs_error = float(jnp.max(jnp.abs(output.astype(jnp.float32) - expected[:, None, None])).item())
    except (AssertionError, RuntimeError, TypeError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"

    median_time = sorted(samples)[len(samples) // 2] if samples else 0.0
    mean_time = sum(samples) / len(samples) if samples else 0.0
    logical_flops = 2 * sum(ACTIVE_GROUP_SIZES) * shape.m * shape.n
    tflops = logical_flops / median_time / 1e12 if median_time else 0.0
    row = BenchmarkRow(
        kernel="grouped_wgrad",
        implementation="transformer_engine_jax_v2_bf16",
        package_version=TRANSFORMER_ENGINE_VERSION,
        shape=f"{shape.name}:({ROUTED_ROWS},{shape.m})T@({ROUTED_ROWS},{shape.n})->(3,{shape.m},{shape.n})",
        dtype="bfloat16",
        device_type=jax.devices()[0].device_kind,
        device_count=jax.device_count(),
        active_group_sizes=ACTIVE_GROUP_SIZES,
        padded_group_sizes=PADDED_GROUP_SIZES,
        compile_time=compile_time,
        samples=samples,
        median_time=median_time,
        mean_time=mean_time,
        tflops=tflops,
        max_abs_error=max_abs_error,
        error=error,
        git_sha=launch_provenance().tree_hash,
        jax_version=importlib.metadata.version("jax"),
        jaxlib_version=importlib.metadata.version("jaxlib"),
        backend_env={
            "NVTE_JAX_ENFORCE_V2_GROUPED_GEMM": os.environ.get(
                "NVTE_JAX_ENFORCE_V2_GROUPED_GEMM",
                "",
            ),
            "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
        },
    )
    logger.info("transformer_engine_weight_grad_benchmark %s", row.model_dump_json())
    return row


def run_benchmark(config: TransformerEngineWeightGradBenchmarkConfig) -> None:
    te_jax, install_time, install_stdout, install_stderr, install_error, cuda_home, cccl_include = (
        _install_transformer_engine()
    )
    cuda_version = None
    cublas_lt_version = None
    grouped_gemm_workspace_size = None
    capability_error = None
    rows: list[BenchmarkRow] = []
    if te_jax is not None:
        try:
            cuda_version = int(te_jax.get_cuda_version())
            cublas_lt_version = int(te_jax.get_cublasLt_version())
            grouped_gemm_workspace_size = int(te_jax.get_grouped_gemm_setup_workspace_size(3))
        except RuntimeError as exc:
            capability_error = f"{type(exc).__name__}: {exc}"
        rows = [_benchmark_shape(shape) for shape in TARGET_SHAPES]

    result = BenchmarkResult(
        package_version=TRANSFORMER_ENGINE_VERSION,
        install_time=install_time,
        install_error=install_error,
        install_stdout_tail=install_stdout,
        install_stderr_tail=install_stderr,
        platform_machine=os.uname().machine,
        cuda_home=cuda_home,
        cccl_include=cccl_include,
        cuda_version=cuda_version,
        cublas_lt_version=cublas_lt_version,
        grouped_gemm_workspace_size=grouped_gemm_workspace_size,
        capability_error=capability_error,
        rows=rows,
    )
    logger.info("transformer_engine_weight_grad_result %s", result.model_dump_json())
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fs.open(prefix_join(config.output_path, "results.json"), "w") as handle:
        json.dump(result.model_dump(mode="json"), handle, indent=2)


def build_benchmark(
    *,
    version: str | None = None,
) -> ArtifactStep[TransformerEngineWeightGradBenchmarkResult]:
    name = "benchmarks/transformer-engine-ragged-weight-grad-gb200"
    version = resolve_version(name, version)

    def build_config(ctx: StepContext) -> TransformerEngineWeightGradBenchmarkConfig:
        return TransformerEngineWeightGradBenchmarkConfig(output_path=ctx.output_path)

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=TransformerEngineWeightGradBenchmarkResult,
        run=remote(
            run_benchmark,
            name="transformer-engine-ragged-weight-grad-gb200",
            resources=BENCHMARK_RESOURCES,
            env_vars={"NVTE_JAX_ENFORCE_V2_GROUPED_GEMM": "1"},
        ),
        build_config=build_config,
    )


@click.command()
@build_options
def main() -> ArtifactStep[TransformerEngineWeightGradBenchmarkResult]:
    return build_benchmark()


if __name__ == "__main__":
    main()
