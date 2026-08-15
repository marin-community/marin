# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build and run an exact-local-shape Mixture-of-Kittens oracle on four GB200s."""

import dataclasses
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

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

MOK_REPOSITORY = "https://github.com/cursor/mixture-of-kittens.git"
MOK_COMMIT = "6438bf48f88094d305972fbe0fa6deba0f7d4d1a"
TORCH_VERSION = "2.10.0+cu130"
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
CUDA_NVCC_VERSION = "13.0.88"
CUDA_CCCL_VERSION = "13.0.85"
LOG_TAIL = 12_000
BENCHMARK_RESOURCES = ResourceConfig.with_gpu(
    "GB200",
    count=4,
    cpu=32,
    ram="256g",
    disk="256g",
    regions=[ANY_REGION],
)


class ScheduleResult(BaseModel):
    name: str
    forward_communication_sms: int
    backward_communication_sms: int
    minibatch_size: int
    macrobatch_size: int
    forward_samples_ms: list[float]
    backward_samples_ms: list[float]
    combined_samples_ms: list[float]
    median_forward_ms: float
    median_backward_ms: float
    median_combined_ms: float


class HeroShape(BaseModel):
    num_local_tokens: int
    hidden_size: int
    logical_intermediate_size: int
    physical_intermediate_size: int
    top_k: int
    num_local_experts: int
    num_global_experts: int
    includes_shared_expert: bool


class WorkerEnvironment(BaseModel):
    torch_version: str
    torch_cuda_version: str
    device_name: str
    world_size: int
    processes_per_gpu: int


class WorkerResult(BaseModel):
    environment: WorkerEnvironment
    shape: HeroShape
    finite_on_all_ranks: bool
    repeat_deterministic_on_all_ranks: bool
    configs: list[ScheduleResult]


class MokHeroResult(BaseModel):
    source_commit: str
    torch_version: str
    git_sha: str
    install_time: float
    build_time: float
    smoke_time: float
    benchmark_time: float
    install_stdout_tail: str
    install_stderr_tail: str
    build_stdout_tail: str
    build_stderr_tail: str
    smoke_stdout_tail: str
    smoke_stderr_tail: str
    benchmark_stdout_tail: str
    benchmark_stderr_tail: str
    error_stage: str | None
    error: str | None
    error_elapsed: float
    error_stdout_tail: str
    error_stderr_tail: str
    worker_result: WorkerResult | None


class MokHeroBenchmarkArtifact(Artifact):
    """Machine-readable process-per-GPU MoK hero-layer benchmark."""


@dataclasses.dataclass(frozen=True)
class MokHeroBenchmarkConfig:
    output_path: str


@dataclasses.dataclass(frozen=True, slots=True)
class StageOutput:
    elapsed: float
    stdout: str
    stderr: str


class BenchmarkStageError(RuntimeError):
    def __init__(self, stage: str, output: StageOutput, returncode: int):
        super().__init__(f"{stage} exited {returncode}")
        self.stage = stage
        self.output = output


def _run_stage(
    stage: str,
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> StageOutput:
    logger.info("Starting MoK stage %s", stage)
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    output = StageOutput(
        elapsed=time.perf_counter() - start,
        stdout=completed.stdout[-LOG_TAIL:],
        stderr=completed.stderr[-LOG_TAIL:],
    )
    if completed.returncode != 0:
        raise BenchmarkStageError(stage, output, completed.returncode)
    logger.info("Completed MoK stage %s in %.3f seconds", stage, output.elapsed)
    return output


def _cuda_environment(venv: Path, source: Path) -> dict[str, str]:
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = venv / "lib" / python_version / "site-packages"
    cuda_home = site_packages / "nvidia" / "cu13"
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"CUDA 13 NVCC was not installed at {nvcc}")

    torch_lib = site_packages / "torch" / "lib"
    env = dict(os.environ)
    env["CUDA_HOME"] = str(cuda_home)
    env["MOK_NVCC"] = str(nvcc)
    env["MOK_ARCH"] = "SM100"
    env["PATH"] = f"{venv}/bin:{cuda_home}/bin:{env['PATH']}"
    env["LIBRARY_PATH"] = f"{cuda_home}/lib:{env.get('LIBRARY_PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{cuda_home}/lib:{torch_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    env["PYTHONPATH"] = f"{source / 'tests'}:{Path.cwd()}:{env.get('PYTHONPATH', '')}"
    env["OMP_NUM_THREADS"] = "1"
    return env


def _empty_result() -> dict[str, object]:
    return {
        "install_time": 0.0,
        "build_time": 0.0,
        "smoke_time": 0.0,
        "benchmark_time": 0.0,
        "install_stdout_tail": "",
        "install_stderr_tail": "",
        "build_stdout_tail": "",
        "build_stderr_tail": "",
        "smoke_stdout_tail": "",
        "smoke_stderr_tail": "",
        "benchmark_stdout_tail": "",
        "benchmark_stderr_tail": "",
    }


def run_benchmark(config: MokHeroBenchmarkConfig) -> None:
    working_dir = Path(tempfile.mkdtemp(prefix="ra2a-mok-hero-"))
    venv = working_dir / "venv"
    source = working_dir / "mixture-of-kittens"
    worker_output = working_dir / "hero-result.json"
    stage_data = _empty_result()
    worker_result = None
    error_stage = None
    error = None
    error_elapsed = 0.0
    error_stdout_tail = ""
    error_stderr_tail = ""

    try:
        _run_stage("create_venv", ["uv", "venv", "--python", sys.executable, str(venv)])
        install = _run_stage(
            "install_torch",
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(venv / "bin" / "python"),
                "--extra-index-url",
                TORCH_INDEX,
                "--index-strategy",
                "unsafe-best-match",
                f"torch=={TORCH_VERSION}",
                f"nvidia-cuda-nvcc=={CUDA_NVCC_VERSION}",
                f"nvidia-cuda-crt=={CUDA_NVCC_VERSION}",
                f"nvidia-nvvm=={CUDA_NVCC_VERSION}",
                f"nvidia-cuda-cccl=={CUDA_CCCL_VERSION}",
                "numpy",
                "setuptools>=80",
            ],
        )
        stage_data.update(
            install_time=install.elapsed,
            install_stdout_tail=install.stdout,
            install_stderr_tail=install.stderr,
        )
        _run_stage("clone", ["git", "clone", "--recurse-submodules", MOK_REPOSITORY, str(source)])
        _run_stage("checkout", ["git", "checkout", MOK_COMMIT], cwd=source)
        _run_stage("submodules", ["git", "submodule", "update", "--init", "--recursive"], cwd=source)

        environment = _cuda_environment(venv, source)
        build = _run_stage(
            "build",
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(venv / "bin" / "python"),
                "--no-build-isolation",
                "--no-deps",
                str(source),
            ],
            env=environment,
        )
        stage_data.update(
            build_time=build.elapsed,
            build_stdout_tail=build.stdout,
            build_stderr_tail=build.stderr,
        )

        torchrun = [
            str(venv / "bin" / "python"),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
        ]
        smoke = _run_stage(
            "smoke",
            [
                str(venv / "bin" / "python"),
                "-c",
                "import mok; from mok import functional; print(mok.__file__, functional.__file__)",
            ],
            cwd=working_dir,
            env=environment,
        )
        stage_data.update(
            smoke_time=smoke.elapsed,
            smoke_stdout_tail=smoke.stdout,
            smoke_stderr_tail=smoke.stderr,
        )

        worker = Path.cwd() / "experiments" / "grug" / "moe_hero_ep" / "mok_hero_benchmark_worker.py"
        benchmark = _run_stage(
            "benchmark",
            [*torchrun, str(worker), "--json-output", str(worker_output)],
            cwd=working_dir,
            env=environment,
        )
        stage_data.update(
            benchmark_time=benchmark.elapsed,
            benchmark_stdout_tail=benchmark.stdout,
            benchmark_stderr_tail=benchmark.stderr,
        )
        with worker_output.open(encoding="utf-8") as handle:
            worker_result = WorkerResult.model_validate(json.load(handle))
    except BenchmarkStageError as exc:
        error_stage = exc.stage
        error = str(exc)
        error_elapsed = exc.output.elapsed
        error_stdout_tail = exc.output.stdout
        error_stderr_tail = exc.output.stderr
    except RuntimeError as exc:
        error_stage = "environment"
        error = str(exc)

    result = MokHeroResult(
        source_commit=MOK_COMMIT,
        torch_version=TORCH_VERSION,
        git_sha=launch_provenance().tree_hash,
        error_stage=error_stage,
        error=error,
        error_elapsed=error_elapsed,
        error_stdout_tail=error_stdout_tail,
        error_stderr_tail=error_stderr_tail,
        worker_result=worker_result,
        **stage_data,
    )
    logger.info("mok_hero_result %s", result.model_dump_json())
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fs.open(prefix_join(config.output_path, "results.json"), "w") as handle:
        json.dump(result.model_dump(mode="json"), handle, indent=2)


def build_benchmark(*, version: str | None = None) -> ArtifactStep[MokHeroBenchmarkArtifact]:
    name = "benchmarks/mok-hero-layer-gb200x4"
    version = resolve_version(name, version)

    def build_config(ctx: StepContext) -> MokHeroBenchmarkConfig:
        return MokHeroBenchmarkConfig(output_path=ctx.output_path)

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=MokHeroBenchmarkArtifact,
        run=remote(
            run_benchmark,
            name="mok-hero-layer-gb200x4",
            resources=BENCHMARK_RESOURCES,
        ),
        build_config=build_config,
    )


@click.command()
@build_options
def main() -> ArtifactStep[MokHeroBenchmarkArtifact]:
    return build_benchmark()


if __name__ == "__main__":
    main()
