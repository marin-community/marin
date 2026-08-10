# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the exact Grug context-parallel attention benchmark on GB200 gangs."""

from __future__ import annotations

import os
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path

import click
from fray.cluster import ResourceConfig
from iris.cluster.setup_scripts import cuda_toolchain_setup_script, default_setup_script
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name

from experiments.grug.dispatch import dispatch_grug_training_run

HERO_NODES_PER_RACK = 16
HERO_GPUS_PER_NODE = 4
HERO_PROCESSES_PER_TASK = HERO_GPUS_PER_NODE
TRANSFORMER_ENGINE_SETUP_SCRIPT = r"""set -e
cd "$IRIS_WORKDIR"
uv pip install --python "$IRIS_VENV/bin/python" \
  nvidia-cuda-cccl==13.3.3.4.1 \
  nvidia-cudnn-frontend==1.25.0 \
  transformer_engine_cu13==2.17.1
nccl_lib="$IRIS_VENV/lib/python3.12/site-packages/nvidia/nccl/lib"
if [ ! -e "$nccl_lib/libnccl.so" ]; then
  ln -s libnccl.so.2 "$nccl_lib/libnccl.so"
fi
uv pip install --python "$IRIS_VENV/bin/python" \
  --no-build-isolation --no-deps transformer_engine_jax==2.17.1
uv pip install --python "$IRIS_VENV/bin/python" --no-deps transformer_engine==2.17.1
"""
TRANSFORMER_ENGINE_BUILD_ENV = {
    # TE 2.17.1 falls back to CUDA 12 when JAX 0.11 does not expose its private runtime-version API.
    "CUDA_VERSION": "13.0",
    # The GPU workspace contains CUDA 12 packages for optional dependencies. cuDNN frontend must
    # bind to the CUDA 13 runtime used by JAX and the TE core.
    "CUDNN_FRONTEND_CUDART_LIB_NAME": "libcudart.so.13",
    # CUDA 13's unified pip layout nests the CCCL headers below the directory TE discovers.
    "CPLUS_INCLUDE_PATH": (
        "/app/.venv/lib/python3.12/site-packages/nvidia/cu13/include/cccl:"
        "/app/.venv/lib/python3.12/site-packages/nvidia/nvtx/include:"
        "/app/.venv/lib/python3.12/site-packages/include"
    ),
    "LIBRARY_PATH": "/app/.venv/lib/python3.12/site-packages/nvidia/nccl/lib",
    "LD_LIBRARY_PATH": "/app/.venv/lib/python3.12/site-packages/nvidia/nccl/lib",
    "NVTE_BUILD_USE_NVIDIA_WHEELS": "1",
    "NVTE_CUDA_ARCHS": "100",
    "NVTE_WITH_NCCL_EP": "0",
}
BENCHMARK_PATH = Path("lib/levanter/scripts/bench/bench_grug_context_parallel_attention.py")


@dataclass(frozen=True)
class ContextParallelAttentionBenchmarkConfig:
    run_id: str
    resources: ResourceConfig
    seq_lens: str
    cases: str
    strategies: str
    context_parallel_size: int
    segments_per_sequence: int
    all_gather_stripe_size: int
    warmup: int
    steps: int


class ContextParallelAttentionBenchmarkResult(Artifact):
    """Marker artifact for an exact-shape context-parallel attention benchmark."""


def _run_benchmark_process(config: ContextParallelAttentionBenchmarkConfig) -> None:
    """Join the Iris JAX mesh and run one distributed benchmark process."""
    os.environ.setdefault("NVTE_FUSED_RING_ATTENTION_USE_SCAN", "0")

    # Deferred so the CPU coordinator never initializes JAX or Transformer Engine.
    from iris.runtime.jax_init import initialize_jax  # noqa: PLC0415

    initialize_jax(endpoint_name="grug_cp_attention_benchmark")

    benchmark_path = Path.cwd() / BENCHMARK_PATH
    argv = [
        str(benchmark_path),
        "--seq-lens",
        config.seq_lens,
        "--cases",
        config.cases,
        "--strategies",
        config.strategies,
        "--context-parallel-size",
        str(config.context_parallel_size),
        "--device-count",
        str(config.resources.replicas * HERO_GPUS_PER_NODE),
        "--segments-per-sequence",
        str(config.segments_per_sequence),
        "--all-gather-stripe-size",
        str(config.all_gather_stripe_size),
        "--warmup",
        str(config.warmup),
        "--steps",
        str(config.steps),
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(str(benchmark_path), run_name="__main__")
    finally:
        sys.argv = old_argv


def run_context_parallel_attention_benchmark(config: ContextParallelAttentionBenchmarkConfig) -> None:
    """Dispatch and wait for the multi-node GB200 benchmark gang."""
    dispatch_grug_training_run(
        run_id=config.run_id,
        config=config,
        local_entrypoint=_run_benchmark_process,
        resources=config.resources,
        max_retries_failure=0,
        processes_per_task=HERO_PROCESSES_PER_TASK,
        extra_env_vars=TRANSFORMER_ENGINE_BUILD_ENV,
        setup_scripts=(
            default_setup_script(extras=("gpu",), python_version=f"{sys.version_info.major}.{sys.version_info.minor}"),
            cuda_toolchain_setup_script(),
            TRANSFORMER_ENGINE_SETUP_SCRIPT,
        ),
    )


def build_context_parallel_attention_benchmark(
    *,
    run_id: str,
    nodes: int,
    seq_lens: str,
    cases: str,
    strategies: str,
    context_parallel_size: int,
    segments_per_sequence: int,
    all_gather_stripe_size: int,
    warmup: int,
    steps: int,
    version: str | None = None,
) -> ArtifactStep[ContextParallelAttentionBenchmarkResult]:
    """Build an exact-shape Grug attention benchmark allocation."""
    resources = ResourceConfig.with_gpu(
        "GB200",
        count=HERO_GPUS_PER_NODE,
        cpu=120,
        ram="850g",
        disk="1t",
        replicas=nodes,
    )
    name = f"grug/{run_id}"
    version = resolve_version(name, version)

    def build_config(ctx: StepContext) -> ContextParallelAttentionBenchmarkConfig:
        return ContextParallelAttentionBenchmarkConfig(
            run_id=run_id,
            resources=ctx.runtime_arg("benchmark_resources"),
            seq_lens=seq_lens,
            cases=cases,
            strategies=strategies,
            context_parallel_size=context_parallel_size,
            segments_per_sequence=segments_per_sequence,
            all_gather_stripe_size=all_gather_stripe_size,
            warmup=warmup,
            steps=steps,
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=ContextParallelAttentionBenchmarkResult,
        run=run_context_parallel_attention_benchmark,
        build_config=build_config,
        deps=(),
        runtime_args={"benchmark_resources": resources},
    )


@click.command()
@click.option("--run-id", required=True)
@click.option("--nodes", type=click.IntRange(min=1), default=HERO_NODES_PER_RACK, show_default=True)
@click.option("--seq-lens", default="262144", show_default=True)
@click.option("--cases", default="local,global", show_default=True)
@click.option("--strategies", default="ring,all_gather", show_default=True)
@click.option("--context-parallel-size", type=click.IntRange(min=2), default=4, show_default=True)
@click.option("--segments-per-sequence", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--all-gather-stripe-size", type=click.IntRange(min=1), default=512, show_default=True)
@click.option("--warmup", type=click.IntRange(min=0), default=1, show_default=True)
@click.option("--steps", type=click.IntRange(min=1), default=3, show_default=True)
@build_options
def main(
    run_id: str,
    nodes: int,
    seq_lens: str,
    cases: str,
    strategies: str,
    context_parallel_size: int,
    segments_per_sequence: int,
    all_gather_stripe_size: int,
    warmup: int,
    steps: int,
) -> ArtifactStep[ContextParallelAttentionBenchmarkResult]:
    return build_context_parallel_attention_benchmark(
        run_id=run_id,
        nodes=nodes,
        seq_lens=seq_lens,
        cases=cases,
        strategies=strategies,
        context_parallel_size=context_parallel_size,
        segments_per_sequence=segments_per_sequence,
        all_gather_stripe_size=all_gather_stripe_size,
        warmup=warmup,
        steps=steps,
    )


if __name__ == "__main__":
    main()
