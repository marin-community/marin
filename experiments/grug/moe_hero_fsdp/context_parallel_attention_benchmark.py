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
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name

from experiments.grug.dispatch import dispatch_grug_training_run

HERO_NODES_PER_RACK = 16
HERO_GPUS_PER_NODE = 4
HERO_PROCESSES_PER_TASK = HERO_GPUS_PER_NODE
TRANSFORMER_ENGINE_PIP_ARGS = (
    # The aarch64 JAX package is an sdist. Its isolated build omits the NVTX headers and cannot
    # detect the CUDA 13 toolchain already installed by Marin's GPU extra.
    "--no-build-isolation",
    "transformer_engine[jax]==2.17.1",
)
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
        pip_packages=TRANSFORMER_ENGINE_PIP_ARGS,
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
