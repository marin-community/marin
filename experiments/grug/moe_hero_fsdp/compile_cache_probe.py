# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-node XLA compilation-cache probe on the hero FSDP code path.

The hero run's compile is minutes long, so a caching change costs an hour per
iteration to evaluate. This launcher runs the same ``run_grug`` entrypoint, the
same ``sonic_cute`` / ``gpu_fa4_cute`` kernels, and the same PGLE and mesh
wiring against a four-layer model, which brings the loop down to a few minutes
while keeping every cache key input that matters: XLA flags, device topology,
and PGLE's two-key scheme.

Compile time does not scale down with the model, so the probe measures cache
hits and keys, not the cost of a hero compile. Four layers of custom-call
matmuls also leave XLA little to autotune, so it is the wrong instrument for
measuring the per-fusion autotune cache.

Two nodes is the smallest shape that exercises cross-node behavior — JAX writes
persistent cache entries only from process 0, so a single node cannot show
whether the other nodes hit or miss.

Read the outcome from the task logs, which carry JAX's own accounting when the
launcher is submitted with ``-e JAX_EXPLAIN_CACHE_MISSES 1``:

    PERSISTENT COMPILATION CACHE MISS for '<module>' with key '<key>'
    Persistent compilation cache hit for '<module>' with key '<key>'

Point successive runs at their own cache prefix with
``-e JAX_COMPILATION_CACHE_DIR <prefix>`` to make "cold" reproducible.
"""

import dataclasses
import math

import click
import jmp
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.progress_watchdog import ProgressWatchdogConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.tracker.telemetry import TelemetryConfig
from levanter.trainer import TrainerConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.data import mixture
from marin.experiment.namespacing import user_namespaced_name
from rigging.filesystem import marin_temp_bucket, prefix_join

from experiments.grug.moe_hero_fsdp.heuristic import MoeHeuristic
from experiments.grug.moe_hero_fsdp.launch import (
    _SLIMPAJAMA_SHUFFLE,
    HERO_GPUS_PER_TASK,
    HERO_MIXED_PRECISION,
    HERO_PROCESS_STALL_TIMEOUT,
    HERO_PROCESSES_PER_TASK,
    HERO_TRAIN_STEP_TIMEOUT,
    _hero_node_resources,
    _slimpajama_6b_dataset,
)
from experiments.grug.moe_hero_fsdp.model import GrugModelConfig
from experiments.grug.moe_hero_fsdp.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe_hero_fsdp.train import GrugRunConfig, GrugTrainerConfig, run_grug

DEFAULT_PROBE_NODES = 2
# PGLE recompiles with an FDO profile only after `jax_pgle_profiling_runs` (3)
# executions, and that recompile is what populates the second of PGLE's two
# cache keys. Stay well clear of the boundary.
DEFAULT_PROBE_STEPS = 8
PROBE_SEQUENCES_PER_DEVICE = 8
PROBE_OUTPUT_TTL_DAYS = 1


class CompileCacheProbeResult(Artifact):
    """Task logs from a compilation-cache probe run."""


def build_probe_configs(*, num_train_steps: int, batch_size: int) -> tuple[GrugModelConfig, GrugMoeMuonHConfig]:
    """Build the model and optimizer for a short multi-node compilation-cache probe."""
    hidden_dim = 2048
    model = GrugModelConfig(
        vocab_size=128_256,
        hidden_dim=hidden_dim,
        intermediate_dim=hidden_dim,
        shared_expert_intermediate_dim=hidden_dim,
        num_shared_experts=1,
        num_experts=32,
        num_experts_per_token=1,
        num_layers=4,
        num_heads=16,
        num_kv_heads=4,
        local_kv_heads=4,
        global_kv_heads=2,
        head_dim=128,
        max_seq_len=2048,
        sliding_window=512,
        global_every=4,
        capacity_factor=1.0,
        initializer_std=0.5 / math.sqrt(hidden_dim),
        qk_mult=1.3,
        sconv=True,
        attention_implementation="gpu_fa4_cute",
        moe_implementation="sonic_cute",
        expert_chunks=4,
        report_capacity_overflow=True,
        rope_fused=True,
    )
    optimizer = MoeHeuristic().build_optimizer_config(
        num_train_steps=num_train_steps,
        batch_size=batch_size,
        hidden_dim=model.hidden_dim,
        seq_len=model.max_seq_len,
    )
    return model, optimizer


def build_compile_cache_probe_run(
    *,
    run_id: str,
    nodes: int,
    num_steps: int,
    data_seed: int | None = None,
    version: str | None = None,
) -> ArtifactStep[CompileCacheProbeResult]:
    """Build the multi-node compilation-cache probe.

    Change ``data_seed`` between runs to select a different first shuffled data
    block without changing compilation keys.
    """
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if nodes < 2:
        raise ValueError(f"nodes must be at least 2 to exercise cross-node caching, got {nodes}")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    batch_size = nodes * HERO_GPUS_PER_TASK * PROBE_SEQUENCES_PER_DEVICE
    model, optimizer = build_probe_configs(num_train_steps=num_steps, batch_size=batch_size)
    # One DP replica per node: parameters are FSDP-sharded over each node's four
    # GPUs and replicated across nodes, the hero's rack-local layout at node scale.
    grug_trainer = GrugTrainerConfig(
        data_seed=data_seed,
        log_every=1,
        ema_beta=None,
        z_loss_weight=1e-4,
        offload_opt_state=False,
        save_checkpoints=False,
        expert_axis_size=1,
        replica_axis_size=nodes,
        sharding_dump_path=None,
    )
    train_resources = _hero_node_resources(nodes)
    name = f"grug/compile-cache-probe/{run_id}"
    version = resolve_version(name, version)
    step_name = user_namespaced_name(name, version)
    slim = _slimpajama_6b_dataset()
    output_path = marin_temp_bucket(ttl_days=PROBE_OUTPUT_TTL_DAYS, prefix=prefix_join(step_name, version))

    def build_config(ctx: StepContext) -> GrugRunConfig:
        trainer = TrainerConfig(
            id=run_id,
            seed=0,
            train_batch_size=batch_size,
            num_train_steps=num_steps,
            profiler=ProfilerConfig(enabled=False, start_step=8, num_steps=0),
            mp=jmp.get_policy(HERO_MIXED_PRECISION),
            tracker=TelemetryConfig(),
            watch=WatchConfig(interval=20),
            progress_watchdog=ProgressWatchdogConfig(
                step_timeout=HERO_TRAIN_STEP_TIMEOUT,
                process_timeout=HERO_PROCESS_STALL_TIMEOUT,
            ),
            use_explicit_mesh_axes=True,
            require_accelerator=True,
            allow_nondivisible_batch_size=False,
            checkpointer=CheckpointerConfig(
                base_path=prefix_join(ctx.output_path, "checkpoints"),
                temporary_base_path=None,
                save_interval=None,
                keep=None,
                append_run_id_to_base_path=False,
            ),
        )
        return GrugRunConfig(
            model=model,
            data=mixture(ctx, {slim: 1.0}, shuffle=_SLIMPAJAMA_SHUFFLE),
            resources=ctx.runtime_arg("train_resources"),
            optimizer=optimizer,
            trainer=dataclasses.replace(grug_trainer, trainer=trainer),
            eval=None,
            processes_per_task=HERO_PROCESSES_PER_TASK,
        )

    return ArtifactStep(
        name=step_name,
        version=version,
        artifact_type=CompileCacheProbeResult,
        run=run_grug,
        build_config=build_config,
        deps=(slim,),
        runtime_args={"train_resources": train_resources},
        override_path=output_path,
    )


@click.command()
@click.option("--run-id", required=True, help="Run identifier for artifact and telemetry names.")
@click.option(
    "--nodes",
    type=click.IntRange(min=2),
    default=DEFAULT_PROBE_NODES,
    show_default=True,
    help="GB200 nodes, four GPUs each. One data-parallel replica per node.",
)
@click.option(
    "--num-steps",
    type=click.IntRange(min=1),
    default=DEFAULT_PROBE_STEPS,
    show_default=True,
    help="Number of training steps.",
)
@click.option(
    "--data-seed",
    type=int,
    default=None,
    help="Dataset shuffle seed. Use distinct values to measure cold first-block reads on reused nodes.",
)
@build_options
def main(run_id: str, nodes: int, num_steps: int, data_seed: int | None) -> ArtifactStep[CompileCacheProbeResult]:
    return build_compile_cache_probe_run(
        run_id=run_id,
        nodes=nodes,
        num_steps=num_steps,
        data_seed=data_seed,
    )


if __name__ == "__main__":
    main()
