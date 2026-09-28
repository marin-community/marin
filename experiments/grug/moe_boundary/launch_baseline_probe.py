# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fresh baseline throughput probe at d1280 (diagnostic for the Phase 0 -6.5% gap).

Reruns the May Recipe d1280 baseline model (boundary operator OFF, PKO on,
long RoPE on, seq 4096, z-loss off, v4-32 EP=1) for a handful of steps on
today's stack, to distinguish "the boundary variant is slower" from "the
stack drifted since June".

Interpretation: June baseline steady-state was 171.9k tok/s (median), the
September variant 160.7k. If this probe lands near 172k, the variant is the
cause; near 160k, it's stack drift.

Submit (v4-32)::

    .venv/bin/iris --cluster=marin job run --no-wait \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe_boundary.launch_baseline_probe \\
            --version dev --run
"""

import dataclasses

import click
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.data import mixture
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint

from experiments.grug.moe_boundary.launch import (
    GrugMoeLaunchConfig,
    grug_moe_boundary_mix,
    run_grug_moe_trial,
)
from experiments.grug.moe_boundary.launch_compute_opt import _EP, baseline_recipe
from experiments.grug.moe_boundary.train import GrugTrainerConfig

_SEQ: int = 4096
_TRAIN_RESOURCES = ResourceConfig.with_tpu("v4-32")


def baseline_probe_cell(*, hidden_dim: int, steps: int, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Baseline (boundary-off) throughput probe cell."""
    model, optimizer, _ = baseline_recipe(hidden_dim)
    # Pin the June baseline measurement conditions: PKO on for long layers,
    # RoPE on long layers (the heuristic's CURRENT defaults disable both,
    # so they must be overridden explicitly).
    model = dataclasses.replace(model, max_seq_len=_SEQ, disable_pko=False, disable_long_rope=False)
    name = f"grug/moe_baseline_probe_d{hidden_dim}_ep{_EP}"
    version = resolve_version(name, version)
    train, validation = grug_moe_boundary_mix()
    run_id = f"moe_baseline_probe_d{hidden_dim}_ep{_EP}"

    def build_config(ctx: StepContext) -> GrugMoeLaunchConfig:
        return GrugMoeLaunchConfig(
            model=model,
            data=mixture(ctx, train, validation=validation),
            output_path=ctx.output_path,
            run_id=run_id,
            resources=ctx.runtime_arg("train_resources"),
            steps=steps,
            batch_size=256,
            seed=0,
            mp="params=float32,compute=bfloat16,output=bfloat16",
            tracker=WandbConfig(
                project="marin_moe",
                tags=["moe", "baseline-probe", "throughput", f"d{hidden_dim}"],
                group="moe-boundary",
                name=None,
            ),
            optimizer=optimizer,
            # Disposable throughput probe: no periodic saves, node-local
            # checkpoints only.
            checkpointer=CheckpointerConfig(base_path=f"{ctx.output_path}/checkpoints", save_interval=None),
            grug_trainer=GrugTrainerConfig(z_loss_weight=0.0, ema_beta=None, log_every=1),
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_moe_trial,
        build_config=build_config,
        deps=(*train, *validation),
        runtime_args={"train_resources": _TRAIN_RESOURCES},
    )


@click.command()
@click.option("--dim", type=click.Choice(["512", "768", "1024", "1280"]), default="1280")
@click.option("--steps", type=int, default=64, help="Probe length in steps (~10 min of compute).")
@build_options
def build(dim: str, steps: int):
    """Build one baseline (boundary-off) throughput probe cell."""
    return baseline_probe_cell(hidden_dim=int(dim), steps=steps)


if __name__ == "__main__":
    build()
