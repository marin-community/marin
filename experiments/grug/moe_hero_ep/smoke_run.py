# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-node smoke run of the MoK arm at the hero's two token widths.

The hero shape needs a whole NVL72 rack; this is the smallest configuration that still exercises
what the two-width ABI added -- ``hidden_dim=6144`` with ``latent_dim=3072``, both real shared
experts fused, a real expert axis (EP4 across one GB200 node), and the full train step through the
fused forward and backward. Everything else is shrunk: four layers, twelve routed experts, a small
batch, a handful of steps.

It reuses ``build_mok_hero_run`` rather than restating the hero config, patching only the two
module constants that hardcode the rack -- so a drift in the launcher shows up here as an error
rather than as a silently different shape.
"""

from __future__ import annotations

import dataclasses
import os

import click
from marin.execution.lazy import materialized_config
from rigging.filesystem.cluster_config import marin_prefix

from experiments.grug.moe_hero_ep import launch as launch_module
from experiments.grug.moe_hero_ep.train import _apply_hero_ep_runtime_defaults, _run_grug_local

SMOKE_NODES = 1
SMOKE_GPUS_PER_NODE = 4
SMOKE_EXPERT_AXIS = SMOKE_NODES * SMOKE_GPUS_PER_NODE


def _echo_metrics_to_stdout() -> None:
    """Mirror the per-step training metrics into the job log.

    The tracker's own record goes to W&B (offline here) and to a summary-only JSONL, neither of
    which shows the loss trajectory in the Iris log. The trajectory is the point of a smoke run,
    so tee the two metrics that matter as they are logged.
    """
    import levanter.tracker  # noqa: PLC0415

    original_log = levanter.tracker.log

    def log_and_echo(metrics, *args, **kwargs):
        if isinstance(metrics, dict):
            step = kwargs.get("step", args[0] if args else None)
            if "train/loss" in metrics:
                print(f"[smoke] step={step} train/loss={float(metrics['train/loss']):.6f}", flush=True)
            if "moe/drop_fraction" in metrics:
                print(
                    f"[smoke] step={step} moe/drop_fraction={float(metrics['moe/drop_fraction']):.6g} "
                    f"moe/dropped_assignments={int(metrics['moe/dropped_assignments'])}",
                    flush=True,
                )
        return original_log(metrics, *args, **kwargs)

    levanter.tracker.log = log_and_echo


@click.command()
@click.option("--run-id", required=True)
@click.option("--num-steps", type=click.IntRange(min=1), default=8, show_default=True)
@click.option("--batch-size", type=click.IntRange(min=1), default=8, show_default=True)
@click.option("--num-layers", type=click.IntRange(min=1), default=4, show_default=True)
@click.option("--num-experts", type=click.IntRange(min=1), default=12, show_default=True)
@click.option("--num-experts-per-token", type=click.IntRange(min=1), default=4, show_default=True)
@click.option("--intermediate-dim", type=click.IntRange(min=1), default=1024, show_default=True)
@click.option(
    "--latent-dim",
    type=click.IntRange(min=0),
    default=3072,
    show_default=True,
    help="Routed width. 3072 against hidden 6144 is the hero's two-width shape; 0 is the control.",
)
@click.option(
    "--mok-schedule-capacity-multiplier",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    help=(
        "MoK schedule capacity multiplier. The per-rank schedule holds tokens*topk*max(2, ceil(ep*m)) "
        "assignments while the mean received is tokens*topk, so the hero's 0.5 buys 32x headroom at "
        "EP64 and only 2x at this run's EP4 -- which an unbalanced early router can exceed."
    ),
)
@click.option("--backend", type=click.Choice(("mok", "fixed")), default="mok", show_default=True)
@click.option("--dry-run", is_flag=True, help="Print the resolved config and data paths, then exit.")
def main(
    run_id: str,
    num_steps: int,
    batch_size: int,
    num_layers: int,
    num_experts: int,
    num_experts_per_token: int,
    intermediate_dim: int,
    latent_dim: int,
    mok_schedule_capacity_multiplier: float | None,
    backend: str,
    dry_run: bool,
) -> None:
    # The hero launcher hardcodes the 16-node rack and its 64-way expert axis. Narrow both to one
    # node before building, so the expert-axis divisibility checks run against the real axis size.
    launch_module.HERO_EP_NODES = SMOKE_NODES
    launch_module.HERO_EP_EXPERT_AXIS_SIZE = SMOKE_EXPERT_AXIS

    builder = launch_module.build_mok_hero_run if backend == "mok" else launch_module.build_multiprocess_hero_run
    kwargs = dict(
        run_id=run_id,
        num_steps=num_steps,
        batch_size=batch_size,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        intermediate_dim=intermediate_dim,
        latent_dim=latent_dim,
        version="smoke-dev",
    )
    if backend == "mok":
        kwargs["mok_package"] = "mixture-of-kittens"
    step = builder(**kwargs)
    config = materialized_config(step, marin_prefix())

    model_overrides = {}
    if mok_schedule_capacity_multiplier is not None:
        model_overrides["mok_schedule_capacity_multiplier"] = mok_schedule_capacity_multiplier
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(
            config.model,
            num_layers=num_layers,
            **model_overrides,
            # Waves are a pooled-transport knob; one wave keeps the local expert count legal at
            # every bank size this script accepts.
            num_expert_waves=1,
        ),
        trainer=dataclasses.replace(
            config.trainer,
            # Host offload and pinned master params are rack-scale memory tactics, not part of what
            # this run is checking, and each is its own failure mode.
            offload_opt_state=False,
            expert_axis_size=SMOKE_EXPERT_AXIS,
            replica_axis_size=1,
        ),
        runtime_pip_packages=(),
    )
    print(
        f"[smoke] backend={config.model.moe_implementation} hidden={config.model.hidden_dim} "
        f"latent={config.model.latent_dim} layers={config.model.num_layers} "
        f"experts={config.model.num_experts} topk={config.model.num_experts_per_token} "
        f"intermediate={config.model.intermediate_dim} shared={config.model.num_shared_experts}"
        f"x{config.model.shared_expert_intermediate_dim} batch={config.trainer.trainer.train_batch_size} "
        f"seq={config.model.max_seq_len} steps={num_steps} expert_axis={config.trainer.expert_axis_size} "
        f"capacity_multiplier={config.model.mok_schedule_capacity_multiplier} "
        f"minibatch={config.model.mok_minibatch_size} macrobatch={config.model.mok_macrobatch_size}",
        flush=True,
    )
    if dry_run:
        print(f"[smoke] data={config.data}", flush=True)
        return
    _echo_metrics_to_stdout()
    _apply_hero_ep_runtime_defaults(
        inline_watch_enabled=config.trainer.trainer.watch.is_enabled,
        processes_per_task=config.processes_per_task,
    )
    _run_grug_local(config, stop_after_steps=num_steps)
    print("[smoke] training loop returned", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("WANDB_MODE", "offline")
    main()
