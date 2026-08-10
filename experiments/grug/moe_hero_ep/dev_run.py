# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the MoK hero directly inside a reserved multi-node dev-GPU session."""

import dataclasses

import click
from marin.execution.lazy import materialized_config
from rigging.filesystem import marin_prefix

from experiments.grug.moe_hero_ep.launch import (
    HERO_PROFILE_NUM_STEPS,
    build_mok_hero_run,
    build_multiprocess_hero_run,
)
from experiments.grug.moe_hero_ep.train import _apply_hero_ep_runtime_defaults, _run_grug_local


@click.command()
@click.option("--run-id", required=True)
@click.option(
    "--num-steps",
    type=click.IntRange(min=1),
    default=25,
    show_default=True,
    help="Configure the full training and optimizer horizon.",
)
@click.option(
    "--stop-after-steps",
    type=click.IntRange(min=1),
    default=None,
    help="Exit after this many completed steps without shortening the configured optimizer horizon.",
)
@click.option("--backend", type=click.Choice(("mok", "fixed")), default="mok", show_default=True)
@click.option(
    "--mok-expert-placement",
    type=click.Choice(("contiguous", "r9_profile_hot_cold")),
    default="contiguous",
    show_default=True,
    help="Relabel the r9 expert bank at initialization so hot/cold pairs share an EP rank.",
)
@click.option(
    "--profile-start-step",
    type=click.IntRange(min=1),
    default=None,
    help="Start an XProf capture at this training step; omitted disables profiling.",
)
@click.option(
    "--profile-num-steps",
    type=click.IntRange(min=1),
    default=HERO_PROFILE_NUM_STEPS,
    show_default=True,
    help="Number of training steps to capture.",
)
@click.option(
    "--profile-all-processes",
    is_flag=True,
    help="Capture and upload a distinct XPlane from every JAX process for rack-level PGLE aggregation.",
)
@click.option(
    "--mok-fwd-num-comm-sms",
    type=click.IntRange(min=1),
    default=None,
    help="Override MoK forward communication SMs; must be even.",
)
@click.option(
    "--mok-bwd-num-comm-sms",
    type=click.IntRange(min=1),
    default=None,
    help="Override MoK backward communication SMs; must be even.",
)
@click.option(
    "--mok-minibatch-size",
    type=click.IntRange(min=1),
    default=None,
    help="Override MoK's minibatch size; must be divisible by 256.",
)
@click.option(
    "--mok-macrobatch-size",
    type=click.IntRange(min=1),
    default=None,
    help="Override MoK's macrobatch size; must be a multiple of the effective minibatch size.",
)
@click.option(
    "--mok-schedule-capacity-multiplier",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    help="Override MoK's schedule capacity multiplier; omitted uses the model default (0.5).",
)
@click.option(
    "--mok-all-gather-top-experts-chunk-bytes",
    type=click.IntRange(min=1),
    default=None,
    help="Override MoK's all-gather chunk size; must be 16-byte aligned.",
)
def main(
    run_id: str,
    num_steps: int,
    stop_after_steps: int | None,
    backend: str,
    mok_expert_placement: str,
    profile_start_step: int | None,
    profile_num_steps: int,
    profile_all_processes: bool,
    mok_fwd_num_comm_sms: int | None,
    mok_bwd_num_comm_sms: int | None,
    mok_minibatch_size: int | None,
    mok_macrobatch_size: int | None,
    mok_schedule_capacity_multiplier: float | None,
    mok_all_gather_top_experts_chunk_bytes: int | None,
) -> None:
    """Materialize the normal hero config and execute it in the current Iris task."""

    mok_overrides = {
        option: value
        for option, value in (
            ("--mok-fwd-num-comm-sms", mok_fwd_num_comm_sms),
            ("--mok-bwd-num-comm-sms", mok_bwd_num_comm_sms),
            ("--mok-minibatch-size", mok_minibatch_size),
            ("--mok-macrobatch-size", mok_macrobatch_size),
            ("--mok-schedule-capacity-multiplier", mok_schedule_capacity_multiplier),
            ("--mok-all-gather-top-experts-chunk-bytes", mok_all_gather_top_experts_chunk_bytes),
        )
        if value is not None
    }
    if backend == "fixed" and (mok_overrides or mok_expert_placement != "contiguous"):
        if mok_expert_placement != "contiguous":
            raise click.UsageError("--mok-expert-placement is only valid with --backend mok")
        option = next(iter(mok_overrides))
        raise click.UsageError(f"{option} is only valid with --backend mok")
    if stop_after_steps is not None and stop_after_steps > num_steps:
        raise click.UsageError("--stop-after-steps cannot exceed --num-steps")
    if profile_all_processes and profile_start_step is None:
        raise click.UsageError("--profile-all-processes requires --profile-start-step")
    for option, value, multiple in (
        ("--mok-fwd-num-comm-sms", mok_fwd_num_comm_sms, 2),
        ("--mok-bwd-num-comm-sms", mok_bwd_num_comm_sms, 2),
        ("--mok-minibatch-size", mok_minibatch_size, 256),
        ("--mok-all-gather-top-experts-chunk-bytes", mok_all_gather_top_experts_chunk_bytes, 16),
    ):
        if value is not None and value % multiple:
            raise click.BadParameter(f"must be divisible by {multiple}", param_hint=option)

    if backend == "mok":
        step = build_mok_hero_run(
            run_id=run_id,
            num_steps=num_steps,
            mok_package="mixture-of-kittens",
            mok_expert_placement=mok_expert_placement,
            profile_start_step=profile_start_step,
            profile_num_steps=profile_num_steps,
            version="dev",
        )
    else:
        step = build_multiprocess_hero_run(
            run_id=run_id,
            num_steps=num_steps,
            profile_start_step=profile_start_step,
            profile_num_steps=profile_num_steps,
            version="dev",
        )
    config = materialized_config(step, marin_prefix())
    model_overrides = {
        name: value
        for name, value in (
            ("mok_fwd_num_comm_sms", mok_fwd_num_comm_sms),
            ("mok_bwd_num_comm_sms", mok_bwd_num_comm_sms),
            ("mok_minibatch_size", mok_minibatch_size),
            ("mok_macrobatch_size", mok_macrobatch_size),
            ("mok_schedule_capacity_multiplier", mok_schedule_capacity_multiplier),
            ("mok_all_gather_top_experts_chunk_bytes", mok_all_gather_top_experts_chunk_bytes),
        )
        if value is not None
    }
    if model_overrides:
        effective_minibatch_size = model_overrides.get("mok_minibatch_size", config.model.mok_minibatch_size)
        effective_macrobatch_size = model_overrides.get("mok_macrobatch_size", config.model.mok_macrobatch_size)
        if effective_macrobatch_size % effective_minibatch_size:
            raise click.BadParameter(
                "must be a multiple of the effective MoK minibatch size",
                param_hint="--mok-macrobatch-size",
            )
        config = dataclasses.replace(
            config,
            model=dataclasses.replace(config.model, **model_overrides),
        )
    if profile_all_processes:
        config = dataclasses.replace(
            config,
            trainer=dataclasses.replace(
                config.trainer,
                trainer=dataclasses.replace(
                    config.trainer.trainer,
                    profiler=dataclasses.replace(config.trainer.trainer.profiler, process_index=None),
                ),
            ),
        )
    config = dataclasses.replace(
        config,
        runtime_pip_packages=(),
    )
    _apply_hero_ep_runtime_defaults()
    _run_grug_local(config, stop_after_steps=stop_after_steps)


if __name__ == "__main__":
    main()
