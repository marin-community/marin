# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Four-GPU training proxy preserving hero per-device attention and expert shapes."""

import dataclasses
import functools

import click
from levanter.tracker.json_logger import JsonLoggerConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options

from experiments.benchmarks.fa4.hero_forward import STEPS, _run_study_local
from experiments.benchmarks.hero_profile import PRODUCTION_SCHEDULE_STEPS, run_restored_benchmark
from experiments.grug.moe_hero_ep.hero_recipe import HeroThroughputResult
from experiments.grug.moe_hero_ep.launch_diagnostics import build_diagnostic_run
from experiments.grug.moe_hero_ep.train import GrugRunConfig, TrainingDataMode, _run_grug_local

PROXY_GPUS = 4
PROXY_EXPERTS = 24
PROXY_LAYERS = 4
PROXY_BATCH = 64


@click.command()
@click.option("--run-id", required=True)
@build_options
def main(run_id: str) -> ArtifactStep[HeroThroughputResult]:
    step = build_diagnostic_run(
        run_id=run_id,
        dp_racks=1,
        num_steps=STEPS,
        schedule_steps=PRODUCTION_SCHEDULE_STEPS,
        batch_size=PROXY_BATCH,
        training_data_mode=TrainingDataMode.MIXTURE,
        save_checkpoints=False,
        profile_steps=0,
    )

    def build_config(ctx: StepContext) -> GrugRunConfig:
        config = step.build_config(ctx)
        return dataclasses.replace(
            config,
            model=dataclasses.replace(config.model, num_layers=PROXY_LAYERS, num_experts=PROXY_EXPERTS),
            trainer=dataclasses.replace(
                config.trainer,
                expert_axis_size=PROXY_GPUS,
                trainer=dataclasses.replace(config.trainer.trainer, load_checkpoint=False, tracker=JsonLoggerConfig()),
            ),
            max_retries_failure=0,
        )

    return dataclasses.replace(
        step,
        build_config=build_config,
        runtime_args={
            **step.runtime_args,
            "train_resources": dataclasses.replace(step.runtime_args["train_resources"], replicas=1),
        },
        run=functools.partial(
            run_restored_benchmark,
            local_entrypoint=functools.partial(_run_study_local, train=_run_grug_local),
        ),
    )


if __name__ == "__main__":
    main()
