# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the production Mochi SFT recipe at eleven, three, or one rack."""

from dataclasses import replace
from pathlib import Path

import click
from fray.types import ANY_REGION, ResourceConfig
from marin.execution.lazy import lower
from marin.execution.step_runner import StepRunner
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.grug.moe_hero_ep import launch_hero_sft
from experiments.grug.moe_hero_ep.expert_placement import ExpertPlacement
from experiments.grug.moe_hero_ep.sft_recipe import MochiSFT
from experiments.grug.moe_hero_ep.train import RouterBiasMode
from experiments.post_training.mochi_tokenizer import MOCHI_TOKENIZER_REVISION, MOCHI_TOKENIZER_URI
from experiments.sft.launcher import SFTPreset

# This reproduces the eleven-rack `hero-sft-fullab-step393k-v2` run through step 400407.
MOCHI_SFT = MochiSFT(
    run_id="hero-sft-fullab-step393k-v2",
    initialize_from=(
        "s3://marin-us-east-02a/marin/grug/hero-lce-log65536sq-step390k/2026.08.19.2/checkpoints/step-393491"
    ),
    muon_lr=3.2905e-5,
    adam_lr=3.2905e-5,
    max_grad_norm=1.0,
    replay_fraction=0.2035,
    dp_racks=11,
    router_bias_mode=RouterBiasMode.FROZEN,
    capacity_factor=launch_hero_sft.SFT_CAPACITY_FACTOR,
    eval_every=500,
    checkpoints="periodic",
    expert_placement=ExpertPlacement.from_json(
        Path(launch_hero_sft.__file__).with_name("hero_sft_expert_placement_step393491.json").read_text()
    ),
    version="2026.11.10",
    source_revision="406dff47f217637ceca499b266989588cb0df5ab",
    export_resources=ResourceConfig.with_cpu(cpu=64, ram="256g", disk="512g", regions=(ANY_REGION,)),
    # The pinned Hub tokenizer, config, and chat template match the training copy at S3 version 2026.09.18 by SHA-256.
    tokenizer_uri=MOCHI_TOKENIZER_URI,
    tokenizer_revision=MOCHI_TOKENIZER_REVISION,
)


def _preset(recipe: MochiSFT) -> SFTPreset[MochiSFT]:
    resources = launch_hero_sft.sft_resources(recipe.dp_racks)
    # The coordinator selects the Iris cluster where its child jobs run.
    return SFTPreset(
        f"mixture-262k-{resources.replicas}node",
        recipe,
        replace(resources, regions=(ANY_REGION,)),
    )


def _short_preset(racks: int) -> SFTPreset[MochiSFT]:
    return _preset(
        replace(MOCHI_SFT, run_id=f"mochi-sft-{racks}-rack", dp_racks=racks, stop_after_steps=3, checkpoints="none")
    )


MIXTURE_262K_176NODE = _preset(MOCHI_SFT)
MIXTURE_262K_16NODE = _short_preset(1)
MIXTURE_262K_48NODE = _short_preset(3)
MOCHI_SFT_PRESETS = (MIXTURE_262K_176NODE, MIXTURE_262K_16NODE, MIXTURE_262K_48NODE)
_BY_NAME = {preset.name: preset for preset in MOCHI_SFT_PRESETS}


@click.command()
@click.option("--preset", type=click.Choice(tuple(_BY_NAME)), required=True)
@click.option("--steps", type=click.IntRange(min=1), help="Stop after this many SFT updates.")
@click.option("--export", is_flag=True, help="Save the final native checkpoint and export it to Hugging Face format.")
@click.option("--prefix", help="Output storage URI or absolute directory. Bounded runs default to three-day storage.")
@click.option("--run", is_flag=True, help="Launch the selected workflow.")
def main(preset: str, steps: int | None, export: bool, prefix: str | None, run: bool) -> None:
    configure_coreweave_s3()
    chosen = _BY_NAME[preset]
    try:
        if export:
            terminal = chosen.recipe.trained(chosen.resources, steps=steps, prefix=prefix).step
        else:
            terminal = chosen.recipe.train_artifact(chosen.resources, steps=steps, prefix=prefix)
    except ValueError as error:
        raise click.UsageError(str(error)) from error
    StepRunner().run([lower(terminal)], dry_run=not run)


if __name__ == "__main__":
    main()
