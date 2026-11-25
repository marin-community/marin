# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose a versioned SFT mix and an RL preset into one artifact graph."""

from dataclasses import replace

import click
from marin.execution.artifact import is_mutable_version
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.rl.cli import rl_build_options
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.grug.moe_hero_ep.sft_recipe import MochiSFT
from experiments.post_training import rl, rl_compose
from experiments.post_training.mochi_rl import MOCHI
from experiments.post_training.mochi_sft import MOCHI_SFT_PRESETS
from experiments.post_training.rl_presets import ModelPresets, Preset
from experiments.post_training.snowball_rl import SNOWBALL
from experiments.post_training.snowball_sft import SNOWBALL_SFT
from experiments.sft.launcher import PreparedData, SFTPreset, SFTSpec


def build_arm(
    sft: SFTPreset[MochiSFT] | SFTPreset[SFTSpec],
    preset: Preset,
    *,
    table: ModelPresets,
    data_mix: PreparedData,
    version: str,
    sft_steps: int,
    rl_steps: int,
    tasks: ArtifactStep | None = None,
) -> rl.RLExperiment:
    """Train a mix with preset hyperparameters, then select and export the best RL policy."""
    if is_mutable_version(version):
        raise ValueError("Long-loop arms require an immutable version")
    recipe = sft.recipe.with_data_mix(data_mix)
    if isinstance(recipe, MochiSFT):
        recipe = replace(recipe, version=version, stop_after_steps=sft_steps, checkpoints="final")
    else:
        recipe = replace(recipe, version=version, num_train_steps=sft_steps)
    if not preset.evaluations:
        raise ValueError("Best-policy selection requires an RL preset with held-out evaluation")
    preset = replace(
        preset.with_updates(rl_steps),
        revision=version,
        checkpoints="best",
        export="best",
        checkpoint_every=preset.checkpoint_every or preset.evaluations[0].every or rl_steps,
    )
    source = recipe.trained(
        sft.resources,
        coordinator_cluster=preset.recipe.layout.cluster,
        coordinator_prefix=rl_compose.run_prefix(preset),
    )
    return rl.build_experiment(preset, table=table, tasks=tasks, init=source, sft_preset=sft.name)


@click.command(help=__doc__)
@click.option("--model", type=click.Choice(("mochi", "snowball")), required=True)
@click.option("--sft-preset", required=True)
@click.option("--rl-preset", required=True)
@click.option("--data-mix", required=True, help="Immutable mix artifact NAME@VERSION.")
@click.option("--data-uri", required=True, help="Published regional mix artifact directory.")
@click.option("--data-file", required=True, help="LM data JSON manifest relative to the mix directory.")
@click.option("--data-tokenizer", required=True, help="Tokenizer identity recorded in the data manifest.")
@click.option("--sft-steps", type=click.IntRange(min=1), required=True)
@click.option("--rl-steps", type=click.IntRange(min=1), required=True)
@click.option("--tasks", help="Immutable curated RL task selection NAME@VERSION; defaults to the RL preset.")
@rl_build_options
def main(
    model: str,
    sft_preset: str,
    rl_preset: str,
    data_mix: str,
    data_uri: str,
    data_file: str,
    data_tokenizer: str,
    sft_steps: int,
    rl_steps: int,
    tasks: str | None,
) -> tuple[ArtifactStep, ...]:
    configure_coreweave_s3()
    name, separator, mix_version = data_mix.rpartition("@")
    if not name or not separator:
        raise click.BadParameter("data-mix requires NAME@VERSION")
    table = MOCHI if model == "mochi" else SNOWBALL
    sfts = {preset.name: preset for preset in (MOCHI_SFT_PRESETS if model == "mochi" else SNOWBALL_SFT)}
    if sft_preset not in sfts or rl_preset not in table.by_name:
        raise click.BadParameter("select SFT and RL presets belonging to the requested model")
    try:
        arm = build_arm(
            sfts[sft_preset],
            table.by_name[rl_preset],
            table=table,
            data_mix=PreparedData(ArtifactStep.adopt(name, mix_version, data_uri), data_file, data_tokenizer),
            version=resolve_version("long-loop", None),
            sft_steps=sft_steps,
            rl_steps=rl_steps,
            tasks=rl_compose.selection_artifact(tasks) if tasks is not None else None,
        )
    except ValueError as error:
        raise click.UsageError(str(error)) from error
    return arm.runs


if __name__ == "__main__":
    main()
