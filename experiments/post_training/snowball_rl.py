# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a Snowball RL experiment from a named preset.

\b
  learning-medium  EvalLite score before and after 25 async RLOO-N updates; 40 H100.

Without --run, print the plan.

\b
Examples, each after `uv run python -m experiments.post_training.snowball_rl`:
  --preset learning-medium --version 2026.11.14
  --preset learning-medium --version 2026.11.14 --run
  --preset learning-medium --tasks my-selection@2026.11.02 --version 2026.11.14 --run
  --preset learning-medium --init checkpoints/my-sft@2026.11.10 --version 2026.11.14 --run
"""

from __future__ import annotations

import click
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import SkyRLRun

from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_recipe as recipe


def build_run(
    preset: presets.Preset, *, tasks: ArtifactStep | None = None, init: ArtifactStep | None = None
) -> ArtifactStep[SkyRLRun]:
    """Construct one Snowball run of a preset, optionally on other training tasks or from other initial weights."""
    name = user_owned_name(f"{presets.RUN_PREFIX}/{preset.name}")
    spec = recipe.run_spec(preset, name=name, version=resolve_version(name, None), tasks=tasks, init=init)
    return recipe.run_step(preset, spec)


@click.command(help=__doc__)
@click.option(
    "--preset",
    "preset_name",
    required=True,
    type=click.Choice(sorted(presets.PRESETS)),
    help="Experiment to run.",
)
@click.option(
    "--tasks",
    "tasks_name",
    metavar="NAME@VERSION",
    help="Curated task selection to train on, at an immutable version. Defaults to the preset's pinned selection.",
)
@click.option(
    "--init",
    "init_name",
    metavar="NAME@VERSION",
    help="Initial weights: a published model artifact with an HF export. The initial weights are "
    "evaluated too, so the run shows the gain.",
)
@rl_build_options
def main(preset_name: str, tasks_name: str | None, init_name: str | None) -> ArtifactStep[SkyRLRun]:
    return build_run(
        presets.PRESETS[preset_name],
        tasks=presets.selection_artifact(tasks_name) if tasks_name else None,
        init=presets.published_artifact(init_name) if init_name else None,
    )


if __name__ == "__main__":
    main()
