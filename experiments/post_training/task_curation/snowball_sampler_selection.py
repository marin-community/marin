# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the four-source sampler pilot from Snowball's already-admitted task releases."""

import click
from finestore.rl.source_identity import canonical_source_id
from marin.execution.lazy import ArtifactStep, run
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.rl_compose import selection_artifact, selection_summary
from experiments.post_training.task_curation import CANONICAL_ARTIFACT_ROOT
from experiments.post_training.task_curation.local import artifact_prefix
from experiments.post_training.task_curation.selection import (
    FULL_VALIDATION_FILENAME,
    HeldoutSource,
    PanelFormat,
    SelectionRecipe,
    task_selection_step,
)

PARENT = selection_artifact("snowball-100k@2026.11.19.1")
EVAL_LITE = selection_artifact("eval-lite@2026.11.17.1")
REASONING_GYM = "Reasoning Gym:reasoning_gym"
SOURCES = {
    REASONING_GYM: "arithmetic",
    "TaskTrove:laion__nemotron-gym-agent-calendar-v2": "calendar",
    "TaskTrove:laion__nemotron-gym-instruction-following-calendar-v3": "instruction_calendar",
    "TaskTrove:laion__nemotron-gym-instruction-following-structured-v3": "structural",
}


def full_panel(step: ArtifactStep, rows: int) -> HeldoutSource:
    return HeldoutSource(
        uri=str(StoragePath(step.path(CANONICAL_ARTIFACT_ROOT)) / FULL_VALIDATION_FILENAME),
        version=step.version,
        rows=rows,
        format=PanelFormat.TASK_SPEC,
    )


PILOT = SelectionRecipe(
    name="snowball-sampler-four",
    version="2026.11.25.1",
    sources=SOURCES,
    family_limits=dict.fromkeys(SOURCES.values(), 256),
    grader_task_names={REASONING_GYM: ("chain_sum", "base_conversion", "basic_arithmetic", "leg_counting")},
    heldout=(full_panel(PARENT, 910),),
    exclusions=(full_panel(EVAL_LITE, 1500),),
    seed=20261019,
    validation_per_family=20,
)


def pilot_selection_step() -> ArtifactStep:
    """Bind the original source roots published in the parent selection's receipt."""
    sources = {}
    for row in selection_summary(PARENT)["sources"]:
        key = canonical_source_id(row["source_id"])
        if key in SOURCES:
            sources[key] = ArtifactStep.adopt(
                f"rl-data/sampler-inputs/{key}", version=PARENT.version, source=row["root"]
            )
    return task_selection_step(sources, PILOT)


@click.command(help=__doc__)
@click.option("--run", "build", is_flag=True, help="Materialize the pinned selection.")
def main(build: bool) -> None:
    configure_coreweave_s3()
    with artifact_prefix(CANONICAL_ARTIFACT_ROOT):
        step = pilot_selection_step()
        click.echo(run(step)[0].path if build else step.path())


if __name__ == "__main__":
    main()
