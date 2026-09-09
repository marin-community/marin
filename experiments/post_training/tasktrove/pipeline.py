# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove conversion step graph.

    python -m experiments.post_training.tasktrove.pipeline --version 2026.09.09
    python -m experiments.post_training.tasktrove.pipeline --version 2026.09.09 --run
    python -m experiments.post_training.tasktrove.pipeline --version 2026.09.09 --stage templates --run

Stages: ``raw`` downloads the active parquets; ``fingerprints`` records one template id per task;
``templates`` extracts one exemplar per template for converter authors; ``converted`` applies the
registered converters; ``validated`` writes the ledger and fails on malformed output.
"""

from dataclasses import dataclass

import click
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from marin.experiment.data import hf_download

from experiments.post_training.tasktrove.convert import convert_tasks
from experiments.post_training.tasktrove.fingerprint import build_template_index, fingerprint_tasks
from experiments.post_training.tasktrove.sources import TASKTROVE_HF_ID, TASKTROVE_REVISION
from experiments.post_training.tasktrove.validate import validate_converted

STAGES = ("raw", "fingerprints", "templates", "converted", "validated")
ACTIVE_PARQUETS = ("*/tasks.parquet",)
"""The default config; ``deprecated/*`` is excluded."""


@dataclass(frozen=True)
class TaskTroveWorkflow:
    raw: ArtifactStep
    fingerprints: ArtifactStep
    templates: ArtifactStep
    converted: ArtifactStep
    validated: ArtifactStep


def build_workflow() -> TaskTroveWorkflow:
    coordinator = ResourceConfig.with_cpu(cpu=4, ram="16g")
    raw = hf_download("raw/tasktrove", hf_id=TASKTROVE_HF_ID, revision=TASKTROVE_REVISION, urls_glob=ACTIVE_PARQUETS)
    fingerprints = apply(
        "tasktrove/fingerprints",
        remote(fingerprint_tasks, resources=coordinator),
        input_path=raw,
        output_path=OUT,
    )
    templates = apply(
        "tasktrove/templates",
        remote(build_template_index, resources=coordinator),
        input_path=raw,
        fingerprints_path=fingerprints,
        output_path=OUT,
    )
    converted = apply(
        "tasktrove/converted",
        remote(convert_tasks, resources=coordinator),
        input_path=raw,
        output_path=OUT,
    )
    validated = apply(
        "tasktrove/validated",
        remote(validate_converted, resources=coordinator),
        converted_path=converted,
        output_path=OUT,
        artifact_type=Artifact,
    )
    return TaskTroveWorkflow(raw, fingerprints, templates, converted, validated)


@click.command(help=__doc__)
@click.option("--stage", type=click.Choice(STAGES), default="validated", show_default=True)
@build_options
def main(stage: str) -> ArtifactStep:
    return getattr(build_workflow(), stage)


if __name__ == "__main__":
    main()
