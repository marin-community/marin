# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the pinned TaskTrove release.

- ``uv run python -m experiments.post_training.tasktrove.pipeline`` prints the build plan.
- Add ``--run`` to build or reuse the pinned release.
- Add ``--stage templates --run`` to stop after a named stage.

Stages:

- ``raw`` downloads the pinned Hugging Face Parquet files.
- ``summaries`` groups tasks by template.
- ``templates`` extracts exemplars and checks converter coverage.
- ``converted`` normalizes retained sources into Harbor tasks.
- ``filtered`` deduplicates rows and removes tasks that fail verifier checks.
- ``release`` writes the task Parquet, rejection ledger, manifest, and report.
"""

from dataclasses import dataclass

import click
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import OUT, ArtifactStep, apply, lower, run
from marin.execution.remote import remote
from marin.experiment.data import hf_download

from experiments.post_training.tasktrove.convert import convert_tasks
from experiments.post_training.tasktrove.dataset import TASKS_GLOB, TASKTROVE_HF_ID, TASKTROVE_REVISION
from experiments.post_training.tasktrove.publish import publish_release
from experiments.post_training.tasktrove.task_templates import build_template_index, summarize_templates
from experiments.post_training.tasktrove.verify import filter_tasks

STAGES = ("raw", "summaries", "templates", "converted", "filtered", "release")
PIPELINE_VERSION = "2026.09.10.8"
VERIFY_TOOL_REF = "b2b68d8b0a770cdc0ab3903780172c4b3eea81b1"
RAW_VERSION = "2026.09.09"
"""Pinned download version; bump only when ``TASKTROVE_REVISION`` changes, so reruns reuse the download."""


@dataclass(frozen=True)
class TaskTroveWorkflow:
    raw: ArtifactStep
    summaries: ArtifactStep
    templates: ArtifactStep
    converted: ArtifactStep
    filtered: ArtifactStep
    release: ArtifactStep


def build_workflow() -> TaskTroveWorkflow:
    coordinator = ResourceConfig.with_cpu(cpu=4, ram="16g")
    raw = hf_download(
        "raw/tasktrove", hf_id=TASKTROVE_HF_ID, revision=TASKTROVE_REVISION, version=RAW_VERSION, urls_glob=(TASKS_GLOB,)
    )
    summaries = apply(
        "tasktrove/template_summaries",
        remote(summarize_templates, resources=coordinator),
        version=PIPELINE_VERSION,
        input_path=raw,
        output_path=OUT,
    )
    templates = apply(
        "tasktrove/templates",
        remote(build_template_index, resources=coordinator),
        version=PIPELINE_VERSION,
        input_path=raw,
        summaries_path=summaries,
        output_path=OUT,
    )
    converted = apply(
        "tasktrove/converted",
        remote(convert_tasks, resources=coordinator),
        version=PIPELINE_VERSION,
        input_path=raw,
        templates_path=templates,
        output_path=OUT,
        tool_ref=VERIFY_TOOL_REF,
    )
    filtered = apply(
        "tasktrove/graded",
        remote(filter_tasks, resources=coordinator),
        version=PIPELINE_VERSION,
        converted_path=converted,
        output_path=OUT,
        max_tasks_per_source=None,
    )
    release = apply(
        "tasktrove/clean",
        remote(publish_release, resources=coordinator),
        version=PIPELINE_VERSION,
        filtered_path=filtered,
        output_path=OUT,
        tool_ref=VERIFY_TOOL_REF,
        artifact_type=Artifact,
    )
    return TaskTroveWorkflow(raw, summaries, templates, converted, filtered, release)


@click.command(help=__doc__)
@click.option("--stage", type=click.Choice(STAGES), default="release", show_default=True)
@click.option("--run", "do_run", is_flag=True, help="Build the selected stage; the default prints its plan.")
@click.option("--max-concurrent", type=int, default=8, show_default=True)
def main(stage: str, do_run: bool, max_concurrent: int) -> None:
    target = getattr(build_workflow(), stage)
    if do_run:
        run(target, max_concurrent=max_concurrent)
    else:
        click.echo(lower(target))


if __name__ == "__main__":
    main()
