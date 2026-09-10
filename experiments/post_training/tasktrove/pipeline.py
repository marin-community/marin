# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove Clean step graph.

    python -m experiments.post_training.tasktrove.pipeline --version 2026.09.09 --verify-tool-ref <sha>
    python -m experiments.post_training.tasktrove.pipeline --version 2026.09.09 --verify-tool-ref <sha> --run
    python -m experiments.post_training.tasktrove.pipeline --version 2026.09.09 --verify-tool-ref <sha> \\
        --stage templates --run

Steps: ``raw`` downloads the parquets; ``fingerprints`` records one template id per task;
``templates`` extracts one exemplar per template and writes the converter coverage; ``converted``
applies the registered converters; ``deduped`` drops repeated instructions; ``verified`` throws
away tasks whose grader does not hold up; ``clean`` assembles the output.
"""

from dataclasses import dataclass

import click
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from marin.experiment.data import hf_download

from experiments.post_training.tasktrove.clean import build_clean
from experiments.post_training.tasktrove.convert import convert_tasks
from experiments.post_training.tasktrove.dedup import dedup_tasks
from experiments.post_training.tasktrove.fingerprint import TASKS_GLOB, build_template_index, fingerprint_tasks
from experiments.post_training.tasktrove.sources import TASKTROVE_HF_ID, TASKTROVE_REVISION
from experiments.post_training.tasktrove.verify import verify_tasks

STAGES = ("raw", "fingerprints", "templates", "converted", "deduped", "verified", "clean")


@dataclass(frozen=True)
class TaskTroveWorkflow:
    raw: ArtifactStep
    fingerprints: ArtifactStep
    templates: ArtifactStep
    converted: ArtifactStep
    deduped: ArtifactStep
    verified: ArtifactStep
    clean: ArtifactStep


def build_workflow(tool_ref: str, max_tasks_per_source: int | None) -> TaskTroveWorkflow:
    coordinator = ResourceConfig.with_cpu(cpu=4, ram="16g")
    raw = hf_download("raw/tasktrove", hf_id=TASKTROVE_HF_ID, revision=TASKTROVE_REVISION, urls_glob=(TASKS_GLOB,))
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
        templates_path=templates,
        output_path=OUT,
        tool_ref=tool_ref,
    )
    deduped = apply(
        "tasktrove/deduped",
        remote(dedup_tasks, resources=coordinator),
        converted_path=converted,
        output_path=OUT,
        max_tasks_per_source=max_tasks_per_source,
    )
    verified = apply(
        "tasktrove/verified",
        remote(verify_tasks, resources=coordinator),
        deduped_path=deduped,
        output_path=OUT,
    )
    clean = apply(
        "tasktrove/clean",
        remote(build_clean, resources=coordinator),
        deduped_path=deduped,
        verified_path=verified,
        output_path=OUT,
        tool_ref=tool_ref,
        artifact_type=Artifact,
    )
    return TaskTroveWorkflow(raw, fingerprints, templates, converted, deduped, verified, clean)


@click.command(help=__doc__)
@click.option("--stage", type=click.Choice(STAGES), default="clean", show_default=True)
@click.option("--verify-tool-ref", required=True, help="git ref of lib/tasktrove-verify baked into every Dockerfile")
@click.option("--max-tasks-per-source", type=int, default=None, help="seeded cap per source; default keeps all")
@build_options
def main(stage: str, verify_tool_ref: str, max_tasks_per_source: int | None) -> ArtifactStep:
    return getattr(build_workflow(verify_tool_ref, max_tasks_per_source), stage)


if __name__ == "__main__":
    main()
