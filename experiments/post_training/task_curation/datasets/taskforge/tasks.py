# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Taskforge testgen20261013: generated tasks across four catalog subjects.

The release (``release.py``) freezes the run's accepted tasks, one file per subject and acceptance
band. Each row carries the TaskSpec Taskforge built at schema 0.25; conversion upgrades it to the
current schema without changing its prompt, resources or grader. The run validated every task with
eight solver attempts, two adversaries and the author's grader controls; ``too-easy`` sources hold
the tasks every solver attempt passed, which this run accepted by policy.

The author's known-correct reference control becomes the golden control: a final reply, or workspace
files. A reference reached through shell calls cannot be replayed as one submission, so those tasks
check their grader on an empty submission.
"""

import json
from collections.abc import Mapping
from typing import Any

from taskcompendium.models import CommandSemantics, TaskResource, TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import (
    Controls,
    ControlSubmission,
    ImportRejection,
    IntendedUse,
    RawRow,
    Reply,
    WorkspaceFiles,
)
from taskcompendium.runtime.resources import inline_resource, resource_bytes

from experiments.post_training.task_curation.datasets.taskforge.release import SUBJECTS, Band
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

RUN_LABEL = "testgen20261013"
RELEASE_REPO = "open-athena/taskforge-testgen20261013"
RELEASE_REVISION = "dbddda1bd387384e82e72803b2919580933f45d6"
CONVERTER_VERSION = "1"
SOURCE_SCHEMA_VERSION = "0.25"
SHELL_CAPABILITY = "shell"
REFERENCE_PATH = "taskforge/reference_control.json"
REMOVED_FIELDS = ("compatible_backends", "output_directories")

RUBRIC = """
Taskforge generated this task from a capability description and validated it before release: eight solver attempts
(solve rate), two adversaries probing grader shortcuts, and author-written positive and negative grader controls.
The grader is a private script. Hidden grader fixtures are intentionally absent from the prompt.

Judge whether the prompt fully specifies one gradable answer, whether the grader accepts every correct answer the
prompt permits, and whether its checks are scientifically correct. Generated scenarios are expected; a fictional
setting is not a defect. Difficulty, or a task every solver passed, is not a quality defect.
"""


def _upgraded_environment(environment: Mapping[str, Any]) -> dict[str, Any]:
    """Replace 0.25 backend lists with command semantics: images run Linux processes, image-less shells simulate."""
    if environment["compatible_backends"]:
        raise ValueError(f"Unexpected compatible backends {environment['compatible_backends']}")
    upgraded = {key: value for key, value in environment.items() if key != "compatible_backends"}
    if environment.get("docker_image") is not None or environment.get("packages_lock") is not None:
        upgraded["command_semantics"] = CommandSemantics.LINUX_PROCESS
    elif SHELL_CAPABILITY in environment["capabilities"]:
        upgraded["command_semantics"] = CommandSemantics.SHELL_SIMULATOR
    return upgraded


def upgraded_task(task: Mapping[str, Any]) -> dict[str, Any]:
    """A Taskforge 0.25 TaskSpec payload at the current schema, rejecting any field the upgrade would lose."""
    if task["schema_version"] != SOURCE_SCHEMA_VERSION:
        raise ValueError(f"Expected TaskSpec schema {SOURCE_SCHEMA_VERSION}, got {task['schema_version']}")
    if task["output_directories"]:
        raise ValueError("Output directories have no current TaskSpec equivalent")
    upgraded = {key: value for key, value in task.items() if key not in REMOVED_FIELDS}
    upgraded["environment_requirements"] = _upgraded_environment(task["environment_requirements"])
    grader = dict(task["grader"])
    if grader.get("environment") is not None:
        grader["environment"] = _upgraded_environment(grader["environment"])
    upgraded["grader"] = grader
    upgraded["schema_version"] = TaskSpec.model_fields["schema_version"].default
    return upgraded


def reference_control(controls: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    references = [
        control
        for control in controls
        if control["kind"] == "positive" and control["category"] == "known_correct" and control["concern"] == "reference"
    ]
    if len(references) != 1:
        raise ValueError(f"Expected one known-correct reference control, found {len(references)}")
    return references[0]["payload"]


def convert_taskforge(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    data = row.data
    task = upgraded_task(data["task"])
    reference = json.dumps(reference_control(data["controls"]), sort_keys=True).encode()
    resources = dict(task["resources"])
    resources["oracle"] = [*resources["oracle"], inline_resource(REFERENCE_PATH, reference).model_dump(mode="json")]
    tags = (
        *task["tags"],
        data["id"],
        f"capability:{data['capability']}",
        f"subject:{data['subject']}",
        f"band:{data['band']}",
    )
    return TaskSpec.model_validate(
        {**task, "id": row.id, "source": row.source.model_dump(mode="json"), "resources": resources, "tags": tags}
    )


def taskforge_golden(task: TaskSpec) -> ControlSubmission | None:
    """The author's reference: its final reply, or its workspace files; None for shell-call references."""
    resource = next(resource for resource in task.resources.oracle if resource.path == REFERENCE_PATH)
    payload = json.loads(resource_bytes(resource))
    if "files" in payload:
        files = (TaskResource.model_validate(entry) for entry in payload["files"])
        return WorkspaceFiles({f"/{file.path}": resource_bytes(file) for file in files})
    turns = payload["turns"]
    if len(turns) != 1 or turns[0]["type"] != "message":
        return None
    return Reply(TextMessage(role="assistant", content=turns[0]["content"]))


TASKFORGE_CONTROLS = Controls(golden=taskforge_golden)


def _source(subject: str, band: Band) -> RlDataSource[CurationRecipe]:
    group = subject if band == Band.CALIBRATED else f"{subject}-{band}"
    band_note = (
        "Tasks inside the run's solve-rate band (1 to 7 of 8 solver attempts passed)."
        if band == Band.CALIBRATED
        else "Tasks every solver attempt passed; accepted by this run's policy, kept apart for mixing decisions."
    )
    return RlDataSource(
        pipeline=process_rows,
        info=SourceInfo(
            id=f"taskforge:{group}:{RUN_LABEL}",
            title=f"Taskforge {RUN_LABEL} · {SUBJECTS[subject]}{' · too easy' if band == Band.TOO_EASY else ''}",
            origin="Taskforge",
            family=f"taskforge-{subject}",
            tags=("rlvr", "generated", "multi-turn", "shell", f"subject:{subject}", f"band:{band}"),
            notes=band_note,
        ),
        config=CurationRecipe(
            name=f"taskforge-{RUN_LABEL}-{group}",
            source=HfSource(RELEASE_REPO, RELEASE_REVISION, (f"tasks/{group}.jsonl",), SourceFormat.JSONL),
            convert=convert_taskforge,
            version=CONVERTER_VERSION,
            intended_use=IntendedUse.TRAIN,
            rubric=RUBRIC,
            controls=TASKFORGE_CONTROLS,
        ),
    )


def sources() -> list[RlDataSource[CurationRecipe]]:
    return [_source(subject, band) for subject in SUBJECTS for band in Band]
