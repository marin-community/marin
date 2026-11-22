# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned ScienceAgentBench Verified annotations and paired actor inputs.

Conversion requires prepared images and an audited replay/grader package.
Registration additionally requires the official Pi evaluation route.
"""

import json
from collections.abc import Iterator
from enum import StrEnum
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict
from rigging.filesystem.storage_path import StoragePath

from taskcompendium.grader import GraderPackage
from taskcompendium.importers.fewshot import parquet_documents
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    PrivateVerifierResource,
    ResourceGroups,
    ScienceAgentBenchFile,
    ScienceAgentBenchVerifierFile,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.inputs import ConversionContext, SourceFiles, SourceFormat
from taskcompendium.runtime.resources import inline_resource

ANNOTATION_REVISION = "9c6e96c9e74572e979b0930ee735041cef528cb7"
ANNOTATION_FILE = "data/verified-00000-of-00001.parquet"
VERIFIED_INSTANCE_IDS = frozenset(range(1, 103))
WORKING_DIRECTORY = "/testbed"
SUBMISSION_PATH = "/testbed/solution.py"
REFERENCE_PATH = "/controls/reference.py"


class KnowledgeCondition(StrEnum):
    ON = "knowledge-on"
    OFF = "knowledge-off"


class VerifiedAnnotation(BaseModel):
    """Released annotation fields; reference metadata stays outside the actor prompt."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    instance_id: int
    domain: str
    subtask_categories: str
    github_name: str
    task_inst: str
    domain_knowledge: str | None
    dataset_folder_tree: str
    dataset_preview: str | None
    src_file_or_path: str | None
    gold_program_name: str
    output_fname: str
    eval_script_name: str


def read_verified_annotations(path: StoragePath, _context: ConversionContext) -> Iterator[dict[str, Any]]:
    """Read the complete pinned panel without sampling, rewriting, or dropping rows.

    Invalid input yields no rows, so a failed import cannot publish a partial panel.
    """
    rows = list(parquet_documents(path))
    annotations = tuple(VerifiedAnnotation.model_validate(row) for row in rows)
    identities = tuple(annotation.instance_id for annotation in annotations)
    if len(identities) != len(set(identities)) or set(identities) != VERIFIED_INSTANCE_IDS:
        raise ValueError("ScienceAgentBench Verified requires each instance ID 1 through 102 exactly once")
    yield from rows


VERIFIED_SOURCE = SourceFiles(
    dataset="osunlp/ScienceAgentBench",
    revision=ANNOTATION_REVISION,
    patterns=(ANNOTATION_FILE,),
    format=SourceFormat.PARQUET,
    read=read_verified_annotations,
)


def dataset_directory(annotation: VerifiedAnnotation) -> str:
    """Resolve the dataset root from the released directory-tree header."""
    header = annotation.dataset_folder_tree.splitlines()[0]
    if not header.startswith("|-- ") or not header.endswith("/"):
        raise ValueError(f"Unrecognized dataset tree for instance {annotation.instance_id}")
    name = header[4:-1]
    if not name or name in {".", ".."} or len(PurePosixPath(name).parts) != 1 or "/" in name:
        raise ValueError(f"Invalid dataset directory for instance {annotation.instance_id}")
    return f"benchmark/datasets/{name}"


@cache
def _dataset_manifest() -> dict[str, tuple[ScienceAgentBenchFile, ...]]:
    path = Path(__file__).with_name("scienceagentbench_assets") / "dataset_manifest.json"
    manifest = json.loads(path.read_text())
    return {
        root: tuple(ScienceAgentBenchFile.model_validate(row) for row in rows)
        for root, rows in manifest["datasets"].items()
    }


def dataset_files(annotation: VerifiedAnnotation) -> tuple[ScienceAgentBenchFile, ...]:
    """Select the pinned files from the task's released dataset directory."""
    root = dataset_directory(annotation)
    files = _dataset_manifest()[root]
    if not files or any(not file.member.startswith(root + "/") for file in files):
        raise ValueError(f"Invalid input manifest for {root}")
    return files


class PublishedEvaluator(BaseModel):
    """Publisher assets and scoring method for one task."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    evaluator: ScienceAgentBenchVerifierFile
    gold: tuple[ScienceAgentBenchVerifierFile, ...]
    visual: bool

    def resources(self) -> tuple[PrivateVerifierResource, ...]:
        return (
            PrivateVerifierResource(path="private_evaluator.py", source=self.evaluator),
            *(
                PrivateVerifierResource(path=file.member.removeprefix("benchmark/eval_programs/"), source=file)
                for file in self.gold
            ),
        )


@cache
def _verifier_manifest() -> dict[int, PublishedEvaluator]:
    path = Path(__file__).with_name("scienceagentbench_assets") / "verifier_manifest.json"
    return {int(key): PublishedEvaluator.model_validate(row) for key, row in json.loads(path.read_text()).items()}


def published_evaluator(annotation: VerifiedAnnotation) -> PublishedEvaluator:
    record = _verifier_manifest()[annotation.instance_id]
    if record.evaluator.member != f"benchmark/eval_programs/{annotation.eval_script_name}":
        raise ValueError(f"Evaluator mismatch for task {annotation.instance_id}")
    return record


def actor_context(annotation: VerifiedAnnotation, condition: KnowledgeCondition) -> ConversationInput:
    """Expose task inputs only, preserving complete instructions and previews.

    Reference filenames, source-code locations, and raw annotations are excluded.
    Image preparation must separately verify that no private metadata or expert
    text is baked into actor images or mounted into knowledge-off workspaces.
    """
    parts = [
        "Write a self-contained Python program that completes the scientific task below.",
        f"## Task\n\n{annotation.task_inst}",
    ]
    if condition is KnowledgeCondition.ON and annotation.domain_knowledge:
        parts.append(f"## Expert knowledge\n\n{annotation.domain_knowledge}")
    parts.append(
        f"## Input data\n\nThe working directory is `{WORKING_DIRECTORY}`. "
        f"The dataset is at `{dataset_directory(annotation)}` relative to it.\n\n"
        f"Directory structure:\n```\n{annotation.dataset_folder_tree}\n```"
    )
    if annotation.dataset_preview:
        parts.append(f"## Data preview\n\n{annotation.dataset_preview}")
    parts.append(
        f"## Submission\n\nSave the complete program to `{SUBMISSION_PATH}`. "
        f"It must run with `python {SUBMISSION_PATH}` from `{WORKING_DIRECTORY}` "
        f"and produce `{annotation.output_fname}` as specified in the task. "
        "Create required output directories. The submitted program will be run again in a clean environment."
    )
    return ConversationInput(events=(TextMessage(role="user", content="\n\n".join(parts)),))


def scienceagentbench_task(
    annotation: VerifiedAnnotation,
    condition: KnowledgeCondition,
    *,
    environment: EnvironmentRequirements,
    package: GraderPackage,
    reference_program: bytes | None,
    importer_revision: str,
) -> TaskSpec:
    """Convert a Verified row with prepared task-specific data and grading assets.

    The actor receives scientific software and the declared input files. Replay
    uses the same environment, and the package owns the evaluator, gold assets,
    output decoding, and judge contract. Reference validation can copy
    ``REFERENCE_PATH`` to ``SUBMISSION_PATH`` before using ordinary grading.
    """
    grader = package.grader
    if not isinstance(grader, ScriptGrader) or grader.replay is None:
        raise ValueError("ScienceAgentBench requires isolated replay before script grading")
    if (environment.docker_image is None and environment.docker_build is None) or (
        environment.working_directory != WORKING_DIRECTORY
    ):
        raise ValueError("ScienceAgentBench requires a prepared image with /testbed as its working directory")
    if grader.replay.environment != environment:
        raise ValueError("ScienceAgentBench actor and replay environments must match")
    labels = {"knowledge_condition": condition.value, "domain": annotation.domain}
    if annotation.instance_id in (53, 58):
        labels["grading_caveat"] = "official_inputs_contain_exact_target"
    return TaskSpec(
        id=f"scienceagentbench-verified/{condition.value}/{annotation.instance_id}",
        source=Source(
            dataset=VERIFIED_SOURCE.dataset,
            revision=ANNOTATION_REVISION,
            row=str(annotation.instance_id),
            importer_revision=importer_revision,
        ),
        context=actor_context(annotation, condition),
        environment_requirements=environment,
        answer_type=AnswerType.FILE,
        answer_format=PlainText(),
        output_paths=(SUBMISSION_PATH,),
        grader=grader,
        resources=ResourceGroups(
            oracle=(
                (inline_resource(REFERENCE_PATH.lstrip("/"), reference_program),)
                if reference_program is not None
                else ()
            ),
            verifier=package.resources,
        ),
        labels=labels,
    )
