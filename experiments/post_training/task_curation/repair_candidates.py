# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load FWeindel/validated-tasks repair-candidate folders as preview TaskSpecs.

A candidate folder (for example ``bugsinpy_v2`` + ``bugsinpy_v3``) ships Harbor task archives
in ``tasks.parquet``. This loader unpacks each archive and lowers it to a TaskSpec for the
repair lane's matching and comparative review. The lowering is a preview: it keeps the
archive's own ``tests/test.sh`` verifier as grading terms under ``NoGrader``, so nothing is
graded or admitted on this converter's word and every preview's grading contract carries the
same caveat. Production admission of any replaced or added task runs the normal pipeline
with a converter that owns its grader binding and controls.
"""

import logging
import tomllib
from dataclasses import dataclass

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from taskcompendium.convert.answers import unsupported
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportRejection

from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import (
    TASK_TOML,
    TEST_SH,
    archive_file,
    archive_resources,
    unpack_task_binary,
)

logger = logging.getLogger(__name__)

CANDIDATE_REPO = "FWeindel/validated-tasks"
BUGSINPY_FOLDERS = ("bugsinpy_v2", "bugsinpy_v3")
BUGSINPY_REQUIRED_GRADER_FILES = ("tests/run_test.sh", "tests/fixed-tests.tar.gz")
PREVIEW_IMPORTER_REVISION = "repair-candidate-preview-v1"

GRADER_FILES = (TEST_SH, *BUGSINPY_REQUIRED_GRADER_FILES)


class CandidateTask(BaseModel):
    """One candidate row with its preview TaskSpec and provenance fields."""

    path: str
    source_task_id: str | None
    content_sha256: str | None
    task: TaskSpec


@dataclass(frozen=True)
class CandidatePopulation:
    source_id: str
    tasks: list[CandidateTask]


def load_folder_rows(repo: str, folders: tuple[str, ...], *, revision: str | None = None) -> list[dict]:
    """Download and concatenate every folder's ``tasks.parquet`` rows at the pinned revision."""
    rows: list[dict] = []
    for folder in folders:
        path = hf_hub_download(repo, f"{folder}/tasks.parquet", repo_type="dataset", revision=revision)
        table = pq.read_table(path)
        rows.extend(table.to_pylist())
        logger.info("Loaded %d rows from %s/%s", table.num_rows, repo, folder)
    return rows


def preview_task(row: dict, *, candidate_source_id: str, revision: str) -> TaskSpec | ImportRejection:
    """Lower one unpacked FWeindel archive row into a preview TaskSpec."""
    prepared = unpack_task_binary(row, ConversionContext({}, None))
    instruction = prepared["instruction"]
    if not instruction.strip():
        return unsupported("missing_instruction", "The archive has no instruction")
    identifier = f"{candidate_source_id}:{prepared['path']}"
    source = Source(
        dataset=candidate_source_id,
        revision=revision,
        row=prepared["path"],
        importer_revision=PREVIEW_IMPORTER_REVISION,
    )
    return TaskSpec(
        id=identifier,
        source=source,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS),
        resources=archive_resources(prepared),
        answer_type=AnswerType.FILE,
        answer_format=PlainText(),
        grader=_preview_grader(prepared),
        tags=("repair-candidate",),
    )


def _preview_grader(prepared: dict) -> NoGrader:
    """Keep the archive's own verifier terms; qualification is never this preview's word."""
    limits = _task_toml_limits(prepared)
    verifier = limits.get("verifier", {})
    return NoGrader(
        reason=(
            "Repair-candidate preview: the archive's own tests/test.sh verifier is retained as "
            "grading terms; a runnable grader binding is the normal admission pipeline's work."
        ),
        contract={
            "evaluator": "archive_test_sh",
            "required_files": list(GRADER_FILES),
            "verifier_timeout_sec": verifier.get("timeout_sec"),
        },
    )


def _task_toml_limits(prepared: dict) -> dict:
    encoded = archive_file(prepared, TASK_TOML)
    if encoded is None:
        return {}
    return tomllib.loads(encoded.decode())


def load_candidate_population(
    *,
    folders: tuple[str, ...] = BUGSINPY_FOLDERS,
    repo: str = CANDIDATE_REPO,
    revision: str,
    candidate_source_id: str,
) -> CandidatePopulation:
    """Load one FWeindel folder pair as preview TaskSpecs, retaining per-row rejections."""
    tasks: list[CandidateTask] = []
    for row in load_folder_rows(repo, folders, revision=revision):
        lowered = preview_task(row, candidate_source_id=candidate_source_id, revision=revision)
        if isinstance(lowered, ImportRejection):
            logger.warning("Candidate row rejected in preview: %s %s", row.get("path"), lowered.reason)
            continue
        tasks.append(
            CandidateTask(
                path=row["path"],
                source_task_id=row.get("source_task_id"),
                content_sha256=row.get("content_sha256"),
                task=lowered,
            )
        )
    logger.info("Preview-loaded %d candidate tasks from %s", len(tasks), candidate_source_id)
    return CandidatePopulation(source_id=candidate_source_id, tasks=tasks)
