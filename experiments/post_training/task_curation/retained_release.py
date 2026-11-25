# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import a verified retention release without changing its stored task contracts.

The Atlas seals each retained release with its census, lexical, dedup and grading
receipts in the pinned registry; those admission gates ran upstream. This reader
fails closed on the bytes (release digest, per-task identity, row count) and on
the lexical qualification the publication carries, so a run can only consume the
population the publication read back.
"""

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.curation import (
    DatasetRelease,
    LexicalContaminationEvidence,
    OversightEvidence,
    SourceUse,
)
from huggingface_hub import hf_hub_download
from marin.execution.lazy import ArtifactStep
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.audit_schema import NORMALIZED_SCHEMA
from taskcompendium.pipeline.conversion import normalization_columns
from taskcompendium.pipeline.models import IntendedUse, SourceStatus
from taskcompendium.pipeline.source_processing import SourceProcessingMode

from experiments.post_training.task_curation.admission import require_lexical_qualification
from experiments.post_training.task_curation.campaign import CampaignArtifact, PipelineResult
from experiments.post_training.task_curation.config import PipelineOptions
from experiments.post_training.task_curation.prebuilt import require_runnable_grader
from experiments.post_training.task_curation.source import EXCLUDED_TAG, RlDataSource, SourceReference

RETAINED_TASKS_FILE = "tasks.parquet"
"""The published cohort file the reader downloads and every view derives from."""
RETAINED_PART_FILE = "part-000000.parquet"
"""The single-shard output basename shared by the retained views."""
FINAL_VIEW = "final"

RETAINED_MANIFEST_FILE = "manifest.json"
"""The import manifest the writer publishes and the evidence pointer names."""
RETAINED_IMPORT_VERSION = "2026.11.13.1"
RETAINED_ADMISSION_VERSION = "2026.11.13.2"
RETAINED_SCHEMA = NORMALIZED_SCHEMA.append(pa.field("runtime", pa.large_string()))

EMPTY_EVIDENCE_TABLE = pa.table(
    {
        "task_id": pa.array([], pa.string()),
        "admission": pa.array([], pa.string()),
        "filter_reasons": pa.array([], pa.list_(pa.string())),
    }
)


@dataclass(frozen=True)
class RetainedCohort:
    name: str
    version: str
    release: DatasetRelease
    original_dataset: SourceReference
    expected_rows: int
    tasks_parquet_sha256: str
    intended_use: IntendedUse
    evidence_url: str

    @property
    def dataset(self) -> SourceReference:
        return SourceReference(self.release.repo, self.release.revision, self.release.url)

    @property
    def files(self) -> tuple[str, ...]:
        return (RETAINED_TASKS_FILE,)


@dataclass(frozen=True)
class RetainedImport:
    cohort: RetainedCohort
    output_path: str
    mode: SourceProcessingMode
    oversight: OversightEvidence | None
    inherited_admission: DatasetRelease | None
    lexical_contamination: LexicalContaminationEvidence | None


def _write_evidence_views(
    output: StoragePath, *, normalized: StoragePath, mode: SourceProcessingMode
) -> tuple[dict[str, str], dict[str, str]]:
    """Write the final, review and verify views; return the outputs and datasets entries.

    The frozen retained population is all-admitted: the review and verify views
    carry no rows, so the assessment contract still holds and its excluded cohort
    samples zero. QUICK writes an empty final view, matching its admitted count.
    """
    final_view = output / FINAL_VIEW
    final_view.mkdirs(exist_ok=True)
    outputs: dict[str, str] = {}
    admitted = mode is not SourceProcessingMode.QUICK
    if admitted:
        final = final_view / RETAINED_PART_FILE
        with normalized.open("rb") as source, final.open("wb", auto_mkdir=True) as destination:
            shutil.copyfileobj(source, destination)
        outputs[FINAL_VIEW] = str(final.parent)
    else:
        with (final_view / RETAINED_PART_FILE).open("wb") as stream:
            pq.write_table(RETAINED_SCHEMA.empty_table(), stream)
    views = {}
    for view in ("review", "verify"):
        view_dir = output / view
        view_dir.mkdirs(exist_ok=True)
        with (view_dir / RETAINED_PART_FILE).open("wb") as stream:
            pq.write_table(EMPTY_EVIDENCE_TABLE, stream)
        views[view] = str(view_dir)
    return outputs, {"normalize": str(normalized.parent), **views, FINAL_VIEW: str(final_view)}


def _retained_manifest(invocation: RetainedImport, *, seen: int, views: dict[str, str]) -> dict[str, object]:
    """The import manifest: the assessment contract plus the retention provenance."""
    cohort = invocation.cohort
    return {
        "mode": invocation.mode.value,
        "status": SourceStatus.COMPLETED.value,
        "intended_use": cohort.intended_use.value,
        "source": cohort.name,
        "datasets": views,
        "input_rows": seen,
        "converted_rows": seen,
        "admitted_rows": seen if invocation.mode is not SourceProcessingMode.QUICK else 0,
        "processed_rows": seen,
        "unprocessed_rows": 0,
        "qualification_newly_executed": False,
        "inherited_oversight": invocation.oversight.model_dump(mode="json") if invocation.oversight else None,
        "inherited_admission": (
            invocation.inherited_admission.model_dump(mode="json") if invocation.inherited_admission else None
        ),
        "original_dataset": {
            "repo": cohort.original_dataset.name,
            "revision": cohort.original_dataset.revision,
            "url": cohort.original_dataset.url,
        },
        "lexical_contamination": (
            invocation.lexical_contamination.model_dump(mode="json") if invocation.lexical_contamination else None
        ),
        "rejections": {},
        "repo": cohort.release.repo,
        "revision": cohort.release.revision,
        "tasks_parquet_sha256": cohort.tasks_parquet_sha256,
        "retention_evidence": cohort.evidence_url,
    }


def run_retained_import(invocation: RetainedImport) -> CampaignArtifact:
    """Read exact published bytes, preserving the original qualification provenance."""
    cohort = invocation.cohort
    admitted = invocation.mode is not SourceProcessingMode.QUICK
    if admitted:
        if cohort.expected_rows <= 0:
            raise ValueError("An admitted source requires a nonempty retained population")
        require_lexical_qualification(
            invocation.lexical_contamination,
            cohort.release.repo,
            cohort.release.revision,
            SourceUse(cohort.intended_use.value),
        )
    local = Path(
        hf_hub_download(cohort.release.repo, cohort.files[0], repo_type="dataset", revision=cohort.release.revision)
    )
    if file_sha256(StoragePath(str(local))) != cohort.tasks_parquet_sha256:
        raise ValueError("Retained release differs from its publication readback receipt")
    output = StoragePath(invocation.output_path)
    normalized = output / "normalize" / RETAINED_PART_FILE
    seen = set()
    with normalized.open("wb", auto_mkdir=True) as stream, pq.ParquetWriter(stream, RETAINED_SCHEMA) as writer:
        for batch in pq.ParquetFile(local).iter_batches(batch_size=16):
            records = []
            for row in batch.to_pylist():
                encoded = row["task_spec"]
                task = TaskSpec.model_validate_json(encoded)
                if admitted:
                    require_runnable_grader(task)
                if task.id in seen or hashlib.sha256(encoded.encode()).hexdigest() != row["task_sha256"]:
                    raise ValueError(f"Repeated or changed retained task: {task.id}")
                seen.add(task.id)
                records.append(
                    {
                        **normalization_columns(task.id, task.source, task, None, (), task.source.row),
                        "task_json": encoded,
                        "runtime": row["runtime"],
                    }
                )
            writer.write_table(pa.Table.from_pylist(records, schema=RETAINED_SCHEMA))
    if len(seen) != cohort.expected_rows:
        raise ValueError("Retained release count differs from its frozen plan")
    (output / RETAINED_TASKS_FILE).upload_from(str(local))
    outputs = {"tasks": str(output / RETAINED_TASKS_FILE), "normalize": str(normalized.parent)}
    view_outputs, views = _write_evidence_views(output, normalized=normalized, mode=invocation.mode)
    outputs.update(view_outputs)
    manifest = _retained_manifest(invocation, seen=len(seen), views=views)
    (output / RETAINED_MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")
    return CampaignArtifact(
        status="completed",
        result=PipelineResult(
            status=SourceStatus.COMPLETED,
            outputs=outputs,
            evidence={"manifest": str(output / RETAINED_MANIFEST_FILE)},
            stages=("download", "validate", "normalize"),
        ),
    )


def import_retained(source: RlDataSource[RetainedCohort], options: PipelineOptions) -> ArtifactStep[CampaignArtifact]:
    """Import one retained cohort for selection, after the declaration's license gate."""
    source.require_license()
    if options.mode is not SourceProcessingMode.QUICK:
        if EXCLUDED_TAG in source.info.tags:
            raise ValueError(f"Retained cohort is excluded pending qualification: {source.name}")
    config = source.config
    assert config is not None

    def build_config(context) -> RetainedImport:
        return RetainedImport(
            config,
            context.output_path,
            options.mode,
            source.info.oversight,
            source.info.inherited_admission,
            source.info.lexical_contamination,
        )

    return ArtifactStep(
        name=f"retained/{options.mode.value}/{config.release.repo}/{config.release.revision}",
        version=RETAINED_IMPORT_VERSION if options.mode is SourceProcessingMode.QUICK else RETAINED_ADMISSION_VERSION,
        artifact_type=CampaignArtifact,
        run=run_retained_import,
        build_config=build_config,
        deps=(),
    )
