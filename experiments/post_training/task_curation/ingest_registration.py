# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read Atlas ingest qualifications through the existing retained-source binding."""

from dataclasses import replace
from pathlib import Path

from finestore.rl.curation import TaskFormat
from pydantic import BaseModel, ConfigDict

from experiments.post_training.task_curation.ingest_stages.converter_binding import load_converter
from experiments.post_training.task_curation.ingest_stages.ingest_config import IngestConfig, IngestMode
from experiments.post_training.task_curation.oversight_binding import bind_derived_oversight
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.receipt_shards import receipts_from_documents
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.remediation_binding import RetainedReleaseReceipt, bind_retained_release
from experiments.post_training.task_curation.sample_qc_binding import (
    SampleQCReceipt,
    bind_sample_result,
    require_sample_qualification,
)
from experiments.post_training.task_curation.source import RlDataSource

REGISTRATION_DIRECTORY = Path(__file__).parent / "datasets/ingest_admissions"


class IngestAdmission(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    policy: IngestConfig
    retained: RetainedReleaseReceipt
    sample: SampleQCReceipt | None = None


def original_source(record: IngestAdmission) -> RlDataSource[CurationRecipe]:
    policy = record.policy
    binding = load_converter(policy.converter_module)
    recipe = CurationRecipe(
        name=policy.source_id,
        source=HfSource(
            repo=policy.upstream.repository,
            revision=policy.upstream.revision,
            files=(policy.upstream.path,),
            format=binding.source_format,
        ),
        convert=binding.convert,
        version=record.retained.original_pipeline.version,
        intended_use=policy.intended_use[0],
        controls=binding.controls,
        grader=binding.grader,
    )
    return RlDataSource(info=record.retained.original_info, config=recipe, pipeline=process_rows)


def qualified_source(record: IngestAdmission) -> RlDataSource:
    if record.source_id != record.policy.source_id or record.source_id != record.retained.source_id:
        raise ValueError("Ingest qualification contains another source identity")
    original = original_source(record)
    retained = bind_retained_release(original, record.retained)
    # A conversion's original blend format is provenance; its private published
    # population now consists of verified TaskSpec rows.
    retained = replace(retained, info=replace(retained.info, task_format=TaskFormat.TASKSPEC, oversight=None))
    if record.sample is not None:
        require_sample_qualification(record.sample)
        retained = bind_sample_result(retained, record.sample)
    return bind_derived_oversight(retained, record.sample, None)


def registered_sources(sources: dict[str, RlDataSource]) -> dict[str, RlDataSource]:
    """Apply source-specific immutable qualifications after the legacy receipt chains."""
    documents = {path.name: read_registry_json(path) for path in sorted(REGISTRATION_DIRECTORY.glob("*.json"))}
    records = receipts_from_documents(documents, IngestAdmission, "ingest admission")
    result = dict(sources)
    for source_id, record in records.items():
        if record.policy.mode is IngestMode.INGEST and source_id in result:
            raise ValueError("An ingested source now conflicts with another Atlas declaration")
        if record.policy.mode is IngestMode.REPAIR:
            previous = result.get(source_id)
            dataset = None if previous is None else previous.dataset
            if dataset is None or (dataset.name, dataset.revision) != (
                record.policy.upstream.repository,
                record.policy.upstream.revision,
            ):
                raise ValueError("Repair qualification no longer matches its original Atlas pin")
        result[source_id] = qualified_source(record)
    return result
