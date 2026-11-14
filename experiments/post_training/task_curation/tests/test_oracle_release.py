# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Filtration changes the active population without inheriting its old admission."""

import hashlib
import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import ContentEvidenceReference, OracleFiltrationEvidence, SourceDeclaration
from pydantic import TypeAdapter
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.models import IntendedUse
from taskcompendium.pipeline.oracle_filter import MechanicalOracleTaskReceipt, OracleFilterReceipt

from experiments.post_training.task_curation.oracle_release import (
    OracleFilteredReleaseReceipt,
    bind_oracle_filtered_release,
    verify_oracle_release,
)
from experiments.post_training.task_curation.retained_release import RETAINED_IMPORT_VERSION
from experiments.post_training.task_curation.source import SourceInfo
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source
from experiments.post_training.task_curation.tests.prebuilt_fixtures import prebuilt_task


@pytest.fixture
def zero_release(tmp_path):
    source = number_source(tmp_path / "input")
    release = source.current_release.model_copy(
        update={
            "repo": "open-athena/filtered",
            "revision": "f" * 40,
            "url": "https://huggingface.co/datasets/open-athena/filtered/tree/" + "f" * 40,
        }
    )
    evidence = OracleFiltrationEvidence(
        original_release=source.current_release,
        original_cohort=ContentEvidenceReference(uri="s3://test/original.parquet", sha256="a" * 64),
        original_count=source.info.within_source_dedup.unique_count,
        oracle_filter=ContentEvidenceReference(uri="s3://test/oracle-filter.json", sha256="b" * 64),
        oracle_filter_version=3,
        filtered_release=release,
        private=True,
        tasks_parquet=ContentEvidenceReference(uri="s3://test/tasks.parquet", sha256="c" * 64),
        admitted_count=0,
        exclusion_counts={"unpinned_environment": source.info.within_source_dedup.unique_count},
        validation=ContentEvidenceReference(uri="s3://test/validation.json", sha256="d" * 64),
    )
    dedup = source.info.within_source_dedup.model_copy(
        update={
            "source_dataset_id": release.repo,
            "source_revision": release.revision,
            "converter_version": RETAINED_IMPORT_VERSION,
            "raw_count": 0,
            "unique_count": 0,
            "within_source_duplicates": 0,
        }
    )
    receipt = OracleFilteredReleaseReceipt(
        source_id=source.info.id,
        original_pipeline=source.pipeline_reference,
        original_info=source.info,
        filtration=evidence,
        output_dedup=dedup,
    )
    return source, receipt


@pytest.mark.parametrize("intended_use", [IntendedUse.TRAIN, IntendedUse.EVAL])
def test_zero_filtered_pin_is_visible_held_and_preserves_prior_retention(zero_release, intended_use):
    source, receipt = zero_release
    source = replace(source, config=replace(source.config, intended_use=intended_use))
    bound = bind_oracle_filtered_release(source, receipt)
    declaration = bound.catalog_declaration
    assert declaration is not None
    assert declaration.current_release == receipt.filtration.filtered_release
    assert declaration.task_count == 0 and declaration.status == "Excluded"
    assert bound.config.intended_use is intended_use
    assert declaration.daytona_safety is None and declaration.actor_census_input is None
    assert bound.info.oversight is None and bound.info.lexical_contamination is None
    assert any("oracle filtration retained no tasks" in reason for reason in bound.admission_blockers)
    restored = SourceDeclaration.model_validate_json(declaration.model_dump_json())
    assert restored.oracle_filtration == receipt.filtration
    with pytest.raises(ValueError, match="cannot remain admitted"):
        SourceDeclaration.model_validate({**declaration.model_dump(), "status": "Available"})


def test_filter_binding_rejects_changed_configured_input(zero_release):
    source, receipt = zero_release
    changed = replace(source, config=replace(source.config, version="another-converter"))
    with pytest.raises(ValueError, match="configured conversion"):
        bind_oracle_filtered_release(changed, receipt)


def test_existing_source_records_omit_absent_filtration(zero_release):
    source, _ = zero_release
    assert "oracle_filtration" not in source.declaration.model_dump(mode="json")
    original = TypeAdapter(SourceInfo).dump_python(source.info, mode="json")
    assert "oracle_filtration" not in original
    assert TypeAdapter(SourceInfo).validate_python(original) == source.info


def test_physical_oracle_publication_requires_exact_original_tasks(tmp_path, zero_release):
    _, record = zero_release
    task = TaskSpec.model_validate(json.loads(prebuilt_task("original")["lowered_task_spec"])["task"])
    row = {"id": task.id, "task_spec": task.model_dump_json(), "runtime": "{}"}
    original, output, filtered = (tmp_path / name for name in ("original.parquet", "tasks.parquet", "oracle.json"))
    table = pa.Table.from_pylist([row])
    pq.write_table(table, original)
    pq.write_table(table.slice(0, 0), output)
    oracle = OracleFilterReceipt(
        recipe_identity=None,
        input_cohort_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        input_rows=1,
        conversion_rejections=0,
        tasks={
            task.id: MechanicalOracleTaskReceipt(
                task_sha256=canonical_sha256(task.model_dump(mode="json")),
                runtime_sha256=canonical_sha256({}),
                has_oracle=False,
                exclusion_reason="unpinned_environment",
                evidence=(ContentEvidenceReference(uri="s3://test/proof", sha256="e" * 64),),
            )
        },
    )
    filtered.write_text(oracle.model_dump_json())

    def reference(path):
        return ContentEvidenceReference(uri=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    evidence = OracleFiltrationEvidence.model_validate(
        {
            **record.filtration.model_dump(),
            "original_count": 1,
            "exclusion_counts": {"unpinned_environment": 1},
            "original_cohort": reference(original),
            "oracle_filter": reference(filtered),
            "tasks_parquet": reference(output),
        }
    )
    observed = verify_oracle_release(evidence, original, filtered, output)
    assert observed.tasks[task.id].exclusion_reason == "unpinned_environment"
    changed = task.model_copy(update={"id": "substituted"})
    pq.write_table(pa.Table.from_pylist([{**row, "id": changed.id, "task_spec": changed.model_dump_json()}]), original)
    changed_sha = hashlib.sha256(original.read_bytes()).hexdigest()
    oracle = oracle.model_copy(update={"input_cohort_sha256": changed_sha})
    filtered.write_text(oracle.model_dump_json())
    changed_evidence = evidence.model_copy(
        update={"original_cohort": reference(original), "oracle_filter": reference(filtered)}
    )
    with pytest.raises(ValueError, match="coverage"):
        verify_oracle_release(changed_evidence, original, filtered, output)
