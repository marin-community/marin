# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A ledger-declared taskspec pin binds census, lexical and grading, and audits clean.

Coordinator ruling 2026-11-25 ~23:35 EST (the declared-pin re-pin route): the harbor
cohorts' retirement-ledger entries move to the taskspec conversion releases, the
declaration carries the conversion identity from the entry, and the census, lexical and
grading receipts bind at the declared pin. A ledger entry whose conversion identity
differs from the receipts fails the declaration audit.
"""

from pathlib import Path

import pytest
from finestore.rl.curation import (
    DataRestriction,
    DeconReceiptStatus,
    LexicalUse,
    OracleValidation,
    SourceOrigin,
    TaskFormat,
    WithinSourceDedupEvidence,
)

from experiments.post_training.task_curation import retirement
from experiments.post_training.task_curation.census_binding import ActorCensusReceipt, bind_actor_census
from experiments.post_training.task_curation.datasets import native_releases
from experiments.post_training.task_curation.datasets.native_releases import sources as ledger_sources
from experiments.post_training.task_curation.declaration_audit import (
    CanonicalLayer,
    DeclaredPin,
    audit_families,
    declared_pins,
    selected_pins,
)
from experiments.post_training.task_curation.grading_binding import GradingCensusReceipt
from experiments.post_training.task_curation.lexical_binding import (
    LexicalMeasurementReceipt,
    bind_lexical_measurement,
)
from experiments.post_training.task_curation.prebuilt import PrebuiltCohort
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo
from experiments.post_training.task_curation.within_source_binding import bind_within_source_dedup

pytest.importorskip("taskcompendium")

from taskcompendium.pipeline.models import IntendedUse

SOURCE_ID = "TaskTrove:fixture__ledger-cohort"
RELEASE_REPO = "open-athena/fixture__ledger-cohort-00000000-taskspec-20261124g"
RELEASE_REVISION = "a" * 40
CONVERSION_VERSION = "harbor-cohort-taskspec-v1"
CONVERTER_REVISION = "c" * 40
POPULATION_DIGEST = "d" * 40


def ledger_entry(conversion_version: str = CONVERSION_VERSION) -> dict:
    return {
        "id": SOURCE_ID,
        "name": "fixture__ledger-cohort",
        "display_name": "Fixture ledger cohort",
        "dataset_id": "fixture__ledger-cohort",
        "dataset_revision": "f" * 40,
        "url": "https://example.test/upstream",
        "origin": "Task Trove",
        "family": "ci-workflow",
        "tags": ["agentic"],
        "notes": "fixture",
        "independent_qc": None,
        "disposition": "quick_converted",
        "disposition_release": {
            "repo": RELEASE_REPO,
            "revision": RELEASE_REVISION,
            "converted_rows": 2,
            "input_rows": 2,
            "rejections": {},
            "converter_identity": CONVERTER_REVISION,
            "conversion_version": conversion_version,
            "daytona_measurement": {
                "dataset_id": RELEASE_REPO,
                "dataset_revision": RELEASE_REVISION,
                "daytona_snapshot_count": 0,
                "daytona_machine_slots": 2,
                "daytona_execution_failures": 0,
                "daytona_count_basis": "Fixture census",
                "daytona_unsafe_reasons": [],
                "daytona_unsafe_details": {},
                "daytona_unsafe_counts": {},
                "daytona_roles": {"agent": {}, "grader": {}},
            },
            "grading_environment": {
                "assessed_tasks": 2,
                "required_counts": {"separate": 2},
                "declared_counts": {"separate": 2},
                "status_counts": {"matched": 2},
                "reason_counts": {"fixture": 2},
                "evidence_url": "https://example.test/grading-census",
            },
        },
    }


def census_receipt(repo=RELEASE_REPO, revision=RELEASE_REVISION, version=CONVERSION_VERSION) -> dict:
    return {
        "source_id": SOURCE_ID,
        "input": {
            "dataset_id": repo,
            "dataset_revision": revision,
            "converter_version": version,
            "converter_revision": CONVERTER_REVISION,
            "measured_tasks": 2,
            "task_specs_sha256": POPULATION_DIGEST,
            "evidence_url": "https://example.test/census",
        },
        "measurement": {
            "dataset_id": repo,
            "dataset_revision": revision,
            "daytona_snapshot_count": 0,
            "daytona_machine_slots": 2,
            "daytona_execution_failures": 0,
            "daytona_count_basis": "Fixture census",
            "daytona_unsafe_reasons": [],
            "daytona_unsafe_details": {},
            "daytona_unsafe_counts": {},
            "daytona_roles": {"agent": {}, "grader": {}},
        },
        "integrity_contracts": None,
    }


def lexical_receipt() -> dict:
    return {
        "source_id": SOURCE_ID,
        "input": {
            "dataset_id": RELEASE_REPO,
            "dataset_revision": RELEASE_REVISION,
            "converter_version": CONVERSION_VERSION,
            "converter_revision": CONVERTER_REVISION,
            "measured_tasks": 2,
            "task_specs_sha256": POPULATION_DIGEST,
            "evidence_url": "https://example.test/lexical",
        },
        "use": "train",
        "measurement": {
            "version": 1,
            "input_dataset_id": RELEASE_REPO,
            "input_dataset_revision": RELEASE_REVISION,
            "converter_revision": CONVERTER_REVISION,
            "text_renderer_version": "taskcompendium-format-conversation-v1",
            "measured_tasks": 2,
            "contaminated_tasks": 0,
            "contaminated_fraction": 0.0,
            "max_overlap": 0.0,
            "mean_max_overlap": 0.0,
            "p95_max_overlap": 0.0,
            "matched_evals": {},
            "eval_corpus_version": "aa-index-v4.1.1-v3",
            "eval_bloom_uri": "s3://bloom/example",
            "eval_hash_index_uri": "s3://bloom/example/_bloom/eval_hash_index.parquet",
            "feature_filter_version": 4,
            "ngram_length": 13,
            "overlap_threshold": 0.5,
            "min_matched_features": 2,
            "drop_set_uris": ["s3://drops/example"],
            "task_flags_uri": "s3://flags/example",
            "evidence_url": "https://example.test/lexical-report",
        },
    }


def grading_receipt() -> dict:
    return {
        "source_id": SOURCE_ID,
        "input": census_receipt()["input"],
        "population_path": "s3://population/example",
        "scanned_task_specs_sha256": POPULATION_DIGEST,
        "evidence": ledger_entry()["disposition_release"]["grading_environment"],
        "task_evidence_uri": "s3://grading/example/tasks.parquet",
        "unassessable_reasons": {},
    }


def dedup_evidence() -> dict:
    return {
        "raw_count": 2,
        "unique_count": 2,
        "within_source_duplicates": 0,
        "source_dataset_id": RELEASE_REPO,
        "source_revision": RELEASE_REVISION,
        "converter_revision": CONVERTER_REVISION,
        "converter_version": CONVERSION_VERSION,
        "drop_manifest_uri": "s3://drops/example/empty.parquet",
        "evidence_url": "https://example.test/population",
    }


def declared_source(entry: dict) -> RlDataSource:
    release = entry["disposition_release"]
    cohort = PrebuiltCohort(
        intended_use=IntendedUse.TRAIN,
        name=entry["name"],
        version=release.get("conversion_version") or "2026.11.12.1",
        repo=release["repo"],
        revision=release["revision"],
        expected_rows=release["converted_rows"],
    )
    return RlDataSource(
        config=cohort,
        pipeline=None,
        info=SourceInfo(
            data_restriction=DataRestriction.OPEN,
            id=entry["id"],
            origin=SourceOrigin.TASKTROVE,
            upstream_dataset=entry["dataset_id"],
            upstream_name=entry["display_name"],
            upstream_url=entry["url"],
            build_path="Native Harbor QUICK conversion",
            family=entry["family"],
            tags=tuple(tag for tag in entry["tags"] if tag != "excluded"),
            task_format=TaskFormat.HARBOR,
            count=release["converted_rows"],
            dataset=cohort.dataset,
            oracle_validation=OracleValidation.SKIPPED,
            daytona_snapshot_count=release["daytona_measurement"]["daytona_snapshot_count"],
            daytona_machine_slots=release["daytona_measurement"]["daytona_machine_slots"],
            daytona_execution_failures=release["daytona_measurement"]["daytona_execution_failures"],
            daytona_count_basis=release["daytona_measurement"]["daytona_count_basis"],
            daytona_roles=release["daytona_measurement"]["daytona_roles"],
            daytona_unsafe_reasons=tuple(release["daytona_measurement"]["daytona_unsafe_reasons"]),
            daytona_unsafe_details=release["daytona_measurement"]["daytona_unsafe_details"],
            daytona_unsafe_counts=release["daytona_measurement"]["daytona_unsafe_counts"],
        ),
    )


def bound_source(conversion_version: str = CONVERSION_VERSION):
    entry = ledger_entry(conversion_version)
    source = declared_source(entry)
    actor = ActorCensusReceipt.model_validate(census_receipt(version=conversion_version))
    source = bind_actor_census(source, actor)
    source = bind_lexical_measurement(
        source,
        __import__(
            "experiments.post_training.task_curation.lexical_binding", fromlist=["LexicalMeasurementReceipt"]
        ).LexicalMeasurementReceipt.model_validate(lexical_receipt()),
        LexicalUse.TRAIN,
    )
    source = bind_within_source_dedup(
        source,
        __import__(
            "finestore.rl.curation", fromlist=["WithinSourceDedupEvidence"]
        ).WithinSourceDedupEvidence.model_validate(dedup_evidence()),
    )
    return entry, source


def test_ledger_declared_taskspec_pin_binds_and_audits_clean():
    """The ledger-declared taskspec pin binds census, lexical and grading, and the
    declaration audit carries no finding for the source."""
    _, source = bound_source()
    assert source.dataset.name == RELEASE_REPO and source.dataset.revision == RELEASE_REVISION
    assert source.version == CONVERSION_VERSION
    assert source.info.actor_census_input is not None
    assert source.info.lexical_contamination is not None
    assert source.info.within_source_dedup is not None

    grading = GradingCensusReceipt.model_validate(grading_receipt())
    assert grading.evidence.assessed_tasks == source.info.actor_census_input.measured_tasks

    layer = CanonicalLayer(
        source_ids=frozenset({SOURCE_ID}),
        declared=declared_pins([source]),
        selected=selected_pins([source]),
        retired=frozenset(),
        archived_releases={},
    )
    pin = DeclaredPin(
        source_id=SOURCE_ID,
        dataset_id=RELEASE_REPO,
        dataset_revision=RELEASE_REVISION,
        converter_version=CONVERSION_VERSION,
    )
    findings, _ = audit_families(
        {
            "census receipts": {SOURCE_ID: pin},
            "lexical receipts": {SOURCE_ID: pin},
            "grading census receipts": {SOURCE_ID: pin},
        },
        layer=layer,
    )
    assert findings == []


def test_ledger_entry_with_mismatched_conversion_identity_fails_closed():
    """A ledger entry whose conversion identity differs from the receipts refuses the
    binding chain: the dedup census does not match the declared source."""
    entry = ledger_entry(conversion_version="2026.11.12.1")
    source = declared_source(entry)
    source = bind_actor_census(source, ActorCensusReceipt.model_validate(census_receipt(version="2026.11.12.1")))
    with pytest.raises(ValueError, match="does not match configured source"):
        bind_within_source_dedup(source, WithinSourceDedupEvidence.model_validate(dedup_evidence()))


def test_ledger_sources_read_the_conversion_identity_from_the_entry(monkeypatch):
    """native_releases reads the conversion identity from the ledger entry instead of the
    hard-coded 2026.11.12.1; a taskspec entry declares the executable TaskSpec release."""

    monkeypatch.setattr(retirement, "RETIREMENTS_PATH", Path("fixture-ledger"))
    monkeypatch.setattr(native_releases, "read_registry_json", lambda path: {"entries": [ledger_entry()]})
    monkeypatch.setattr(
        native_releases,
        "lexical_measurements",
        lambda: {SOURCE_ID: LexicalMeasurementReceipt.model_validate(lexical_receipt())},
    )
    declared = {source.info.id: source for source in ledger_sources()}
    source = declared[SOURCE_ID]
    assert source.version == CONVERSION_VERSION
    assert source.dataset.name == RELEASE_REPO
    assert source.info.task_format is TaskFormat.TASKSPEC
    assert source.pipeline is native_releases.import_taskcompendium_prebuilt
    lexical = source.info.lexical_contamination
    assert lexical is not None and lexical.receipt is not None
    assert lexical.receipt.status is DeconReceiptStatus.CLEAN
    assert (lexical.receipt.release.dataset_id, lexical.receipt.release.dataset_revision) == (
        RELEASE_REPO,
        RELEASE_REVISION,
    )
    assert lexical.admission_blocker(RELEASE_REPO, RELEASE_REVISION) is None


def test_taskspec_entry_without_a_lexical_receipt_fails_closed(monkeypatch):
    """A taskspec conversion entry whose lexical measurement was never registered refuses
    the declaration instead of admitting an uncertified population."""
    monkeypatch.setattr(retirement, "RETIREMENTS_PATH", Path("fixture-ledger"))
    monkeypatch.setattr(native_releases, "read_registry_json", lambda path: {"entries": [ledger_entry()]})
    monkeypatch.setattr(native_releases, "lexical_measurements", lambda: {})
    with pytest.raises(ValueError, match="no registered lexical measurement receipt"):
        ledger_sources()


def test_taskspec_lexical_evidence_requires_a_clean_measurement():
    """A flagged population can never carry the clean release receipt."""
    receipt = LexicalMeasurementReceipt.model_validate(lexical_receipt())
    flagged = receipt.measurement.model_copy(update={"contaminated_tasks": 1, "contaminated_fraction": 0.5})
    flagged_receipt = LexicalMeasurementReceipt.model_validate(
        {**receipt.model_dump(mode="json"), "measurement": flagged.model_dump(mode="json")}
    )
    with pytest.raises(ValueError, match="not clean"):
        native_releases.taskspec_lexical_evidence(
            {"repo": RELEASE_REPO, "revision": RELEASE_REVISION}, flagged_receipt
        )
