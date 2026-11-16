# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the source-status taxonomy, exclusion claims and the status extras artifact."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from finestore.rl.curation import OversightEvidence, OversightLevel

from experiments.post_training.task_curation import source_status
from experiments.post_training.task_curation.exclusion_registry import MANUAL_EXCLUSION_CLAIMS, NEEDS_RULING_SOURCES
from experiments.post_training.task_curation.metadata_exclusions import metadata_exclusions
from experiments.post_training.task_curation.source import lexical_blocker_reason, population_blocker_reason
from experiments.post_training.task_curation.sources import PENDING_MEASUREMENT_SOURCES, all_sources
from finestore.rl.status_taxonomy import (
    ExclusionReason,
    REASON_EXPLANATIONS,
    SourceStatus,
)
from infra.marina.apps.rl_data_catalog.server.licensing import source_license_ban

APP_DIST = Path("infra/marina/apps/rl_data_catalog/dist")
EXTRAS_PATH = APP_DIST / "measurements" / "source-status-extras.json"
LICENSE_CHECKS = Path(__file__).parents[1] / "tests" / "fixtures" / "license_checks.json"

# Every admission blocker code path maps to one reason; the real-catalog test
# below asserts each emitted blocker's reason appears among the row's claims.
BLOCKER_REASON_RULES = (
    ("license ban: ", ExclusionReason.LICENSE_BAN),
    ("oracle filtration retained no tasks: ", ExclusionReason.ORACLE_FILTRATION_RETAINED_ZERO),
    ("actor census does not cover the current configured input", ExclusionReason.ACTOR_CENSUS_STALE),
    ("actor census does not cover the current retained release", ExclusionReason.ACTOR_CENSUS_STALE),
    ("admission requires a nonempty retained population", ExclusionReason.RETAINED_POPULATION_EMPTY),
    ("actor census does not cover the retained population", ExclusionReason.ACTOR_CENSUS_STALE),
    ("grading census does not cover the complete retained population", ExclusionReason.GRADING_CENSUS_INCOMPLETE),
    (
        "grading census counts do not cover the complete retained population",
        ExclusionReason.GRADING_CENSUS_COUNTS_INCOMPLETE,
    ),
    ("admitted source has tasks without runnable graders", ExclusionReason.GRADER_MISSING),
    (
        "Evaluation requires a measured lexical receipt for the current dataset pin",
        ExclusionReason.LEXICAL_RECEIPT_MISSING,
    ),
    ("Decontamination", ExclusionReason.LEXICAL_REMEDIATION_REQUIRED),
    ("Atlas ", ExclusionReason.LEXICAL_REMEDIATION_REQUIRED),
    ("actor census evidence is missing", ExclusionReason.ACTOR_CENSUS_MISSING),
    ("evidence-derived SAMPLE or FULL oversight is missing", ExclusionReason.OVERSIGHT_MISSING),
    ("accepted agentic task format is missing", ExclusionReason.AGENTIC_TASK_FORMAT_MISSING),
    ("production snapshot identity measurement is missing", ExclusionReason.SNAPSHOT_IDENTITY_MISSING),
    ("execution requirement measurement is missing", ExclusionReason.EXECUTION_REQUIREMENT_MISSING),
)

LEXICAL_MESSAGES = {
    "Evaluation requires a measured lexical receipt for the current dataset pin": (
        ExclusionReason.LEXICAL_RECEIPT_MISSING
    ),
    "Decontamination measurement and private-release admission receipt are missing": (
        ExclusionReason.LEXICAL_REMEDIATION_REQUIRED
    ),
    "Decontamination admission blocked: clean or remediated decontamination receipt is missing": (
        ExclusionReason.LEXICAL_REMEDIATION_REQUIRED
    ),
    "Decontamination admission blocked: benchmark measurement does not certify a training release": (
        ExclusionReason.LEXICAL_REMEDIATION_REQUIRED
    ),
    "Decontamination admission blocked: decontamination receipt belongs to a different dataset pin": (
        ExclusionReason.LEXICAL_REMEDIATION_REQUIRED
    ),
    "Atlas admission requires a private open-athena TaskCompendium release": (
        ExclusionReason.LEXICAL_REMEDIATION_REQUIRED
    ),
    "Atlas remediation requires the original private open-athena snapshot": (
        ExclusionReason.LEXICAL_REMEDIATION_REQUIRED
    ),
}


@pytest.mark.data_integration
def test_selection_semantics_do_not_change():
    """The refactor folds Not selectable into Excluded; Available stays exactly the same set."""
    before = {}
    after = {}
    for source in all_sources().values():
        info = source.info
        dataset = source.dataset
        ban = source_license_ban(info.id, dataset.name if dataset else None, info.license_receipt)
        blockers = source.admission_blockers
        before[info.id] = (
            "Excluded"
            if ban is not None
            else next(
                (label for tag, label in (("deprecated", "Deprecated"), ("excluded", "Excluded")) if tag in info.tags),
                "Not selectable" if blockers else "Available",
            )
        )
        after[info.id] = source.status.value
    assert {key for key, value in before.items() if value == "Available"} == {
        key for key, value in after.items() if value == "Available"
    }
    assert set(before.values()) <= {"Available", "Deprecated", "Excluded", "Not selectable"}
    assert set(after.values()) <= {status.value for status in SourceStatus}


@pytest.mark.data_integration
def test_every_excluded_catalog_row_carries_evidenced_claims():
    for source in all_sources().values():
        declaration = source.declaration
        if declaration.status != SourceStatus.EXCLUDED:
            continue
        assert declaration.exclusion_claims, f"Excluded source without a claim: {declaration.id}"
        for claim in declaration.exclusion_claims:
            assert claim.evidence.message.strip(), f"Evidence without a message: {declaration.id}"


@pytest.mark.data_integration
def test_enum_covers_every_blocker_code_path():
    for source in all_sources().values():
        declaration = source.declaration
        reasons = {claim.reason for claim in declaration.exclusion_claims}
        for blocker in source.admission_blockers:
            expected = next(
                (reason for prefix, reason in BLOCKER_REASON_RULES if blocker.startswith(prefix)),
                None,
            )
            assert expected is not None, f"Unmapped blocker code path: {blocker}"
            assert expected in reasons, f"Blocker {blocker!r} lacks its {expected} claim: {declaration.id}"


@pytest.mark.data_integration
def test_every_manual_exclusion_registry_row_is_excluded_with_its_claims():
    for source in all_sources().values():
        claims = MANUAL_EXCLUSION_CLAIMS.get(source.info.id)
        if claims is None:
            continue
        declaration = source.declaration
        assert declaration.status == SourceStatus.EXCLUDED, declaration.id
        for claim in claims:
            assert claim in declaration.exclusion_claims, declaration.id


@pytest.mark.data_integration
def test_pending_rows_cover_the_intake_queue():
    rows = source_status.pending_rows()
    assert {row["id"] for row in rows} == {source.info.id for source in PENDING_MEASUREMENT_SOURCES}
    for row in rows:
        assert row["status"] == SourceStatus.PENDING
        assert row["stage"] in {stage.value for stage in source_status.PendingStage}


def test_blocker_reason_mappings_are_total():
    for blocker, reason in {
        "admission requires a nonempty retained population": ExclusionReason.RETAINED_POPULATION_EMPTY,
        "actor census does not cover the retained population": ExclusionReason.ACTOR_CENSUS_STALE,
        "grading census does not cover the complete retained population": ExclusionReason.GRADING_CENSUS_INCOMPLETE,
        "grading census counts do not cover the complete retained population": (
            ExclusionReason.GRADING_CENSUS_COUNTS_INCOMPLETE
        ),
        "admitted source has tasks without runnable graders": ExclusionReason.GRADER_MISSING,
    }.items():
        assert population_blocker_reason(blocker) is reason
    with pytest.raises(ValueError, match="Unmapped population admission blocker"):
        population_blocker_reason("an unmapped blocker")
    for message, reason in LEXICAL_MESSAGES.items():
        assert lexical_blocker_reason(message) is reason
    with pytest.raises(ValueError, match="Unmapped lexical qualification failure"):
        lexical_blocker_reason("an unmapped lexical failure")


def test_every_reason_has_one_canonical_explanation():
    assert set(REASON_EXPLANATIONS) == set(ExclusionReason)
    for explanation in REASON_EXPLANATIONS.values():
        assert explanation.label.strip()
        assert explanation.meaning.strip()
        assert explanation.clears.strip()


@pytest.mark.data_integration
def test_unspecified_legacy_is_gated_to_recorded_rulings():
    legacy = {
        source.info.id
        for source in all_sources().values()
        if any(claim.reason is ExclusionReason.UNSPECIFIED_LEGACY for claim in source.declaration.exclusion_claims)
    }
    extras = source_status.status_extras()
    legacy |= {
        row["id"]
        for row in [*extras["pending"], *extras["metadata_exclusions"]]
        if any(claim["reason"] == ExclusionReason.UNSPECIFIED_LEGACY.value for claim in row["reasons"])
    }
    assert legacy == set(NEEDS_RULING_SOURCES), "Record a user ruling and drop the row from NEEDS_RULING_SOURCES"


def test_committed_extras_match_the_static_taxonomy():
    committed = json.loads(EXTRAS_PATH.read_text())
    assert committed["reason_explanations"] == source_status.reason_explanations()
    assert committed["metadata_exclusions"] == source_status.metadata_rows()


@pytest.mark.data_integration
def test_committed_extras_match_the_full_taxonomy():
    committed = json.loads(EXTRAS_PATH.read_text())
    assert committed == source_status.status_extras()


def test_metadata_exclusion_pins_match_the_license_check_fixtures():
    fixtures = json.loads(LICENSE_CHECKS.read_text())
    by_id = {row["source_id"]: row for row in fixtures}
    pins = {
        "GPUMODE:kernelbot-data": ("P0:GPUMODE/kernelbot-data", "E1"),
        "CyBench:andyzorigin-cybench": ("P0:andyzorigin/cybench", "F1"),
        "HAL:princeton-pli-hal-harness": ("P0:princeton-pli/hal-harness", "F2"),
        "MiroVerse:miromind-ai-MiroVerse-v0.1": ("P0:miromind-ai/MiroVerse-v0.1", "F1"),
    }
    for row_id, (fixture_id, rule) in pins.items():
        row = metadata_exclusions()[row_id]
        fixture = by_id[fixture_id]
        assert row.dataset_revision == fixture["revision"]
        assert rule in row.claims[0].evidence.message


def test_metadata_exclusions_carry_evidence_and_never_ingest():
    for row in metadata_exclusions().values():
        assert row.claims, f"Metadata exclusion without a claim: {row.id}"
        for claim in row.claims:
            assert claim.evidence.message.strip()
            assert claim.evidence.uri, f"Metadata exclusion evidence without a link: {row.id}"


def test_pending_stage_derivation():
    receipts = {"actor census": {"a"}, "dedup": {"a", "b"}}
    assert source_status.pending_stage("a", receipts) is source_status.PendingStage.ADMISSION_PENDING
    assert source_status.pending_stage("b", receipts) is source_status.PendingStage.MEASUREMENTS_PENDING
    assert source_status.pending_stage("c", receipts) is source_status.PendingStage.INTAKE_QUEUED


def test_status_enum_semantics(tmp_path):
    from experiments.post_training.task_curation.tests.admission_fixtures import (  # noqa: PLC0415
        with_admission_census,
    )
    from experiments.post_training.task_curation.tests.numbers_pipeline import number_source  # noqa: PLC0415

    source = number_source(tmp_path / "input")
    assert source.status is SourceStatus.EXCLUDED
    source = replace(
        source,
        info=replace(
            source.info,
            oversight=OversightEvidence(
                level=OversightLevel.SAMPLE,
                campaign="fixture",
                verification="Fixture ingestion and grader controls executed.",
            ),
        ),
    )
    source = with_admission_census(source)
    assert source.status is SourceStatus.AVAILABLE
    deprecated = replace(source, info=replace(source.info, tags=(*source.info.tags, "deprecated")))
    assert deprecated.status is SourceStatus.DEPRECATED
    tagged = replace(source, info=replace(source.info, tags=(*source.info.tags, "excluded")))
    with pytest.raises(ValueError, match="without a typed exclusion claim"):
        _ = tagged.status
