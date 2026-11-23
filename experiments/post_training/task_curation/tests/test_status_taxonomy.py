# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the source-status taxonomy, exclusion claims and the status extras artifact."""

import ast
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
from finestore.rl.curation import OversightEvidence, OversightLevel, SourceUse, TaskFormat, require_typed_status_fields
from finestore.rl.difficulty import DIFFICULTY_GAPS
from finestore.rl.status_taxonomy import (
    FORMAT_EXPLANATIONS,
    METHOD_EXPLANATIONS,
    REASON_EXPLANATIONS,
    STAGE_EXPLANATIONS,
    STATUS_EXPLANATIONS,
    USE_EXPLANATIONS,
    ExclusionReason,
    PendingStage,
    SourceMethod,
    SourceStatus,
)

from experiments.post_training.task_curation import source_status
from experiments.post_training.task_curation.exclusion_registry import (
    MANUAL_EXCLUSION_CLAIMS,
    NEEDS_RULING_SOURCES,
    manual_exclusion_claims,
)
from experiments.post_training.task_curation.metadata_exclusions import metadata_exclusions
from experiments.post_training.task_curation.source import (
    POPULATION_BLOCKER_REASONS,
    lexical_blocker_reason,
    population_blocker_reason,
)
from experiments.post_training.task_curation.sources import PENDING_MEASUREMENT_SOURCES, all_sources, configured_sources
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
    ("evidence-derived oversight with verification is missing", ExclusionReason.OVERSIGHT_MISSING),
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
    "Atlas remediation requires the original private open-athena snapshot": ExclusionReason.LEXICAL_REMEDIATION_REQUIRED,
}


@pytest.mark.data_integration
def test_selection_semantics_match_the_ruled_baseline():
    """Available equals the pre-taxonomy string logic minus the user-ruled exclusions.

    The taxonomy folded Not selectable into Excluded and moved lexical-only
    remediation rows to Pending without changing selection. The ruled
    exclusions (manual registry claims, e.g. the 2026-11-16 NO_SUPPORTED_BACKEND
    ruling for the Nemotron math_tir sources) are the only deliberate delta.
    """
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
    ruled = {info_id for info_id in before if manual_exclusion_claims(info_id)}
    assert {key for key, value in after.items() if value == "Available"} == {
        key for key, value in before.items() if value == "Available"
    } - ruled
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


@pytest.mark.data_integration
@pytest.mark.data_integration
def test_status_page_composition_has_unique_ids():
    """The served catalog plus the extras compose exactly one row per source ID."""
    extras = source_status.status_extras()
    catalog = {source.info.id: source for source in all_sources().values()}
    queued = [*extras["pending"], *extras["metadata_exclusions"]]
    by_id = {source_id: source.status.value for source_id, source in catalog.items()}
    for item in queued:
        assert item["id"] not in by_id, f"Extras row duplicates a catalog source: {item['id']}"
        by_id[item["id"]] = item["status"]
    assert len(by_id) == len(catalog) + len(queued)

    def catalog_count(status: SourceStatus) -> int:
        return sum(1 for source in catalog.values() if source.status is status)

    statuses = Counter(by_id.values())
    assert statuses["Available"] == catalog_count(SourceStatus.AVAILABLE)
    assert statuses["Pending"] == catalog_count(SourceStatus.PENDING) + len(extras["pending"])
    assert statuses["Excluded"] == catalog_count(SourceStatus.EXCLUDED) + len(extras["metadata_exclusions"])
    assert "Deprecated" not in statuses or statuses["Deprecated"] == catalog_count(SourceStatus.DEPRECATED)
    for source_id in extras["stages"]:
        assert source_id in catalog, f"Stage overlay targets a non-catalog source: {source_id}"


@pytest.mark.data_integration
def test_remediation_rows_are_configured_pending_sources():
    remediation = {row["id"]: row for row in source_status.remediation_rows()}
    intake = {source.info.id for source in PENDING_MEASUREMENT_SOURCES}
    configured = {source.info.id for source in all_sources().values()}
    assert set(remediation) <= configured - intake
    for row in remediation.values():
        assert row["status"] == SourceStatus.PENDING
        assert row["stage"] == source_status.PendingStage.LEXICAL_REMEDIATION_QUEUED.value
        assert row["reasons"], f"Remediation row without its claim: {row['id']}"


REBUILT_LEXICAL_ONLY_SOURCES = (
    "TaskTrove:tasktrove-curriculum-easy-rebuilt-v1",
    "TaskTrove:tasktrove-curriculum-medium-rebuilt-v1",
    "TaskTrove:tasktrove-e2egit-v2-rebuilt",
    "TaskTrove:tasktrove-e2egit-large-rebuilt-v1",
    "TaskTrove:tasktrove-multifile-rebuilt-v1",
    "TaskTrove:tasktrove-stack-pytest-rebuilt-v1",
)


@pytest.mark.data_integration
def test_lexical_only_sources_queue_as_pending_with_the_remediation_stage():
    """User ruling 2026-11-17: lexical-remediation sources are Pending, not Excluded.

    A source whose only exclusion claim is the missing lexical remediation
    receipt queues for remediation when no user ruling excludes it and the
    population is nonempty — a stale legacy ``excluded`` tag must not keep it
    Excluded. The six rebuilt cohorts named in the excluded-triage brief were
    the regression rows; their remediation completed, so they must now be
    Available or Pending, never Excluded.
    """
    lexical_only = {}
    for source in all_sources().values():
        claims = source.exclusion_claims
        if {claim.reason for claim in claims} != {ExclusionReason.LEXICAL_REMEDIATION_REQUIRED}:
            continue
        if manual_exclusion_claims(source.info.id):
            continue
        if (source.info.within_source_dedup.unique_count or 0) <= 0:
            continue
        lexical_only[source.info.id] = source
        assert (
            source.status is SourceStatus.PENDING
        ), f"Lexical-remediation-only source must queue as Pending: {source.info.id}"
    remediation = {row["id"]: row for row in source_status.remediation_rows()}
    for expected in REBUILT_LEXICAL_ONLY_SOURCES:
        status = {source.info.id: source.status for source in all_sources().values()}[expected]
        assert status in {
            SourceStatus.AVAILABLE,
            SourceStatus.PENDING,
        }, f"Rebuilt lexical-only source regressed to Excluded: {expected}"
        if status is SourceStatus.PENDING:
            row = remediation.get(expected)
            assert row is not None and row["stage"] == source_status.PendingStage.LEXICAL_REMEDIATION_QUEUED.value
    for source in lexical_only.values():
        row = remediation.get(source.info.id)
        assert row is not None, f"Remediation queue lacks the queued source: {source.info.id}"
        assert row["stage"] == source_status.PendingStage.LEXICAL_REMEDIATION_QUEUED.value
        assert {claim["reason"] for claim in row["reasons"]} == {ExclusionReason.LEXICAL_REMEDIATION_REQUIRED.value}


def test_admission_blocker_strings_stay_mapped():
    """Every blocker string the population check can emit keeps its typed reason."""
    admission = ast.parse((Path("infra/marina/apps/rl_data_catalog/server/admission.py")).read_text())
    function = next(
        node
        for node in admission.body
        if isinstance(node, ast.FunctionDef) and node.name == "population_admission_blocker"
    )
    emitted = {
        node.value.value
        for node in ast.walk(function)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
    }
    assert emitted == set(POPULATION_BLOCKER_REASONS)
    for blocker in emitted:
        assert population_blocker_reason(blocker) is POPULATION_BLOCKER_REASONS[blocker]
    with pytest.raises(ValueError, match="Unmapped population admission blocker"):
        population_blocker_reason("an unmapped blocker")
    for message, reason in LEXICAL_MESSAGES.items():
        assert lexical_blocker_reason(message) is reason
    with pytest.raises(ValueError, match="Unmapped lexical qualification failure"):
        lexical_blocker_reason("an unmapped lexical failure")


@pytest.mark.parametrize(
    ("table", "members"),
    [
        (REASON_EXPLANATIONS, set(ExclusionReason)),
        (STATUS_EXPLANATIONS, set(SourceStatus)),
        (STAGE_EXPLANATIONS, set(PendingStage)),
    ],
)
def test_every_state_has_one_canonical_explanation(table, members):
    assert set(table) == members
    for explanation in table.values():
        assert explanation.label.strip()
        assert explanation.meaning.strip()
        assert explanation.clears.strip()


def test_column_explanations_cover_the_column_enums():
    """The Use, Method and Format label tables cover exactly their enum values."""
    assert set(USE_EXPLANATIONS) == {use.value for use in SourceUse}
    assert set(METHOD_EXPLANATIONS) == set(SourceMethod)
    assert set(FORMAT_EXPLANATIONS) == {item.value for item in TaskFormat}
    for table in (USE_EXPLANATIONS, FORMAT_EXPLANATIONS):
        for item in table.values():
            assert item.label.strip()
            assert item.meaning.strip()
    for item in METHOD_EXPLANATIONS.values():
        assert item.label.strip()
        assert item.meaning.strip()


def test_method_tag_aliases_validate_as_enum_members():
    """The stored legacy tags rl and sft validate as the canonical Method values."""
    assert SourceMethod("rl") is SourceMethod.RL
    assert SourceMethod("sft") is SourceMethod.SFT_VIABLE
    with pytest.raises(ValueError):
        SourceMethod("distillation")


def test_oversight_missing_explanation_carries_the_quick_receipt_path():
    # QUICK oversight admits a source: the explanation cannot read as if only
    # SAMPLE or FULL evidence could clear the reason.
    explanation = REASON_EXPLANATIONS[ExclusionReason.OVERSIGHT_MISSING]
    assert "QUICK" in explanation.clears
    assert "QUICK" in explanation.meaning


def test_difficulty_gap_mapping_targets_defined_reasons():
    mapped = {spec.exclusion for spec in DIFFICULTY_GAPS.values() if spec.exclusion is not None}
    assert mapped, "The typed gap table must keep at least one SOURCE-owned exclusion mapping"
    for reason in mapped:
        assert reason in ExclusionReason
        assert reason in REASON_EXPLANATIONS  # every mapped reason carries its canonical explanation


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


def test_reason_explanations_cover_every_typed_reason():
    explanations = source_status.reason_explanations()
    assert set(explanations) == {reason.value for reason in ExclusionReason}
    for explanation in explanations.values():
        assert set(explanation) == {"label", "meaning", "clears"}
        assert explanation["label"] and explanation["meaning"] and explanation["clears"]


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


@pytest.mark.data_integration
def test_metadata_exclusions_never_enter_the_catalog():
    catalog_ids = {source.info.id for source in all_sources().values()} | {
        source.info.id for source in configured_sources()
    }
    for row in metadata_exclusions().values():
        assert row.id not in catalog_ids, f"Metadata exclusion declared as a catalog source: {row.id}"
        assert row.claims, f"Metadata exclusion without a claim: {row.id}"
        for claim in row.claims:
            assert claim.evidence.message.strip()
            assert claim.evidence.uri, f"Metadata exclusion evidence without a link: {row.id}"


def test_pending_stage_derivation():
    receipts = {"actor census": {"a"}, "dedup": {"a", "b"}}
    assert source_status.pending_stage("a", receipts) is source_status.PendingStage.ADMISSION_PENDING
    assert source_status.pending_stage("b", receipts) is source_status.PendingStage.MEASUREMENTS_PENDING
    assert source_status.pending_stage("c", receipts) is source_status.PendingStage.INTAKE_QUEUED


def _admitted_source(tmp_path):
    from experiments.post_training.task_curation.tests.admission_fixtures import (  # noqa: PLC0415
        with_admission_census,
    )
    from experiments.post_training.task_curation.tests.numbers_pipeline import number_source  # noqa: PLC0415

    source = number_source(tmp_path / "input")
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
    return with_admission_census(source)


def test_pending_declaration_without_stage_fails_the_build(tmp_path):
    """Regression: a Pending source without a PendingStage fails the build.

    The loader's typed Pending downgrade copies declarations with
    ``model_copy`` and re-checks them explicitly; the same check guards the
    model itself, so no Pending declaration without a stage can stage, load or
    land.
    """
    source = _admitted_source(tmp_path)
    assert source.status is SourceStatus.AVAILABLE
    copy = source.declaration.model_copy(update={"status": SourceStatus.PENDING, "stage": None})
    with pytest.raises(ValueError, match="carries no PendingStage"):
        require_typed_status_fields(copy)
    # The load-time downgrade assigns the stage from the backend evidence.
    staged = copy.model_copy(update={"stage": PendingStage.BACKEND_VERIFICATION_PENDING})
    require_typed_status_fields(staged)
    with pytest.raises(ValueError, match="Only a Pending source carries a PendingStage"):
        require_typed_status_fields(staged.model_copy(update={"status": SourceStatus.AVAILABLE}))


def test_excluded_declaration_without_claim_fails_the_build(tmp_path):
    source = _admitted_source(tmp_path)
    copy = source.declaration.model_copy(update={"status": SourceStatus.EXCLUDED, "exclusion_claims": []})
    with pytest.raises(ValueError, match="carries no ExclusionReason"):
        require_typed_status_fields(copy)


def test_pending_extras_row_without_stage_fails_the_build():
    with pytest.raises(ValueError, match="carries no PendingStage"):
        source_status.require_typed_rows([{"id": "Fixture:x", "status": "Pending", "stage": None, "reasons": []}])
    with pytest.raises(ValueError, match="carries no ExclusionReason"):
        source_status.require_typed_rows([{"id": "Fixture:y", "status": "Excluded", "stage": None, "reasons": []}])
    source_status.require_typed_rows(
        [
            {"id": "Fixture:ok", "status": "Pending", "stage": "Intake queued", "reasons": []},
            {"id": "Fixture:excluded", "status": "Excluded", "stage": None, "reasons": [{"reason": "license-ban"}]},
        ]
    )


@pytest.mark.data_integration
def test_pending_rows_carry_typed_columns():
    uses = {use.value for use in SourceUse}
    methods = {method.value for method in SourceMethod}
    stages = {stage.value for stage in PendingStage}
    formats = {item.value for item in TaskFormat}
    for row in source_status.pending_rows():
        assert row["use"] in uses, f"Pending row Use outside SourceUse: {row['id']}"
        assert row["task_format"] in formats, f"Pending row Format outside TaskFormat: {row['id']}"
        assert row["method_tags"] and set(row["method_tags"]) <= methods, row["id"]
        assert row["stage"] in stages, f"Pending row stage outside PendingStage: {row['id']}"


def test_status_enum_semantics(tmp_path, monkeypatch):
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
    lexical = replace(source, info=replace(source.info, lexical_contamination=None))
    assert lexical.status is SourceStatus.PENDING  # remediation work queues the row, not excludes it
    quick = replace(
        source,
        info=replace(
            source.info,
            oversight=OversightEvidence(
                level=OversightLevel.QUICK, campaign="fixture", verification="Fixture conversion executed."
            ),
        ),
    )
    assert quick.status is SourceStatus.AVAILABLE  # user ruling: QUICK QC admits to the Atlas
    import experiments.post_training.task_curation.sources as source_catalog  # noqa: PLC0415
    from experiments.post_training.task_curation.sources import runnable_sources  # noqa: PLC0415

    monkeypatch.setattr(source_catalog, "all_sources", lambda: {quick.info.id: quick})
    assert quick.name not in runnable_sources()  # QUICK stays out of the default training selection
