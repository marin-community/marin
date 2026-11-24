# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from click.testing import CliRunner
from finestore.rl.curation import OversightEvidence, OversightLevel
from finestore.rl.licenses import LicenseClass, LicenseDecision, LicenseDecisionKind, LicenseReceipt, require_license
from taskcompendium.pipeline.source_processing import SourceProcessingMode

from experiments.post_training.task_curation import driver
from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.config import PipelineOptions
from experiments.post_training.task_curation.license_binding import (
    BANNED_SOURCES_ISSUE,
    LICENSE_RECEIPT_INDEX,
    append_license_receipt,
    bind_license_receipt,
)
from experiments.post_training.task_curation.license_resolution import license_evidence
from experiments.post_training.task_curation.pipeline import process_rows
from experiments.post_training.task_curation.source import SourceOrigin
from experiments.post_training.task_curation.tests.admission_fixtures import with_admission_census
from experiments.post_training.task_curation.tests.license_fixtures import fixture_license, with_license
from experiments.post_training.task_curation.tests.numbers_pipeline import NUMBERS_DATASET, number_source
from infra.marina.apps.rl_data_catalog.server import licensing
from infra.marina.apps.rl_data_catalog.server.admission import require_catalog_declaration, require_declaration_admission
from infra.marina.apps.rl_data_catalog.server.filtration import filtration
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id
from infra.marina.apps.rl_data_catalog.tests.filtration_fixtures import count_evidence


@pytest.mark.parametrize(
    "license_id,classification,allowed",
    [
        ("MIT", LicenseClass.PERMISSIVE, True),
        ("Apache-2.0", LicenseClass.PERMISSIVE, True),
        ("CC-BY-SA-4.0", LicenseClass.PERMISSIVE, True),
        ("CC0-1.0", LicenseClass.PERMISSIVE, True),
        ("openmdw-1.1", LicenseClass.PERMISSIVE, True),
        ("CC-BY-NC-4.0", LicenseClass.RESTRICTED, False),
        ("CC-BY-ND-4.0", LicenseClass.RESTRICTED, False),
        ("unknown", LicenseClass.UNKNOWN, False),
        (None, LicenseClass.MISSING, False),
    ],
)
def test_license_evidence_controls_permission(license_id, classification, allowed):
    document = f"---\nlicense: {license_id}\n---\n".encode() if license_id else b"# Dataset without declared terms\n"
    receipt = fixture_license("Marin:fixture", "fixture/data", "a" * 40).model_copy(
        update={"evidence": [license_evidence("https://example.test/README.md", document, "dataset_card")]}
    )
    receipt = LicenseReceipt.model_validate_json(receipt.model_dump_json())
    assert receipt.classification is classification
    if allowed:
        require_license(receipt, "Marin:fixture", "fixture/data", "a" * 40, "train", ban=None)
        assert receipt.evidence[0].sha256
    else:
        with pytest.raises(ValueError, match="blocks ingest and admission"):
            require_license(receipt, "Marin:fixture", "fixture/data", "a" * 40, "train", ban=None)


def test_restrictive_file_overrides_permissive_card_and_affirmative_terms_do_not_restrict():
    card = license_evidence("https://example.test/card", b"---\nlicense: mit\n---\n", "dataset_card")
    terms = license_evidence(
        "https://example.test/LICENSE", b"Research-only. Redistribution prohibited.", "license_file"
    )
    receipt = fixture_license("Marin:fixture", "fixture/data", "a" * 40).model_copy(update={"evidence": [card, terms]})
    assert receipt.classification is LicenseClass.RESTRICTED
    permitted = license_evidence(
        "https://example.test/card",
        b"---\nlicense: cc-by-4.0\n---\nThis dataset is ready for commercial or non-commercial uses.",
        "dataset_card",
    )
    assert receipt.model_copy(update={"evidence": [permitted]}).classification is LicenseClass.PERMISSIVE


def test_scoped_exception_is_preserved_and_cannot_override_a_ban():
    receipt = fixture_license("Marin:fixture", "fixture/data", "a" * 40).model_copy(
        update={"evidence": [license_evidence("https://example.test/card", b"# No license declared", "dataset_card")]}
    )
    approval = LicenseDecision(
        kind=LicenseDecisionKind.EXCEPTION,
        approved=True,
        who="coordinator",
        when=datetime.now(UTC).isoformat(),
        source_id=receipt.source_id,
        dataset_id=receipt.dataset_id,
        revision=receipt.revision,
        scope=["train"],
        rationale="Explicit permission from the rights holder at this pin",
        evidence_urls=["https://example.test/permission"],
        issue_url=BANNED_SOURCES_ISSUE,
    )
    receipt = LicenseReceipt.model_validate_json(receipt.model_copy(update={"decisions": [approval]}).model_dump_json())
    require_license(receipt, receipt.source_id, receipt.dataset_id, receipt.revision, "train", ban=None)
    assert receipt.decisions[0].issue_url == BANNED_SOURCES_ISSUE
    with pytest.raises(ValueError, match="blocks ingest"):
        require_license(receipt, receipt.source_id, receipt.dataset_id, receipt.revision, "eval", ban=None)
    with pytest.raises(ValueError, match="another source revision"):
        require_license(receipt, receipt.source_id, receipt.dataset_id, "b" * 40, "train", ban=None)
    # Informational provenance flags alone do not override a dataset-level grant or scoped user decision.
    for flag in ("bundled_third_party", "model_generated_outputs"):
        require_license(
            receipt.model_copy(update={flag: True}),
            receipt.source_id,
            receipt.dataset_id,
            receipt.revision,
            "train",
            ban=None,
        )
    ban = approval.model_copy(update={"kind": LicenseDecisionKind.BAN, "approved": False})
    with pytest.raises(ValueError, match="bans this source"):
        require_license(
            receipt.model_copy(update={"decisions": [approval, ban]}),
            receipt.source_id,
            receipt.dataset_id,
            receipt.revision,
            "train",
            ban=None,
        )


def test_source_rename_carries_its_license_receipt(tmp_path):
    """A rename rebinds the reviewed receipt to the canonical identity.

    The registry accepts only canonical receipt identities at its write boundary,
    the catalog layer binds by the canonical id, and the re-keyed receipt keeps
    the same evidence bytes: a rename never needs a new license judgment.
    """
    historical_id = "MarinSkyRL:nemotron_if"
    canonical_id = canonical_source_id(historical_id)
    assert canonical_id == "Nemotron:nemotron_if"

    receipt = fixture_license(historical_id, NUMBERS_DATASET.name, NUMBERS_DATASET.revision)
    # The registry write boundary refuses a receipt whose identity is not canonical,
    # so a registry row can never keep a pre-rename id.
    with pytest.raises(ValueError, match="License registry key differs from the receipt source identity"):
        append_license_receipt(LICENSE_RECEIPT_INDEX, receipt, message="test: refuse the pre-rename identity")

    # The maintained rebind re-points only the receipt identity; decisions follow it.
    renamed = LicenseReceipt.model_validate_json(
        receipt.model_copy(
            update={
                "source_id": canonical_id,
                "decisions": [
                    decision.model_copy(update={"source_id": canonical_source_id(decision.source_id)})
                    for decision in receipt.decisions
                ],
            }
        ).model_dump_json()
    )
    assert renamed.evidence == receipt.evidence

    input_path = tmp_path / "input.txt"
    input_path.write_text("1\n2\n")
    source = number_source(input_path)
    source = replace(
        source,
        info=replace(
            source.info,
            id=canonical_id,
            origin=SourceOrigin("Nemotron"),
            upstream_dataset="nemotron_if",
            upstream_name="nemotron_if",
            title=None,
        ),
    )
    # A registry row still keyed by the pre-rename id no longer binds the renamed
    # identity, and the gate fails closed on the missing evidence.
    stale_registry = {historical_id: receipt}
    assert bind_license_receipt(source, stale_registry.get(source.info.id)).info.license_receipt is None
    with pytest.raises(ValueError, match="License evidence is missing"):
        bind_license_receipt(source, stale_registry.get(source.info.id)).require_license()
    # The re-keyed canonical row binds the same evidence; the rename carries its license.
    bound = bind_license_receipt(source, {canonical_id: renamed}.get(source.info.id))
    assert bound.info.license_receipt is not None
    assert bound.info.license_receipt.evidence == receipt.evidence
    bound.require_license()


def test_license_gate_precedes_recipe_validation_and_ingestion(tmp_path, monkeypatch):
    input_path = tmp_path / "input.txt"
    input_path.write_text("1\n2\n")
    source = number_source(input_path)
    source = replace(source, info=replace(source.info, license_receipt=None))
    # No recipe settings or CurationRecipe: stage 0 must fail before either is inspected.
    with pytest.raises(ValueError, match="License evidence is missing"):
        process_rows(source, PipelineOptions(SourceProcessingMode.FULL, CampaignRuntime()))
    monkeypatch.setattr(driver, "runnable_sources", lambda: {source.name: source})
    output = tmp_path / "output"
    result = CliRunner().invoke(
        driver.main, ["--source", source.name, "--mode", "quick", "--run", "--output-root", str(output)]
    )
    assert isinstance(result.exception, ValueError)
    assert "License evidence is missing" in str(result.exception)
    assert not output.exists()


def test_permissive_source_runs_real_ingestion_and_receipt_file_controls_access(tmp_path, monkeypatch):
    input_path = tmp_path / "input.txt"
    input_path.write_text("3\n5\n")
    source = with_license(number_source(input_path))
    reviewed = fixture_license(source.info.id, NUMBERS_DATASET.name, NUMBERS_DATASET.revision)
    registry = tmp_path / "license-receipts"
    (registry / "part-000.json").parent.mkdir(parents=True)
    (registry / "part-000.json").write_text(
        json.dumps({source.info.id: reviewed.model_dump(mode="json")}, sort_keys=True) + "\n"
    )
    source = replace(source, info=replace(source.info, license_receipt=None))
    monkeypatch.setattr(driver, "runnable_sources", lambda: {source.name: source})
    output = tmp_path / "output"
    arguments = [
        "--source",
        source.name,
        "--mode",
        "quick",
        "--run",
        "--output-root",
        str(output),
        "--license-receipts",
        str(registry),
    ]
    result = CliRunner().invoke(driver.main, arguments)
    assert result.exit_code == 0, result.output + str(result.exception)
    assert [json.loads(path.read_text()) for path in output.rglob("numbers.json")] == [[6, 10]]
    restricted = reviewed.model_copy(
        update={
            "evidence": [
                license_evidence("https://example.test/LICENSE", b"---\nlicense: cc-by-nc-4.0\n---\n", "dataset_card")
            ]
        }
    )
    (registry / "part-000.json").write_text(
        json.dumps({source.info.id: restricted.model_dump(mode="json")}, sort_keys=True) + "\n"
    )
    denied = CliRunner().invoke(driver.main, arguments)
    assert isinstance(denied.exception, ValueError)
    assert "restricted blocks ingest and admission" in str(denied.exception)


def test_new_admission_requires_permission_even_with_complete_population_evidence(tmp_path):
    source = with_admission_census(number_source(tmp_path / "input.txt"))
    declaration = source.declaration
    require_declaration_admission(declaration)
    with pytest.raises(ValueError, match="License evidence is missing"):
        require_declaration_admission(declaration.model_copy(update={"license_receipt": None}))
    assert declaration.license_receipt.revision == declaration.dataset_revision


@pytest.mark.parametrize("status", ["Excluded", "Deprecated"])
def test_catalog_publishes_licensed_unavailable_metadata_without_admitting_population(tmp_path, status):
    declaration = with_admission_census(number_source(tmp_path / "input.txt")).declaration.model_copy(
        update={"status": status, "actor_census_input": None}
    )
    # Complete inventories retain unavailable metadata before population qualification exists.
    require_catalog_declaration(declaration)
    with pytest.raises(ValueError, match="Admission census belongs to another current population"):
        require_declaration_admission(declaration)
    with pytest.raises(ValueError, match="Admission census belongs to another current population"):
        require_catalog_declaration(declaration.model_copy(update={"status": "Available"}))
    with pytest.raises(ValueError, match="License evidence is missing"):
        require_catalog_declaration(declaration.model_copy(update={"license_receipt": None}))


def test_user_ban_blocks_ingestion_and_counts_but_preserves_archived_evidence(tmp_path, monkeypatch):

    source = with_admission_census(number_source(tmp_path / "input.txt"))
    # the archived state is an ADMITTED source: QUICK oversight evidence clears
    # the oversight-missing claim, so the pre-ban population admits
    source = replace(
        source,
        info=replace(
            source.info,
            oversight=OversightEvidence(
                level=OversightLevel.QUICK, campaign="fixture", verification="Fixture conversion executed."
            ),
        ),
    )
    archived = source.declaration.model_dump(mode="json")
    before_bytes = json.dumps(archived, sort_keys=True)
    counts = count_evidence(source.declaration, 0, 0)
    assert filtration(archived, counts)["admitted_count"] == 1
    ban = {
        "source_id": source.info.id,
        "upstream_id": NUMBERS_DATASET.name,
        "upstream_revision": "a" * 40,
        "classification": "unknown",
        "license_ids": ["unknown"],
        "reason": "Upstream task-data permission has not been established.",
        "evidence_urls": ["https://example.test/terms"],
        "decided_at": "2026-11-15T17:00:00+00:00",
        "decision_by": "user via coordinator",
        "issue_url": BANNED_SOURCES_ISSUE,
    }
    monkeypatch.setattr(licensing, "LICENSE_BANS", json.dumps({"version": "fixture-2", "bans": [ban]}))
    licensing.license_bans.cache_clear()
    try:
        with pytest.raises(ValueError, match="License banned"):
            source.require_license()
        renamed = replace(
            source,
            info=replace(
                source.info,
                id="Marin:renamed",
                upstream_dataset="renamed",
                upstream_name="renamed",
                title=None,
                license_receipt=fixture_license("Marin:renamed", NUMBERS_DATASET.name, NUMBERS_DATASET.revision),
            ),
        )
        with pytest.raises(ValueError, match="License banned"):
            renamed.require_license()
        with pytest.raises(ValueError, match="License banned"):
            require_declaration_admission(source.declaration)
        excluded = source.declaration
        require_catalog_declaration(excluded)
        projected = licensing.license_projection(archived)
        assert projected["status"] == excluded.status == "Excluded"
        assert projected["license_ban"]["issue_url"] == BANNED_SOURCES_ISSUE
        assert filtration(archived, counts)["admitted_count"] == 0
        assert json.dumps(archived, sort_keys=True) == before_bytes
        assert archived["task_count"] == excluded.task_count == 1
    finally:
        licensing.license_bans.cache_clear()
