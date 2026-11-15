# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from datetime import UTC, datetime

import pytest
from click.testing import CliRunner
from finestore.rl.licenses import LicenseClass, LicenseDecision, LicenseDecisionKind, LicenseReceipt, require_license
from taskcompendium.pipeline.source_processing import SourceProcessingMode

from experiments.post_training.task_curation import driver
from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.config import PipelineOptions
from experiments.post_training.task_curation.license_binding import BANNED_SOURCES_ISSUE
from experiments.post_training.task_curation.license_resolution import license_evidence
from experiments.post_training.task_curation.pipeline import process_rows
from experiments.post_training.task_curation.tests.admission_fixtures import with_admission_census
from experiments.post_training.task_curation.tests.license_fixtures import fixture_license, with_license
from experiments.post_training.task_curation.tests.numbers_pipeline import NUMBERS_DATASET, number_source
from infra.marina.apps.rl_data_catalog.server.admission import require_declaration_admission


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
        require_license(receipt, "Marin:fixture", "fixture/data", "a" * 40, "train")
        assert receipt.evidence[0].sha256
    else:
        with pytest.raises(ValueError, match="blocks ingest and admission"):
            require_license(receipt, "Marin:fixture", "fixture/data", "a" * 40, "train")


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


def test_scoped_exception_is_preserved_and_cannot_override_a_ban_or_review_flag():
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
    require_license(receipt, receipt.source_id, receipt.dataset_id, receipt.revision, "train")
    assert receipt.decisions[0].issue_url == BANNED_SOURCES_ISSUE
    with pytest.raises(ValueError, match="blocks ingest"):
        require_license(receipt, receipt.source_id, receipt.dataset_id, receipt.revision, "eval")
    with pytest.raises(ValueError, match="another source revision"):
        require_license(receipt, receipt.source_id, receipt.dataset_id, "b" * 40, "train")
    for flag in ("bundled_third_party", "model_generated_outputs"):
        with pytest.raises(ValueError, match="requires a recorded decision"):
            require_license(
                receipt.model_copy(update={flag: True}), receipt.source_id, receipt.dataset_id, receipt.revision, "train"
            )
    ban = approval.model_copy(update={"kind": LicenseDecisionKind.BAN, "approved": False})
    with pytest.raises(ValueError, match="bans this source"):
        require_license(
            receipt.model_copy(update={"decisions": [approval, ban]}),
            receipt.source_id,
            receipt.dataset_id,
            receipt.revision,
            "train",
        )


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
    registry = tmp_path / "licenses.json"
    registry.write_text('{"' + source.info.id + '":' + source.info.license_receipt.model_dump_json() + "}")
    source = replace(source, info=replace(source.info, license_receipt=None))
    monkeypatch.setattr(driver, "runnable_sources", lambda: {source.name: source})
    output = tmp_path / "output"
    result = CliRunner().invoke(
        driver.main,
        [
            "--source",
            source.name,
            "--mode",
            "quick",
            "--run",
            "--output-root",
            str(output),
            "--license-receipts",
            str(registry),
        ],
    )
    assert result.exit_code == 0, result.output + str(result.exception)
    assert list(output.rglob("numbers.json"))
    assert NUMBERS_DATASET.revision in registry.read_text()


def test_new_admission_requires_permission_even_with_complete_population_evidence(tmp_path):
    source = with_admission_census(number_source(tmp_path / "input.txt"))
    declaration = source.declaration
    require_declaration_admission(declaration)
    with pytest.raises(ValueError, match="License evidence is missing"):
        require_declaration_admission(declaration.model_copy(update={"license_receipt": None}))
    assert declaration.license_receipt.revision == declaration.dataset_revision
