# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Admission counts production backends only; dev-only verdicts never satisfy it."""

import pytest
from finestore.rl.curation import IndependentQC

from experiments.post_training.task_curation.tests.admission_fixtures import (
    with_declaration_admission_census,
)
from infra.marina.apps.rl_data_catalog.server.admission import (
    backend_admission_review,
    has_runnable_production_support,
    require_available_lexical_clearance,
    require_sample_pin_currency,
)
from infra.marina.apps.rl_data_catalog.tests.contamination_fixtures import (
    contamination_declarations,
)
from lib.taskcompendium.tests.decontamination_fixtures import (
    decontamination_evidence,
)


def test_dev_only_supported_cell_never_satisfies_admission():
    """A DEV_ONLY SUPPORTED verdict is dropped before the runnable check."""
    verdicts = {
        "docker": "supported",
        "docker-gvisor": "supported",
        "iris": "unsupported",
        "daytona": "unsupported",
        "local": "partial",
    }
    # local is production and partial: RUNNABLE_FIT includes partial, so this passes
    assert backend_admission_review(verdicts, "Available") == (None, [])


def test_only_dev_support_still_blocks():
    verdicts = {
        "docker": "supported",
        "docker-gvisor": "supported",
        "iris": "unsupported",
        "daytona": "unsupported",
        "local": "unsupported",
    }
    blocker, _ = backend_admission_review(verdicts, "Available")
    assert blocker is not None and "no surveyed production backend" in blocker
    # the blocked message still names every surveyed backend, dev-only included
    assert "docker=supported" in blocker and "docker-gvisor=supported" in blocker


def test_survey_deployment_classes_override_the_defaults():
    """A survey's own deployment_classes map wins over the mirrored defaults."""
    verdicts = {"custom-prod-backend": "supported"}
    # unknown backends default to dev_only: blocked without a survey map
    assert backend_admission_review(verdicts, "Available")[0] is not None
    # the survey declares it production: admitted
    assert backend_admission_review(
        verdicts,
        "Available",
        deployment_classes={"custom-prod-backend": "production"},
    ) == (None, [])


def test_all_production_role_absent_needs_no_backend():
    verdicts = {"iris": "role_absent", "daytona": "role_absent", "local": "role_absent", "docker": "supported"}
    # the dev-only cell cannot rescue the role_absent production set
    assert backend_admission_review(verdicts, "Available") == (None, [])


def test_non_available_status_needs_no_backend():
    assert backend_admission_review({"docker": "supported", "iris": "unsupported"}, "Excluded") == (None, [])


def test_verify_pending_never_blocks_and_reports_the_backends():
    """VERIFY on production backends is pending evidence, never a fit failure.

    The status decision is the loader's: an already-admitted source keeps its
    status with the marker, a new admission materializes as Pending.
    """
    verdicts = {"daytona": "verify", "iris": "verify", "local": "verify"}
    blocker, pending = backend_admission_review(verdicts, "Available")
    assert blocker is None
    assert pending == ["daytona", "iris", "local"]


def test_verify_with_a_runnable_production_backend_passes_with_a_marker():
    verdicts = {"daytona": "verify", "local": "supported"}
    blocker, pending = backend_admission_review(verdicts, "Available")
    assert blocker is None and pending == ["daytona"]


def test_all_production_unsupported_still_blocks_with_exact_verdicts():
    verdicts = {"daytona": "unsupported", "iris": "unsupported", "local": "unsupported"}
    blocker, pending = backend_admission_review(verdicts, "Available")
    assert blocker == (
        "no surveyed production backend can run the source: " "daytona=unsupported, iris=unsupported, local=unsupported"
    )
    assert pending == []


def test_not_applicable_does_not_negate_a_production_verify():
    """N/A on other production backends cannot negate one pending VERIFY cell."""
    verdicts = {"daytona": "not_applicable", "iris": "not_applicable", "local": "verify"}
    blocker, pending = backend_admission_review(verdicts, "Available")
    assert blocker is None and pending == ["local"]


def test_runnable_production_support_ignores_dev_only_cells():
    """The runnable check filters to production backends before scanning verdicts."""
    assert has_runnable_production_support({"docker": "supported", "iris": "unsupported"}) is False
    assert has_runnable_production_support({"docker": "supported", "iris": "supported"}) is True
    assert has_runnable_production_support({"iris": "verify", "local": "verify"}) is False
    assert (
        has_runnable_production_support(
            {"custom": "supported"},
            deployment_classes={"custom": "production"},
        )
        is True
    )


REV = "b" * 40


def _lexical_declaration(use, evidence=None, *, dataset_id="fixture"):
    """An admitted-census declaration with an explicit use and optional lexical evidence."""

    source = contamination_declarations()[0][1]
    base = source.model_copy(
        update={
            "dataset_id": dataset_id,
            "dataset_revision": REV,
            "intended_use": use,
            "lexical_contamination": evidence,
            "within_source_dedup": source.within_source_dedup.model_copy(
                update={"source_dataset_id": dataset_id, "source_revision": REV}
            ),
        }
    )
    return with_declaration_admission_census(base)


def _eval_receipt_evidence():
    """A benchmark-measured lexical receipt bound to the fixture pin."""

    evidence = decontamination_evidence("fixture", REV, 3)
    payload = evidence.model_dump(mode="json")
    payload["receipt"]["use"] = "eval"
    payload["receipt"]["status"] = "measured (benchmark, not remediated)"
    payload["receipt"]["release"] = None
    return evidence.model_validate(payload)


def test_new_train_admission_requires_the_private_release_receipt():
    with pytest.raises(ValueError, match="Decontamination"):
        require_available_lexical_clearance(_lexical_declaration("train"))


def test_new_train_admission_passes_with_private_release_evidence():
    evidence = decontamination_evidence("open-athena/fixture", REV, 3)
    require_available_lexical_clearance(_lexical_declaration("train", evidence, dataset_id="open-athena/fixture"))


def test_new_evaluation_admission_requires_a_measured_receipt():
    with pytest.raises(ValueError, match="measured lexical receipt"):
        require_available_lexical_clearance(_lexical_declaration("eval"))


def test_new_evaluation_admission_passes_with_a_benchmark_measured_receipt():
    require_available_lexical_clearance(_lexical_declaration("eval", _eval_receipt_evidence()))


def test_null_intended_use_is_refused_before_any_lexical_check():
    with pytest.raises(ValueError, match="explicit intended_use"):
        require_available_lexical_clearance(_lexical_declaration(None))


def _declaration_with_panel(use, panel_currency):
    qc = (
        IndependentQC(
            verdict="good",
            method="SAMPLE content panel: seed 0, 100 selected / 918 current tasks",
            reviewed=100,
            evidence_url="s3://test/panel",
            panel_currency=panel_currency,
        )
        if panel_currency is not None
        else None
    )
    source = with_declaration_admission_census(contamination_declarations()[0][1])
    return source.model_copy(update={"intended_use": use, "independent_qc": qc})


def test_archived_lineage_panel_refuses_a_new_available_admission():

    with pytest.raises(ValueError, match="belongs to an older population pin"):
        require_sample_pin_currency(_declaration_with_panel("eval", "archived-lineage"))


def test_current_pin_panel_passes_the_gate():
    require_sample_pin_currency(_declaration_with_panel("eval", "current-pin"))


def test_missing_or_untyped_panel_is_not_the_sample_gate_to_refuse():
    require_sample_pin_currency(_declaration_with_panel("train", None))
