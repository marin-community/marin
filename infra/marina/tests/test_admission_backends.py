# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Admission counts production backends only; dev-only verdicts never satisfy it."""


from infra.marina.apps.rl_data_catalog.server.admission import backend_admission_review


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
