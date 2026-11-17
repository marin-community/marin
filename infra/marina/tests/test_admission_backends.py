# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Admission counts production backends only; dev-only verdicts never satisfy it."""


from infra.marina.apps.rl_data_catalog.server.admission import backend_support_blocker


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
    assert backend_support_blocker(verdicts, "Available") is None


def test_only_dev_support_still_blocks():
    verdicts = {
        "docker": "supported",
        "docker-gvisor": "supported",
        "iris": "unsupported",
        "daytona": "unsupported",
        "local": "unsupported",
    }
    blocker = backend_support_blocker(verdicts, "Available")
    assert blocker is not None and "no surveyed production backend" in blocker
    # the blocked message still names every surveyed backend, dev-only included
    assert "docker=supported" in blocker and "docker-gvisor=supported" in blocker


def test_survey_deployment_classes_override_the_defaults():
    """A survey's own deployment_classes map wins over the mirrored defaults."""
    verdicts = {"custom-prod-backend": "supported"}
    # unknown backends default to dev_only: blocked without a survey map
    assert backend_support_blocker(verdicts, "Available") is not None
    # the survey declares it production: admitted
    assert (
        backend_support_blocker(
            verdicts,
            "Available",
            deployment_classes={"custom-prod-backend": "production"},
        )
        is None
    )


def test_all_production_role_absent_needs_no_backend():
    verdicts = {"iris": "role_absent", "daytona": "role_absent", "local": "role_absent", "docker": "supported"}
    # the dev-only cell cannot rescue the role_absent production set
    assert backend_support_blocker(verdicts, "Available") is None


def test_non_available_status_needs_no_backend():
    assert backend_support_blocker({"docker": "supported", "iris": "unsupported"}, "Excluded") is None
