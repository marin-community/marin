# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iac.gcp.cloud_run import iap_access_members


def test_iap_access_members_always_admits_openathena_domain():
    assert iap_access_members(()) == ("domain:openathena.ai",)


def test_iap_access_members_normalizes_extras_without_duplicate_domain_grants():
    assert iap_access_members(
        (
            "russell.power@gmail.com",
            "*@openathena.ai",
            "group:ops@openathena.ai",
        )
    ) == (
        "domain:openathena.ai",
        "user:russell.power@gmail.com",
        "group:ops@openathena.ai",
    )
