# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from iac.config import KueueProvisioningSpec
from iris.cluster.platforms.k8s.kueue_manifests import build_cks_values
from pydantic import ValidationError


def test_kueue_default_reserves_two_gibibytes_for_manager():
    spec = KueueProvisioningSpec()
    manager = build_cks_values(["iris"], manager_memory_limit=spec.manager_memory_limit)["kueue"]["controllerManager"][
        "manager"
    ]

    assert manager["resources"] == {
        "limits": {"memory": "2Gi"},
        "requests": {"memory": "2Gi"},
    }


def test_kueue_rejects_manager_memory_below_two_gibibytes():
    with pytest.raises(ValidationError) as exc_info:
        KueueProvisioningSpec(manager_memory_limit="1Gi")

    assert [error["loc"] for error in exc_info.value.errors()] == [("manager_memory_limit",)]


def test_kueue_allows_larger_manager_memory_override():
    spec = KueueProvisioningSpec(manager_memory_limit="4Gi")
    manager = build_cks_values(["iris"], manager_memory_limit=spec.manager_memory_limit)["kueue"]["controllerManager"][
        "manager"
    ]

    assert manager["resources"]["limits"]["memory"] == "4Gi"
    assert manager["resources"]["requests"]["memory"] == "4Gi"
