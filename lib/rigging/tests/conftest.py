# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest
from rigging.grants import Permission, permissions_allow, permissions_cover
from rigging.pytest_config import hermetic_cluster_config as hermetic_cluster_config


@dataclass
class GrantPolicyTable:
    permissions: dict[str, tuple[Permission, ...]]
    denied_resources: set[str] = field(default_factory=set)
    denied_grant_resources: set[str] = field(default_factory=set)

    def allows_grant(self, principal: str, permissions: Sequence[Permission]) -> bool:
        if any(permission.resource in self.denied_grant_resources for permission in permissions):
            return False
        return permissions_cover(self.permissions.get(principal, ()), permissions)

    def check(self, principal: str, service: str, action: str, resource: str) -> bool:
        return resource not in self.denied_resources and permissions_allow(
            self.permissions.get(principal, ()), service=service, action=action, resource=resource
        )


@pytest.fixture
def grant_permissions():
    return (
        Permission("loom", frozenset({"session.create", "session.read"}), "/alice/"),
        Permission("iris", frozenset({"job.submit", "job.read"}), "/alice/"),
        Permission("cw_s3", frozenset({"s3:GetObject"}), "s3://logs/alerts/"),
    )


@pytest.fixture
def grant_policy(grant_permissions):
    return GrantPolicyTable({"user:alice": grant_permissions})
