# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest
import rigging.filesystem.cluster_config as cluster_config
from rigging.config_discovery import find_project_root
from rigging.grants import Permission, permissions_allow, permissions_cover


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


@pytest.fixture(autouse=True)
def _hermetic_cluster_config(monkeypatch):
    """Isolate cluster-config resolution from the host's ambient environment.

    ``~/.config/marin/clusters`` is a real per-user override some dev hosts carry
    for connecting to a live Iris cluster. It takes priority over the
    repo-committed ``config/`` dir, and a `data:`-less document there (e.g. a
    pure Iris connection config) silently swaps the loaded ``DataConfig`` out
    from under any test that expects the committed cluster layout — the same
    class of host-dependent flake as an unmocked GCE metadata lookup. Tests see
    only the repo-committed and bundled config dirs.
    """
    monkeypatch.setattr(
        cluster_config,
        "MARIN_CLUSTER_CONFIG_DIRS",
        tuple(p for p in cluster_config.MARIN_CLUSTER_CONFIG_DIRS if p != cluster_config.PER_USER_CLUSTER_CONFIG_DIR),
    )
    find_project_root.cache_clear()
    cluster_config.reset_data_config_cache()
    yield
    find_project_root.cache_clear()
    cluster_config.reset_data_config_cache()
