# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from iac.config import DOCKER_HUB_UPSTREAM, GcpRemoteRepositorySpec
from iac.gcp.registries import remote_repository_config


@pytest.mark.parametrize(
    ("upstream", "expected_uri"),
    [
        ("https://ghcr.io", "https://ghcr.io"),
        (DOCKER_HUB_UPSTREAM, "https://registry-1.docker.io"),
    ],
)
def test_remote_repository_config_uses_canonical_common_repository(upstream: str, expected_uri: str) -> None:
    spec = GcpRemoteRepositorySpec(name="mirror", docker_upstream=upstream, locations=["us"])

    config = remote_repository_config(spec)

    assert config.common_repository is not None
    assert config.common_repository.uri == expected_uri
    assert config.docker_repository is None
