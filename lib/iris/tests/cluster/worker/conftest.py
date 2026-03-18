# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for Docker-based worker tests."""

import pytest

from iris.cluster.runtime.docker import DockerRuntime


@pytest.fixture
def docker_runtime(tmp_path):
    """DockerRuntime that cleans up its own containers after the test."""
    rt = DockerRuntime(cache_dir=tmp_path / "cache")
    yield rt
    rt.cleanup()
