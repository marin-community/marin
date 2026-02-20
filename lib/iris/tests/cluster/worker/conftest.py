# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for Docker-based worker tests."""

import pytest

from iris.cluster.runtime.docker import DockerRuntime


@pytest.fixture
def docker_runtime():
    """DockerRuntime that cleans up its own containers after the test."""
    rt = DockerRuntime()
    yield rt
    rt.cleanup()
