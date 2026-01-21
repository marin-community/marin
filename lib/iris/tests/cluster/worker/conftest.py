# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared fixtures for Docker-based worker tests."""

import subprocess

import pytest

from iris.cluster.worker.docker import DockerRuntime


def _get_container_ids() -> set[str]:
    """Get all container IDs (running and stopped)."""
    result = subprocess.run(
        ["docker", "ps", "-aq"],
        capture_output=True,
        text=True,
        check=False,
    )
    return set(result.stdout.strip().split()) if result.stdout.strip() else set()


def _get_image_ids() -> set[str]:
    """Get all image IDs."""
    result = subprocess.run(
        ["docker", "images", "-q"],
        capture_output=True,
        text=True,
        check=False,
    )
    return set(result.stdout.strip().split()) if result.stdout.strip() else set()


@pytest.fixture
def docker_runtime():
    """DockerRuntime that cleans up containers and images created during test."""
    containers_before = _get_container_ids()
    images_before = _get_image_ids()

    runtime = DockerRuntime()
    yield runtime

    # Clean up new containers
    containers_after = _get_container_ids()
    for container_id in containers_after - containers_before:
        subprocess.run(["docker", "rm", "-f", container_id], capture_output=True, check=False)

    # Clean up new images
    images_after = _get_image_ids()
    for image_id in images_after - images_before:
        subprocess.run(["docker", "rmi", "-f", image_id], capture_output=True, check=False)
