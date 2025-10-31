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

"""
Central configuration for TPU CI infrastructure.

All configuration values are hardcoded here to ensure consistent deployments.
"""

GCP_PROJECT_ID = "hai-gcp-models"

# Controller VM configuration (runs preemption monitor only)
CONTROLLER_NAME = "tpu-ci-monitor-controller"
CONTROLLER_ZONE = "us-west4-a"
CONTROLLER_MACHINE_TYPE = "e2-micro"
CONTROLLER_DISK_SIZE_GB = 10

# Multi-zone TPU configuration: maps zone -> desired instance count
TPU_ZONES_CONFIG = {
    "us-west4-a": 2,
    "us-east1-c": 2,
    "europe-west4-b": 2,
}

TPU_VM_PREFIX = "tpu-ci"
TPU_ACCELERATOR_TYPE = "v5litepod-4"
TPU_VERSION = "tpu-ubuntu2204-base"

GITHUB_BRANCH = "main"  # branch to use for the controller
GITHUB_URL = "https://github.com/marin-community/marin.git"

RUNNER_LABELS = ["tpu", "self-hosted", "tpu-ci"]

# Docker image configuration
# Images are pushed to the Artifact Registry in each region
ARTIFACT_REGISTRY_REPO_NAME = "marin-ci"
DOCKER_REPOSITORY = f"{GCP_PROJECT_ID}/{ARTIFACT_REGISTRY_REPO_NAME}"
DOCKER_IMAGE_NAME = "tpu-ci"
DOCKER_IMAGE_TAG = "latest"


def get_all_regions() -> list[str]:
    """Extract unique regions from TPU_ZONES_CONFIG."""
    regions = set()
    for zone in TPU_ZONES_CONFIG.keys():
        # Zone format: us-west4-a -> region: us-west4
        region = zone.rsplit("-", 1)[0]
        regions.add(region)
    return sorted(regions)


def get_docker_image_for_zone(zone: str) -> str:
    """
    Get the regional Docker image URL for a specific zone,
    e.g. "us-west4-docker.pkg.dev/hai-gcp-models/marin-ci/tpu-ci:latest"
    """
    region = zone.rsplit("-", 1)[0]
    registry = f"{region}-docker.pkg.dev"
    return f"{registry}/{DOCKER_REPOSITORY}/{DOCKER_IMAGE_NAME}:{DOCKER_IMAGE_TAG}"


INFRA_DIR = "infra/tpu-ci"
DOCKERFILE_TPU_CI_PATH = "docker/marin/Dockerfile.tpu-ci"
