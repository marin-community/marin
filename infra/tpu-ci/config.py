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

# GCP Project Configuration
GCP_PROJECT_ID = "hai-gcp-models"
REGION = "us-west4"
ZONE = "us-west4-a"

# Monitor Controller VM (runs preemption monitor only)
CONTROLLER_NAME = "tpu-ci-monitor-controller"
CONTROLLER_MACHINE_TYPE = "e2-micro"
CONTROLLER_DISK_SIZE_GB = 10

# TPU Configuration
TPU_VM_PREFIX = "tpu-ci"
TPU_VM_COUNT = 2
TPU_ACCELERATOR_TYPE = "v5litepod-4"
TPU_VERSION = "tpu-ubuntu2204-base"

# GitHub Configuration
GITHUB_ORG = "marin-community"
GITHUB_REPO = "marin"
GITHUB_BRANCH = "main"  # Can be overridden via environment variable
GITHUB_CONFIG_URL = f"https://github.com/{GITHUB_ORG}/{GITHUB_REPO}"

# GitHub Actions Runner Configuration
RUNNER_LABELS = ["tpu", "self-hosted", "tpu-ci"]

# Docker Configuration
DOCKER_REGISTRY = f"{REGION}-docker.pkg.dev"
ARTIFACT_REGISTRY_REPO_NAME = "marin-ci"
DOCKER_REPOSITORY = f"{GCP_PROJECT_ID}/{ARTIFACT_REGISTRY_REPO_NAME}"
DOCKER_IMAGE_TAG = "latest"

# TPU CI Docker image (runs tests on TPU VMs)
DOCKER_IMAGE_NAME = "tpu-ci"
DOCKER_IMAGE_FULL = f"{DOCKER_REGISTRY}/{DOCKER_REPOSITORY}/{DOCKER_IMAGE_NAME}:{DOCKER_IMAGE_TAG}"

# Docker run command template for TPU workloads
# TPU containers require --privileged for device access
DOCKER_RUN_TEMPLATE = (
    "docker run --rm --privileged "
    "-e JAX_PLATFORMS=tpu -e PJRT_DEVICE=TPU -e TPU_CI=true "
    "-v /opt/marin:/opt/marin -w /opt/marin "
    f"{DOCKER_IMAGE_FULL}"
)

# Paths
INFRA_DIR = "infra/tpu-ci"
DOCKERFILE_TPU_CI_PATH = "docker/marin/Dockerfile.tpu-ci"
