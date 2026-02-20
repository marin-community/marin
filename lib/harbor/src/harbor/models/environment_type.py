# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class EnvironmentType(str, Enum):
    DOCKER = "docker"
    DAYTONA = "daytona"
    E2B = "e2b"
    MODAL = "modal"
    RUNLOOP = "runloop"
    GKE = "gke"
