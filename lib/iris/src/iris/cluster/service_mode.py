# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum


class ServiceMode(StrEnum):
    DRY_RUN = "dry_run"  # Validate only, return synthetic responses
    LOCAL = "local"  # Validate + create real local resources
    CLOUD = "cloud"  # Validate + call real cloud APIs
