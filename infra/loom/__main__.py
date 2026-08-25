# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for the production Loom host."""

from infrastructure import DeploymentConfig, create_infrastructure  # pyrefly: ignore[missing-import]

create_infrastructure(DeploymentConfig.from_pulumi())
