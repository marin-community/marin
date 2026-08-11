# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for Marin's GitHub resources."""

import os
import sys
from typing import cast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pulumi
from iac.github.credentials import credential_manifest
from iac.github.external_runtime_updater import (
    ExternalRuntimeUpdaterBootstrap,
    external_runtime_updater_config,
    register_external_runtime_updater,
    register_external_runtime_updater_environment,
)
from iac.github.resources import credential_resource_plans, register_credentials


def main() -> None:
    config = pulumi.Config()
    manifest = credential_manifest(
        schema_version=config.require_int("schemaVersion"),
        organization=config.require("organization"),
        repositories=cast(list[str], config.require_object("repositories")),
        credentials=cast(list[dict], config.require_object("credentials")),
    )
    provider_owner = pulumi.Config("github").require("owner")
    if provider_owner != manifest.organization:
        raise ValueError(
            f"github:owner {provider_owner!r} does not match marin-github:organization {manifest.organization!r}"
        )

    plans = credential_resource_plans(manifest)
    register_credentials(manifest)
    updater = external_runtime_updater_config(
        organization=manifest.organization,
        settings=cast(dict[str, object], config.require_object("externalRuntimeUpdater")),
    )
    _, deployment_policy = register_external_runtime_updater_environment(
        updater.organization,
        updater.repository,
    )
    if isinstance(updater, ExternalRuntimeUpdaterBootstrap):
        pulumi.log.warn(
            f"external runtime updater bootstrap is tracked by issue #{updater.issue}; "
            "see infra/pulumi/github/README.md"
        )
    else:
        register_external_runtime_updater(updater, deployment_policy)
    pulumi.export("credential_count", len(plans))
    pulumi.export("external_runtime_updater_enabled", not isinstance(updater, ExternalRuntimeUpdaterBootstrap))


main()
