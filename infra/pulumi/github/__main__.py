# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for Marin's GitHub resources."""

import os
import sys
from typing import cast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pulumi
from iac.github.credentials import credential_manifest
from iac.github.external_runtime_updater import (
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
    updater_environment = register_external_runtime_updater_environment(
        manifest.organization,
        "marin-community/marin",
    )
    updater = external_runtime_updater_config(
        organization=manifest.organization,
        settings=cast(dict[str, object], config.require_object("externalRuntimeUpdater")),
    )
    if updater is not None:
        register_external_runtime_updater(updater, updater_environment)
    else:
        pulumi.log.warn(
            "external runtime updater is awaiting its one-time GitHub App bootstrap; "
            "see infra/pulumi/github/README.md"
        )
    pulumi.export("credential_count", len(plans))
    pulumi.export("external_runtime_updater_enabled", updater is not None)


main()
