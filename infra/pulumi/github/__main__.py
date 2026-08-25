# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for Marin's GitHub resources."""

import os
import sys
from typing import cast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pulumi
import pulumi_github as github
from iac.github.credentials import credential_manifest
from iac.github.dependency_updater import (
    dependency_updater_config,
    register_dependency_updater,
    register_dependency_updater_environment,
)
from iac.github.resources import credential_resource_plans, register_credentials, repository_name

LOOM_STACK = "organization/marin-loom/marin-loom"
FORK_FERRY_FEDERATION = "fork-ferry"


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
    fork_ferry_repository = config.require("forkFerryRepository")
    loom = pulumi.StackReference(LOOM_STACK)
    fork_ferry_profile = loom.require_output("githubFederationProfiles").apply(
        lambda profiles: profiles[FORK_FERRY_FEDERATION]
    )
    github.ActionsVariable(
        "fork-ferry-profile",
        repository=repository_name(manifest.organization, fork_ferry_repository),
        variable_name="LOOM_FORK_FERRY_PROFILE",
        value=fork_ferry_profile,
    )
    updater = dependency_updater_config(
        organization=manifest.organization,
        settings=cast(dict[str, object], config.require_object("dependencyUpdater")),
    )
    _, deployment_policy = register_dependency_updater_environment(
        updater.organization,
        updater.repository,
    )
    register_dependency_updater(updater, deployment_policy)
    pulumi.export("credential_count", len(plans))
    pulumi.export("dependency_updater_enabled", True)
    pulumi.export("fork_ferry_profile", fork_ferry_profile)


main()
