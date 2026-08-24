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
    DependencyUpdaterInstallation,
    dependency_updater_config,
    dependency_updater_installation_id,
    register_dependency_updater_environment,
    register_dependency_updater_installation,
    register_dependency_updater_repository,
    validate_classic_branch_protection,
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
    live_installations = github.get_organization_app_installations()
    installation_id = dependency_updater_installation_id(
        updater,
        tuple(
            DependencyUpdaterInstallation(
                installation_id=installation.id,
                app_id=installation.app_id,
                client_id=installation.client_id,
                app_slug=installation.app_slug,
                permissions=installation.permissions,
                repository_selection=installation.repository_selection,
                suspended=installation.suspended,
            )
            for installation in live_installations.installations
        ),
    )
    installation = register_dependency_updater_installation(updater, installation_id)
    updater_resource_count = 1
    for repository in updater.repositories:
        normalized_repository = repository_name(updater.organization, repository.repository)
        protection_rules = github.get_branch_protection_rules(repository=normalized_repository)
        validate_classic_branch_protection(repository, [rule.pattern for rule in protection_rules.rules])
        _, deployment_policy = register_dependency_updater_environment(
            updater.organization,
            repository.repository,
        )
        updater_resources = register_dependency_updater_repository(
            updater,
            repository,
            deployment_policy,
            installation,
        )
        updater_resource_count += len(updater_resources) + 2
    pulumi.export("credential_count", len(plans))
    pulumi.export("dependency_updater_enabled", True)
    pulumi.export("dependency_updater_repository_count", len(updater.repositories))
    pulumi.export(
        "dependency_updater_repositories_without_private_key",
        [repository.repository for repository in updater.repositories if repository.private_key is None],
    )
    pulumi.export("dependency_updater_resource_count", updater_resource_count)
    pulumi.export("fork_ferry_profile", fork_ferry_profile)


main()
