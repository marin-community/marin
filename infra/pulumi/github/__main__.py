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
CODEHEALTH_REFINEMENT_FEDERATION = "codehealth-refinement"
FORK_FERRY_FEDERATION = "fork-ferry"
PR_REVIEW_FEDERATION = "pr-review"


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
    marin_repository = repository_name(manifest.organization, fork_ferry_repository)
    loom = pulumi.StackReference(LOOM_STACK)
    federation_profiles = loom.require_output("githubFederationProfiles")
    profile_variables = {
        "codehealth-refinement-profile": (
            "LOOM_CODEHEALTH_REFINEMENT_PROFILE",
            CODEHEALTH_REFINEMENT_FEDERATION,
        ),
        "fork-ferry-profile": ("LOOM_FORK_FERRY_PROFILE", FORK_FERRY_FEDERATION),
        "pr-review-profile": ("LOOM_PR_REVIEW_PROFILE", PR_REVIEW_FEDERATION),
    }
    resolved_profiles = {}
    for resource_name, (variable_name, federation_name) in profile_variables.items():
        profile = federation_profiles.apply(lambda profiles, name=federation_name: profiles[name])
        resolved_profiles[federation_name] = profile
        github.ActionsVariable(
            resource_name,
            repository=marin_repository,
            variable_name=variable_name,
            value=profile,
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
    pulumi.export(
        "codehealth_refinement_profile",
        resolved_profiles[CODEHEALTH_REFINEMENT_FEDERATION],
    )
    pulumi.export("dependency_updater_enabled", True)
    pulumi.export("fork_ferry_profile", resolved_profiles[FORK_FERRY_FEDERATION])
    pulumi.export("pr_review_profile", resolved_profiles[PR_REVIEW_FEDERATION])


main()
