# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi resources for externally valued GitHub Actions secrets."""

from dataclasses import dataclass
from urllib.parse import quote

import pulumi
import pulumi_github as github

from iac.github.credentials import (
    Credential,
    CredentialManifest,
    EnvironmentCredential,
    OrganizationCredential,
    OrganizationVisibility,
    Presence,
    RepositoryCredential,
)


@dataclass(frozen=True)
class CredentialResourcePlan:
    credential: Credential
    logical_name: str
    resource_id: str


def _repository_name(organization: str, repository: str) -> str:
    prefix = f"{organization}/"
    if not repository.startswith(prefix):
        raise ValueError(f"repository {repository!r} is not owned by {organization!r}")
    return repository.removeprefix(prefix)


def _logical_name(credential: Credential) -> str:
    return "-".join(credential.key).replace("/", "-").replace("_", "-").lower()


def credential_resource_plans(manifest: CredentialManifest) -> tuple[CredentialResourcePlan, ...]:
    """Compute lookup IDs for the present credential declarations."""
    plans: list[CredentialResourcePlan] = []
    for credential in manifest.credentials:
        if credential.presence is not Presence.PRESENT:
            continue
        if isinstance(credential, OrganizationCredential):
            resource_id = credential.name
        elif isinstance(credential, RepositoryCredential):
            repository = _repository_name(manifest.organization, credential.repository)
            resource_id = f"{repository}:{credential.name}"
        else:
            assert isinstance(credential, EnvironmentCredential)
            repository = _repository_name(manifest.organization, credential.repository)
            resource_id = f"{repository}:{quote(credential.environment, safe='')}:{credential.name}"
        plans.append(
            CredentialResourcePlan(
                credential=credential,
                logical_name=_logical_name(credential),
                resource_id=resource_id,
            )
        )
    return tuple(plans)


def register_credentials(manifest: CredentialManifest) -> tuple[pulumi.CustomResource, ...]:
    """Register existing secrets as external, read-only resources."""
    resources: list[pulumi.CustomResource] = []
    for plan in credential_resource_plans(manifest):
        credential = plan.credential
        if isinstance(credential, OrganizationCredential):
            secret: pulumi.CustomResource = github.ActionsOrganizationSecret.get(
                plan.logical_name,
                id=plan.resource_id,
            )
        elif isinstance(credential, RepositoryCredential):
            secret = github.ActionsSecret.get(plan.logical_name, id=plan.resource_id)
        else:
            assert isinstance(credential, EnvironmentCredential)
            secret = github.ActionsEnvironmentSecret.get(plan.logical_name, id=plan.resource_id)
        resources.append(secret)

        if isinstance(credential, OrganizationCredential) and credential.visibility is OrganizationVisibility.SELECTED:
            resources.append(
                github.ActionsOrganizationSecretRepositories.get(
                    f"{plan.logical_name}-repositories",
                    id=credential.name,
                )
            )
    return tuple(resources)
