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
    Management,
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


def repository_name(organization: str, repository: str) -> str:
    prefix = f"{organization}/"
    if not repository.startswith(prefix):
        raise ValueError(f"repository {repository!r} is not owned by {organization!r}")
    return repository.removeprefix(prefix)


def _logical_name(credential: Credential) -> str:
    return "-".join(credential.key).replace("/", "-").replace("_", "-").lower()


def _credential_resource_plan(manifest: CredentialManifest, credential: Credential) -> CredentialResourcePlan:
    if isinstance(credential, OrganizationCredential):
        resource_id = credential.name
    elif isinstance(credential, RepositoryCredential):
        repository = repository_name(manifest.organization, credential.repository)
        resource_id = f"{repository}:{credential.name}"
    else:
        assert isinstance(credential, EnvironmentCredential)
        repository = repository_name(manifest.organization, credential.repository)
        resource_id = f"{repository}:{quote(credential.environment, safe='')}:{credential.name}"
    return CredentialResourcePlan(
        credential=credential,
        logical_name=_logical_name(credential),
        resource_id=resource_id,
    )


def credential_resource_plans(manifest: CredentialManifest) -> tuple[CredentialResourcePlan, ...]:
    """Compute lookup IDs for present credentials managed outside Pulumi."""
    return tuple(
        _credential_resource_plan(manifest, credential)
        for credential in manifest.credentials
        if credential.presence is Presence.PRESENT and credential.management is Management.EXTERNAL
    )


def _managed_secret(plan: CredentialResourcePlan, organization: str) -> pulumi.CustomResource:
    credential = plan.credential
    assert credential.key_id is not None
    assert credential.value_encrypted is not None
    opts = pulumi.ResourceOptions(import_=plan.resource_id)
    if isinstance(credential, OrganizationCredential):
        return github.ActionsOrganizationSecret(
            plan.logical_name,
            secret_name=credential.name,
            visibility=credential.visibility,
            key_id=credential.key_id,
            value_encrypted=credential.value_encrypted,
            opts=opts,
        )
    repository = repository_name(organization, credential.repository)
    if isinstance(credential, RepositoryCredential):
        return github.ActionsSecret(
            plan.logical_name,
            repository=repository,
            secret_name=credential.name,
            key_id=credential.key_id,
            value_encrypted=credential.value_encrypted,
            opts=opts,
        )
    assert isinstance(credential, EnvironmentCredential)
    return github.ActionsEnvironmentSecret(
        plan.logical_name,
        repository=repository,
        environment=credential.environment,
        secret_name=credential.name,
        key_id=credential.key_id,
        value_encrypted=credential.value_encrypted,
        opts=opts,
    )


def register_credentials(manifest: CredentialManifest) -> tuple[pulumi.CustomResource, ...]:
    """Register declared external lookups and Pulumi-managed sealed secrets."""
    resources: list[pulumi.CustomResource] = []
    plans = [
        _credential_resource_plan(manifest, credential)
        for credential in manifest.credentials
        if credential.presence is Presence.PRESENT and credential.management is not Management.PULUMI
    ]
    for plan in plans:
        credential = plan.credential
        if credential.management is Management.SEALED:
            secret = _managed_secret(plan, manifest.organization)
        elif isinstance(credential, OrganizationCredential):
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
            if credential.management is Management.SEALED:
                repository_ids = [
                    github.get_repository(full_name=repository).repo_id for repository in credential.repositories
                ]
                access = github.ActionsOrganizationSecretRepositories(
                    f"{plan.logical_name}-repositories",
                    secret_name=credential.name,
                    selected_repository_ids=repository_ids,
                    opts=pulumi.ResourceOptions(import_=credential.name, depends_on=[secret]),
                )
            else:
                access = github.ActionsOrganizationSecretRepositories.get(
                    f"{plan.logical_name}-repositories",
                    id=credential.name,
                )
            resources.append(access)
    return tuple(resources)
