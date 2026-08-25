# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read GitHub Actions secret metadata through the GitHub CLI."""

import json
import subprocess

from iac.github.credentials import (
    CredentialManifest,
    EnvironmentLiveSecret,
    LiveSecret,
    OrganizationLiveSecret,
    OrganizationVisibility,
    RepositoryLiveSecret,
)

GITHUB_API_PAGE_SIZE = 100


def _gh_json(*args: str) -> object:
    result = subprocess.run(["gh", *args], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _gh_paginated_items(endpoint: str, collection: str) -> tuple[dict, ...]:
    pages = _gh_json("api", "--paginate", "--slurp", endpoint)
    assert isinstance(pages, list)
    return tuple(item for page in pages for item in page[collection])


def github_secret_inventory(manifest: CredentialManifest) -> tuple[LiveSecret, ...]:
    """Return live organization, repository, and environment secret metadata."""
    secrets: list[LiveSecret] = []
    organization_secrets = _gh_json(
        "secret",
        "list",
        "--org",
        manifest.organization,
        "--json",
        "name,visibility",
    )
    assert isinstance(organization_secrets, list)
    for item in organization_secrets:
        repositories: tuple[str, ...] = ()
        visibility = OrganizationVisibility(item["visibility"])
        if visibility is OrganizationVisibility.SELECTED:
            selected_repositories = _gh_paginated_items(
                f"orgs/{manifest.organization}/actions/secrets/{item['name']}/repositories"
                f"?per_page={GITHUB_API_PAGE_SIZE}",
                "repositories",
            )
            repositories = tuple(sorted(repository["full_name"] for repository in selected_repositories))
        secrets.append(
            OrganizationLiveSecret(
                name=item["name"],
                visibility=visibility,
                repositories=repositories,
            )
        )

    for repository in manifest.repositories:
        repository_secrets = _gh_json(
            "secret",
            "list",
            "--repo",
            repository,
            "--json",
            "name",
        )
        assert isinstance(repository_secrets, list)
        secrets.extend(RepositoryLiveSecret(name=item["name"], repository=repository) for item in repository_secrets)
        environments = _gh_paginated_items(
            f"repos/{repository}/environments?per_page={GITHUB_API_PAGE_SIZE}",
            "environments",
        )
        for environment in environments:
            environment_secrets = _gh_json(
                "secret",
                "list",
                "--repo",
                repository,
                "--env",
                environment["name"],
                "--json",
                "name",
            )
            assert isinstance(environment_secrets, list)
            secrets.extend(
                EnvironmentLiveSecret(
                    name=item["name"],
                    repository=repository,
                    environment=environment["name"],
                )
                for item in environment_secrets
            )
    return tuple(secrets)
