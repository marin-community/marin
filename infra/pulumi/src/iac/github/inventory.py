# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read GitHub secret and ruleset metadata through the GitHub CLI."""

import json
import subprocess
from dataclasses import dataclass

from iac.github.credentials import (
    CredentialManifest,
    EnvironmentLiveSecret,
    LiveSecret,
    OrganizationLiveSecret,
    OrganizationVisibility,
    RepositoryLiveSecret,
)

GITHUB_API_PAGE_SIZE = 100


@dataclass(frozen=True)
class RequiredStatusCheck:
    context: str
    integration_id: int


def _gh_json(*args: str) -> object:
    result = subprocess.run(["gh", *args], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _gh_paginated_items(endpoint: str, collection: str) -> tuple[dict, ...]:
    pages = _gh_json("api", "--paginate", "--slurp", endpoint)
    assert isinstance(pages, list)
    return tuple(item for page in pages for item in page[collection])


def github_required_status_checks(repository: str, ruleset_name: str) -> tuple[RequiredStatusCheck, ...]:
    """Return the required checks from one repository ruleset."""
    rulesets = _gh_json("api", f"repos/{repository}/rulesets?includes_parents=false")
    assert isinstance(rulesets, list)
    matches = [ruleset for ruleset in rulesets if ruleset["name"] == ruleset_name]
    if len(matches) != 1:
        raise ValueError(f"expected one {ruleset_name!r} ruleset for {repository}; found {len(matches)}")
    ruleset = _gh_json("api", f"repos/{repository}/rulesets/{matches[0]['id']}")
    assert isinstance(ruleset, dict)
    status_rules = [rule for rule in ruleset["rules"] if rule["type"] == "required_status_checks"]
    if len(status_rules) != 1:
        raise ValueError(f"expected one required-status-check rule in {ruleset_name!r}; found {len(status_rules)}")
    return tuple(
        RequiredStatusCheck(context=check["context"], integration_id=check["integration_id"])
        for check in status_rules[0]["parameters"]["required_status_checks"]
    )


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
