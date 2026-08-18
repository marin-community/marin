# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
from pathlib import Path

import pulumi
import pytest
from iac.github import resources as github_resources
from iac.github.audit import audit_credentials, discover_secret_references
from iac.github.credentials import (
    CREDENTIAL_SCHEMA_VERSION,
    CredentialManifest,
    CredentialScope,
    Disposition,
    EnvironmentCredential,
    Management,
    OrganizationCredential,
    OrganizationLiveSecret,
    OrganizationVisibility,
    Presence,
    RepositoryCredential,
    RepositoryLiveSecret,
    SecretReference,
    SourceKind,
    ValueSource,
    credential_manifest,
    load_stack_manifest,
)
from iac.github.resources import credential_resource_plans, register_credentials

REPO_ROOT = Path(__file__).resolve().parents[3]
STACK_CONFIG = REPO_ROOT / "infra" / "pulumi" / "github" / "Pulumi.marin-community.yaml"
EXAMPLE_REPOSITORY = "example/repo"

EXPECTED_REMOVAL_CANDIDATES = frozenset(
    {
        "repository:marin-community/marin:DISCORD_WEBHOOK_CODE_REVIEW",
        "repository:marin-community/marin:DUCKY_CW_ACCESS_KEY",
        "repository:marin-community/marin:GCP_ARTIFACT_REPOSITORY_SA_KEY",
        "repository:marin-community/marin:GCP_SA_KEY",
        "repository:marin-community/marin:GCP_SA_LOGGING_KEY",
        "repository:marin-community/marin:GEMINI_API_KEY",
        "repository:marin-community/marin:MARIN_SSH_KEY",
        "repository:marin-community/marin:NEW_GCP_JSON",
        "repository:marin-community/marin:RAY_AUTH_TOKEN",
        "repository:marin-community/marin:SPEEDRUN_LEADERBOARD_PAT",
    }
)


def test_committed_stack_covers_workflow_references_and_isolates_removal_candidates() -> None:
    manifest = load_stack_manifest(STACK_CONFIG)
    references = discover_secret_references(REPO_ROOT)

    report = audit_credentials(manifest, references)

    assert report.errors == ()
    assert set(report.removal_candidates) == EXPECTED_REMOVAL_CANDIDATES
    assert set(report.referenced_missing) == {
        "repository:marin-community/marin:ANTHROPIC_ADMIN_KEY",
        "repository:marin-community/marin:CLAUDE_MAX_OAUTH_TOKEN",
        "repository:marin-community/marin:COREWEAVE_API_TOKEN",
        "repository:marin-community/marin:OPENAI_ADMIN_KEY",
    }
    assert set(report.shadowed) == {"organization:GCP_PROJECT_ID"}


def test_committed_stack_models_every_present_external_secret_as_a_read_only_resource() -> None:
    manifest = load_stack_manifest(STACK_CONFIG)

    plans = credential_resource_plans(manifest)

    expected = {
        credential.key
        for credential in manifest.credentials
        if credential.presence is Presence.PRESENT and credential.management is Management.EXTERNAL
    }
    assert {plan.credential.key for plan in plans} == expected
    discord = next(plan for plan in plans if plan.credential.name == "DISCORD_WEBHOOK_INTERNAL_DISCUSS")
    assert discord.resource_id == "marin:DISCORD_WEBHOOK_INTERNAL_DISCUSS"


def test_registers_each_present_secret_and_selected_repository_access(monkeypatch) -> None:
    calls: list[tuple[str, str, dict]] = []

    class ResourceType:
        def __init__(self, kind: str):
            self.kind = kind

        def get(self, resource_name: str, **kwargs):
            calls.append((f"read-{self.kind}", resource_name, kwargs))
            return object.__new__(pulumi.CustomResource)

        def __call__(self, resource_name: str, **kwargs):
            calls.append((f"manage-{self.kind}", resource_name, kwargs))
            return object.__new__(pulumi.CustomResource)

    monkeypatch.setattr(github_resources.github, "ActionsSecret", ResourceType("repository"))
    monkeypatch.setattr(github_resources.github, "ActionsOrganizationSecret", ResourceType("organization"))
    monkeypatch.setattr(github_resources.github, "ActionsEnvironmentSecret", ResourceType("environment"))
    monkeypatch.setattr(
        github_resources.github,
        "ActionsOrganizationSecretRepositories",
        ResourceType("organization-access"),
    )
    manifest = load_stack_manifest(STACK_CONFIG)

    registered = register_credentials(manifest)

    registered_credentials = [
        credential
        for credential in manifest.credentials
        if credential.presence is Presence.PRESENT and credential.management is not Management.PULUMI
    ]
    external = [credential for credential in registered_credentials if credential.management is Management.EXTERNAL]
    scope_counts = Counter(credential.scope for credential in external)
    selected_organization_credentials = [
        credential
        for credential in registered_credentials
        if isinstance(credential, OrganizationCredential) and credential.visibility is OrganizationVisibility.SELECTED
    ]
    assert len(registered) == len(registered_credentials) + len(selected_organization_credentials)
    assert [kind for kind, _, _ in calls].count("read-organization") == scope_counts[CredentialScope.ORGANIZATION]
    assert [kind for kind, _, _ in calls].count("read-repository") == scope_counts[CredentialScope.REPOSITORY]
    assert [kind for kind, _, _ in calls].count("read-environment") == scope_counts[CredentialScope.ENVIRONMENT]
    assert [kind for kind, _, _ in calls].count("manage-organization") == 1
    assert [kind for kind, _, _ in calls].count("read-organization-access") == len(selected_organization_credentials)
    selected_access = next(kwargs for kind, _, kwargs in calls if kind == "read-organization-access")
    assert selected_access["id"] == "LOOM_TRIGGER_GH_TOKEN"


def test_registers_pulumi_managed_organization_secret_from_sealed_value(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    class ManagedOrganizationSecret:
        def __init__(self, resource_name: str, **kwargs):
            calls.append((resource_name, kwargs))

    monkeypatch.setattr(github_resources.github, "ActionsOrganizationSecret", ManagedOrganizationSecret)
    manifest = CredentialManifest(
        organization="example",
        repositories=(),
        credentials=(
            OrganizationCredential(
                name="SLACK_WEBHOOK_URL",
                presence=Presence.PRESENT,
                management=Management.SEALED,
                source=ValueSource(
                    kind=SourceKind.GCP_SECRET,
                    ref="gcp-secret://projects/example/secrets/slack-webhook/versions/1",
                ),
                disposition=Disposition.KEEP,
                visibility=OrganizationVisibility.ALL,
                key_id="github-actions-key-id",
                value_encrypted="github-sealed-ciphertext",
            ),
        ),
    )

    registered = register_credentials(manifest)

    assert len(registered) == 1
    assert len(calls) == 1
    resource_name, kwargs = calls[0]
    assert resource_name == "organization-slack-webhook-url"
    assert kwargs["secret_name"] == "SLACK_WEBHOOK_URL"
    assert kwargs["visibility"] == "all"
    assert kwargs["key_id"] == "github-actions-key-id"
    assert kwargs["value_encrypted"] == "github-sealed-ciphertext"
    assert kwargs["opts"].import_ == "SLACK_WEBHOOK_URL"


@pytest.mark.parametrize(
    "sealed_value",
    [
        {},
        {"key_id": "github-actions-key-id"},
        {"value_encrypted": "github-sealed-ciphertext"},
    ],
)
def test_manifest_requires_complete_sealed_value_for_pulumi_managed_credentials(sealed_value: dict[str, str]) -> None:
    credential = {
        "name": "TOKEN",
        "scope": "organization",
        "presence": "present",
        "management": "sealed",
        "source_kind": "gcp-secret",
        "source_ref": "gcp-secret://projects/example/secrets/token/versions/1",
        "disposition": "keep",
        "visibility": "all",
        **sealed_value,
    }

    with pytest.raises(ValueError):
        credential_manifest(
            schema_version=CREDENTIAL_SCHEMA_VERSION,
            organization="example",
            repositories=[],
            credentials=[credential],
        )


def test_live_audit_reports_missing_unmanaged_and_scope_drift() -> None:
    manifest = CredentialManifest(
        organization="example",
        repositories=(EXAMPLE_REPOSITORY,),
        credentials=(
            OrganizationCredential(
                name="ORG_TOKEN",
                presence=Presence.PRESENT,
                source=ValueSource(kind=SourceKind.MANUAL, ref="owner-recovery:org-token"),
                disposition=Disposition.KEEP,
                visibility=OrganizationVisibility.SELECTED,
                repositories=(EXAMPLE_REPOSITORY,),
            ),
            RepositoryCredential(
                name="REPO_TOKEN",
                presence=Presence.PRESENT,
                source=ValueSource(kind=SourceKind.MANUAL, ref="owner-recovery:repo-token"),
                disposition=Disposition.KEEP,
                repository=EXAMPLE_REPOSITORY,
            ),
        ),
    )
    live = (
        OrganizationLiveSecret(
            name="ORG_TOKEN",
            visibility=OrganizationVisibility.ALL,
        ),
        RepositoryLiveSecret(name="UNMANAGED", repository=EXAMPLE_REPOSITORY),
    )

    report = audit_credentials(manifest, {}, live)

    assert {finding.code for finding in report.errors} == {
        "declared-secret-missing",
        "repository-access-drift",
        "unmanaged-live-secret",
        "visibility-drift",
    }


def test_environment_secret_does_not_resolve_as_a_repository_secret() -> None:
    manifest = CredentialManifest(
        organization="example",
        repositories=(EXAMPLE_REPOSITORY,),
        credentials=(
            EnvironmentCredential(
                name="DEPLOY_TOKEN",
                presence=Presence.PRESENT,
                source=ValueSource(kind=SourceKind.MANUAL, ref="owner-recovery:deploy-token"),
                disposition=Disposition.KEEP,
                repository=EXAMPLE_REPOSITORY,
                environment="production",
            ),
        ),
    )
    references = {
        "DEPLOY_TOKEN": (SecretReference(path=".github/workflows/deploy.yaml", line=12),),
    }

    report = audit_credentials(manifest, references)

    assert [(finding.code, finding.credential) for finding in report.errors] == [
        ("undeclared-reference", "DEPLOY_TOKEN")
    ]


def test_environment_secret_resolves_inside_its_workflow_job() -> None:
    manifest = CredentialManifest(
        organization="example",
        repositories=(EXAMPLE_REPOSITORY,),
        credentials=(
            EnvironmentCredential(
                name="DEPLOY_TOKEN",
                presence=Presence.PRESENT,
                source=ValueSource(kind=SourceKind.MANUAL, ref="owner-recovery:deploy-token"),
                disposition=Disposition.KEEP,
                repository=EXAMPLE_REPOSITORY,
                environment="production",
            ),
        ),
    )
    references = {
        "DEPLOY_TOKEN": (SecretReference(path=".github/workflows/deploy.yaml", line=12, environment="production"),),
    }

    report = audit_credentials(manifest, references)

    assert report.errors == ()
    assert report.unreferenced == ()


def test_manifest_rejects_unpinned_gcp_secret_version() -> None:
    with pytest.raises(ValueError):
        credential_manifest(
            schema_version=CREDENTIAL_SCHEMA_VERSION,
            organization="example",
            repositories=[EXAMPLE_REPOSITORY],
            credentials=[
                {
                    "name": "TOKEN",
                    "scope": "repository",
                    "repository": EXAMPLE_REPOSITORY,
                    "presence": "present",
                    "source_kind": "gcp-secret",
                    "source_ref": "gcp-secret://projects/p/secrets/s/versions/latest",
                    "disposition": "keep",
                }
            ],
        )


def test_manifest_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError):
        credential_manifest(
            schema_version=CREDENTIAL_SCHEMA_VERSION,
            organization="example",
            repositories=[EXAMPLE_REPOSITORY],
            credentials=[
                {
                    "name": "TOKEN",
                    "scope": "repository",
                    "repository": EXAMPLE_REPOSITORY,
                    "presence": "present",
                    "source_kind": "manual",
                    "source_ref": "owner-recovery:token",
                    "disposition": "keep",
                    "unexpected": True,
                }
            ],
        )


def test_reference_scan_ignores_commented_secrets(tmp_path: Path) -> None:
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "example.yaml").write_text(
        """
env:
  ACTIVE: ${{ secrets.ACTIVE_TOKEN }}
  # RETIRED: ${{ secrets.RETIRED_TOKEN }}
""".lstrip()
    )

    references = discover_secret_references(tmp_path)

    assert set(references) == {"ACTIVE_TOKEN"}
