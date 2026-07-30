# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconcile GitHub Actions credential declarations with references and live metadata."""

import re
from collections import defaultdict
from pathlib import Path

from iac.github.credentials import (
    AuditReport,
    Credential,
    CredentialManifest,
    CredentialScope,
    Disposition,
    EnvironmentCredential,
    Finding,
    LiveSecret,
    OrganizationCredential,
    OrganizationLiveSecret,
    OrganizationVisibility,
    Presence,
    RepositoryCredential,
    SecretReference,
)

BUILTIN_ACTIONS_SECRETS = frozenset({"GITHUB_TOKEN"})
SECRET_REFERENCE = re.compile(r"\bsecrets\.([A-Za-z_][A-Za-z0-9_]*)")


def discover_secret_references(repo_root: Path) -> dict[str, tuple[SecretReference, ...]]:
    """Return Actions secret references in workflow and composite-action YAML."""
    references: dict[str, list[SecretReference]] = defaultdict(list)
    roots = (repo_root / ".github" / "workflows", repo_root / ".github" / "actions")
    paths = sorted(path for root in roots for suffix in ("*.yaml", "*.yml") for path in root.rglob(suffix))
    for path in paths:
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            for match in SECRET_REFERENCE.finditer(line):
                references[match.group(1)].append(
                    SecretReference(path=str(path.relative_to(repo_root)), line=line_number)
                )
    return {name: tuple(found) for name, found in sorted(references.items())}


def _accessible_to_repository(credential: Credential, repository: str) -> bool:
    if isinstance(credential, RepositoryCredential):
        return credential.repository == repository
    if isinstance(credential, EnvironmentCredential):
        return False
    assert isinstance(credential, OrganizationCredential)
    if credential.visibility is OrganizationVisibility.ALL:
        return True
    if credential.visibility is OrganizationVisibility.PRIVATE:
        return False
    return repository in credential.repositories


def _resolved_credential(
    manifest: CredentialManifest,
    name: str,
    repository: str,
) -> Credential | None:
    candidates = [
        credential
        for credential in manifest.credentials
        if credential.name == name and _accessible_to_repository(credential, repository)
    ]
    scope_order = {
        CredentialScope.ENVIRONMENT: 0,
        CredentialScope.REPOSITORY: 1,
        CredentialScope.ORGANIZATION: 2,
    }
    candidates.sort(key=lambda credential: (credential.presence is not Presence.PRESENT, scope_order[credential.scope]))
    return candidates[0] if candidates else None


def _live_findings(manifest: CredentialManifest, live_secrets: tuple[LiveSecret, ...]) -> list[Finding]:
    findings: list[Finding] = []
    declarations = {credential.key: credential for credential in manifest.credentials}
    live = {secret.key: secret for secret in live_secrets}
    for key, credential in declarations.items():
        live_secret = live.get(key)
        if credential.presence is Presence.REFERENCED_MISSING:
            if live_secret is not None:
                findings.append(
                    Finding("expected-missing-present", "secret now exists; mark it present", credential.label)
                )
            continue
        if live_secret is None:
            findings.append(
                Finding("declared-secret-missing", "declared present but absent from GitHub", credential.label)
            )
            continue
        if isinstance(credential, OrganizationCredential):
            assert isinstance(live_secret, OrganizationLiveSecret)
            if live_secret.visibility != credential.visibility:
                findings.append(
                    Finding(
                        "visibility-drift",
                        f"expected {credential.visibility!r}, found {live_secret.visibility!r}",
                        credential.label,
                    )
                )
            if live_secret.repositories != credential.repositories:
                findings.append(
                    Finding(
                        "repository-access-drift",
                        f"expected {list(credential.repositories)!r}, found {list(live_secret.repositories)!r}",
                        credential.label,
                    )
                )
    for key in live:
        if key not in declarations:
            findings.append(
                Finding("unmanaged-live-secret", "present in GitHub but absent from the catalog", ":".join(key))
            )
    return findings


def audit_credentials(
    manifest: CredentialManifest,
    references: dict[str, tuple[SecretReference, ...]],
    live_secrets: tuple[LiveSecret, ...] | None = None,
) -> AuditReport:
    """Compare the catalog with workflow references and optional live metadata."""
    errors: list[Finding] = []
    consumers: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for name, found in references.items():
        if name in BUILTIN_ACTIONS_SECRETS:
            continue
        for repository in manifest.repositories:
            credential = _resolved_credential(manifest, name, repository)
            if credential is not None:
                consumers[credential.key].update(reference.path for reference in found)
                break
        else:
            locations = ", ".join(f"{reference.path}:{reference.line}" for reference in found)
            errors.append(Finding("undeclared-reference", f"{name} is referenced at {locations}", name))

    for credential in manifest.credentials:
        if credential.disposition is Disposition.REMOVE_CANDIDATE and consumers[credential.key]:
            errors.append(
                Finding(
                    "removal-candidate-referenced",
                    f"referenced by {sorted(consumers[credential.key])}",
                    credential.label,
                )
            )
    if live_secrets is not None:
        errors.extend(_live_findings(manifest, live_secrets))

    present = [credential for credential in manifest.credentials if credential.presence is Presence.PRESENT]
    unreferenced = sorted(credential.label for credential in present if not consumers[credential.key])
    removal_candidates = sorted(
        credential.label for credential in present if credential.disposition is Disposition.REMOVE_CANDIDATE
    )
    referenced_missing = sorted(
        credential.label
        for credential in manifest.credentials
        if credential.presence is Presence.REFERENCED_MISSING and consumers[credential.key]
    )
    shadowed = sorted(
        credential.label
        for credential in manifest.credentials
        if isinstance(credential, OrganizationCredential)
        and credential.presence is Presence.PRESENT
        and any(
            other.name == credential.name
            and isinstance(other, RepositoryCredential | EnvironmentCredential)
            and other.presence is Presence.PRESENT
            and _accessible_to_repository(credential, other.repository)
            for other in manifest.credentials
        )
    )
    return AuditReport(
        errors=tuple(errors),
        referenced_missing=tuple(referenced_missing),
        removal_candidates=tuple(removal_candidates),
        unreferenced=tuple(unreferenced),
        shadowed=tuple(shadowed),
        recoverable_credentials=tuple(
            sorted(c.label for c in manifest.credentials if c.source.does_not_require_owner_recovery)
        ),
        manual_recovery_credentials=tuple(
            sorted(c.label for c in manifest.credentials if not c.source.does_not_require_owner_recovery)
        ),
    )
