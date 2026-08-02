# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Diff live GCP IAM against what iam_data.py declares, to catch drift `pulumi preview` can't.

Every grant iam_data.py declares is a non-authoritative `*IAMMember`, so `pulumi preview` only
notices a live binding it already tracks in state. A grant nobody told Pulumi about — a newly
enabled API's service agent, a hand-run `gcloud ... add-iam-policy-binding`, an expanded role on
an existing member — lands invisibly. This module re-enumerates the live IAM surface and reports
what diverges from the checked-in config.

Scope: only non-human members are diffed. `user:<email>` principals are KMS-encrypted ciphertext
in iam_data.py (this file's history is public) and are managed through the encrypted grant
workflow plus code review, so diffing them would need decrypt access this read-only scan
deliberately does not hold. Service accounts, groups, and Google-managed service agents are plain
strings on both sides and diff exactly.
"""

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from iac.gcp.iam import GcpEncryptedMember, GcpRoleGrant

MARKER = "<!-- marin-iam-drift -->"

# Built-in roles far broader than the narrowly-scoped grants this IAM surface otherwise uses.
# A live binding of one of these to a service account is the headline drift case (#7713): owner
# or editor handed to an automation identity is exactly what a hand-run grant tends to leave.
BROAD_ROLES = frozenset(
    {
        "roles/owner",
        "roles/editor",
        "roles/iam.securityAdmin",
    }
)

# Google-managed service-agent domains whose name does not follow the `gcp-sa-<api>` convention,
# mapped to the API whose enablement creates them. Deliberately conservative: a domain absent
# here and not matching `gcp-sa-*` yields no API guess, so its agent is never flagged as
# orphaned. Better to miss a stale agent than to file a false "API disabled" finding.
SYSTEM_AGENT_APIS: dict[str, str] = {
    "compute-system": "compute.googleapis.com",
    "container-engine-robot": "container.googleapis.com",
    "container-analysis": "containeranalysis.googleapis.com",
    "containerregistry": "containerregistry.googleapis.com",
    "cloud-tpu": "tpu.googleapis.com",
    "cloud-filer": "file.googleapis.com",
    "cloud-redis": "redis.googleapis.com",
    "dataproc-accounts": "dataproc.googleapis.com",
    "storage-transfer-service": "storagetransfer.googleapis.com",
    "storageinsights": "storageinsights.googleapis.com",
    "websecurityscanner": "websecurityscanner.googleapis.com",
}

_GSA_SUFFIX = ".iam.gserviceaccount.com"
_GCP_SA_PREFIX = "gcp-sa-"


class Container(StrEnum):
    """The kind of resource a binding sits on — the same six surfaces iam_data.py declares."""

    PROJECT = "project"
    KMS = "kms"
    SECRET = "secret"
    BUCKET = "bucket"
    ARTIFACT_REPOSITORY = "artifact_repository"
    SERVICE_ACCOUNT = "service_account"


@dataclass(frozen=True)
class Binding:
    """One (resource, role, member) triple, the unit both live enumeration and the declared
    config flatten to. Conditions are intentionally not part of the identity: this surface uses
    them rarely and their titles would have to match exactly across live and config to diff."""

    container: Container
    resource: str
    role: str
    member: str


class FindingKind(StrEnum):
    UNDECLARED_BINDING = "undeclared_binding"
    DISABLED_API_AGENT = "disabled_api_agent"


@dataclass(frozen=True)
class Finding:
    """One divergence for triage. `broad` marks an undeclared binding of a `BROAD_ROLES` role,
    which sorts first and reads as the highest-severity drift."""

    kind: FindingKind
    member: str
    detail: str
    container: Container | None = None
    resource: str = ""
    role: str = ""
    broad: bool = False


def is_human(member: str) -> bool:
    """A `user:<email>` principal — out of scope for the diff (see the module docstring)."""
    return member.startswith("user:")


def redact(member: str) -> str:
    """Mask a human email's local part before it reaches a public issue. Service accounts,
    groups, and service agents are not personal identities and pass through unchanged."""
    if not is_human(member):
        return member
    email = member.removeprefix("user:")
    local, _, domain = email.partition("@")
    masked = f"{local[0]}***" if local else "***"
    return f"user:{masked}@{domain}" if domain else f"user:{masked}"


def service_agent_api(member: str, project: str) -> str | None:
    """The API whose enablement would create this Google-managed service agent, or None if
    `member` is not a recognizable agent (a user-managed SA in this project, another project's
    SA, or an unmapped domain). None means "do not judge this identity's API state"."""
    if not member.startswith("serviceAccount:"):
        return None
    _, _, domain = member.removeprefix("serviceAccount:").partition("@")
    if domain in ("cloudservices.gserviceaccount.com",) or domain.endswith("developer.gserviceaccount.com"):
        # The core Google APIs / default compute agents; never tied to a single toggleable API.
        return None
    if not domain.endswith(_GSA_SUFFIX):
        return None
    name = domain[: -len(_GSA_SUFFIX)]
    if name == project:
        # A user-managed service account this project owns, not a Google agent.
        return None
    if name.startswith(_GCP_SA_PREFIX):
        return f"{name.removeprefix(_GCP_SA_PREFIX)}.googleapis.com"
    return SYSTEM_AGENT_APIS.get(name)


def declared_bindings(
    project: str,
    crypto_key_id: str,
    *,
    project_grants: Iterable[GcpRoleGrant],
    kms_grants: Iterable[GcpRoleGrant],
    secrets: Iterable,
    buckets: Iterable,
    artifact_repositories: Iterable,
    service_accounts: Iterable,
) -> set[Binding]:
    """Flatten iam_data.py's declared grants to the same `Binding` triples live enumeration
    produces. Human members (encrypted, or a stray plain `user:` string) are dropped: they are
    out of scope for the diff, so keeping them would only add keys the live side never matches."""
    bindings: set[Binding] = set()

    def add(container: Container, resource: str, grants: Iterable[GcpRoleGrant]) -> None:
        for grant in grants:
            for member in grant.members:
                if isinstance(member, GcpEncryptedMember) or is_human(member):
                    continue
                bindings.add(Binding(container, resource, grant.role, member))

    add(Container.PROJECT, project, project_grants)
    add(Container.KMS, crypto_key_id, kms_grants)
    for secret in secrets:
        add(Container.SECRET, secret.secret, secret.grants)
    for bucket in buckets:
        add(Container.BUCKET, bucket.bucket, bucket.grants)
    for repo in artifact_repositories:
        add(Container.ARTIFACT_REPOSITORY, f"{repo.location}/{repo.repository}", repo.grants)
    for account in service_accounts:
        add(Container.SERVICE_ACCOUNT, account.email, account.grants)
    return bindings


def classify(
    live: Iterable[Binding],
    declared: set[Binding],
    *,
    project: str,
    enabled_services: frozenset[str] | None,
) -> list[Finding]:
    """Diff live bindings against the declared set and flag orphaned service agents.

    `enabled_services` is None when the scan lacks `serviceusage.services.list` — the disabled-API
    check is skipped rather than guessed, and only undeclared-binding findings are produced.
    """
    findings: list[Finding] = []
    live = list(live)

    for binding in live:
        if is_human(binding.member) or binding in declared:
            continue
        broad = binding.role in BROAD_ROLES
        detail = "broad role bound outside Pulumi config" if broad else "live binding absent from Pulumi config"
        findings.append(
            Finding(
                kind=FindingKind.UNDECLARED_BINDING,
                member=binding.member,
                detail=detail,
                container=binding.container,
                resource=binding.resource,
                role=binding.role,
                broad=broad,
            )
        )

    if enabled_services is not None:
        for member in sorted({binding.member for binding in live}):
            api = service_agent_api(member, project)
            if api is None or api in enabled_services:
                continue
            findings.append(
                Finding(
                    kind=FindingKind.DISABLED_API_AGENT,
                    member=member,
                    detail=f"backing API {api} looks disabled; GCP does not remove the agent automatically",
                )
            )

    findings.sort(key=_severity)
    return findings


def _severity(finding: Finding) -> tuple[int, str, str, str]:
    # Broad undeclared grants first, then orphaned agents, then ordinary undeclared bindings.
    if finding.kind is FindingKind.UNDECLARED_BINDING and finding.broad:
        rank = 0
    elif finding.kind is FindingKind.DISABLED_API_AGENT:
        rank = 1
    else:
        rank = 2
    return rank, finding.resource, finding.role, finding.member


def fingerprint(findings: Iterable[Finding]) -> str:
    """Stable content hash of the finding set, embedded in the tracking issue so a re-scan that
    turns up the same drift updates nothing and re-notifies no one."""
    canonical = "\n".join(
        sorted(f"{f.kind}|{f.container or ''}|{f.resource}|{f.role}|{f.member}|{f.broad}" for f in findings)
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _table(rows: list[tuple[str, ...]], header: tuple[str, ...]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(cell or "—" for cell in row) + " |" for row in rows]
    return lines


def render_markdown(findings: list[Finding], *, project: str, run_url: str | None) -> str:
    """The tracking-issue body: the marker, a fingerprint, and one table per finding kind.

    Callers pass this only when `findings` is non-empty; an empty scan resolves the issue rather
    than rendering a body.
    """
    broad = [f for f in findings if f.kind is FindingKind.UNDECLARED_BINDING and f.broad]
    plain = [f for f in findings if f.kind is FindingKind.UNDECLARED_BINDING and not f.broad]
    orphaned = [f for f in findings if f.kind is FindingKind.DISABLED_API_AGENT]

    lines = [
        MARKER,
        f"<!-- fingerprint:{fingerprint(findings)} -->",
        f"# GCP IAM drift on `{project}`",
        "",
        f"The scheduled scan found {len(findings)} binding(s) live in GCP that diverge from "
        "`infra/pulumi/src/iac/gcp/iam_data.py`. These are non-authoritative grants, so "
        "`pulumi preview` cannot see them; add each legitimate one to `iam_data.py` (or revoke "
        "it in GCP) to clear this issue. Human `user:` grants are managed separately and are not "
        "diffed here.",
        "",
    ]

    if broad:
        lines += ["## Broad roles bound outside Pulumi", ""]
        lines += _table(
            [(f.resource, f.role, redact(f.member)) for f in broad],
            ("Resource", "Role", "Member"),
        )
        lines += [""]

    if orphaned:
        lines += ["## Service agents whose API looks disabled", ""]
        lines += _table(
            [(f.member, f.detail) for f in orphaned],
            ("Service agent", "Note"),
        )
        lines += [""]

    if plain:
        lines += ["## Undeclared bindings", ""]
        lines += _table(
            [(str(f.container or ""), f.resource, f.role, redact(f.member)) for f in plain],
            ("Container", "Resource", "Role", "Member"),
        )
        lines += [""]

    if run_url:
        lines += [f"_Scan run: {run_url}_"]
    return "\n".join(lines).rstrip() + "\n"
