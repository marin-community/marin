#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Round-trip encrypted IAM principals through a local JSON audit file.

    uv run --package marin-iac --extra deploy python infra/pulumi/iam_audit.py \
      decrypt --out /tmp/iam_members.json
    # review, or edit an "email" field, then:
    uv run --package marin-iac --extra deploy python infra/pulumi/iam_audit.py \
      encrypt --in /tmp/iam_members.json

The JSON file holds real email addresses in plaintext. Keep it out of the repo
and delete it when the audit is complete.
"""

import argparse
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from google.cloud import kms_v1
from iac.gcp.iam import GcpEncryptedMember, GcpRoleGrant
from iac.gcp.iam_config import (
    IAM_DATA_PATH,
    GcpIamConfig,
    GcpPrincipal,
    load_iam_config,
    replace_principals,
    write_iam_config,
)
from iac.gcp.iam_kms import PROJECT, crypto_key_id, decrypt_member, encrypt_email

_NONDETERMINISM_WARNING = (
    "NOTE: GCP KMS encryption is non-deterministic. Re-encrypting changes the "
    "ciphertext line for every audited principal, even when its email is unchanged."
)


@dataclass(frozen=True, order=True)
class AuditGrant:
    """One resource grant attached to a decrypted audit principal."""

    container: str
    resource: str
    role: str

    def json_object(self) -> dict[str, str]:
        return {
            "container": self.container,
            "resource": self.resource,
            "role": self.role,
        }

    @classmethod
    def from_json_object(cls, value: object, path: str) -> "AuditGrant":
        fields = _json_fields(value, path, frozenset({"container", "resource", "role"}))
        return cls(
            container=_json_string(fields["container"], f"{path}.container"),
            resource=_json_string(fields["resource"], f"{path}.resource"),
            role=_json_string(fields["role"], f"{path}.role"),
        )


@dataclass(frozen=True)
class AuditPrincipal:
    """One decrypted principal and all grants that reference it."""

    principal_id: str
    email: str
    grants: tuple[AuditGrant, ...]

    def json_object(self) -> dict[str, object]:
        return {
            "principal_id": self.principal_id,
            "email": self.email,
            "grants": [grant.json_object() for grant in self.grants],
        }

    @classmethod
    def from_json_object(cls, value: object, path: str) -> "AuditPrincipal":
        fields = _json_fields(value, path, frozenset({"principal_id", "email", "grants"}))
        raw_grants = fields["grants"]
        if not isinstance(raw_grants, list):
            raise ValueError(f"{path}.grants must be a list")
        return cls(
            principal_id=_json_string(fields["principal_id"], f"{path}.principal_id"),
            email=_json_string(fields["email"], f"{path}.email"),
            grants=tuple(
                AuditGrant.from_json_object(grant, f"{path}.grants[{index}]") for index, grant in enumerate(raw_grants)
            ),
        )


def _json_fields(value: object, path: str, required: frozenset[str]) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} must be an object")
    missing = required - value.keys()
    unknown = value.keys() - required
    if missing:
        raise ValueError(f"{path} is missing field(s): {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{path} has unknown field(s): {', '.join(sorted(unknown))}")
    return value


def _json_string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    return value


def _audit_principals(path: Path) -> tuple[AuditPrincipal, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list")
    return tuple(AuditPrincipal.from_json_object(record, f"{path}[{index}]") for index, record in enumerate(raw))


def _iter_grants(config: GcpIamConfig) -> Iterator[tuple[str, str, GcpRoleGrant]]:
    """Yield (container, resource, grant) for every declared role grant."""
    for grant in config.project_grants:
        yield "project_grants", PROJECT, grant
    key_id = crypto_key_id(config)
    for grant in config.kms_grants:
        yield "kms_grants", key_id, grant
    for secret in config.secrets:
        for grant in secret.grants:
            yield "secrets", secret.secret, grant
    for bucket in config.buckets:
        for grant in bucket.grants:
            yield "buckets", bucket.bucket, grant
    for repo in config.artifact_repositories:
        for grant in repo.grants:
            yield "artifact_repositories", f"{repo.location}/{repo.repository}", grant
    for account in config.service_accounts:
        for grant in account.grants:
            yield "service_accounts", account.email, grant


def decrypt(out_path: Path) -> None:
    """Write one audit record per opaque principal ID."""
    print(_NONDETERMINISM_WARNING)

    config = load_iam_config()
    client = kms_v1.KeyManagementServiceClient()
    key_id = crypto_key_id(config)
    principal_ids = {principal.ciphertext: principal.principal_id for principal in config.principals}
    grants_by_id: dict[str, list[AuditGrant]] = {principal.principal_id: [] for principal in config.principals}
    for container, resource, grant in _iter_grants(config):
        for member in grant.members:
            if not isinstance(member, GcpEncryptedMember):
                continue
            grants_by_id[principal_ids[member.ciphertext]].append(
                AuditGrant(container=container, resource=resource, role=grant.role)
            )

    records = tuple(
        sorted(
            (
                AuditPrincipal(
                    principal_id=principal.principal_id,
                    email=decrypt_member(client, key_id, principal.ciphertext),
                    grants=tuple(sorted(grants_by_id[principal.principal_id])),
                )
                for principal in config.principals
            ),
            key=lambda record: record.email,
        )
    )
    out_path.write_text(
        json.dumps([record.json_object() for record in records], indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(records)} decrypted principals "
        f"({sum(len(record.grants) for record in records)} grants) to {out_path}"
    )


def encrypt(in_path: Path) -> None:
    """Re-encrypt every principal from an audit file back into the YAML registry."""
    print(_NONDETERMINISM_WARNING)

    records = _audit_principals(in_path)
    config = load_iam_config()
    file_ids = {principal.principal_id for principal in config.principals}
    record_ids = {record.principal_id for record in records}
    if len(record_ids) != len(records):
        raise SystemExit(f"{in_path} contains duplicate principal IDs")
    if missing := file_ids - record_ids:
        raise SystemExit(f"{len(missing)} principal(s) in {IAM_DATA_PATH} are missing from {in_path}")
    if unknown := record_ids - file_ids:
        raise SystemExit(f"{len(unknown)} principal(s) in {in_path} are missing from {IAM_DATA_PATH}")

    emails = {record.principal_id: record.email for record in records}
    client = kms_v1.KeyManagementServiceClient()
    key_id = crypto_key_id(config)
    principals = tuple(
        GcpPrincipal(
            principal_id=principal.principal_id,
            ciphertext=encrypt_email(
                client,
                key_id,
                emails[principal.principal_id].removeprefix("user:"),
            ),
        )
        for principal in config.principals
    )
    write_iam_config(replace_principals(config, principals))
    print(f"re-encrypted {len(principals)} principals into {IAM_DATA_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    decrypt_parser = subparsers.add_parser("decrypt", help="Decrypt IAM principals to a local JSON audit file.")
    decrypt_parser.add_argument("--out", type=Path, required=True)

    encrypt_parser = subparsers.add_parser("encrypt", help="Re-encrypt a JSON audit file into the IAM YAML.")
    encrypt_parser.add_argument("--in", dest="in_path", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "decrypt":
        decrypt(args.out)
    else:
        encrypt(args.in_path)


if __name__ == "__main__":
    main()
