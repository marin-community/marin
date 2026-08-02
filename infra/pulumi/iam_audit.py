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
from pathlib import Path

from google.cloud import kms_v1
from iac.gcp import iam_data
from iac.gcp.iam import GcpEncryptedMember, GcpRoleGrant
from iac.gcp.iam_config import (
    IAM_DATA_PATH,
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


def _iter_grants() -> Iterator[tuple[str, str, GcpRoleGrant]]:
    """Yield (container, resource, grant) for every declared role grant."""
    for grant in iam_data.PROJECT_GRANTS:
        yield "PROJECT_GRANTS", PROJECT, grant
    for grant in iam_data.KMS_GRANTS:
        yield "KMS_GRANTS", crypto_key_id(), grant
    for secret in iam_data.SECRETS:
        for grant in secret.grants:
            yield "SECRETS", secret.secret, grant
    for bucket in iam_data.BUCKETS:
        for grant in bucket.grants:
            yield "BUCKETS", bucket.bucket, grant
    for repo in iam_data.ARTIFACT_REPOSITORIES:
        for grant in repo.grants:
            yield "ARTIFACT_REPOSITORIES", f"{repo.location}/{repo.repository}", grant
    for account in iam_data.SERVICE_ACCOUNTS:
        for grant in account.grants:
            yield "SERVICE_ACCOUNTS", account.email, grant


def decrypt(out_path: Path) -> None:
    """Write one audit record per opaque principal ID."""
    print(_NONDETERMINISM_WARNING)

    client = kms_v1.KeyManagementServiceClient()
    key_id = crypto_key_id()
    principal_ids = {principal.ciphertext: principal.principal_id for principal in iam_data.CONFIG.principals}
    records = {
        principal.principal_id: {
            "principal_id": principal.principal_id,
            "ciphertext": principal.ciphertext,
            "email": decrypt_member(client, key_id, principal.ciphertext),
            "grants": [],
        }
        for principal in iam_data.CONFIG.principals
    }
    for container, resource, grant in _iter_grants():
        for member in grant.members:
            if not isinstance(member, GcpEncryptedMember):
                continue
            records[principal_ids[member.ciphertext]]["grants"].append(
                {"container": container, "resource": resource, "role": grant.role}
            )

    ordered_records = sorted(records.values(), key=lambda record: record["email"])
    for record in ordered_records:
        record["grants"].sort(key=lambda grant: (grant["container"], grant["resource"], grant["role"]))
    out_path.write_text(json.dumps(ordered_records, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {len(ordered_records)} decrypted principals "
        f"({sum(len(record['grants']) for record in ordered_records)} grants) to {out_path}"
    )


def encrypt(in_path: Path) -> None:
    """Re-encrypt every principal from an audit file back into the YAML registry."""
    print(_NONDETERMINISM_WARNING)

    records = json.loads(in_path.read_text(encoding="utf-8"))
    config = load_iam_config()
    file_ids = {principal.principal_id for principal in config.principals}
    record_ids = {record["principal_id"] for record in records}
    if missing := file_ids - record_ids:
        raise SystemExit(f"{len(missing)} principal(s) in {IAM_DATA_PATH} are missing from {in_path}")
    if unknown := record_ids - file_ids:
        raise SystemExit(f"{len(unknown)} principal(s) in {in_path} are missing from {IAM_DATA_PATH}")

    emails = {record["principal_id"]: record["email"] for record in records}
    client = kms_v1.KeyManagementServiceClient()
    key_id = crypto_key_id()
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
