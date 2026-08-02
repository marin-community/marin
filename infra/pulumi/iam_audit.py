#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Round-trip iam_data.py's encrypted `user:<email>` members through a local JSON file, for
offline audit and small edits (fixing a typo, rotating a person out) without hand-decrypting
KMS ciphertext.

    uv run --package marin-iac --extra deploy python infra/pulumi/iam_audit.py decrypt --out /tmp/iam_members.json
    # review, or edit an "email" field, then:
    uv run --package marin-iac --extra deploy python infra/pulumi/iam_audit.py encrypt --in /tmp/iam_members.json

The JSON file holds real email addresses in plaintext — keep it out of the repo and delete it
once you're done. `encrypt` only rotates existing entries, matched by their original ciphertext;
adding or removing a member still needs a direct edit to iam_data.py, matching that file's own
docstring guidance for structural changes.
"""

import argparse
import base64
import json
import re
from collections.abc import Iterator
from pathlib import Path

from google.cloud import kms_v1
from iac.gcp import iam_data
from iac.gcp.iam import GcpEncryptedMember, GcpRoleGrant

# Matches lib/iris/config/marin.yaml's provisioning.gcp.project. iam_data.KMS_LOCATION/
# KMS_KEY_RING/KMS_KEY only cover the key ring/key within it, not the project it lives under.
PROJECT = "hai-gcp-models"

IAM_DATA_PATH = Path(__file__).resolve().parent / "src" / "iac" / "gcp" / "iam_data.py"

_CIPHERTEXT_RE = re.compile(r'GcpEncryptedMember\(\s*ciphertext="([^"]+)"\s*\)')

_NONDETERMINISM_WARNING = (
    "NOTE: GCP KMS encryption is non-deterministic — re-encrypting produces different "
    "ciphertext for every entry, even ones whose email you didn't change, so the resulting "
    "diff in iam_data.py won't be limited to the member(s) you actually edited."
)


def _crypto_key_id() -> str:
    return (
        f"projects/{PROJECT}/locations/{iam_data.KMS_LOCATION}/keyRings/{iam_data.KMS_KEY_RING}/"
        f"cryptoKeys/{iam_data.KMS_KEY}"
    )


def _iter_grants() -> Iterator[tuple[str, str, GcpRoleGrant]]:
    """Yield (container, resource, grant) for every GcpRoleGrant iam_data.py declares."""
    for grant in iam_data.PROJECT_GRANTS:
        yield "PROJECT_GRANTS", PROJECT, grant
    for grant in iam_data.KMS_GRANTS:
        yield "KMS_GRANTS", _crypto_key_id(), grant
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
    """One record per unique ciphertext (i.e. per person, not per grant) — the same person's
    ciphertext is reused verbatim everywhere they're granted access, so grouping this way is
    both a more useful audit view and what makes `encrypt` unambiguous: rotating a person's
    email replaces every occurrence of their old ciphertext in one pass."""
    print(_NONDETERMINISM_WARNING)

    client = kms_v1.KeyManagementServiceClient()
    crypto_key_id = _crypto_key_id()
    by_ciphertext: dict[str, dict] = {}
    for container, resource, grant in _iter_grants():
        for member in grant.members:
            if not isinstance(member, GcpEncryptedMember):
                continue
            record = by_ciphertext.get(member.ciphertext)
            if record is None:
                response = client.decrypt(name=crypto_key_id, ciphertext=base64.b64decode(member.ciphertext))
                record = {
                    "ciphertext": member.ciphertext,
                    "email": f"user:{response.plaintext.decode('utf-8')}",
                    "grants": [],
                }
                by_ciphertext[member.ciphertext] = record
            record["grants"].append({"container": container, "resource": resource, "role": grant.role})

    records = sorted(by_ciphertext.values(), key=lambda r: r["email"])
    for record in records:
        record["grants"].sort(key=lambda g: (g["container"], g["resource"], g["role"]))
    out_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} decrypted members ({sum(len(r['grants']) for r in records)} grants) to {out_path}")


def encrypt(in_path: Path) -> None:
    print(_NONDETERMINISM_WARNING)

    records = json.loads(in_path.read_text(encoding="utf-8"))
    client = kms_v1.KeyManagementServiceClient()
    crypto_key_id = _crypto_key_id()
    text = IAM_DATA_PATH.read_text(encoding="utf-8")

    file_ciphertexts = set(_CIPHERTEXT_RE.findall(text))
    seen = set()
    for record in records:
        old_ciphertext = record["ciphertext"]
        if old_ciphertext not in file_ciphertexts:
            raise SystemExit(
                f"ciphertext for {record['email']} not found in {IAM_DATA_PATH} — adding a new "
                "member isn't supported by this tool, edit iam_data.py directly"
            )
        seen.add(old_ciphertext)

        email = record["email"]
        assert email.startswith("user:"), f"expected a user:<email> principal, got {email!r}"
        response = client.encrypt(name=crypto_key_id, plaintext=email.removeprefix("user:").encode("utf-8"))
        new_ciphertext = base64.b64encode(response.ciphertext).decode("ascii")
        text = text.replace(old_ciphertext, new_ciphertext)

    missing = file_ciphertexts - seen
    if missing:
        raise SystemExit(
            f"{len(missing)} member(s) in {IAM_DATA_PATH} are missing from {in_path} — removing a "
            "member isn't supported by this tool, edit iam_data.py directly"
        )

    IAM_DATA_PATH.write_text(text, encoding="utf-8")
    print(f"re-encrypted {len(records)} members into {IAM_DATA_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    decrypt_parser = subparsers.add_parser("decrypt", help="Decrypt iam_data.py's encrypted members to a JSON file.")
    decrypt_parser.add_argument("--out", type=Path, required=True)

    encrypt_parser = subparsers.add_parser("encrypt", help="Re-encrypt a JSON file's members back into iam_data.py.")
    encrypt_parser.add_argument("--in", dest="in_path", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "decrypt":
        decrypt(args.out)
    else:
        encrypt(args.in_path)


if __name__ == "__main__":
    main()
