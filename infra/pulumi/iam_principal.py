#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grant project roles or decrypt changed IAM principals.

    # Add one person to existing or new project-level role grants.
    uv run --package marin-iac --extra deploy \
      python infra/pulumi/iam_principal.py grant alice@openathena.ai \
      --project-role roles/logging.viewer \
      --project-role roles/monitoring.viewer

    # Reveal the people behind principal references changed by a grant PR.
    git diff origin/main...HEAD -- infra/pulumi/src/iac/gcp/iam_data.yaml \
      | uv run --package marin-iac --extra deploy \
          python infra/pulumi/iam_principal.py decrypt --diff

These commands use the marin-iac KMS key. Plaintext emails remain outside Git.
"""

import argparse
import re
import sys
from pathlib import Path

import yaml
from google.cloud import kms_v1
from iac.gcp.iam_config import (
    IAM_DATA_PATH,
    PRINCIPAL_ID_PATTERN,
    grant_project_roles,
    load_iam_config,
    register_principal,
    write_iam_config,
)
from iac.gcp.iam_kms import crypto_key_id, decrypt_member, encrypt_email
from iris.cluster.config import PRINCIPAL_REFERENCE_PREFIX, config_to_dict, load_config

_PRINCIPAL_REFERENCE_PATTERNS = (
    re.compile(rf"^[+-]\s*-\s+principal:\s+({PRINCIPAL_ID_PATTERN})\s*$"),
    re.compile(rf'principals\["({PRINCIPAL_ID_PATTERN})"\]'),
    re.compile(rf"{re.escape(PRINCIPAL_REFERENCE_PREFIX)}({PRINCIPAL_ID_PATTERN})"),
)
_PRINCIPAL_RECORD_RE = re.compile(rf"^[+-]\s*({PRINCIPAL_ID_PATTERN}):\s+(\S+)\s*$")


def grant(email: str, project_roles: tuple[str, ...]) -> None:
    """Update project role grants for one encrypted human principal."""
    config = load_iam_config()
    client = kms_v1.KeyManagementServiceClient()
    key_id = crypto_key_id(config)
    updated = grant_project_roles(
        config,
        email,
        project_roles,
        decrypt_ciphertext=lambda ciphertext: decrypt_member(client, key_id, ciphertext),
        encrypt_email=lambda plaintext: encrypt_email(client, key_id, plaintext),
    )
    if updated == config:
        print("all requested grants already exist")
        return
    write_iam_config(updated)
    print(f"updated {IAM_DATA_PATH} with {len(project_roles)} project role grant(s)")


def register(email: str) -> None:
    """Register one encrypted human principal and print its opaque ID."""
    config = load_iam_config()
    client = kms_v1.KeyManagementServiceClient()
    key_id = crypto_key_id(config)
    registration = register_principal(
        config,
        email,
        decrypt_ciphertext=lambda ciphertext: decrypt_member(client, key_id, ciphertext),
        encrypt_email=lambda plaintext: encrypt_email(client, key_id, plaintext),
    )
    if registration.created:
        write_iam_config(registration.config)
    print(registration.principal.principal_id)


def decrypt_ciphertexts(ciphertexts: list[str]) -> None:
    """Print plaintext principals for explicit ciphertext arguments."""
    config = load_iam_config()
    client = kms_v1.KeyManagementServiceClient()
    key_id = crypto_key_id(config)
    for ciphertext in ciphertexts:
        print(decrypt_member(client, key_id, ciphertext))


def render_iris(config_path: Path, output_path: Path) -> None:
    """Resolve IAM references in Iris identity fields into a private deployment file."""
    output_path = output_path.resolve()
    if output_path.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("Rendered Iris configs must stay outside the checkout")
    config = load_config(config_path)
    identities = {user for tier in config.user_budgets for user in tier.user_ids}
    if config.auth is not None:
        identities.update((*config.auth.user_roles, *config.auth.admin_users, *config.auth.allowed_submitters))
    iam = load_iam_config()
    ciphertexts = {principal.principal_id: principal.ciphertext for principal in iam.principals}
    client = kms_v1.KeyManagementServiceClient()
    key_id = crypto_key_id(iam)
    emails = {
        reference: (
            decrypt_member(client, key_id, ciphertexts[reference.removeprefix(PRINCIPAL_REFERENCE_PREFIX)]).removeprefix(
                "user:"
            )
        )
        for reference in identities
        if reference.startswith(PRINCIPAL_REFERENCE_PREFIX)
    }
    if config.auth is not None:
        config.auth.user_roles = {emails.get(user, user): role for user, role in config.auth.user_roles.items()}
        config.auth.admin_users = [emails.get(user, user) for user in config.auth.admin_users]
        config.auth.allowed_submitters = [emails.get(user, user) for user in config.auth.allowed_submitters]
    for tier in config.user_budgets:
        tier.user_ids = [emails.get(user, user) for user in tier.user_ids]
    rendered = yaml.safe_dump(config_to_dict(config), sort_keys=False)
    output_path.touch(mode=0o600, exist_ok=False)
    output_path.write_text(rendered)


def decrypt_diff() -> None:
    """Annotate changed YAML or Python principal references without printing unchanged people."""
    lines = sys.stdin.readlines()
    config = load_iam_config()
    current_ciphertexts = {principal.principal_id: principal.ciphertext for principal in config.principals}
    changed_records: dict[tuple[str, str], str] = {}
    changed_references: list[tuple[str, str]] = []
    for line in lines:
        if not line.startswith(("+", "-")) or line.startswith(("+++", "---")):
            continue
        if match := _PRINCIPAL_RECORD_RE.match(line):
            changed_records[(line[0], match.group(1))] = match.group(2)
            continue
        for reference_pattern in _PRINCIPAL_REFERENCE_PATTERNS:
            matches = tuple(reference_pattern.finditer(line))
            if matches:
                changed_references.extend((line[0], match.group(1)) for match in matches)
                break

    client = kms_v1.KeyManagementServiceClient()
    key_id = crypto_key_id(config)
    printed = 0
    referenced_ids = {principal_id for _, principal_id in changed_references}
    for marker, principal_id in changed_references:
        ciphertext = changed_records.get((marker, principal_id))
        if ciphertext is None:
            ciphertext = current_ciphertexts.get(principal_id)
        if ciphertext is None:
            opposite_marker = "-" if marker == "+" else "+"
            ciphertext = changed_records.get((opposite_marker, principal_id))
        if ciphertext is None:
            raise SystemExit(f"cannot resolve changed principal {principal_id}")
        print(f"{marker} {decrypt_member(client, key_id, ciphertext)}")
        printed += 1

    for (marker, principal_id), ciphertext in changed_records.items():
        if principal_id in referenced_ids:
            continue
        print(f"{marker} {decrypt_member(client, key_id, ciphertext)}")
        printed += 1

    if printed == 0:
        print("no changed IAM principals in the diff", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    grant_parser = subparsers.add_parser("grant", help="Grant one encrypted person one or more project roles.")
    grant_parser.add_argument("email")
    grant_parser.add_argument(
        "--project-role",
        dest="project_roles",
        action="append",
        required=True,
        help="Project role to grant; repeat for multiple roles.",
    )

    register_parser = subparsers.add_parser(
        "register", help="Register one person and print the reusable opaque principal ID."
    )
    register_parser.add_argument("email")

    decrypt_parser = subparsers.add_parser("decrypt", help="Decrypt ciphertexts or annotate a YAML diff.")
    decrypt_parser.add_argument("ciphertexts", nargs="*")
    decrypt_parser.add_argument("--diff", action="store_true", help="Read a unified YAML diff from stdin.")

    render_parser = subparsers.add_parser("render-iris", help="Resolve encrypted identities for an Iris deployment.")
    render_parser.add_argument("config", type=Path)
    render_parser.add_argument("--output", type=Path, required=True, help="New private file outside the checkout")

    args = parser.parse_args()
    if args.command == "grant":
        grant(args.email, tuple(args.project_roles))
    elif args.command == "register":
        register(args.email)
    elif args.command == "render-iris":
        render_iris(args.config, args.output)
    elif args.diff:
        if args.ciphertexts:
            parser.error("pass either --diff (stdin) or positional ciphertexts, not both")
        decrypt_diff()
    elif args.ciphertexts:
        decrypt_ciphertexts(args.ciphertexts)
    else:
        parser.error("decrypt needs ciphertext arguments or --diff")


if __name__ == "__main__":
    main()
