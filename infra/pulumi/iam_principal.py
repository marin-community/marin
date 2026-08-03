#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Encrypt or decrypt one `user:<email>` principal against the marin-iac KMS key.

Adding a grant needs the ciphertext for a new person's email; reviewing a grant PR needs the
plaintext behind the encrypted members in the diff. This does both, one principal at a time:

    # Adding a grant: turn a new person's email into the snippet to paste into iam_data.py.
    uv run --package marin-iac --extra deploy python infra/pulumi/iam_principal.py encrypt alice@openathena.ai

    # Reviewing a grant PR: reveal the real email(s) behind changed GcpEncryptedMember lines.
    git diff origin/main...HEAD -- infra/pulumi/src/iac/gcp/iam_data.py \
      | uv run --package marin-iac --extra deploy python infra/pulumi/iam_principal.py decrypt --diff

`decrypt` also accepts bare ciphertext strings as positional arguments. Both operations need
`roles/cloudkms.cryptoKeyEncrypterDecrypter` (encrypt) or `.cryptoKeyDecrypter` (decrypt) on
the key — the same access `pulumi up`/`preview` already requires (see infra/pulumi/README.md).

GCP KMS encryption is non-deterministic: encrypting the same email twice yields different
ciphertext.
"""

import argparse
import re
import sys

from google.cloud import kms_v1
from iac.gcp.iam_kms import crypto_key_id, decrypt_member, encrypt_email

# A GcpEncryptedMember spans two lines in source (the wrapper and the `ciphertext="..."` arg),
# and a unified diff prefixes each line independently, so match the ciphertext token alone
# rather than the whole `GcpEncryptedMember(...)` wrapper, which a diff line can't carry.
_CIPHERTEXT_RE = re.compile(r'ciphertext="([^"]+)"')


def encrypt(emails: list[str]) -> None:
    """Print the GcpEncryptedMember snippet for each new email, ready to paste into iam_data.py."""
    client = kms_v1.KeyManagementServiceClient()
    key_id = crypto_key_id()
    for email in emails:
        plaintext = email.removeprefix("user:").strip()
        assert "@" in plaintext, f"expected an email, got {email!r}"
        ciphertext = encrypt_email(client, key_id, plaintext)
        print(f'GcpEncryptedMember(ciphertext="{ciphertext}"),')


def decrypt_ciphertexts(ciphertexts: list[str]) -> None:
    client = kms_v1.KeyManagementServiceClient()
    key_id = crypto_key_id()
    for ciphertext in ciphertexts:
        print(decrypt_member(client, key_id, ciphertext))


def decrypt_diff() -> None:
    """Annotate a unified diff on stdin: for every added/removed line that carries a
    GcpEncryptedMember ciphertext, print the +/- marker and the decrypted principal, so a
    reviewer sees the real grant instead of opaque ciphertext."""
    client = kms_v1.KeyManagementServiceClient()
    key_id = crypto_key_id()
    printed = 0
    for line in sys.stdin:
        if not (line.startswith("+") or line.startswith("-")) or line.startswith(("+++", "---")):
            continue
        match = _CIPHERTEXT_RE.search(line)
        if match is None:
            continue
        principal = decrypt_member(client, key_id, match.group(1))
        print(f"{line[0]} {principal}")
        printed += 1
    if printed == 0:
        print("no changed GcpEncryptedMember lines in the diff", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    encrypt_parser = subparsers.add_parser("encrypt", help="Encrypt new email(s) into GcpEncryptedMember snippets.")
    encrypt_parser.add_argument("emails", nargs="+")

    decrypt_parser = subparsers.add_parser("decrypt", help="Decrypt ciphertext(s), or annotate a diff with --diff.")
    decrypt_parser.add_argument("ciphertexts", nargs="*")
    decrypt_parser.add_argument("--diff", action="store_true", help="Read a unified diff from stdin and annotate it.")

    args = parser.parse_args()
    if args.command == "encrypt":
        encrypt(args.emails)
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
