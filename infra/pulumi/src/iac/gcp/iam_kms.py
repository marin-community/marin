# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared KMS helpers for the IAM YAML principal tooling (`iam_audit.py`,
`iam_principal.py`). Both encrypt/decrypt `user:<email>` principals against the marin-iac key;
this is the one place the project, key path, and base64 framing live."""

import base64

from google.cloud import kms_v1

from iac.gcp.iam_config import load_iam_config

# The YAML names only the key ring/key, not the project the key lives under. That
# project matches lib/iris/config/marin.yaml's provisioning.gcp.project.
PROJECT = "hai-gcp-models"


def crypto_key_id() -> str:
    config = load_iam_config()
    return (
        f"projects/{PROJECT}/locations/{config.kms_location}/keyRings/{config.kms_key_ring}/"
        f"cryptoKeys/{config.kms_key}"
    )


def encrypt_email(client: kms_v1.KeyManagementServiceClient, crypto_key_id: str, email: str) -> str:
    """Encrypt a bare `<email>` (no `user:` prefix) to base64 ciphertext for a GcpEncryptedMember."""
    response = client.encrypt(name=crypto_key_id, plaintext=email.encode("utf-8"))
    return base64.b64encode(response.ciphertext).decode("ascii")


def decrypt_member(client: kms_v1.KeyManagementServiceClient, crypto_key_id: str, ciphertext: str) -> str:
    """Decrypt base64 ciphertext back to a `user:<email>` principal string."""
    response = client.decrypt(name=crypto_key_id, ciphertext=base64.b64decode(ciphertext))
    return f"user:{response.plaintext.decode('utf-8')}"
