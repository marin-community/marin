# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read and verify the sealed Micro210 evaluation protocol."""

import hashlib
import json
from dataclasses import dataclass

from rigging.filesystem.storage_path import StoragePath

LEGACY_POLICY_DIGEST = "e0faa32c68994e8bf01d0ab7177893aa9c29f8c4309b9183de5c3dfcb6fa67aa"
LEGACY_POLICY_REVISION = "2026.11.18"
LEGACY_POLICY_COMMIT = "6ae1211cf3aa80beb0ca364b171f149a9506c05a"

POLICY_REPO = "open-athena/micro210-policy"
POLICY_DIGEST = "9d850b9f05c13c7a56df4c0bd9db23e7c805c505018cd330b175519ddca2057e"
POLICY_VERSION = "2026.11.23"
POLICY_REVISION = f"{POLICY_VERSION}-together"
POLICY_COMMIT = "7ab57cf5e33cacc98440bfcc2dfd452b94c44344"


@dataclass(frozen=True)
class Micro210PolicyReference:
    uri: str
    revision: str
    sha256: str
    byte_size: int


# Human-readable copy (sha256: e0faa32c68994e8bf01d0ab7177893aa9c29f8c4309b9183de5c3dfcb6fa67aa): https://gist.github.com/penfever/b9b7be780c14969634e897621454deef/0b5add0ef2843418eaded61258a91f51f3b1b69a
LEGACY_POLICY_REFERENCE = Micro210PolicyReference(
    uri=f"hf://datasets/{POLICY_REPO}@{LEGACY_POLICY_COMMIT}/policy.json",
    revision=LEGACY_POLICY_REVISION,
    sha256=LEGACY_POLICY_DIGEST,
    byte_size=122243,
)


POLICY_REFERENCE = Micro210PolicyReference(
    uri=f"hf://datasets/{POLICY_REPO}@{POLICY_COMMIT}/policy.json",
    revision=POLICY_REVISION,
    sha256=POLICY_DIGEST,
    byte_size=131430,
)


# The prepared table re-hosts benchmark items that must not be public, so it lives in a
# private repository that CoreWeave tasks read with the standard HF_TOKEN.
PREPARED_REPO = "open-athena/micro210-prepared-tasks"
PREPARED_COMMIT = "2d33d7be129e6a93bb4279af8a4998a6915771a9"


def read_micro210_policy(reference: Micro210PolicyReference) -> dict:
    """Read and verify a policy from a current or historical evaluation record."""
    raw = StoragePath(reference.uri).read_bytes()
    if len(raw) != reference.byte_size or hashlib.sha256(raw).hexdigest() != reference.sha256:
        raise ValueError("Canonical Micro210 specification hash mismatch")
    return json.loads(raw)
